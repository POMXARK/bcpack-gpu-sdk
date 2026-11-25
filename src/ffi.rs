use crate::gpu_decoder::{
    cmd_barriers, create_committed_buffer, pack_batch, pack_bcn_batch, transition_barrier,
    Bc6hFullGpuDecoder, Bc7FullGpuDecoder, Bc7RansGpuDecoder, BcnGpuDecoder, BcpackGpuDecoder,
    Gfx10DetileGpu, GpuContext,
};
use bcpack::bc1::{QuadTable, Window};
use bcpack::format::Mode as BcpMode;
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::copy_nonoverlapping;
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC3_UNORM, DXGI_FORMAT_BC4_UNORM,
    DXGI_FORMAT_BC5_UNORM, DXGI_FORMAT_BC6H_UF16, DXGI_FORMAT_BC7_UNORM,
};
use windows::Win32::System::Threading::{WaitForSingleObject, INFINITE};

pub const BCPACK_FORMAT_BC1: u32 = 0;
pub const BCPACK_FORMAT_BC7_RANS: u32 = 1;
pub const BCPACK_FORMAT_BC3: u32 = 2;
pub const BCPACK_FORMAT_BC4: u32 = 3;
pub const BCPACK_FORMAT_BC5: u32 = 4;
pub const BCPACK_FORMAT_BC7: u32 = 5;
pub const BCPACK_FORMAT_BC6H: u32 = 6;

pub const BCPACK_SWIZZLE_LINEAR: u32 = 0;
pub const BCPACK_SWIZZLE_GFX10_64KB_D: u32 = 1;
pub const BCPACK_SWIZZLE_GFX10_4KB_S: u32 = 2;

pub struct BcPackContext {
    ctx: GpuContext,
    bc1_dec: BcpackGpuDecoder,
    bc7_dec: Bc7RansGpuDecoder,
    bcn_dec: BcnGpuDecoder,
    bc7_full: Bc7FullGpuDecoder,
    bc6h_full: Bc6hFullGpuDecoder,
    bc1_use_full_bcn: bool,
    detile: Gfx10DetileGpu,
    detile_out_default: ID3D12Resource,
    detile_readback: ID3D12Resource,
    row_pad_default: ID3D12Resource,
    detile_alloc: ID3D12CommandAllocator,
    detile_cmd: ID3D12GraphicsCommandList,
    tex_copy_alloc: ID3D12CommandAllocator,
    tex_copy_cmd: ID3D12GraphicsCommandList,
    max_output_bytes: usize,
    last_kernel_ms: f64,
}

#[inline]
fn format_layout(format: u32) -> Option<(usize, DXGI_FORMAT)> {
    match format {
        BCPACK_FORMAT_BC1 => Some((8, DXGI_FORMAT_BC1_UNORM)),
        BCPACK_FORMAT_BC3 => Some((16, DXGI_FORMAT_BC3_UNORM)),
        BCPACK_FORMAT_BC4 => Some((8, DXGI_FORMAT_BC4_UNORM)),
        BCPACK_FORMAT_BC5 => Some((16, DXGI_FORMAT_BC5_UNORM)),
        BCPACK_FORMAT_BC7 => Some((16, DXGI_FORMAT_BC7_UNORM)),
        BCPACK_FORMAT_BC6H => Some((16, DXGI_FORMAT_BC6H_UF16)),
        _ => None,
    }
}

#[inline]
fn detile_mode_for(block_bytes: usize, swizzle_mode: u32) -> Result<u32, i32> {
    match (block_bytes, swizzle_mode) {
        (8, BCPACK_SWIZZLE_GFX10_64KB_D) => Ok(0),
        (8, BCPACK_SWIZZLE_GFX10_4KB_S) => Ok(1),
        (16, BCPACK_SWIZZLE_GFX10_64KB_D) => Ok(2),
        _ => Err(-6),
    }
}

#[inline]
fn bcn_mode_from_format(format: u32) -> Option<BcpMode> {
    match format {
        BCPACK_FORMAT_BC1 => Some(BcpMode::Bc1),
        BCPACK_FORMAT_BC3 => Some(BcpMode::Bc3),
        BCPACK_FORMAT_BC4 => Some(BcpMode::Bc4),
        BCPACK_FORMAT_BC5 => Some(BcpMode::Bc5),
        _ => None,
    }
}

unsafe fn init_context_with_gpu(
    mut ctx: GpuContext,
    max_chunks: u32,
    max_blocks: u32,
    max_stream_bytes: u32,
) -> *mut BcPackContext {
    let chunks = if max_chunks == 0 { 2048 } else { max_chunks as usize };
    let blocks = if max_blocks == 0 { 8_388_608 } else { max_blocks as usize };
    let stream_cap = if max_stream_bytes == 0 {
        96 * 1024 * 1024
    } else {
        max_stream_bytes as usize
    };

    let window = Window::default_table();
    let quads = QuadTable::default_table();
    let bc1_dec = BcpackGpuDecoder::new(&mut ctx, &window, &quads, chunks, blocks, stream_cap);
    let bc7_sym_rows = (blocks / 2).max(4096);
    let out_bytes = (blocks * 8).max(bc7_sym_rows * 32).max(16 * 1024 * 1024);
    let bc7_dec = Bc7RansGpuDecoder::new(&mut ctx, chunks, bc7_sym_rows, stream_cap);
    let bcn_dec = BcnGpuDecoder::new(&mut ctx, chunks, bc7_sym_rows, out_bytes, stream_cap);
    let bc7_full = Bc7FullGpuDecoder::new(&mut ctx, chunks, (out_bytes / 16).max(4096), stream_cap);
    let bc6h_full = Bc6hFullGpuDecoder::new(&mut ctx, chunks, (out_bytes / 16).max(4096), stream_cap);
    let detile = Gfx10DetileGpu::new(&ctx, 16 * 1024 * 1024);
    let detile_out_default = create_committed_buffer(
        &ctx.device,
        D3D12_HEAP_TYPE_DEFAULT,
        out_bytes as u64,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
    );
    let detile_readback = create_committed_buffer(
        &ctx.device,
        D3D12_HEAP_TYPE_READBACK,
        out_bytes as u64,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    let row_pad_default = create_committed_buffer(
        &ctx.device,
        D3D12_HEAP_TYPE_DEFAULT,
        16 * 1024 * 1024,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COMMON,
    );
    let detile_alloc: ID3D12CommandAllocator = ctx
        .device
        .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)
        .expect("CreateCommandAllocator FFI COMPUTE");
    let detile_cmd: ID3D12GraphicsCommandList = ctx
        .device
        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &detile_alloc, None)
        .expect("CreateCommandList FFI COMPUTE");
    detile_cmd.Close().unwrap();

    let tex_copy_alloc: ID3D12CommandAllocator = ctx
        .device
        .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
        .expect("CreateCommandAllocator FFI DIRECT");
    let tex_copy_cmd: ID3D12GraphicsCommandList = ctx
        .device
        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &tex_copy_alloc, None)
        .expect("CreateCommandList FFI DIRECT");
    tex_copy_cmd.Close().unwrap();

    Box::into_raw(Box::new(BcPackContext {
        ctx,
        bc1_dec,
        bc7_dec,
        bcn_dec,
        bc7_full,
        bc6h_full,
        bc1_use_full_bcn: false,
        detile,
        detile_out_default,
        detile_readback,
        row_pad_default,
        detile_alloc,
        detile_cmd,
        tex_copy_alloc,
        tex_copy_cmd,
        max_output_bytes: out_bytes,
        last_kernel_ms: 0.0,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn bcpack_gpu_create(
    max_chunks: u32,
    max_blocks: u32,
    max_stream_bytes: u32,
) -> *mut BcPackContext {
    let res = catch_unwind(|| {
        let ctx = GpuContext::new(false);
        init_context_with_gpu(ctx, max_chunks, max_blocks, max_stream_bytes)
    });
    res.unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn bcpack_gpu_create_from_device(
    d3d12_device: *mut c_void,
    d3d12_compute_queue: *mut c_void,
    max_chunks: u32,
    max_blocks: u32,
    max_stream_bytes: u32,
) -> *mut BcPackContext {
    if d3d12_device.is_null() {
        return std::ptr::null_mut();
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let dev: ID3D12Device = (*ManuallyDrop::new(ID3D12Device::from_raw(d3d12_device))).clone();
        let cq = if d3d12_compute_queue.is_null() {
            None
        } else {
            let q: ID3D12CommandQueue = (*ManuallyDrop::new(ID3D12CommandQueue::from_raw(d3d12_compute_queue))).clone();
            Some(q)
        };
        let ctx = GpuContext::from_external_device(dev, cq);
        init_context_with_gpu(ctx, max_chunks, max_blocks, max_stream_bytes)
    }));
    res.unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn bcpack_gpu_destroy(ctx: *mut BcPackContext) {
    if !ctx.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            let state = Box::from_raw(ctx);
            if state.ctx.compute_fence_value > 0 {
                let _ = state.ctx.wait_compute_cpu(state.ctx.compute_fence_value, INFINITE);
            }
            if state.ctx.fence_value > 0 {
                state.ctx.wait_direct_cpu(state.ctx.fence_value);
            }
            drop(state);
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn bcpack_gpu_set_bc1_tables(
    ctx: *mut BcPackContext,
    rank1_256: *const u8,
    dict_ep_255x6: *const u8,
    dict_ix_255x16: *const u8,
) -> i32 {
    if ctx.is_null() || rank1_256.is_null() || dict_ep_255x6.is_null() || dict_ix_255x16.is_null() {
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| -> i32 {
        let state = &mut *ctx;
        if state.ctx.compute_fence_value > 0 {
            let _ = state.ctx.wait_compute_cpu(state.ctx.compute_fence_value, INFINITE);
        }
        let r1: &[u8; 256] = &*(rank1_256 as *const [u8; 256]);
        let ep = std::slice::from_raw_parts(dict_ep_255x6, 255 * 6);
        let ix = std::slice::from_raw_parts(dict_ix_255x16, 255 * 16);
        state.bcn_dec.update_bc1_tables(&mut state.ctx, r1, ep, ix);
        state.bc1_use_full_bcn = true;
        0
    }));
    res.unwrap_or(-99)
}

#[no_mangle]
pub unsafe extern "C" fn bcpack_gpu_set_bc7_tables(
    ctx: *mut BcPackContext,
    ctrl7_512: *const u16,
    dict_subsets_255: *const u8,
    dict_ep8_255x24: *const u8,
    dict_ix4_255x32: *const u8,
) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| -> i32 {
        let state = &mut *ctx;
        if state.ctx.compute_fence_value > 0 {
            let _ = state.ctx.wait_compute_cpu(state.ctx.compute_fence_value, INFINITE);
        }
        let ctls = if ctrl7_512.is_null() {
            None
        } else {
            Some(&*(ctrl7_512 as *const [u16; 512]))
        };
        let subs = if dict_subsets_255.is_null() {
            None
        } else {
            Some(&*(dict_subsets_255 as *const [u8; 255]))
        };
        let ep8 = if dict_ep8_255x24.is_null() {
            None
        } else {
            Some(std::slice::from_raw_parts(dict_ep8_255x24, 255 * 24))
        };
        let ix4 = if dict_ix4_255x32.is_null() {
            None
        } else {
            Some(std::slice::from_raw_parts(dict_ix4_255x32, 255 * 32))
        };
        state.bc7_full.update_tables(&mut state.ctx, ctls, subs, ep8, ix4);
        0
    }));
    res.unwrap_or(-99)
}

#[no_mangle]
pub unsafe extern "C" fn bcpack_gpu_set_bc6h_tables(
    ctx: *mut BcPackContext,
    dict_ep16_255x12: *const u16,
    dict_ix_mode_255: *const u8,
    dict_ix16_255x16: *const i16,
) -> i32 {
    if ctx.is_null()
        || dict_ep16_255x12.is_null()
        || dict_ix_mode_255.is_null()
        || dict_ix16_255x16.is_null()
    {
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| -> i32 {
        let state = &mut *ctx;
        if state.ctx.compute_fence_value > 0 {
            let _ = state.ctx.wait_compute_cpu(state.ctx.compute_fence_value, INFINITE);
        }
        let ep16 = std::slice::from_raw_parts(dict_ep16_255x12, 255 * 12);
        let ix_mode: &[u8; 255] = &*(dict_ix_mode_255 as *const [u8; 255]);
        let ix16 = std::slice::from_raw_parts(dict_ix16_255x16, 255 * 16);
        state.bc6h_full.update_tables(&mut state.ctx, ep16, ix_mode, ix16);
        0
    }));
    res.unwrap_or(-99)
}

#[no_mangle]
pub unsafe extern "C" fn BcPackGetComputeFence(ctx: *const BcPackContext) -> *mut c_void {
    if ctx.is_null() {
        return std::ptr::null_mut();
    }
    (*ctx).ctx.compute_fence.as_raw()
}

#[no_mangle]
pub unsafe extern "C" fn BcPackGetDirectFence(ctx: *const BcPackContext) -> *mut c_void {
    if ctx.is_null() {
        return std::ptr::null_mut();
    }
    (*ctx).ctx.fence.as_raw()
}

#[no_mangle]
pub unsafe extern "C" fn BcPackWaitDirectFence(
    ctx: *mut BcPackContext,
    fence_value: u64,
    timeout_ms: u32,
) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let state = &*ctx;
    if fence_value == 0 || state.ctx.fence.GetCompletedValue() >= fence_value {
        return 0;
    }
    if state
        .ctx
        .fence
        .SetEventOnCompletion(fence_value, state.ctx.fence_event)
        .is_err()
    {
        return -2;
    }
    let ms = if timeout_ms == u32::MAX { INFINITE } else { timeout_ms };
    if WaitForSingleObject(state.ctx.fence_event, ms).0 == 0 {
        0
    } else {
        1
    }
}

/// Runs Async Compute decode (+ optional Gfx10 detile in FIFO compute order) leaving linear blocks in
/// `DEFAULT` heap `COMMON` state; returns `(linear_src_res, block_bytes, dxgi_fmt, expected_bytes, compute_fence_val)`.
unsafe fn submit_format_compute_to_internal(
    state: &mut BcPackContext,
    stream: &[u8],
    num_blocks: u32,
    format: u32,
    swizzle_mode: u32,
    width_blocks: u32,
    height_blocks: u32,
) -> Result<(ID3D12Resource, usize, DXGI_FORMAT, usize, u64), i32> {
    let (block_bytes, dxgi_fmt) = format_layout(format).ok_or(-4)?;
    let need_detile = swizzle_mode != BCPACK_SWIZZLE_LINEAR;
    let linear_blocks = if need_detile {
        (width_blocks as usize) * (height_blocks as usize)
    } else {
        num_blocks as usize
    };
    let expected_bytes = linear_blocks * block_bytes;
    if expected_bytes > state.max_output_bytes {
        return Err(-5);
    }

    let (decoded_res, mut compute_val) = if format == BCPACK_FORMAT_BC1 && !state.bc1_use_full_bcn {
        let batch = pack_batch(std::iter::once((stream, num_blocks as usize)));
        let fv = state.bc1_dec.submit_async_compute(&mut state.ctx, &batch);
        state.last_kernel_ms = 0.0;
        (state.bc1_dec.slots[0].out_blocks_default.clone(), fv)
    } else if format == BCPACK_FORMAT_BC7 {
        let sym_rows = (num_blocks as usize) * 2;
        let batch = pack_batch(std::iter::once((stream, sym_rows)));
        let fv = state.bc7_full.submit_async_compute(&mut state.ctx, &batch);
        (state.bc7_full.out_blocks_default.clone(), fv)
    } else if format == BCPACK_FORMAT_BC6H {
        let sym_rows = (num_blocks as usize) * 2;
        let batch = pack_batch(std::iter::once((stream, sym_rows)));
        let fv = state.bc6h_full.submit_async_compute(&mut state.ctx, &batch);
        (state.bc6h_full.out_blocks_default.clone(), fv)
    } else {
        let m = bcn_mode_from_format(format).ok_or(-4)?;
        let batch = pack_bcn_batch(std::iter::once((stream, m, num_blocks as usize)));
        let fv = state.bcn_dec.submit_async_compute(&mut state.ctx, &batch);
        (state.bcn_dec.out_blocks_default.clone(), fv)
    };

    if !need_detile {
        return Ok((decoded_res, block_bytes, dxgi_fmt, expected_bytes, compute_val));
    }

    let tile_mode = detile_mode_for(block_bytes, swizzle_mode)?;
    state.detile_alloc.Reset().unwrap();
    state.detile_cmd.Reset(&state.detile_alloc, None).unwrap();
    cmd_barriers(
        &state.detile_cmd,
        vec![
            transition_barrier(&decoded_res, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition_barrier(&state.detile_out_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ],
    );
    state.detile.record_detile(
        &state.detile_cmd,
        decoded_res.GetGPUVirtualAddress(),
        state.detile_out_default.GetGPUVirtualAddress(),
        width_blocks,
        height_blocks,
        tile_mode,
        block_bytes as u32,
        0,
        0,
        num_blocks,
    );
    cmd_barriers(
        &state.detile_cmd,
        vec![
            transition_barrier(&decoded_res, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON),
            transition_barrier(&state.detile_out_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
        ],
    );
    state.detile_cmd.Close().unwrap();
    let lists = [Some(state.detile_cmd.cast::<ID3D12CommandList>().unwrap())];
    state.ctx.compute_queue.ExecuteCommandLists(&lists);
    compute_val = state.ctx.signal_compute();

    Ok((
        state.detile_out_default.clone(),
        block_bytes,
        dxgi_fmt,
        expected_bytes,
        compute_val,
    ))
}

#[no_mangle]
pub unsafe extern "C" fn BcPackDecodeAsync(
    ctx: *mut BcPackContext,
    compressed_data: *const u8,
    compressed_size: u32,
    num_blocks: u32,
    format: u32,
    swizzle_mode: u32,
    width_blocks: u32,
    height_blocks: u32,
    out_host_buffer: *mut u8,
    out_capacity: u32,
    out_fence_value: *mut u64,
) -> i32 {
    if ctx.is_null() || compressed_data.is_null() || compressed_size == 0 || num_blocks == 0 {
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| -> i32 {
        let state = &mut *ctx;
        let stream = std::slice::from_raw_parts(compressed_data, compressed_size as usize);

        if format == BCPACK_FORMAT_BC7_RANS {
            let sym_rows = (num_blocks as usize) * 2;
            let expected_bytes = sym_rows * 32;
            if !out_host_buffer.is_null() && (out_capacity as usize) < expected_bytes {
                return -2;
            }
            let batch = pack_batch(std::iter::once((stream, sym_rows)));
            if out_host_buffer.is_null() {
                let fence_val = state.bc7_dec.submit_async_compute(&mut state.ctx, &batch);
                if !out_fence_value.is_null() {
                    *out_fence_value = fence_val;
                }
                return 0;
            }
            let (decoded_u16, statuses, kms, _) = state.bc7_dec.decode_batch_and_readback(&mut state.ctx, &batch);
            if statuses.first().map(|s| s.code).unwrap_or(99) != 0 {
                return -3;
            }
            state.last_kernel_ms = kms;
            copy_nonoverlapping(decoded_u16.as_ptr() as *const u8, out_host_buffer, expected_bytes);
            if !out_fence_value.is_null() {
                *out_fence_value = state.ctx.compute_fence_value;
            }
            return 0;
        }

        let (block_bytes, _) = match format_layout(format) {
            Some(v) => v,
            None => return -4,
        };
        let need_detile = swizzle_mode != BCPACK_SWIZZLE_LINEAR;
        let linear_blocks = if need_detile {
            (width_blocks as usize) * (height_blocks as usize)
        } else {
            num_blocks as usize
        };
        let expected_bytes = linear_blocks * block_bytes;
        if expected_bytes > state.max_output_bytes {
            return -5;
        }
        if !out_host_buffer.is_null() && (out_capacity as usize) < expected_bytes {
            return -2;
        }

        if !need_detile && !out_host_buffer.is_null() {
            let (decoded, statuses, kms) = match format {
                BCPACK_FORMAT_BC1 if !state.bc1_use_full_bcn => {
                    let batch = pack_batch(std::iter::once((stream, num_blocks as usize)));
                    let (d, st, k, _) = state.bc1_dec.decode_batch_and_readback(&mut state.ctx, &batch);
                    (d, st, k)
                }
                BCPACK_FORMAT_BC7 => {
                    let batch = pack_batch(std::iter::once((stream, (num_blocks as usize) * 2)));
                    let (d, st, k, _) = state.bc7_full.decode_batch_and_readback(&mut state.ctx, &batch);
                    (d, st, k)
                }
                BCPACK_FORMAT_BC6H => {
                    let batch = pack_batch(std::iter::once((stream, (num_blocks as usize) * 2)));
                    let (d, st, k, _) = state.bc6h_full.decode_batch_and_readback(&mut state.ctx, &batch);
                    (d, st, k)
                }
                _ => {
                    let m = bcn_mode_from_format(format).unwrap();
                    let batch = pack_bcn_batch(std::iter::once((stream, m, num_blocks as usize)));
                    let (d, st, k, _) = state.bcn_dec.decode_batch_and_readback(&mut state.ctx, &batch);
                    (d, st, k)
                }
            };
            if statuses.first().map(|s| s.code).unwrap_or(99) != 0 {
                return -3;
            }
            state.last_kernel_ms = kms;
            copy_nonoverlapping(decoded.as_ptr(), out_host_buffer, expected_bytes);
            if !out_fence_value.is_null() {
                *out_fence_value = state.ctx.compute_fence_value;
            }
            return 0;
        }

        let (src_res, _, _, exp_b, fence_val) = match submit_format_compute_to_internal(
            state,
            stream,
            num_blocks,
            format,
            swizzle_mode,
            width_blocks,
            height_blocks,
        ) {
            Ok(v) => v,
            Err(code) => return code,
        };

        if out_host_buffer.is_null() {
            if !out_fence_value.is_null() {
                *out_fence_value = fence_val;
            }
            return 0;
        }

        state.detile_alloc.Reset().unwrap();
        state.detile_cmd.Reset(&state.detile_alloc, None).unwrap();
        cmd_barriers(
            &state.detile_cmd,
            vec![transition_barrier(&src_res, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)],
        );
        state.detile_cmd.CopyBufferRegion(&state.detile_readback, 0, &src_res, 0, exp_b as u64);
        cmd_barriers(
            &state.detile_cmd,
            vec![transition_barrier(&src_res, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)],
        );
        state.detile_cmd.Close().unwrap();
        let lists = [Some(state.detile_cmd.cast::<ID3D12CommandList>().unwrap())];
        state.ctx.compute_queue.ExecuteCommandLists(&lists);
        state.ctx.signal_and_wait_compute();

        let mut ptr: *mut c_void = std::ptr::null_mut();
        state.detile_readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: exp_b }), Some(&mut ptr)).unwrap();
        copy_nonoverlapping(ptr as *const u8, out_host_buffer, exp_b);
        state.detile_readback.Unmap(0, None);

        state.last_kernel_ms = match format {
            BCPACK_FORMAT_BC1 if !state.bc1_use_full_bcn => state.bc1_dec.read_last_kernel_ms(&state.ctx),
            BCPACK_FORMAT_BC7 => state.bc7_full.read_last_kernel_ms(&state.ctx),
            BCPACK_FORMAT_BC6H => state.bc6h_full.read_last_kernel_ms(&state.ctx),
            _ => state.bcn_dec.read_last_kernel_ms(&state.ctx),
        };
        if !out_fence_value.is_null() {
            *out_fence_value = state.ctx.compute_fence_value;
        }
        0
    }));
    res.unwrap_or(-99)
}

#[no_mangle]
pub unsafe extern "C" fn BcPackDecodeToTexture2D(
    ctx: *mut BcPackContext,
    compressed_data: *const u8,
    compressed_size: u32,
    num_blocks: u32,
    format: u32,
    swizzle_mode: u32,
    width_blocks: u32,
    height_blocks: u32,
    dst_texture2d: *mut c_void,
    subresource_index: u32,
    dst_x_blocks: u32,
    dst_y_blocks: u32,
    out_fence_value: *mut u64,
) -> i32 {
    if ctx.is_null()
        || compressed_data.is_null()
        || compressed_size == 0
        || num_blocks == 0
        || width_blocks == 0
        || height_blocks == 0
        || dst_texture2d.is_null()
    {
        return -1;
    }
    if format_layout(format).is_none() {
        return -4;
    }
    let res = catch_unwind(AssertUnwindSafe(|| -> i32 {
        let state = &mut *ctx;
        if state.ctx.compute_fence_value > 0 {
            let _ = state.ctx.wait_compute_cpu(state.ctx.compute_fence_value, INFINITE);
        }
        if state.ctx.fence_value > 0 {
            state.ctx.wait_direct_cpu(state.ctx.fence_value);
        }
        let dst_tex: ID3D12Resource = (*ManuallyDrop::new(ID3D12Resource::from_raw(dst_texture2d))).clone();
        let stream = std::slice::from_raw_parts(compressed_data, compressed_size as usize);

        let (src_buf, block_bytes, dxgi_fmt, _, compute_val) = match submit_format_compute_to_internal(
            state,
            stream,
            num_blocks,
            format,
            swizzle_mode,
            width_blocks,
            height_blocks,
        ) {
            Ok(v) => v,
            Err(code) => return code,
        };

        // Pure GPU cross-queue Wait: DIRECT queue awaits Async Compute completion
        state
            .ctx
            .queue
            .Wait(&state.ctx.compute_fence, compute_val)
            .expect("DIRECT queue Wait compute_fence");

        let raw_row = width_blocks * (block_bytes as u32);
        let aligned_pitch = (raw_row + 255) & !255;
        let total_padded_bytes = (aligned_pitch as u64) * (height_blocks as u64);
        if total_padded_bytes > 16 * 1024 * 1024 {
            return -5;
        }

        state.tex_copy_alloc.Reset().unwrap();
        state.tex_copy_cmd.Reset(&state.tex_copy_alloc, None).unwrap();

        let copy_src_res = if aligned_pitch == raw_row {
            cmd_barriers(
                &state.tex_copy_cmd,
                vec![
                    transition_barrier(&src_buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    transition_barrier(&dst_tex, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
                ],
            );
            &src_buf
        } else {
            cmd_barriers(
                &state.tex_copy_cmd,
                vec![
                    transition_barrier(&src_buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    transition_barrier(&state.row_pad_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
                ],
            );
            for r in 0..height_blocks {
                state.tex_copy_cmd.CopyBufferRegion(
                    &state.row_pad_default,
                    (r as u64) * (aligned_pitch as u64),
                    &src_buf,
                    (r as u64) * (raw_row as u64),
                    raw_row as u64,
                );
            }
            cmd_barriers(
                &state.tex_copy_cmd,
                vec![
                    transition_barrier(&src_buf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                    transition_barrier(&state.row_pad_default, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    transition_barrier(&dst_tex, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
                ],
            );
            &state.row_pad_default
        };

        let mut dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(dst_tex.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: subresource_index,
            },
        };
        let mut src_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(copy_src_res.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: dxgi_fmt,
                        Width: width_blocks * 4,
                        Height: height_blocks * 4,
                        Depth: 1,
                        RowPitch: aligned_pitch,
                    },
                },
            },
        };
        state.tex_copy_cmd.CopyTextureRegion(
            &dst_loc,
            dst_x_blocks * 4,
            dst_y_blocks * 4,
            0,
            &src_loc,
            None,
        );
        let _ = ManuallyDrop::take(&mut dst_loc.pResource);
        let _ = ManuallyDrop::take(&mut src_loc.pResource);

        if aligned_pitch == raw_row {
            cmd_barriers(
                &state.tex_copy_cmd,
                vec![
                    transition_barrier(&src_buf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                    transition_barrier(&dst_tex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON),
                ],
            );
        } else {
            cmd_barriers(
                &state.tex_copy_cmd,
                vec![
                    transition_barrier(&state.row_pad_default, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                    transition_barrier(&dst_tex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON),
                ],
            );
        }

        state.tex_copy_cmd.Close().unwrap();
        let lists = [Some(state.tex_copy_cmd.cast::<ID3D12CommandList>().unwrap())];
        state.ctx.queue.ExecuteCommandLists(&lists);
        let direct_val = state.ctx.signal_direct_async();
        if !out_fence_value.is_null() {
            *out_fence_value = direct_val;
        }
        0
    }));
    res.unwrap_or(-99)
}

#[no_mangle]
pub unsafe extern "C" fn BcPackDecodeToBuffer(
    ctx: *mut BcPackContext,
    compressed_data: *const u8,
    compressed_size: u32,
    num_blocks: u32,
    format: u32,
    swizzle_mode: u32,
    width_blocks: u32,
    height_blocks: u32,
    dst_d3d12_resource: *mut c_void,
    dst_byte_offset: u64,
    out_fence_value: *mut u64,
) -> i32 {
    if ctx.is_null()
        || compressed_data.is_null()
        || compressed_size == 0
        || num_blocks == 0
        || dst_d3d12_resource.is_null()
    {
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| -> i32 {
        let state = &mut *ctx;
        if state.ctx.compute_fence_value > 0 {
            let _ = state.ctx.wait_compute_cpu(state.ctx.compute_fence_value, INFINITE);
        }
        let dst_res: ID3D12Resource = (*ManuallyDrop::new(ID3D12Resource::from_raw(dst_d3d12_resource))).clone();
        let stream = std::slice::from_raw_parts(compressed_data, compressed_size as usize);

        if format == BCPACK_FORMAT_BC7_RANS {
            let sym_rows = (num_blocks as usize) * 2;
            let expected_bytes = sym_rows * 32;
            if expected_bytes > state.max_output_bytes {
                return -5;
            }
            let batch = pack_batch(std::iter::once((stream, sym_rows)));
            let fence_val = state.bc7_dec.submit_async_to_buffer(&mut state.ctx, &batch, &dst_res, dst_byte_offset);
            if !out_fence_value.is_null() {
                *out_fence_value = fence_val;
            }
            return 0;
        }

        let (src_res, _, _, expected_bytes, _) = match submit_format_compute_to_internal(
            state,
            stream,
            num_blocks,
            format,
            swizzle_mode,
            width_blocks,
            height_blocks,
        ) {
            Ok(v) => v,
            Err(code) => return code,
        };

        state.detile_alloc.Reset().unwrap();
        state.detile_cmd.Reset(&state.detile_alloc, None).unwrap();
        cmd_barriers(
            &state.detile_cmd,
            vec![
                transition_barrier(&src_res, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                transition_barrier(&dst_res, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
            ],
        );
        state.detile_cmd.CopyBufferRegion(&dst_res, dst_byte_offset, &src_res, 0, expected_bytes as u64);
        cmd_barriers(
            &state.detile_cmd,
            vec![
                transition_barrier(&src_res, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                transition_barrier(&dst_res, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON),
            ],
        );
        state.detile_cmd.Close().unwrap();
        let lists = [Some(state.detile_cmd.cast::<ID3D12CommandList>().unwrap())];
        state.ctx.compute_queue.ExecuteCommandLists(&lists);
        let fence_val = state.ctx.signal_compute();
        if !out_fence_value.is_null() {
            *out_fence_value = fence_val;
        }
        0
    }));
    res.unwrap_or(-99)
}

#[no_mangle]
pub unsafe extern "C" fn BcPackWaitFence(
    ctx: *mut BcPackContext,
    fence_value: u64,
    timeout_ms: u32,
) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let state = &*ctx;
    let ms = if timeout_ms == u32::MAX { INFINITE } else { timeout_ms };
    if state.ctx.wait_compute_cpu(fence_value, ms) {
        0
    } else {
        1
    }
}

#[no_mangle]
pub unsafe extern "C" fn BcPackGetLastTimingMs(
    ctx: *const BcPackContext,
    out_kernel_ms: *mut f64,
) -> i32 {
    if ctx.is_null() || out_kernel_ms.is_null() {
        return -1;
    }
    *out_kernel_ms = (*ctx).last_kernel_ms;
    0
}
