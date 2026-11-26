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
    max_chunks: usize,
    max_blocks: usize,
    max_bcn_sym_rows: usize,
    max_bc7_blocks: usize,
    max_bc6h_blocks: usize,
    max_stream_bytes: usize,
    max_output_bytes: usize,
    max_row_pad_bytes: usize,
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
        out_bytes as u64,
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
        max_chunks: chunks,
        max_blocks: blocks,
        max_bcn_sym_rows: bc7_sym_rows,
        max_bc7_blocks: (out_bytes / 16).max(4096),
        max_bc6h_blocks: (out_bytes / 16).max(4096),
        max_stream_bytes: stream_cap,
        max_output_bytes: out_bytes,
        max_row_pad_bytes: out_bytes,
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
    let (chk_block_bytes, _) = match format_layout(format) {
        Some(v) => v,
        None => return -4,
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> i32 {
        let state = &mut *ctx;
        let raw_row = match width_blocks.checked_mul(chk_block_bytes as u32) {
            Some(v) => v,
            None => return -5,
        };
        let aligned_pitch = match raw_row.checked_add(255) {
            Some(v) => v & !255,
            None => return -5,
        };
        let total_padded_bytes = (aligned_pitch as u64) * (height_blocks as u64);
        if aligned_pitch != raw_row && total_padded_bytes > (state.max_row_pad_bytes as u64) {
            return -5;
        }

        if state.ctx.compute_fence_value > 0 {
            let _ = state.ctx.wait_compute_cpu(state.ctx.compute_fence_value, INFINITE);
        }
        if state.ctx.fence_value > 0 {
            state.ctx.wait_direct_cpu(state.ctx.fence_value);
        }
        let dst_tex: ID3D12Resource = (*ManuallyDrop::new(ID3D12Resource::from_raw(dst_texture2d))).clone();
        let stream = std::slice::from_raw_parts(compressed_data, compressed_size as usize);

        let (src_buf, _, dxgi_fmt, _, compute_val) = match submit_format_compute_to_internal(
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

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BcPackBatchItem {
    pub compressed_data: *const u8,
    pub compressed_size: u32,
    pub num_blocks: u32,
    pub format: u32,
    pub swizzle_mode: u32,
    pub width_blocks: u32,
    pub height_blocks: u32,
    pub out_host_buffer: *mut u8,
    pub out_capacity: u32,
    pub status: i32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BatchFamily {
    Bc1Fast,
    Bcn,
    Bc7,
    Bc6h,
}

#[inline]
fn align4(len: usize) -> usize {
    (len + 3) & !3
}

#[no_mangle]
pub unsafe extern "C" fn BcPackDecodeBatch(
    ctx: *mut BcPackContext,
    items: *mut BcPackBatchItem,
    count: u32,
    out_total_kernel_ms: *mut f64,
) -> i32 {
    if count > 0 && !items.is_null() {
        for item in std::slice::from_raw_parts_mut(items, count as usize).iter_mut() {
            item.status = -1;
        }
    }
    if ctx.is_null() {
        return -1;
    }
    if count == 0 {
        let state = &mut *ctx;
        state.last_kernel_ms = 0.0;
        if !out_total_kernel_ms.is_null() {
            *out_total_kernel_ms = 0.0;
        }
        return 0;
    }
    if items.is_null() {
        return -1;
    }

    let res = catch_unwind(AssertUnwindSafe(|| -> i32 {
        let state = &mut *ctx;
        if state.ctx.compute_fence_value > 0 {
            let _ = state.ctx.wait_compute_cpu(state.ctx.compute_fence_value, INFINITE);
        }

        let slice = std::slice::from_raw_parts_mut(items, count as usize);
        let mut bc1_indices: Vec<usize> = Vec::new();
        let mut bcn_indices: Vec<usize> = Vec::new();
        let mut bc7_indices: Vec<usize> = Vec::new();
        let mut bc6h_indices: Vec<usize> = Vec::new();

        let max_chunks = state.max_chunks.max(1);
        let max_stream = state.max_stream_bytes;

        for (idx, item) in slice.iter_mut().enumerate() {
            item.status = -1;
            if item.compressed_data.is_null()
                || item.compressed_size == 0
                || item.num_blocks == 0
                || item.out_host_buffer.is_null()
            {
                item.status = -1;
                continue;
            }
            if item.swizzle_mode != BCPACK_SWIZZLE_LINEAR {
                item.status = -4;
                continue;
            }
            let (block_bytes, _) = match format_layout(item.format) {
                Some(v) => v,
                None => {
                    item.status = -4;
                    continue;
                }
            };
            let blocks = item.num_blocks as usize;
            let out_bytes = match blocks.checked_mul(block_bytes) {
                Some(v) => v,
                None => {
                    item.status = -5;
                    continue;
                }
            };
            if (item.out_capacity as usize) < out_bytes {
                item.status = -2;
                continue;
            }

            let stream_padded = match align4(item.compressed_size as usize).checked_add(256) {
                Some(v) => v,
                None => {
                    item.status = -5;
                    continue;
                }
            };
            if stream_padded > max_stream {
                item.status = -5;
                continue;
            }

            let family = match item.format {
                BCPACK_FORMAT_BC1 if !state.bc1_use_full_bcn => {
                    if blocks > state.max_blocks {
                        item.status = -5;
                        continue;
                    }
                    BatchFamily::Bc1Fast
                }
                BCPACK_FORMAT_BC7 => {
                    if blocks > state.max_bc7_blocks || out_bytes > state.max_output_bytes {
                        item.status = -5;
                        continue;
                    }
                    BatchFamily::Bc7
                }
                BCPACK_FORMAT_BC6H => {
                    if blocks > state.max_bc6h_blocks || out_bytes > state.max_output_bytes {
                        item.status = -5;
                        continue;
                    }
                    BatchFamily::Bc6h
                }
                _ => {
                    let mode = match bcn_mode_from_format(item.format) {
                        Some(m) => m,
                        None => {
                            item.status = -4;
                            continue;
                        }
                    };
                    let sym_rows = match blocks.checked_mul(mode.symbols_per_block()) {
                        Some(v) => v,
                        None => {
                            item.status = -5;
                            continue;
                        }
                    };
                    if sym_rows > state.max_bcn_sym_rows || out_bytes > state.max_output_bytes {
                        item.status = -5;
                        continue;
                    }
                    BatchFamily::Bcn
                }
            };

            match family {
                BatchFamily::Bc1Fast => bc1_indices.push(idx),
                BatchFamily::Bcn => bcn_indices.push(idx),
                BatchFamily::Bc7 => bc7_indices.push(idx),
                BatchFamily::Bc6h => bc6h_indices.push(idx),
            }
        }

        let mut succeeded = 0i32;
        let mut total_kernel_ms = 0.0f64;

        let groups = [
            (BatchFamily::Bc1Fast, bc1_indices),
            (BatchFamily::Bcn, bcn_indices),
            (BatchFamily::Bc7, bc7_indices),
            (BatchFamily::Bc6h, bc6h_indices),
        ];

        for (family, indices) in groups {
            if indices.is_empty() {
                continue;
            }
            let (unit_cap, out_cap) = match family {
                BatchFamily::Bc1Fast => (state.max_blocks, state.max_blocks * 8),
                BatchFamily::Bcn => (state.max_bcn_sym_rows, state.max_output_bytes),
                BatchFamily::Bc7 => (state.max_bc7_blocks * 2, state.max_output_bytes),
                BatchFamily::Bc6h => (state.max_bc6h_blocks * 2, state.max_output_bytes),
            };

            let mut start = 0usize;
            while start < indices.len() {
                let mut end = start;
                let mut sub_units = 0usize;
                let mut sub_out = 0usize;
                let mut sub_stream = 0usize;

                while end < indices.len() && (end - start) < max_chunks {
                    let item = &slice[indices[end]];
                    let blocks = item.num_blocks as usize;
                    let (item_units, item_out) = match family {
                        BatchFamily::Bc1Fast => (blocks, blocks * 8),
                        BatchFamily::Bc7 | BatchFamily::Bc6h => (blocks * 2, blocks * 16),
                        BatchFamily::Bcn => {
                            let mode = bcn_mode_from_format(item.format).unwrap();
                            (blocks * mode.symbols_per_block(), blocks * mode.block_bytes())
                        }
                    };
                    let item_stream = align4(item.compressed_size as usize);
                    if end > start
                        && (sub_units + item_units > unit_cap
                            || sub_out + item_out > out_cap
                            || sub_stream + item_stream + 256 > max_stream)
                    {
                        break;
                    }
                    sub_units += item_units;
                    sub_out += item_out;
                    sub_stream += item_stream;
                    end += 1;
                }

                let sub = &indices[start..end];
                let (decoded, statuses, kms) = match family {
                    BatchFamily::Bc1Fast => {
                        let batch = pack_batch(sub.iter().map(|&i| {
                            let it = &slice[i];
                            let s = std::slice::from_raw_parts(it.compressed_data, it.compressed_size as usize);
                            (s, it.num_blocks as usize)
                        }));
                        let (d, st, k, _) = state.bc1_dec.decode_batch_and_readback(&mut state.ctx, &batch);
                        (d, st, k)
                    }
                    BatchFamily::Bc7 => {
                        let batch = pack_batch(sub.iter().map(|&i| {
                            let it = &slice[i];
                            let s = std::slice::from_raw_parts(it.compressed_data, it.compressed_size as usize);
                            (s, (it.num_blocks as usize) * 2)
                        }));
                        let (d, st, k, _) = state.bc7_full.decode_batch_and_readback(&mut state.ctx, &batch);
                        (d, st, k)
                    }
                    BatchFamily::Bc6h => {
                        let batch = pack_batch(sub.iter().map(|&i| {
                            let it = &slice[i];
                            let s = std::slice::from_raw_parts(it.compressed_data, it.compressed_size as usize);
                            (s, (it.num_blocks as usize) * 2)
                        }));
                        let (d, st, k, _) = state.bc6h_full.decode_batch_and_readback(&mut state.ctx, &batch);
                        (d, st, k)
                    }
                    BatchFamily::Bcn => {
                        let batch = pack_bcn_batch(sub.iter().map(|&i| {
                            let it = &slice[i];
                            let s = std::slice::from_raw_parts(it.compressed_data, it.compressed_size as usize);
                            let mode = bcn_mode_from_format(it.format).unwrap();
                            (s, mode, it.num_blocks as usize)
                        }));
                        let (d, st, k, _) = state.bcn_dec.decode_batch_and_readback(&mut state.ctx, &batch);
                        (d, st, k)
                    }
                };

                total_kernel_ms += kms;
                let mut byte_off = 0usize;
                for (k, &idx) in sub.iter().enumerate() {
                    let item = &mut slice[idx];
                    let (blk_bytes, _) = format_layout(item.format).unwrap();
                    let item_out = (item.num_blocks as usize) * blk_bytes;
                    let code = statuses.get(k).map(|s| s.code).unwrap_or(99);
                    if code == 0 && byte_off + item_out <= decoded.len() {
                        copy_nonoverlapping(decoded.as_ptr().add(byte_off), item.out_host_buffer, item_out);
                        item.status = 0;
                        succeeded += 1;
                    } else {
                        item.status = -3;
                    }
                    byte_off += item_out;
                }

                start = end;
            }
        }

        state.last_kernel_ms = total_kernel_ms;
        if !out_total_kernel_ms.is_null() {
            *out_total_kernel_ms = total_kernel_ms;
        }
        succeeded
    }));
    res.unwrap_or(-99)
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BcPackTextureBatchItem {
    pub compressed_data: *const u8,
    pub compressed_size: u32,
    pub num_blocks: u32,
    pub format: u32,
    pub swizzle_mode: u32,
    pub width_blocks: u32,
    pub height_blocks: u32,
    pub dst_texture2d: *mut c_void,
    pub subresource_index: u32,
    pub dst_x_blocks: u32,
    pub dst_y_blocks: u32,
    pub status: i32,
}

#[inline]
fn align512(len: usize) -> usize {
    (len + 511) & !511
}

#[no_mangle]
pub unsafe extern "C" fn BcPackDecodeBatchToTextures(
    ctx: *mut BcPackContext,
    items: *mut BcPackTextureBatchItem,
    count: u32,
    out_direct_fence_value: *mut u64,
) -> i32 {
    if count > 0 && !items.is_null() {
        for item in std::slice::from_raw_parts_mut(items, count as usize).iter_mut() {
            item.status = -1;
        }
    }
    if ctx.is_null() {
        return -1;
    }
    if count == 0 {
        if !out_direct_fence_value.is_null() {
            *out_direct_fence_value = 0;
        }
        return 0;
    }
    if items.is_null() {
        return -1;
    }

    let res = catch_unwind(AssertUnwindSafe(|| -> i32 {
        let state = &mut *ctx;
        let slice = std::slice::from_raw_parts_mut(items, count as usize);

        let mut bc1_indices: Vec<usize> = Vec::new();
        let mut bcn_indices: Vec<usize> = Vec::new();
        let mut bc7_indices: Vec<usize> = Vec::new();
        let mut bc6h_indices: Vec<usize> = Vec::new();

        let max_chunks = state.max_chunks.max(1);
        let max_stream = state.max_stream_bytes;

        for (idx, item) in slice.iter_mut().enumerate() {
            item.status = -1;
            if item.compressed_data.is_null()
                || item.compressed_size == 0
                || item.num_blocks == 0
                || item.width_blocks == 0
                || item.height_blocks == 0
                || item.dst_texture2d.is_null()
            {
                item.status = -1;
                continue;
            }
            let (block_bytes, _) = match format_layout(item.format) {
                Some(v) => v,
                None => {
                    item.status = -4;
                    continue;
                }
            };
            let need_detile = item.swizzle_mode != BCPACK_SWIZZLE_LINEAR;
            if need_detile && detile_mode_for(block_bytes, item.swizzle_mode).is_err() {
                item.status = -6;
                continue;
            }
            let linear_blocks = match (item.width_blocks as usize).checked_mul(item.height_blocks as usize) {
                Some(v) if v > 0 => v,
                _ => {
                    item.status = -5;
                    continue;
                }
            };
            if !need_detile && (item.num_blocks as usize) < linear_blocks {
                item.status = -5;
                continue;
            }
            let raw_row = match (item.width_blocks as usize).checked_mul(block_bytes) {
                Some(v) => v,
                None => {
                    item.status = -5;
                    continue;
                }
            };
            let aligned_pitch = match raw_row.checked_add(255) {
                Some(v) => v & !255,
                None => {
                    item.status = -5;
                    continue;
                }
            };
            let pad_bytes = match aligned_pitch.checked_mul(item.height_blocks as usize) {
                Some(v) => align512(v),
                None => {
                    item.status = -5;
                    continue;
                }
            };
            if pad_bytes > state.max_row_pad_bytes {
                item.status = -5;
                continue;
            }
            let out_bytes = match (item.num_blocks as usize).max(linear_blocks).checked_mul(block_bytes) {
                Some(v) => v,
                None => {
                    item.status = -5;
                    continue;
                }
            };
            if out_bytes > state.max_output_bytes {
                item.status = -5;
                continue;
            }
            let stream_padded = match align4(item.compressed_size as usize).checked_add(256) {
                Some(v) => v,
                None => {
                    item.status = -5;
                    continue;
                }
            };
            if stream_padded > max_stream {
                item.status = -5;
                continue;
            }

            let blocks = item.num_blocks as usize;
            let family = match item.format {
                BCPACK_FORMAT_BC1 if !state.bc1_use_full_bcn => {
                    if blocks > state.max_blocks || blocks * 8 > state.max_output_bytes {
                        item.status = -5;
                        continue;
                    }
                    BatchFamily::Bc1Fast
                }
                BCPACK_FORMAT_BC7 => {
                    if blocks > state.max_bc7_blocks {
                        item.status = -5;
                        continue;
                    }
                    BatchFamily::Bc7
                }
                BCPACK_FORMAT_BC6H => {
                    if blocks > state.max_bc6h_blocks {
                        item.status = -5;
                        continue;
                    }
                    BatchFamily::Bc6h
                }
                _ => {
                    let mode = match bcn_mode_from_format(item.format) {
                        Some(m) => m,
                        None => {
                            item.status = -4;
                            continue;
                        }
                    };
                    let sym_rows = match blocks.checked_mul(mode.symbols_per_block()) {
                        Some(v) => v,
                        None => {
                            item.status = -5;
                            continue;
                        }
                    };
                    if sym_rows > state.max_bcn_sym_rows {
                        item.status = -5;
                        continue;
                    }
                    BatchFamily::Bcn
                }
            };

            match family {
                BatchFamily::Bc1Fast => bc1_indices.push(idx),
                BatchFamily::Bcn => bcn_indices.push(idx),
                BatchFamily::Bc7 => bc7_indices.push(idx),
                BatchFamily::Bc6h => bc6h_indices.push(idx),
            }
        }

        let mut succeeded = 0i32;
        let mut last_direct_fence = state.ctx.fence_value;

        let groups = [
            (BatchFamily::Bc1Fast, bc1_indices),
            (BatchFamily::Bcn, bcn_indices),
            (BatchFamily::Bc7, bc7_indices),
            (BatchFamily::Bc6h, bc6h_indices),
        ];

        for (family, indices) in groups {
            if indices.is_empty() {
                continue;
            }
            let (unit_cap, out_cap) = match family {
                BatchFamily::Bc1Fast => (state.max_blocks, (state.max_blocks * 8).min(state.max_output_bytes)),
                BatchFamily::Bcn => (state.max_bcn_sym_rows, state.max_output_bytes),
                BatchFamily::Bc7 => (state.max_bc7_blocks * 2, state.max_output_bytes),
                BatchFamily::Bc6h => (state.max_bc6h_blocks * 2, state.max_output_bytes),
            };

            let mut start = 0usize;
            while start < indices.len() {
                // Drain prior sub-batch only when reusing internal scratch/command allocators.
                if state.ctx.compute_fence_value > 0 {
                    let _ = state.ctx.wait_compute_cpu(state.ctx.compute_fence_value, INFINITE);
                }
                if state.ctx.fence_value > 0 {
                    state.ctx.wait_direct_cpu(state.ctx.fence_value);
                }

                let mut end = start;
                let mut sub_units = 0usize;
                let mut sub_out = 0usize;
                let mut sub_stream = 0usize;
                let mut sub_detile = 0usize;
                let mut sub_pad = 0usize;

                while end < indices.len() && (end - start) < max_chunks {
                    let item = &slice[indices[end]];
                    let (blk_bytes, _) = format_layout(item.format).unwrap();
                    let blocks = item.num_blocks as usize;
                    let (item_units, item_out) = match family {
                        BatchFamily::Bc1Fast => (blocks, blocks * 8),
                        BatchFamily::Bc7 | BatchFamily::Bc6h => (blocks * 2, blocks * 16),
                        BatchFamily::Bcn => {
                            let mode = bcn_mode_from_format(item.format).unwrap();
                            (blocks * mode.symbols_per_block(), blocks * mode.block_bytes())
                        }
                    };
                    let item_stream = align4(item.compressed_size as usize);
                    let need_detile = item.swizzle_mode != BCPACK_SWIZZLE_LINEAR;
                    let lin_bytes = (item.width_blocks as usize) * (item.height_blocks as usize) * blk_bytes;
                    let item_detile = if need_detile { align512(lin_bytes) } else { 0 };
                    let raw_row = (item.width_blocks as usize) * blk_bytes;
                    let aligned_pitch = (raw_row + 255) & !255;
                    let may_need_pad = aligned_pitch != raw_row || (!need_detile && (sub_out % 512 != 0));
                    let item_pad = if may_need_pad {
                        align512(aligned_pitch * (item.height_blocks as usize))
                    } else {
                        0
                    };

                    if end > start
                        && (sub_units + item_units > unit_cap
                            || sub_out + item_out > out_cap
                            || sub_stream + item_stream + 256 > max_stream
                            || sub_detile + item_detile > state.max_output_bytes
                            || sub_pad + item_pad > state.max_row_pad_bytes)
                    {
                        break;
                    }
                    sub_units += item_units;
                    sub_out += item_out;
                    sub_stream += item_stream;
                    sub_detile += item_detile;
                    sub_pad += item_pad;
                    end += 1;
                }

                let sub = &indices[start..end];

                // 1) Compute decode whole sub-batch (single submission on compute_queue)
                let (decoded_res, mut compute_val) = match family {
                    BatchFamily::Bc1Fast => {
                        let batch = pack_batch(sub.iter().map(|&i| {
                            let it = &slice[i];
                            let s = std::slice::from_raw_parts(it.compressed_data, it.compressed_size as usize);
                            (s, it.num_blocks as usize)
                        }));
                        let fv = state.bc1_dec.submit_async_compute(&mut state.ctx, &batch);
                        (state.bc1_dec.slots[0].out_blocks_default.clone(), fv)
                    }
                    BatchFamily::Bc7 => {
                        let batch = pack_batch(sub.iter().map(|&i| {
                            let it = &slice[i];
                            let s = std::slice::from_raw_parts(it.compressed_data, it.compressed_size as usize);
                            (s, (it.num_blocks as usize) * 2)
                        }));
                        let fv = state.bc7_full.submit_async_compute(&mut state.ctx, &batch);
                        (state.bc7_full.out_blocks_default.clone(), fv)
                    }
                    BatchFamily::Bc6h => {
                        let batch = pack_batch(sub.iter().map(|&i| {
                            let it = &slice[i];
                            let s = std::slice::from_raw_parts(it.compressed_data, it.compressed_size as usize);
                            (s, (it.num_blocks as usize) * 2)
                        }));
                        let fv = state.bc6h_full.submit_async_compute(&mut state.ctx, &batch);
                        (state.bc6h_full.out_blocks_default.clone(), fv)
                    }
                    BatchFamily::Bcn => {
                        let batch = pack_bcn_batch(sub.iter().map(|&i| {
                            let it = &slice[i];
                            let s = std::slice::from_raw_parts(it.compressed_data, it.compressed_size as usize);
                            let mode = bcn_mode_from_format(it.format).unwrap();
                            (s, mode, it.num_blocks as usize)
                        }));
                        let fv = state.bcn_dec.submit_async_compute(&mut state.ctx, &batch);
                        (state.bcn_dec.out_blocks_default.clone(), fv)
                    }
                };

                // Compute per-item source offsets and optional detile / row-pad plans
                struct CopyPlan {
                    from_detile: bool,
                    src_off: usize,
                    pad_off: Option<usize>,
                    raw_row: u32,
                    aligned_pitch: u32,
                    width_blocks: u32,
                    height_blocks: u32,
                    dxgi_fmt: DXGI_FORMAT,
                    dst_tex: ID3D12Resource,
                    subresource_index: u32,
                    dst_x_blocks: u32,
                    dst_y_blocks: u32,
                }

                let mut plans: Vec<CopyPlan> = Vec::with_capacity(sub.len());
                let mut dec_cursor = 0usize;
                let mut det_cursor = 0usize;
                let mut pad_cursor = 0usize;
                let mut any_detile = false;
                let mut any_linear = false;
                let mut any_pad = false;

                for &idx in sub {
                    let it = &slice[idx];
                    let (blk_bytes, dxgi_fmt) = format_layout(it.format).unwrap();
                    let dec_bytes = (it.num_blocks as usize) * blk_bytes;
                    let dec_off = dec_cursor;
                    dec_cursor += dec_bytes;

                    let need_detile = it.swizzle_mode != BCPACK_SWIZZLE_LINEAR;
                    let src_off = if need_detile {
                        any_detile = true;
                        let off = det_cursor;
                        let lin_bytes = (it.width_blocks as usize) * (it.height_blocks as usize) * blk_bytes;
                        det_cursor += align512(lin_bytes);
                        off
                    } else {
                        any_linear = true;
                        dec_off
                    };

                    let raw_row = (it.width_blocks as usize) * blk_bytes;
                    let aligned_pitch = (raw_row + 255) & !255;
                    let pad_off = if aligned_pitch != raw_row || src_off % 512 != 0 {
                        any_pad = true;
                        let off = pad_cursor;
                        pad_cursor += align512(aligned_pitch * (it.height_blocks as usize));
                        Some(off)
                    } else {
                        None
                    };

                    let dst_tex: ID3D12Resource =
                        (*ManuallyDrop::new(ID3D12Resource::from_raw(it.dst_texture2d))).clone();

                    plans.push(CopyPlan {
                        from_detile: need_detile,
                        src_off,
                        pad_off,
                        raw_row: raw_row as u32,
                        aligned_pitch: aligned_pitch as u32,
                        width_blocks: it.width_blocks,
                        height_blocks: it.height_blocks,
                        dxgi_fmt,
                        dst_tex,
                        subresource_index: it.subresource_index,
                        dst_x_blocks: it.dst_x_blocks,
                        dst_y_blocks: it.dst_y_blocks,
                    });
                }

                // 2) Single compute command list for all Gfx10 detiles in this sub-batch
                if any_detile {
                    state.detile_alloc.Reset().unwrap();
                    state.detile_cmd.Reset(&state.detile_alloc, None).unwrap();
                    cmd_barriers(
                        &state.detile_cmd,
                        vec![
                            transition_barrier(&decoded_res, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                            transition_barrier(&state.detile_out_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                        ],
                    );
                    let mut dec_off = 0usize;
                    let mut det_idx = 0usize;
                    for &idx in sub {
                        let it = &slice[idx];
                        let (blk_bytes, _) = format_layout(it.format).unwrap();
                        let dec_bytes = (it.num_blocks as usize) * blk_bytes;
                        let plan = &plans[det_idx];
                        det_idx += 1;
                        if plan.from_detile {
                            let tile_mode = detile_mode_for(blk_bytes, it.swizzle_mode).unwrap();
                            state.detile.record_detile(
                                &state.detile_cmd,
                                decoded_res.GetGPUVirtualAddress(),
                                state.detile_out_default.GetGPUVirtualAddress(),
                                it.width_blocks,
                                it.height_blocks,
                                tile_mode,
                                blk_bytes as u32,
                                dec_off as u32,
                                plan.src_off as u32,
                                it.num_blocks,
                            );
                        }
                        dec_off += dec_bytes;
                    }
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
                }

                // 3) Single GPU cross-queue Wait + single DIRECT command list for all CopyTextureRegions
                state
                    .ctx
                    .queue
                    .Wait(&state.ctx.compute_fence, compute_val)
                    .expect("DIRECT queue Wait compute_fence");

                state.tex_copy_alloc.Reset().unwrap();
                state.tex_copy_cmd.Reset(&state.tex_copy_alloc, None).unwrap();

                let mut pre_barriers: Vec<D3D12_RESOURCE_BARRIER> = Vec::new();
                if any_linear {
                    pre_barriers.push(transition_barrier(
                        &decoded_res,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    ));
                }
                if any_detile {
                    pre_barriers.push(transition_barrier(
                        &state.detile_out_default,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    ));
                }
                if any_pad {
                    pre_barriers.push(transition_barrier(
                        &state.row_pad_default,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    ));
                }
                if !pre_barriers.is_empty() {
                    cmd_barriers(&state.tex_copy_cmd, pre_barriers);
                }

                if any_pad {
                    for p in &plans {
                        if let Some(pad_off) = p.pad_off {
                            let src_buf = if p.from_detile {
                                &state.detile_out_default
                            } else {
                                &decoded_res
                            };
                            if p.aligned_pitch == p.raw_row {
                                state.tex_copy_cmd.CopyBufferRegion(
                                    &state.row_pad_default,
                                    pad_off as u64,
                                    src_buf,
                                    p.src_off as u64,
                                    (p.raw_row as u64) * (p.height_blocks as u64),
                                );
                            } else {
                                for r in 0..p.height_blocks {
                                    state.tex_copy_cmd.CopyBufferRegion(
                                        &state.row_pad_default,
                                        (pad_off as u64) + (r as u64) * (p.aligned_pitch as u64),
                                        src_buf,
                                        (p.src_off as u64) + (r as u64) * (p.raw_row as u64),
                                        p.raw_row as u64,
                                    );
                                }
                            }
                        }
                    }
                    cmd_barriers(
                        &state.tex_copy_cmd,
                        vec![transition_barrier(
                            &state.row_pad_default,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_COPY_SOURCE,
                        )],
                    );
                }

                // Deduplicate destination ID3D12Resource transitions
                let mut unique_dsts: Vec<ID3D12Resource> = Vec::new();
                for p in &plans {
                    let raw = p.dst_tex.as_raw() as usize;
                    if !unique_dsts.iter().any(|d| d.as_raw() as usize == raw) {
                        unique_dsts.push(p.dst_tex.clone());
                    }
                }
                let dst_to_copy: Vec<D3D12_RESOURCE_BARRIER> = unique_dsts
                    .iter()
                    .map(|d| transition_barrier(d, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST))
                    .collect();
                cmd_barriers(&state.tex_copy_cmd, dst_to_copy);

                for p in &plans {
                    let (copy_src, copy_off) = if let Some(pad_off) = p.pad_off {
                        (&state.row_pad_default, pad_off as u64)
                    } else if p.from_detile {
                        (&state.detile_out_default, p.src_off as u64)
                    } else {
                        (&decoded_res, p.src_off as u64)
                    };
                    let mut dst_loc = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: ManuallyDrop::new(Some(p.dst_tex.clone())),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            SubresourceIndex: p.subresource_index,
                        },
                    };
                    let mut src_loc = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: ManuallyDrop::new(Some(copy_src.clone())),
                        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                                Offset: copy_off,
                                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                    Format: p.dxgi_fmt,
                                    Width: p.width_blocks * 4,
                                    Height: p.height_blocks * 4,
                                    Depth: 1,
                                    RowPitch: p.aligned_pitch,
                                },
                            },
                        },
                    };
                    state.tex_copy_cmd.CopyTextureRegion(
                        &dst_loc,
                        p.dst_x_blocks * 4,
                        p.dst_y_blocks * 4,
                        0,
                        &src_loc,
                        None,
                    );
                    let _ = ManuallyDrop::take(&mut dst_loc.pResource);
                    let _ = ManuallyDrop::take(&mut src_loc.pResource);
                }

                let mut post_barriers: Vec<D3D12_RESOURCE_BARRIER> = unique_dsts
                    .iter()
                    .map(|d| transition_barrier(d, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON))
                    .collect();
                if any_linear {
                    post_barriers.push(transition_barrier(
                        &decoded_res,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    ));
                }
                if any_detile {
                    post_barriers.push(transition_barrier(
                        &state.detile_out_default,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    ));
                }
                if any_pad {
                    post_barriers.push(transition_barrier(
                        &state.row_pad_default,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    ));
                }
                cmd_barriers(&state.tex_copy_cmd, post_barriers);

                state.tex_copy_cmd.Close().unwrap();
                let lists = [Some(state.tex_copy_cmd.cast::<ID3D12CommandList>().unwrap())];
                state.ctx.queue.ExecuteCommandLists(&lists);
                last_direct_fence = state.ctx.signal_direct_async();

                for &idx in sub {
                    slice[idx].status = 0;
                    succeeded += 1;
                }

                start = end;
            }
        }

        state.last_kernel_ms = 0.0;
        if !out_direct_fence_value.is_null() {
            *out_direct_fence_value = last_direct_fence;
        }
        succeeded
    }));
    res.unwrap_or(-99)
}
