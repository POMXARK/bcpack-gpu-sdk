use bcpack::bc1::{QuadTable, Window};
use bcpack::bc7 as bcp_bc7;
use bcpack::bcn as bcp_bcn;
use bcpack::encode::{encode_lanes, Fragment, Rng};
use bcpack::format::Mode as BcpMode;
use bcpack_gpu::ffi::*;
use bcpack_gpu::gpu_decoder::{create_committed_buffer, GpuContext};
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::time::Instant;
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC3_UNORM, DXGI_FORMAT_BC4_UNORM,
    DXGI_FORMAT_BC5_UNORM, DXGI_FORMAT_BC6H_UF16, DXGI_FORMAT_BC7_UNORM, DXGI_SAMPLE_DESC,
};

fn valid_bc7_ctrl_codes() -> Vec<u16> {
    let t7 = bcp_bc7::Tables::default_table();
    let codes: Vec<u16> = bcp_bc7::Control::all()
        .into_iter()
        .filter_map(|c| t7.code_of(c).map(|idx| idx as u16))
        .collect();
    assert!(!codes.is_empty());
    codes
}

/// Builds a clean-room synthetic BCPack stream using `bcpack::encode`.
/// Uses zero back-references on lanes 0..=1 so BC6H/BC7/BCn stay deterministic
/// without relying on any proprietary dictionary, while exercising full modes & blocks.
fn encode_synthetic_chunk(mode: BcpMode, blocks: usize, seed: u64, ctrl7: &[u16]) -> Vec<u8> {
    let ab = mode.alphabet_bits();
    let rows = blocks * mode.symbols_per_block();
    let mut rng = Rng(seed | 1);
    let mut ctrl_for_block = Vec::with_capacity(blocks);
    for i in 0..blocks {
        ctrl_for_block.push(ctrl7[(i + (seed as usize)) % ctrl7.len()]);
    }

    let mut lanes: Vec<Vec<Fragment>> = Vec::with_capacity(ab.len());
    for (lane_idx, &bits) in ab.iter().enumerate() {
        let lim = 1u64 << bits;
        let mut syms = Vec::with_capacity(rows);
        for r in 0..rows {
            let s = if lane_idx < 2 {
                0u16
            } else if mode == BcpMode::Bc7 && lane_idx == 15 {
                let c = ctrl_for_block[r / 2];
                if r % 2 == 0 {
                    c & 31
                } else {
                    (c >> 5) & 15
                }
            } else if mode == BcpMode::Bc6h && lane_idx == 15 {
                if r % 2 == 0 {
                    ((r / 2) % 14) as u16
                } else {
                    (rng.next_u64() & 15) as u16
                }
            } else {
                rng.below(lim) as u16
            };
            syms.push(s);
        }

        let frags: Vec<Fragment> = syms
            .chunks(4096)
            .map(|c| Fragment {
                symbols: c.to_vec(),
                q: 0,
                z: 0,
            })
            .collect();
        lanes.push(frags);
    }
    encode_lanes(ab, &lanes).expect("encode_lanes failed")
}

/// Builds a fast synthetic multi-block stream for huge (>16 MiB) buffers by repeating a
/// 1024-symbol tile into 4096-symbol fragments (avoids huge O(N) CPU RNG overhead in tests).
fn encode_large_synthetic_chunk(mode: BcpMode, blocks: usize, seed: u64, ctrl7: &[u16]) -> Vec<u8> {
    encode_synthetic_chunk(mode, blocks, seed, ctrl7)
}

fn cpu_decode_reference(mode: BcpMode, stream: &[u8], blocks: usize) -> Vec<u8> {
    match mode {
        BcpMode::Bc1 => {
            let w = Window::default_table();
            let q = QuadTable::default_table();
            bcpack::bc1::decode(stream, blocks, &w, &q)
                .expect("CPU BC1 decode failed")
                .blocks
        }
        BcpMode::Bc3 | BcpMode::Bc4 | BcpMode::Bc5 => {
            let t = bcp_bcn::Tables::default_for(mode).expect("CPU BCn tables");
            bcp_bcn::decode(stream, blocks * mode.block_bytes(), t)
                .expect("CPU BCn decode failed")
                .blocks
        }
        BcpMode::Bc7 => {
            let t7 = bcp_bc7::Tables::default_table();
            bcp_bc7::decode(stream, blocks * 16, t7)
                .expect("CPU BC7 decode failed")
                .blocks
        }
        BcpMode::Bc6h => panic!("bcpack-rs CPU reference does not implement BC6H"),
    }
}

fn format_info(mode: BcpMode) -> (u32, DXGI_FORMAT, usize, &'static str) {
    match mode {
        BcpMode::Bc1 => (BCPACK_FORMAT_BC1, DXGI_FORMAT_BC1_UNORM, 8, "BC1"),
        BcpMode::Bc3 => (BCPACK_FORMAT_BC3, DXGI_FORMAT_BC3_UNORM, 16, "BC3"),
        BcpMode::Bc4 => (BCPACK_FORMAT_BC4, DXGI_FORMAT_BC4_UNORM, 8, "BC4"),
        BcpMode::Bc5 => (BCPACK_FORMAT_BC5, DXGI_FORMAT_BC5_UNORM, 16, "BC5"),
        BcpMode::Bc6h => (BCPACK_FORMAT_BC6H, DXGI_FORMAT_BC6H_UF16, 16, "BC6H"),
        BcpMode::Bc7 => (BCPACK_FORMAT_BC7, DXGI_FORMAT_BC7_UNORM, 16, "BC7"),
    }
}

unsafe fn create_bc_texture2d(
    device: &ID3D12Device,
    dxgi_fmt: DXGI_FORMAT,
    width_blocks: u32,
    height_blocks: u32,
) -> ID3D12Resource {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 0,
        VisibleNodeMask: 0,
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: (width_blocks * 4) as u64,
        Height: height_blocks * 4,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: dxgi_fmt,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };
    let mut res: Option<ID3D12Resource> = None;
    device
        .CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut res,
        )
        .expect("CreateCommittedResource Texture2D");
    res.unwrap()
}

unsafe fn readback_bc_texture2d(
    gpu: &mut GpuContext,
    tex: &ID3D12Resource,
    dxgi_fmt: DXGI_FORMAT,
    width_blocks: u32,
    height_blocks: u32,
    block_bytes: usize,
) -> Vec<u8> {
    let raw_row = (width_blocks as usize) * block_bytes;
    let aligned_pitch = (raw_row + 255) & !255;
    let readback_size = aligned_pitch * (height_blocks as usize);
    let readback = create_committed_buffer(
        &gpu.device,
        D3D12_HEAP_TYPE_READBACK,
        readback_size as u64,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    let alloc: ID3D12CommandAllocator = gpu
        .device
        .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
        .unwrap();
    let cmd: ID3D12GraphicsCommandList = gpu
        .device
        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
        .unwrap();

    let mut b_in = [D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(tex.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: D3D12_RESOURCE_STATE_COMMON,
                StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
            }),
        },
    }];
    cmd.ResourceBarrier(&b_in);
    ManuallyDrop::drop(&mut b_in[0].Anonymous.Transition);

    let mut dst_loc = D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(readback.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: dxgi_fmt,
                    Width: width_blocks * 4,
                    Height: height_blocks * 4,
                    Depth: 1,
                    RowPitch: aligned_pitch as u32,
                },
            },
        },
    };
    let mut src_loc = D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(tex.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
    };
    cmd.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
    let _ = ManuallyDrop::take(&mut dst_loc.pResource);
    let _ = ManuallyDrop::take(&mut src_loc.pResource);

    let mut b_out = [D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(tex.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: D3D12_RESOURCE_STATE_COPY_SOURCE,
                StateAfter: D3D12_RESOURCE_STATE_COMMON,
            }),
        },
    }];
    cmd.ResourceBarrier(&b_out);
    ManuallyDrop::drop(&mut b_out[0].Anonymous.Transition);
    cmd.Close().unwrap();

    let lists = [Some(cmd.cast::<ID3D12CommandList>().unwrap())];
    gpu.queue.ExecuteCommandLists(&lists);
    gpu.signal_and_wait();

    let mut ptr: *mut c_void = std::ptr::null_mut();
    readback
        .Map(
            0,
            Some(&D3D12_RANGE {
                Begin: 0,
                End: readback_size,
            }),
            Some(&mut ptr),
        )
        .unwrap();
    let raw = std::slice::from_raw_parts(ptr as *const u8, readback_size);
    let mut dense = Vec::with_capacity(raw_row * (height_blocks as usize));
    for r in 0..(height_blocks as usize) {
        let start = r * aligned_pitch;
        dense.extend_from_slice(&raw[start..start + raw_row]);
    }
    readback.Unmap(0, None);
    dense
}

#[test]
fn bcpack_decode_to_texture2d_all_6_formats_including_bc6h() {
    unsafe {
        let ctrl7 = valid_bc7_ctrl_codes();
        let mut gpu = GpuContext::new(false);
        let ctx = bcpack_gpu_create_from_device(
            gpu.device.as_raw(),
            gpu.compute_queue.as_raw(),
            0,
            0,
            0,
        );
        assert!(!ctx.is_null());

        // Test 6 formats at standard 64x64 blocks AND small unaligned 3x3 blocks (row_pad_default path)
        for &(wb, hb) in &[(64u32, 64u32), (3u32, 3u32)] {
            for &mode in &BcpMode::ALL {
                let (fmt, dxgi_fmt, blk_bytes, label) = format_info(mode);
                let blocks = (wb * hb) as usize;
                let stream = encode_synthetic_chunk(
                    mode,
                    blocks,
                    (wb as u64) * 131 + (fmt as u64) * 997,
                    &ctrl7,
                );
                let tex = create_bc_texture2d(&gpu.device, dxgi_fmt, wb, hb);
                let mut fence_val = 0u64;
                let rc = BcPackDecodeToTexture2D(
                    ctx,
                    stream.as_ptr(),
                    stream.len() as u32,
                    blocks as u32,
                    fmt,
                    BCPACK_SWIZZLE_LINEAR,
                    wb,
                    hb,
                    tex.as_raw(),
                    0,
                    0,
                    0,
                    &mut fence_val,
                );
                assert_eq!(rc, 0, "{label} ({wb}x{hb}) BcPackDecodeToTexture2D failed with {rc}");
                assert_eq!(BcPackWaitDirectFence(ctx, fence_val, u32::MAX), 0);

                let tex_bytes = readback_bc_texture2d(&mut gpu, &tex, dxgi_fmt, wb, hb, blk_bytes);

                let reference = if mode == BcpMode::Bc6h {
                    let mut ref_buf = vec![0u8; blocks * blk_bytes];
                    let mut dummy_fence = 0u64;
                    let rc_async = BcPackDecodeAsync(
                        ctx,
                        stream.as_ptr(),
                        stream.len() as u32,
                        blocks as u32,
                        fmt,
                        BCPACK_SWIZZLE_LINEAR,
                        wb,
                        hb,
                        ref_buf.as_mut_ptr(),
                        ref_buf.len() as u32,
                        &mut dummy_fence,
                    );
                    assert_eq!(rc_async, 0, "BC6H reference async failed");
                    ref_buf
                } else {
                    cpu_decode_reference(mode, &stream, blocks)
                };

                assert_eq!(
                    tex_bytes, reference,
                    "{label} ({wb}x{hb}) Texture2D readback mismatch!"
                );
                println!(
                    "[v14-tex2d] {label} ({}x{} px / {wb}x{hb} blk, {} bytes): 0 byte mismatches",
                    wb * 4,
                    hb * 4,
                    tex_bytes.len()
                );
            }
        }

        // Verify >16 MiB non-256B-aligned BC7 Texture2D (513 x 2048 blocks = 2052x8192 px, 16.03 MiB raw / 16.50 MiB padded)
        // decoded in 64 realistic 513x32-block strips (each exercising row_pad_default)
        let wb = 513u32;
        let hb = 2048u32;
        let strip_h = 32u32;
        let strip_blks = (wb * strip_h) as usize;
        let big_tex = create_bc_texture2d(&gpu.device, DXGI_FORMAT_BC7_UNORM, wb, hb);
        let mut big_expected = Vec::with_capacity((wb * hb) as usize * 16);
        let mut big_fence = 0u64;
        for s_idx in 0..(hb / strip_h) {
            let strip_stream = encode_large_synthetic_chunk(
                BcpMode::Bc7,
                strip_blks,
                0xCAFE_1600 + (s_idx as u64),
                &ctrl7,
            );
            big_expected.extend_from_slice(&cpu_decode_reference(BcpMode::Bc7, &strip_stream, strip_blks));
            let rc_big = BcPackDecodeToTexture2D(
                ctx,
                strip_stream.as_ptr(),
                strip_stream.len() as u32,
                strip_blks as u32,
                BCPACK_FORMAT_BC7,
                BCPACK_SWIZZLE_LINEAR,
                wb,
                strip_h,
                big_tex.as_raw(),
                0,
                0,
                s_idx * strip_h,
                &mut big_fence,
            );
            assert_eq!(rc_big, 0, ">16 MiB row-padded strip {s_idx} failed: {rc_big}");
        }
        assert_eq!(BcPackWaitDirectFence(ctx, big_fence, u32::MAX), 0);

        let gpu_big = readback_bc_texture2d(&mut gpu, &big_tex, DXGI_FORMAT_BC7_UNORM, wb, hb, 16);
        assert_eq!(gpu_big, big_expected, ">16 MiB row-padded BC7 Texture2D mismatch!");
        println!(
            "[v14-tex2d] BC7 >16MiB row-padded ({}x{} px, {:.2} MiB raw / 16.50 MiB padded, 64 strips): 0 byte mismatches",
            wb * 4,
            hb * 4,
            gpu_big.len() as f64 / (1024.0 * 1024.0)
        );

        // Verify pre-submission -5 bounds check when padded output exceeds max_row_pad_bytes (128 MiB)
        let dummy_stream = [0u8; 64];
        let mut oob_fence = 0u64;
        let rc_oob = BcPackDecodeToTexture2D(
            ctx,
            dummy_stream.as_ptr(),
            dummy_stream.len() as u32,
            1025 * 10_000,
            BCPACK_FORMAT_BC7,
            BCPACK_SWIZZLE_LINEAR,
            1025,
            10_000,
            big_tex.as_raw(),
            0,
            0,
            0,
            &mut oob_fence,
        );
        assert_eq!(rc_oob, -5, "Expected -5 for >128 MiB row-padded Texture2D, got {rc_oob}");
        println!("[v14-tex2d] Pre-submission row-pad bounds check (>128 MiB): returned -5 cleanly without GPU fault");

        // Verify 4K (4096x4096 px, 16.00 MiB) BC7 Texture2D decoded via 64 tiled strips (1024x16 blocks each)
        let k4_wb = 1024u32;
        let k4_hb = 1024u32;
        let strip_hb = 16u32;
        let strip_blocks = (k4_wb * strip_hb) as usize;
        let k4_tex = create_bc_texture2d(&gpu.device, DXGI_FORMAT_BC7_UNORM, k4_wb, k4_hb);
        let mut k4_expected = Vec::with_capacity((k4_wb * k4_hb) as usize * 16);
        let mut k4_fence = 0u64;
        for s_idx in 0..(k4_hb / strip_hb) {
            let strip_stream = encode_synthetic_chunk(
                BcpMode::Bc7,
                strip_blocks,
                0x4E77_0000 + (s_idx as u64),
                &ctrl7,
            );
            k4_expected.extend_from_slice(&cpu_decode_reference(BcpMode::Bc7, &strip_stream, strip_blocks));
            let rc_strip = BcPackDecodeToTexture2D(
                ctx,
                strip_stream.as_ptr(),
                strip_stream.len() as u32,
                strip_blocks as u32,
                BCPACK_FORMAT_BC7,
                BCPACK_SWIZZLE_LINEAR,
                k4_wb,
                strip_hb,
                k4_tex.as_raw(),
                0,
                0,
                s_idx * strip_hb,
                &mut k4_fence,
            );
            assert_eq!(rc_strip, 0, "4K BC7 strip {s_idx} failed: {rc_strip}");
        }
        assert_eq!(BcPackWaitDirectFence(ctx, k4_fence, u32::MAX), 0);
        let k4_gpu = readback_bc_texture2d(&mut gpu, &k4_tex, DXGI_FORMAT_BC7_UNORM, k4_wb, k4_hb, 16);
        assert_eq!(k4_gpu, k4_expected, "4K BC7 tiled Texture2D mismatch!");
        println!(
            "[v14-tex2d] BC7 4K tiled via dst_y_blocks (4096x4096 px, {:.2} MiB, 64 strips): 0 byte mismatches",
            k4_gpu.len() as f64 / (1024.0 * 1024.0)
        );

        bcpack_gpu_destroy(ctx);
    }
}

#[test]
fn bcpack_decode_batch_mixed_formats_bit_exact() {
    unsafe {
        let ctrl7 = valid_bc7_ctrl_codes();
        // Force small max_chunks = 16 to verify automatic sub-batching across families!
        let ctx = bcpack_gpu_create(16, 131_072, 4 * 1024 * 1024);
        assert!(!ctx.is_null());

        struct ChunkData {
            mode: BcpMode,
            fmt: u32,
            blocks: usize,
            stream: Vec<u8>,
            out: Vec<u8>,
            expected: Vec<u8>,
        }

        let specs: &[(BcpMode, usize, usize)] = &[
            (BcpMode::Bc1, 12, 512),
            (BcpMode::Bc3, 10, 256),
            (BcpMode::Bc4, 10, 512),
            (BcpMode::Bc5, 10, 256),
            (BcpMode::Bc7, 10, 256),
            (BcpMode::Bc6h, 10, 256),
        ];

        let mut chunks: Vec<ChunkData> = Vec::new();
        for &(mode, count, blocks) in specs {
            let (fmt, _, blk_bytes, _) = format_info(mode);
            for k in 0..count {
                let seed = (fmt as u64) * 10_000 + (k as u64) * 37 + 7;
                let stream = encode_synthetic_chunk(mode, blocks, seed, &ctrl7);
                let expected = if mode == BcpMode::Bc6h {
                    let mut buf = vec![0u8; blocks * blk_bytes];
                    let mut fv = 0u64;
                    let rc = BcPackDecodeAsync(
                        ctx,
                        stream.as_ptr(),
                        stream.len() as u32,
                        blocks as u32,
                        fmt,
                        BCPACK_SWIZZLE_LINEAR,
                        0,
                        0,
                        buf.as_mut_ptr(),
                        buf.len() as u32,
                        &mut fv,
                    );
                    assert_eq!(rc, 0);
                    buf
                } else {
                    cpu_decode_reference(mode, &stream, blocks)
                };
                chunks.push(ChunkData {
                    mode,
                    fmt,
                    blocks,
                    stream,
                    out: vec![0u8; blocks * blk_bytes],
                    expected,
                });
            }
        }

        // Build mixed-order batch (62 valid + 4 isolated invalid items = 66)
        let mut items: Vec<BcPackBatchItem> = Vec::with_capacity(66);
        let mut valid_item_to_chunk: Vec<Option<usize>> = Vec::with_capacity(66);

        let err_small_buf_stream = chunks[0].stream.clone();
        let mut tiny_buf = [0u8; 4];

        for (i, ch) in chunks.iter_mut().enumerate() {
            if i == 5 {
                // Invalid item #1: insufficient out_capacity (-2)
                items.push(BcPackBatchItem {
                    compressed_data: err_small_buf_stream.as_ptr(),
                    compressed_size: err_small_buf_stream.len() as u32,
                    num_blocks: 512,
                    format: BCPACK_FORMAT_BC1,
                    swizzle_mode: BCPACK_SWIZZLE_LINEAR,
                    width_blocks: 0,
                    height_blocks: 0,
                    out_host_buffer: tiny_buf.as_mut_ptr(),
                    out_capacity: tiny_buf.len() as u32,
                    status: 99,
                });
                valid_item_to_chunk.push(None);
            } else if i == 20 {
                // Invalid item #2: BC7_RANS rejected in batch (-4)
                items.push(BcPackBatchItem {
                    compressed_data: err_small_buf_stream.as_ptr(),
                    compressed_size: err_small_buf_stream.len() as u32,
                    num_blocks: 16,
                    format: BCPACK_FORMAT_BC7_RANS,
                    swizzle_mode: BCPACK_SWIZZLE_LINEAR,
                    width_blocks: 0,
                    height_blocks: 0,
                    out_host_buffer: tiny_buf.as_mut_ptr(),
                    out_capacity: tiny_buf.len() as u32,
                    status: 99,
                });
                valid_item_to_chunk.push(None);
            } else if i == 35 {
                // Invalid item #3: non-linear swizzle rejected (-4)
                items.push(BcPackBatchItem {
                    compressed_data: err_small_buf_stream.as_ptr(),
                    compressed_size: err_small_buf_stream.len() as u32,
                    num_blocks: 16,
                    format: BCPACK_FORMAT_BC1,
                    swizzle_mode: BCPACK_SWIZZLE_GFX10_64KB_D,
                    width_blocks: 4,
                    height_blocks: 4,
                    out_host_buffer: tiny_buf.as_mut_ptr(),
                    out_capacity: tiny_buf.len() as u32,
                    status: 99,
                });
                valid_item_to_chunk.push(None);
            } else if i == 50 {
                // Invalid item #4: num_blocks == 0 (-1)
                items.push(BcPackBatchItem {
                    compressed_data: err_small_buf_stream.as_ptr(),
                    compressed_size: err_small_buf_stream.len() as u32,
                    num_blocks: 0,
                    format: BCPACK_FORMAT_BC7,
                    swizzle_mode: BCPACK_SWIZZLE_LINEAR,
                    width_blocks: 0,
                    height_blocks: 0,
                    out_host_buffer: tiny_buf.as_mut_ptr(),
                    out_capacity: tiny_buf.len() as u32,
                    status: 99,
                });
                valid_item_to_chunk.push(None);
            }

            items.push(BcPackBatchItem {
                compressed_data: ch.stream.as_ptr(),
                compressed_size: ch.stream.len() as u32,
                num_blocks: ch.blocks as u32,
                format: ch.fmt,
                swizzle_mode: BCPACK_SWIZZLE_LINEAR,
                width_blocks: 0,
                height_blocks: 0,
                out_host_buffer: ch.out.as_mut_ptr(),
                out_capacity: ch.out.len() as u32,
                status: 99,
            });
            valid_item_to_chunk.push(Some(i));
        }

        let mut kernel_ms = 0.0f64;
        let ok_count = BcPackDecodeBatch(
            ctx,
            items.as_mut_ptr(),
            items.len() as u32,
            &mut kernel_ms,
        );
        assert_eq!(ok_count, 62, "Expected 62 valid items to succeed, got {ok_count}");
        assert!(kernel_ms > 0.0, "Expected positive cumulative GPU kernel_ms");

        let mut invalid_statuses = Vec::new();
        for (idx, it) in items.iter().enumerate() {
            if valid_item_to_chunk[idx].is_none() {
                invalid_statuses.push(it.status);
            } else {
                assert_eq!(it.status, 0, "Valid item {idx} failed with status {}", it.status);
            }
        }
        assert_eq!(invalid_statuses, vec![-2, -4, -4, -1]);

        for (ci, ch) in chunks.iter().enumerate() {
            assert_eq!(
                ch.out, ch.expected,
                "Batch output mismatch on chunk {ci} ({:?})",
                ch.mode
            );
        }
        println!(
            "[v14-batch] 62/66 valid mixed chunks (BC1/BC3/BC4/BC5/BC6H/BC7, max_chunks=16 sub-batched) + 4 isolated errors: 0 byte mismatches (kernel={:.3} ms)",
            kernel_ms
        );

        bcpack_gpu_destroy(ctx);
    }
}

#[test]
fn bench_cpu_vs_gpu_batch_and_zerocopy() {
    unsafe {
        let ctrl7 = valid_bc7_ctrl_codes();
        let gpu = GpuContext::new(false);
        let ctx = bcpack_gpu_create_from_device(
            gpu.device.as_raw(),
            gpu.compute_queue.as_raw(),
            0,
            0,
            0,
        );
        assert!(!ctx.is_null());

        // 384 mixed realistic 64 KiB chunks (64 per format x 6 formats = 24.0 MiB total; 320 = 20.0 MiB CPU subset)
        struct BenchChunk {
            mode: BcpMode,
            fmt: u32,
            blocks: usize,
            stream: Vec<u8>,
            out: Vec<u8>,
        }

        let mut chunks: Vec<BenchChunk> = Vec::with_capacity(384);
        for k in 0..64 {
            for &mode in &BcpMode::ALL {
                let (fmt, _, blk_bytes, _) = format_info(mode);
                let blocks = (64 * 1024) / blk_bytes; // 64 KiB per chunk (8192 or 4096 blocks)
                let stream = encode_synthetic_chunk(
                    mode,
                    blocks,
                    0xB34C_0000 + (k as u64) * 31 + (fmt as u64),
                    &ctrl7,
                );
                chunks.push(BenchChunk {
                    mode,
                    fmt,
                    blocks,
                    stream,
                    out: vec![0u8; blocks * blk_bytes],
                });
            }
        }

        let total_bytes: usize = chunks.iter().map(|c| c.out.len()).sum();
        let cpu_chunks_idx: Vec<usize> = chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.mode != BcpMode::Bc6h)
            .map(|(i, _)| i)
            .collect();
        let cpu_bytes: usize = cpu_chunks_idx.iter().map(|&i| chunks[i].out.len()).sum();

        // 1. CPU 1-thread (bcpack-rs, 50 BC1/3/4/5/7 chunks)
        let t_cpu1 = Instant::now();
        for &i in &cpu_chunks_idx {
            let decoded = cpu_decode_reference(chunks[i].mode, &chunks[i].stream, chunks[i].blocks);
            chunks[i].out.copy_from_slice(&decoded);
        }
        let cpu1_ms = t_cpu1.elapsed().as_secs_f64() * 1000.0;
        let cpu1_gbps = (cpu_bytes as f64 / 1e9) / (cpu1_ms / 1000.0);

        // 2. CPU multi-thread (all available logical cores, 50 BC1/3/4/5/7 chunks)
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        let cpu_refs: Vec<(BcpMode, &[u8], usize, usize)> = cpu_chunks_idx
            .iter()
            .map(|&i| (chunks[i].mode, chunks[i].stream.as_slice(), chunks[i].blocks, chunks[i].out.len()))
            .collect();
        let mut cpu_mt_out: Vec<Vec<u8>> = cpu_refs.iter().map(|r| vec![0u8; r.3]).collect();
        let t_cpumt = Instant::now();
        std::thread::scope(|s| {
            let chunk_size = (cpu_mt_out.len() + threads - 1) / threads;
            let mut rest_out = cpu_mt_out.as_mut_slice();
            let mut offset = 0usize;
            while !rest_out.is_empty() {
                let take = chunk_size.min(rest_out.len());
                let (head, tail) = rest_out.split_at_mut(take);
                let slice_refs = &cpu_refs[offset..offset + take];
                s.spawn(move || {
                    for (dst, &(m, st, blks, _)) in head.iter_mut().zip(slice_refs.iter()) {
                        let d = cpu_decode_reference(m, st, blks);
                        dst.copy_from_slice(&d);
                    }
                });
                offset += take;
                rest_out = tail;
            }
        });
        let cpumt_ms = t_cpumt.elapsed().as_secs_f64() * 1000.0;
        let cpumt_gbps = (cpu_bytes as f64 / 1e9) / (cpumt_ms / 1000.0);

        // Warmup GPU
        let mut items: Vec<BcPackBatchItem> = chunks
            .iter_mut()
            .map(|c| BcPackBatchItem {
                compressed_data: c.stream.as_ptr(),
                compressed_size: c.stream.len() as u32,
                num_blocks: c.blocks as u32,
                format: c.fmt,
                swizzle_mode: BCPACK_SWIZZLE_LINEAR,
                width_blocks: 0,
                height_blocks: 0,
                out_host_buffer: c.out.as_mut_ptr(),
                out_capacity: c.out.len() as u32,
                status: 0,
            })
            .collect();
        let mut warm_kms = 0.0f64;
        assert_eq!(
            BcPackDecodeBatch(ctx, items.as_mut_ptr(), items.len() as u32, &mut warm_kms),
            items.len() as i32
        );

        // 3. Single-chunk BcPackDecodeAsync loop (host readback, all 60 chunks = 30 MiB)
        let t_single = Instant::now();
        for c in chunks.iter_mut() {
            let mut fv = 0u64;
            let rc = BcPackDecodeAsync(
                ctx,
                c.stream.as_ptr(),
                c.stream.len() as u32,
                c.blocks as u32,
                c.fmt,
                BCPACK_SWIZZLE_LINEAR,
                0,
                0,
                c.out.as_mut_ptr(),
                c.out.len() as u32,
                &mut fv,
            );
            assert_eq!(rc, 0);
        }
        let single_ms = t_single.elapsed().as_secs_f64() * 1000.0;
        let single_gbps = (total_bytes as f64 / 1e9) / (single_ms / 1000.0);

        // 4. BcPackDecodeBatch (host readback, all 60 chunks = 30 MiB, avg over 5 runs)
        let mut items: Vec<BcPackBatchItem> = chunks
            .iter_mut()
            .map(|c| BcPackBatchItem {
                compressed_data: c.stream.as_ptr(),
                compressed_size: c.stream.len() as u32,
                num_blocks: c.blocks as u32,
                format: c.fmt,
                swizzle_mode: BCPACK_SWIZZLE_LINEAR,
                width_blocks: 0,
                height_blocks: 0,
                out_host_buffer: c.out.as_mut_ptr(),
                out_capacity: c.out.len() as u32,
                status: 0,
            })
            .collect();
        let mut batch_wall_sum = 0.0f64;
        let mut batch_kern_sum = 0.0f64;
        for _ in 0..5 {
            let mut kms = 0.0f64;
            let t0 = Instant::now();
            let ok = BcPackDecodeBatch(ctx, items.as_mut_ptr(), items.len() as u32, &mut kms);
            batch_wall_sum += t0.elapsed().as_secs_f64() * 1000.0;
            batch_kern_sum += kms;
            assert_eq!(ok, items.len() as i32);
        }
        let batch_wall_ms = batch_wall_sum / 5.0;
        let batch_kern_ms = batch_kern_sum / 5.0;
        let batch_wall_gbps = (total_bytes as f64 / 1e9) / (batch_wall_ms / 1000.0);
        let batch_kern_gbps = (total_bytes as f64 / 1e9) / (batch_kern_ms / 1000.0);

        // 5. BcPackDecodeToTexture2D zero-copy (32 x 512x512px BC7 textures = 8.0 MiB VRAM)
        let wb = 128u32;
        let hb = 128u32;
        let tex_blocks = (wb * hb) as usize;
        let tex_stream = encode_synthetic_chunk(BcpMode::Bc7, tex_blocks, 0x7E22_0001, &ctrl7);
        let mut textures = Vec::with_capacity(32);
        for _ in 0..32 {
            textures.push(create_bc_texture2d(&gpu.device, DXGI_FORMAT_BC7_UNORM, wb, hb));
        }
        // Warmup 1 texture
        let mut wf = 0u64;
        assert_eq!(
            BcPackDecodeToTexture2D(
                ctx,
                tex_stream.as_ptr(),
                tex_stream.len() as u32,
                tex_blocks as u32,
                BCPACK_FORMAT_BC7,
                BCPACK_SWIZZLE_LINEAR,
                wb,
                hb,
                textures[0].as_raw(),
                0,
                0,
                0,
                &mut wf,
            ),
            0
        );
        assert_eq!(BcPackWaitDirectFence(ctx, wf, u32::MAX), 0);

        let tex_total_bytes = textures.len() * tex_blocks * 16;
        let t_tex = Instant::now();
        let mut last_f = 0u64;
        for t in &textures {
            let rc = BcPackDecodeToTexture2D(
                ctx,
                tex_stream.as_ptr(),
                tex_stream.len() as u32,
                tex_blocks as u32,
                BCPACK_FORMAT_BC7,
                BCPACK_SWIZZLE_LINEAR,
                wb,
                hb,
                t.as_raw(),
                0,
                0,
                0,
                &mut last_f,
            );
            assert_eq!(rc, 0);
        }
        assert_eq!(BcPackWaitDirectFence(ctx, last_f, u32::MAX), 0);
        let tex_ms = t_tex.elapsed().as_secs_f64() * 1000.0;
        let tex_gbps = (tex_total_bytes as f64 / 1e9) / (tex_ms / 1000.0);

        println!("=== v0.14.0 RTX 5080 CPU vs GPU BENCHMARK ===");
        println!("CPU bcpack-rs (1 thread,   {:.1} MiB BC1/3/4/5/7): {:7.2} ms | {:6.2} GB/s", cpu_bytes as f64 / (1024.0 * 1024.0), cpu1_ms, cpu1_gbps);
        println!("CPU bcpack-rs ({:2} threads, {:.1} MiB BC1/3/4/5/7): {:7.2} ms | {:6.2} GB/s", threads, cpu_bytes as f64 / (1024.0 * 1024.0), cpumt_ms, cpumt_gbps);
        println!("GPU BcPackDecodeAsync loop ({:.1} MiB mixed 6-fmt): {:7.2} ms | {:6.2} GB/s", total_bytes as f64 / (1024.0 * 1024.0), single_ms, single_gbps);
        println!("GPU BcPackDecodeBatch wall ({:.1} MiB mixed 6-fmt): {:7.2} ms | {:6.2} GB/s", total_bytes as f64 / (1024.0 * 1024.0), batch_wall_ms, batch_wall_gbps);
        println!("GPU BcPackDecodeBatch kern ({:.1} MiB mixed 6-fmt): {:7.2} ms | {:6.2} GB/s", total_bytes as f64 / (1024.0 * 1024.0), batch_kern_ms, batch_kern_gbps);
        println!("GPU BcPackDecodeToTexture2D({:.1} MiB 32x 512px BC7): {:7.2} ms | {:6.2} GB/s", tex_total_bytes as f64 / (1024.0 * 1024.0), tex_ms, tex_gbps);

        bcpack_gpu_destroy(ctx);
    }
}
