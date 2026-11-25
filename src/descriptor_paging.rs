use crate::gpu_decoder::{
    cmd_barriers, create_committed_buffer, heap_props, transition_barrier, upload_slice, GpuContext,
};
use std::ffi::c_void;
use std::mem::{size_of, ManuallyDrop};
use std::time::Instant;
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};

pub const BINDLESS_PAGING_CS_DXIL: &[u8] = include_bytes!("../shaders/bindless_paging.dxil");

pub const VIRTUAL_DESC_COUNT: usize = 1_250_000;
pub const PHYS_HEAP_SLOTS: usize = 262_144;
pub const BACKING_TEX_COUNT: usize = 64;

#[inline]
pub fn backing_tex_rgba8(idx: usize) -> [u8; 4] {
    let i = idx % BACKING_TEX_COUNT;
    [
        ((i * 37 + 19) & 0xFF) as u8,
        ((i * 73 + 53) & 0xFF) as u8,
        ((i * 113 + 97) & 0xFF) as u8,
        255u8,
    ]
}

#[inline]
pub fn backing_tex_packed_u32(idx: usize) -> u32 {
    let c = backing_tex_rgba8(idx);
    (c[0] as u32) | ((c[1] as u32) << 8) | ((c[2] as u32) << 16) | ((c[3] as u32) << 24)
}

pub struct VirtualDescriptorPager {
    _backing_textures: Vec<ID3D12Resource>,
    _staging_heap: ID3D12DescriptorHeap,
    pub phys_heap: ID3D12DescriptorHeap,
    staging_base: D3D12_CPU_DESCRIPTOR_HANDLE,
    phys_cpu_base: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_inc: u32,
    pub virt_to_phys: Vec<u32>,
    phys_to_virt: Vec<u32>,
    next_phys_slot: usize,
    pub evictions: u64,
}

impl VirtualDescriptorPager {
    pub fn new(ctx: &mut GpuContext) -> Self {
        unsafe {
            let mut backing_textures = Vec::with_capacity(BACKING_TEX_COUNT);
            let mut cpu_rows = vec![0u8; BACKING_TEX_COUNT * 4 * 256];
            for ti in 0..BACKING_TEX_COUNT {
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: 4,
                    Height: 4,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                    Flags: D3D12_RESOURCE_FLAG_NONE,
                    ..Default::default()
                };
                let mut res: Option<ID3D12Resource> = None;
                ctx.device
                    .CreateCommittedResource(
                        &heap_props(D3D12_HEAP_TYPE_DEFAULT),
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut res,
                    )
                    .expect("CreateCommittedResource backing_tex");
                backing_textures.push(res.unwrap());

                let rgba = backing_tex_rgba8(ti);
                for row in 0..4usize {
                    let row_off = (ti * 4 + row) * 256;
                    for px in 0..4usize {
                        cpu_rows[row_off + px * 4..row_off + px * 4 + 4].copy_from_slice(&rgba);
                    }
                }
            }

            let upload_buf = create_committed_buffer(
                &ctx.device,
                D3D12_HEAP_TYPE_UPLOAD,
                cpu_rows.len() as u64,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            );
            upload_slice(&upload_buf, &cpu_rows);

            let alloc: ID3D12CommandAllocator = ctx
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .unwrap();
            let list: ID3D12GraphicsCommandList = ctx
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                .unwrap();

            for (ti, tex) in backing_textures.iter().enumerate() {
                let mut dst = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: ManuallyDrop::new(Some(tex.clone())),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                let mut src = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: ManuallyDrop::new(Some(upload_buf.clone())),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: (ti * 1024) as u64,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                                Width: 4,
                                Height: 4,
                                Depth: 1,
                                RowPitch: 256,
                            },
                        },
                    },
                };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                let _ = ManuallyDrop::take(&mut dst.pResource);
                let _ = ManuallyDrop::take(&mut src.pResource);
            }
            let to_srv: Vec<_> = backing_textures
                .iter()
                .map(|t| {
                    transition_barrier(
                        t,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    )
                })
                .collect();
            cmd_barriers(&list, to_srv);
            list.Close().unwrap();
            let exec = [Some(list.cast::<ID3D12CommandList>().unwrap())];
            ctx.queue.ExecuteCommandLists(&exec);
            ctx.signal_and_wait();

            // Non-shader-visible staging heap (required source for CopyDescriptorsSimple)
            let staging_heap: ID3D12DescriptorHeap = ctx
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: BACKING_TEX_COUNT as u32,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })
                .expect("CreateDescriptorHeap staging");

            // Shader-visible physical descriptor heap (262,144 slots < ~1M hardware limit)
            let phys_heap: ID3D12DescriptorHeap = ctx
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: PHYS_HEAP_SLOTS as u32,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .expect("CreateDescriptorHeap phys_heap");

            let srv_inc = ctx
                .device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            let staging_base = staging_heap.GetCPUDescriptorHandleForHeapStart();
            let phys_cpu_base = phys_heap.GetCPUDescriptorHandleForHeapStart();

            for (ti, tex) in backing_textures.iter().enumerate() {
                let mut sd = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                    ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    ..Default::default()
                };
                sd.Anonymous.Texture2D = D3D12_TEX2D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                };
                ctx.device.CreateShaderResourceView(
                    tex,
                    Some(&sd),
                    D3D12_CPU_DESCRIPTOR_HANDLE {
                        ptr: staging_base.ptr + ti * srv_inc as usize,
                    },
                );
            }

            Self {
                _backing_textures: backing_textures,
                _staging_heap: staging_heap,
                phys_heap,
                staging_base,
                phys_cpu_base,
                srv_inc,
                virt_to_phys: vec![u32::MAX; VIRTUAL_DESC_COUNT],
                phys_to_virt: vec![u32::MAX; PHYS_HEAP_SLOTS],
                next_phys_slot: 0,
                evictions: 0,
            }
        }
    }

    pub fn page_in(&mut self, device: &ID3D12Device, virt_id: u32) -> (u32, bool) {
        let vid = virt_id as usize;
        let existing = self.virt_to_phys[vid];
        if existing != u32::MAX {
            return (existing, false);
        }

        let phys_slot = self.next_phys_slot;
        self.next_phys_slot = (self.next_phys_slot + 1) % PHYS_HEAP_SLOTS;

        let prev_virt = self.phys_to_virt[phys_slot];
        if prev_virt != u32::MAX {
            self.virt_to_phys[prev_virt as usize] = u32::MAX;
            self.evictions += 1;
        }

        let src = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.staging_base.ptr + (vid % BACKING_TEX_COUNT) * self.srv_inc as usize,
        };
        let dst = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.phys_cpu_base.ptr + phys_slot * self.srv_inc as usize,
        };
        unsafe {
            device.CopyDescriptorsSimple(
                1,
                dst,
                src,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            );
        }

        self.virt_to_phys[vid] = phys_slot as u32;
        self.phys_to_virt[phys_slot] = virt_id;
        (phys_slot as u32, true)
    }
}

pub struct DescriptorPagingVerifyStats {
    pub virtual_count: usize,
    pub physical_slots: usize,
    pub accesses: usize,
    pub evictions: u64,
    pub copy_ms: f64,
    pub gpu_ms: f64,
    pub verified_samples: usize,
}

pub fn run_descriptor_paging_verify(ctx: &mut GpuContext) -> DescriptorPagingVerifyStats {
    let mut pager = VirtualDescriptorPager::new(ctx);

    let copy_t0 = Instant::now();
    // Phase 1: fill all 262,144 physical heap slots with virtual IDs 0..262,143
    for vid in 0..PHYS_HEAP_SLOTS as u32 {
        let _ = pager.page_in(&ctx.device, vid);
    }
    // Phase 2: page in 37,856 high virtual IDs spanning 990,000..1,249,999 (over NVIDIA's 1M cap)
    let high_ops = 37_856usize;
    let mut sampled_vids = Vec::with_capacity(65_536);
    for k in 0..high_ops {
        let vid = (990_000 + (k * 7) % 260_000) as u32;
        let _ = pager.page_in(&ctx.device, vid);
        if sampled_vids.len() < 32_768 {
            sampled_vids.push(vid);
        }
    }
    let copy_ms = copy_t0.elapsed().as_secs_f64() * 1000.0;

    // Fill remaining samples with still-resident low virtual IDs
    let mut low_vid = (high_ops + 16) as u32;
    while sampled_vids.len() < 65_536 {
        if pager.virt_to_phys[low_vid as usize] != u32::MAX {
            sampled_vids.push(low_vid);
        }
        low_vid = (low_vid + 1) % (PHYS_HEAP_SLOTS as u32);
    }

    assert!(
        pager.evictions >= high_ops as u64 - 4096,
        "Expected physical heap evictions, got {}",
        pager.evictions
    );

    unsafe {
        let req_bytes = (sampled_vids.len() * size_of::<u32>()) as u64;
        let map_bytes = (VIRTUAL_DESC_COUNT * size_of::<u32>()) as u64;

        let req_up = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_UPLOAD,
            req_bytes,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        let map_up = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_UPLOAD,
            map_bytes,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        upload_slice(&req_up, &sampled_vids);
        upload_slice(&map_up, &pager.virt_to_phys);

        let req_def = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_DEFAULT,
            req_bytes,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let map_def = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_DEFAULT,
            map_bytes,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let out_def = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_DEFAULT,
            req_bytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        let out_rb = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_READBACK,
            req_bytes,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );

        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: 4,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR {
                        ShaderRegister: 1,
                        RegisterSpace: 0,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];

        let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: params.len() as u32,
            pParameters: params.as_ptr(),
            NumStaticSamplers: 0,
            pStaticSamplers: std::ptr::null(),
            Flags: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED,
        };
        let mut blob = None;
        let mut err = None;
        D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, Some(&mut err))
            .expect("D3D12SerializeRootSignature bindless_paging");
        let blob = blob.unwrap();
        let rs_bytes = std::slice::from_raw_parts(
            blob.GetBufferPointer() as *const u8,
            blob.GetBufferSize(),
        );
        let root_sig: ID3D12RootSignature = ctx
            .device
            .CreateRootSignature(0, rs_bytes)
            .expect("CreateRootSignature bindless_paging");

        let mut pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(root_sig.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: BINDLESS_PAGING_CS_DXIL.as_ptr() as *const c_void,
                BytecodeLength: BINDLESS_PAGING_CS_DXIL.len(),
            },
            NodeMask: 0,
            CachedPSO: D3D12_CACHED_PIPELINE_STATE::default(),
            Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
        };
        let pso: ID3D12PipelineState = ctx
            .device
            .CreateComputePipelineState(&pso_desc)
            .expect("CreateComputePipelineState bindless_paging");
        let _ = ManuallyDrop::take(&mut pso_desc.pRootSignature);

        let mut ts_heap: Option<ID3D12QueryHeap> = None;
        ctx.device
            .CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                    Count: 2,
                    NodeMask: 0,
                },
                &mut ts_heap,
            )
            .unwrap();
        let ts_heap = ts_heap.unwrap();
        let ts_rb = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_READBACK,
            16,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );

        let alloc: ID3D12CommandAllocator = ctx
            .device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .unwrap();
        let cmd: ID3D12GraphicsCommandList = ctx
            .device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
            .unwrap();

        cmd.CopyBufferRegion(&req_def, 0, &req_up, 0, req_bytes);
        cmd.CopyBufferRegion(&map_def, 0, &map_up, 0, map_bytes);
        cmd_barriers(
            &cmd,
            vec![
                transition_barrier(
                    &req_def,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                ),
                transition_barrier(
                    &map_def,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                ),
            ],
        );

        cmd.SetDescriptorHeaps(&[Some(pager.phys_heap.clone())]);
        cmd.SetComputeRootSignature(&root_sig);
        cmd.SetPipelineState(&pso);
        let cb = [
            sampled_vids.len() as u32,
            VIRTUAL_DESC_COUNT as u32,
            PHYS_HEAP_SLOTS as u32,
            0u32,
        ];
        cmd.SetComputeRoot32BitConstants(0, 4, cb.as_ptr() as *const c_void, 0);
        cmd.SetComputeRootShaderResourceView(1, req_def.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(2, map_def.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(3, out_def.GetGPUVirtualAddress());

        cmd.EndQuery(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
        cmd.Dispatch((sampled_vids.len() as u32).div_ceil(64), 1, 1);
        cmd.EndQuery(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);

        cmd_barriers(
            &cmd,
            vec![transition_barrier(
                &out_def,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )],
        );
        cmd.CopyBufferRegion(&out_rb, 0, &out_def, 0, req_bytes);
        cmd.ResolveQueryData(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &ts_rb, 0);
        cmd.Close().unwrap();

        let exec = [Some(cmd.cast::<ID3D12CommandList>().unwrap())];
        ctx.queue.ExecuteCommandLists(&exec);
        ctx.signal_and_wait();

        let mut ts_ptr: *mut c_void = std::ptr::null_mut();
        ts_rb.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 16 }), Some(&mut ts_ptr)).unwrap();
        let ts = std::slice::from_raw_parts(ts_ptr as *const u64, 2);
        let gpu_ms = (ts[1].saturating_sub(ts[0]) as f64 / ctx.timestamp_freq as f64) * 1000.0;
        ts_rb.Unmap(0, None);

        let mut out_ptr: *mut c_void = std::ptr::null_mut();
        out_rb
            .Map(0, Some(&D3D12_RANGE { Begin: 0, End: req_bytes as usize }), Some(&mut out_ptr))
            .unwrap();
        let got = std::slice::from_raw_parts(out_ptr as *const u32, sampled_vids.len());
        for (idx, (&vid, &px)) in sampled_vids.iter().zip(got.iter()).enumerate() {
            let exp = backing_tex_packed_u32(vid as usize);
            assert_eq!(
                px, exp,
                "FAIL descriptor paging sample {idx} (virt_id={vid}): got={px:#010x} exp={exp:#010x}"
            );
        }
        out_rb.Unmap(0, None);

        DescriptorPagingVerifyStats {
            virtual_count: VIRTUAL_DESC_COUNT,
            physical_slots: PHYS_HEAP_SLOTS,
            accesses: PHYS_HEAP_SLOTS + high_ops,
            evictions: pager.evictions,
            copy_ms,
            gpu_ms,
            verified_samples: sampled_vids.len(),
        }
    }
}
