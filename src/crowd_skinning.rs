use crate::gpu_decoder::{
    cmd_barriers, create_committed_buffer, transition_barrier, uav_barrier, upload_slice,
    GpuContext,
};
use std::ffi::c_void;
use std::mem::{size_of, ManuallyDrop};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

pub const CROWD_SKIN_CS_DXIL: &[u8] = include_bytes!("../shaders/crowd_skin.dxil");

pub const NPC_COUNT: usize = 500;
pub const VERTS_PER_NPC: usize = 1024;
pub const BONES_PER_NPC: usize = 64;
pub const TOTAL_SKINNED_VERTS: usize = NPC_COUNT * VERTS_PER_NPC; // 512,000 vertices

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BaseSkinnedVertex {
    pub pos: [f32; 4],
    pub normal: [f32; 4],
    pub bone_idx: [u32; 4],
    pub bone_wt: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BoneMat {
    pub c0: [f32; 4],
    pub c1: [f32; 4],
    pub c2: [f32; 4],
    pub c3: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct OutSkinnedVertex {
    pub pos: [f32; 4],
    pub normal: [f32; 4],
}

pub struct CrowdSkinVerifyStats {
    pub npc_count: usize,
    pub total_verts: usize,
    pub total_bones: usize,
    pub verified_samples: usize,
    pub max_err: f32,
    pub gpu_ms: f64,
}

fn cpu_skin_vertex(bv: &BaseSkinnedVertex, bones: &[BoneMat], bone_base: usize) -> ([f32; 3], [f32; 3]) {
    let mut sp = [0.0f32; 3];
    let mut sn = [0.0f32; 3];
    for k in 0..4 {
        let m = &bones[bone_base + bv.bone_idx[k] as usize];
        let w = bv.bone_wt[k];
        for c in 0..3 {
            sp[c] += (m.c0[c] * bv.pos[0] + m.c1[c] * bv.pos[1] + m.c2[c] * bv.pos[2] + m.c3[c]) * w;
            sn[c] += (m.c0[c] * bv.normal[0] + m.c1[c] * bv.normal[1] + m.c2[c] * bv.normal[2]) * w;
        }
    }
    let len_sq = sn[0] * sn[0] + sn[1] * sn[1] + sn[2] * sn[2];
    if len_sq > 1e-8 {
        let inv = 1.0 / len_sq.sqrt();
        sn = [sn[0] * inv, sn[1] * inv, sn[2] * inv];
    } else {
        sn = [0.0, 1.0, 0.0];
    }
    (sp, sn)
}

pub fn run_crowd_skinning_verify(ctx: &mut GpuContext) -> CrowdSkinVerifyStats {
    let mut base_verts = Vec::with_capacity(VERTS_PER_NPC);
    for vi in 0..VERTS_PER_NPC {
        let t = (vi as f32 / VERTS_PER_NPC as f32) * std::f32::consts::TAU;
        let p = [t.cos() * 0.45, (vi as f32 / VERTS_PER_NPC as f32) * 1.8, t.sin() * 0.45, 1.0];
        let n_raw = [t.cos(), 0.25, t.sin()];
        let nl = (n_raw[0] * n_raw[0] + n_raw[1] * n_raw[1] + n_raw[2] * n_raw[2]).sqrt();
        let b0 = (vi % BONES_PER_NPC) as u32;
        let b1 = ((vi + 7) % BONES_PER_NPC) as u32;
        let b2 = ((vi + 19) % BONES_PER_NPC) as u32;
        let b3 = ((vi + 31) % BONES_PER_NPC) as u32;
        base_verts.push(BaseSkinnedVertex {
            pos: p,
            normal: [n_raw[0] / nl, n_raw[1] / nl, n_raw[2] / nl, 0.0],
            bone_idx: [b0, b1, b2, b3],
            bone_wt: [0.45, 0.30, 0.15, 0.10],
        });
    }

    let total_bones = NPC_COUNT * BONES_PER_NPC;
    let mut bones = Vec::with_capacity(total_bones);
    for npc in 0..NPC_COUNT {
        let px = ((npc % 25) as f32 - 12.0) * 3.2;
        let pz = ((npc / 25) as f32) * 3.5;
        let base_yaw = (npc as f32) * 0.19;
        for b in 0..BONES_PER_NPC {
            let ang = base_yaw + (b as f32) * 0.07;
            let (s, c) = ang.sin_cos();
            let sway = ((npc + b) as f32 * 0.13).sin() * 0.15;
            bones.push(BoneMat {
                c0: [c, 0.0, -s, 0.0],
                c1: [0.0, 1.0, 0.0, 0.0],
                c2: [s, 0.0, c, 0.0],
                c3: [px + sway, 0.05 * (b % 4) as f32, pz, 1.0],
            });
        }
    }

    unsafe {
        let bv_bytes = (base_verts.len() * size_of::<BaseSkinnedVertex>()) as u64;
        let bm_bytes = (bones.len() * size_of::<BoneMat>()) as u64;
        let out_bytes = (TOTAL_SKINNED_VERTS * size_of::<OutSkinnedVertex>()) as u64;

        let bv_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, bv_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        let bm_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, bm_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        upload_slice(&bv_up, &base_verts);
        upload_slice(&bm_up, &bones);

        let bv_def = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, bv_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        let bm_def = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, bm_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        let out_def = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, out_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        let out_rb = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, out_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

        let params = [
            D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, Anonymous: D3D12_ROOT_PARAMETER_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 4 } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
            D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV, Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
            D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV, Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 1, RegisterSpace: 0 } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
            D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV, Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
        ];
        let rs_desc = D3D12_ROOT_SIGNATURE_DESC { NumParameters: params.len() as u32, pParameters: params.as_ptr(), NumStaticSamplers: 0, pStaticSamplers: std::ptr::null(), Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE };
        let mut blob = None;
        D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None).unwrap();
        let blob = blob.unwrap();
        let root_sig: ID3D12RootSignature = ctx.device.CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize())).unwrap();

        let mut pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(root_sig.clone())),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: CROWD_SKIN_CS_DXIL.as_ptr() as *const c_void, BytecodeLength: CROWD_SKIN_CS_DXIL.len() },
            NodeMask: 0,
            CachedPSO: D3D12_CACHED_PIPELINE_STATE::default(),
            Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
        };
        let pso: ID3D12PipelineState = ctx.device.CreateComputePipelineState(&pso_desc).unwrap();
        let _ = ManuallyDrop::take(&mut pso_desc.pRootSignature);

        let mut ts_heap: Option<ID3D12QueryHeap> = None;
        ctx.device.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 2, NodeMask: 0 }, &mut ts_heap).unwrap();
        let ts_heap = ts_heap.unwrap();
        let ts_rb = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, 16, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

        let alloc: ID3D12CommandAllocator = ctx.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let cmd: ID3D12GraphicsCommandList = ctx.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();

        cmd.CopyBufferRegion(&bv_def, 0, &bv_up, 0, bv_bytes);
        cmd.CopyBufferRegion(&bm_def, 0, &bm_up, 0, bm_bytes);
        cmd_barriers(&cmd, vec![
            transition_barrier(&bv_def, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition_barrier(&bm_def, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        ]);

        cmd.SetComputeRootSignature(&root_sig);
        cmd.SetPipelineState(&pso);
        let cb = [NPC_COUNT as u32, VERTS_PER_NPC as u32, BONES_PER_NPC as u32, 0u32];
        cmd.SetComputeRoot32BitConstants(0, 4, cb.as_ptr() as *const c_void, 0);
        cmd.SetComputeRootShaderResourceView(1, bv_def.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(2, bm_def.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(3, out_def.GetGPUVirtualAddress());

        cmd.EndQuery(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
        cmd.Dispatch((TOTAL_SKINNED_VERTS as u32).div_ceil(64), 1, 1);
        cmd.EndQuery(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);

        cmd_barriers(&cmd, vec![
            uav_barrier(&out_def),
            transition_barrier(&out_def, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        ]);
        cmd.CopyBufferRegion(&out_rb, 0, &out_def, 0, out_bytes);
        cmd.ResolveQueryData(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &ts_rb, 0);
        cmd.Close().unwrap();

        let lists = [Some(cmd.cast::<ID3D12CommandList>().unwrap())];
        ctx.queue.ExecuteCommandLists(&lists);
        ctx.signal_and_wait();

        let mut ts_ptr: *mut c_void = std::ptr::null_mut();
        ts_rb.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 16 }), Some(&mut ts_ptr)).unwrap();
        let ts = std::slice::from_raw_parts(ts_ptr as *const u64, 2);
        let gpu_ms = (ts[1].saturating_sub(ts[0]) as f64 / ctx.timestamp_freq as f64) * 1000.0;
        ts_rb.Unmap(0, None);

        let mut out_ptr: *mut c_void = std::ptr::null_mut();
        out_rb.Map(0, Some(&D3D12_RANGE { Begin: 0, End: out_bytes as usize }), Some(&mut out_ptr)).unwrap();
        let got = std::slice::from_raw_parts(out_ptr as *const OutSkinnedVertex, TOTAL_SKINNED_VERTS);

        let mut max_err = 0.0f32;
        let mut verified = 0usize;
        let step = 125usize; // 512,000 / 125 = 4,096 samples across all 500 NPCs
        for idx in (0..TOTAL_SKINNED_VERTS).step_by(step) {
            let npc_id = idx / VERTS_PER_NPC;
            let local_v = idx % VERTS_PER_NPC;
            let (ep, en) = cpu_skin_vertex(&base_verts[local_v], &bones, npc_id * BONES_PER_NPC);
            let gv = &got[idx];
            for k in 0..3 {
                let dp = (gv.pos[k] - ep[k]).abs();
                let dn = (gv.normal[k] - en[k]).abs();
                assert!(!dp.is_nan() && !dn.is_nan(), "NaN in skinned vertex {idx}");
                max_err = max_err.max(dp).max(dn);
            }
            assert!(max_err < 1e-3, "Skinned vertex mismatch at idx {idx}: max_err={max_err}");
            verified += 1;
        }
        out_rb.Unmap(0, None);

        CrowdSkinVerifyStats {
            npc_count: NPC_COUNT,
            total_verts: TOTAL_SKINNED_VERTS,
            total_bones,
            verified_samples: verified,
            max_err,
            gpu_ms,
        }
    }
}
