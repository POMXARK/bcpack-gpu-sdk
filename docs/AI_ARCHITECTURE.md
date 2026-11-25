# `bcpack-gpu-sdk` — AI/LLM Machine-Readable Hardware SDK Architecture (`v0.11.0`)

> **Target audience:** Coding agents, automated integration planners, and GPU systems engineers working on Xbox Series X|S translation layers (`xcl`, `OpenScarlett`, `AnyXboxX`), DirectStorage GPU hooks, or [`Ahmet-Dedeler/bcpack-rs`](https://github.com/Ahmet-Dedeler/bcpack-rs) backends.
> **License & Provenance:** `MIT OR Apache-2.0`. 100% clean-room implementation derived from public specs (`Ahmet-Dedeler/bcpack-rs`, `microsoft/silicon-rans`). Zero proprietary game assets or SDK binaries.

---

## 1. SDK Module & Shader Map

| Component | Path | Purpose | Entrypoints / Bindings |
|---|---|---|---|
| **C-ABI Header** | [`include/bcpack_gpu.h`](../include/bcpack_gpu.h) | Drop-in DX12 Async Compute + Direct upload API (`cdylib`) | `bcpack_gpu_create_from_device`, `BcPackDecodeToBuffer`, `BcPackDecodeToTexture2D`, `BcPackGetComputeFence`, `BcPackGetDirectFence`, `BcPackDecodeAsync` |
| **FFI Wrapper** | [`src/ffi.rs`](../src/ffi.rs) | Thread-safe C-ABI implementation | `BcPackFormat::{Bc1=0, Bc7Rans=1}`, `BcPackSwizzle::{Linear=0, Gfx10_64KbD=1, Gfx10_4KbS=2}` |
| **GPU Decoder Core** | [`src/gpu_decoder.rs`](../src/gpu_decoder.rs) | DX12 adapter discovery (incl. `IDXGIAdapter3` VRAM + VRS Tier 2 probe), Async Compute pipelines, `Bc16GpuStreamer` | `GpuContext`, `BcpackGpuDecoder`, `Bc7RansGpuDecoder`, `Gfx10DetileGpu`, `Bc16GpuStreamer`, `uav_barrier_all` |
| **Descriptor Pager** | [`src/descriptor_paging.rs`](../src/descriptor_paging.rs), [`shaders/bindless_paging.hlsl`](../shaders/bindless_paging.hlsl) | Virtual `1,250,000` SRVs paged onto `262,144` physical Tier-3 slots | `VirtualDescriptorPager`, `ResourceDescriptorHeap[NonUniformResourceIndex(phys)]` |
| **Crowd Skinning** | [`src/crowd_skinning.rs`](../src/crowd_skinning.rs), [`shaders/crowd_skin.hlsl`](../shaders/crowd_skin.hlsl) | `500` NPCs × 4-bone Linear Blend Skinning Compute pass | `t0` (`BaseSkinnedVertex`), `t1` (`BoneMat`), `u0` (`OutSkinnedVertex`), `b0` (`CrowdCB`) |
| **BC1 Wave32 Decoder** | [`shaders/bcpack_decode.hlsl`](../shaders/bcpack_decode.hlsl) | Dual-chunk (`2×12` = `24/32`) zero-barrier rANS + predictor + quad reconstruction | `[WaveSize(32)]`, `t0` (`g_Models`), `t1` (`g_Window`), `t2` (`g_QuadMasks`), `t3` (`g_Streams`), `t4` (`g_Chunks`), `u0` (`uint2`), `u1` (`g_Status`) |
| **BC7 Wave32 rANS** | [`shaders/bcpack_bc7_rans.hlsl`](../shaders/bcpack_bc7_rans.hlsl) | Dual-chunk (`2×16` = `32/32` 100%) 16-lane rANS entropy stage | `[WaveSize(32)]`, `t0` (`g_Models`), `t1` (`g_Streams`), `t2` (`g_Chunks`), `u0` (`uint` = `8` DWORDs/row), `u1` (`g_Status`) |
| **16B Block Packer** | [`shaders/bc16_pack.hlsl`](../shaders/bc16_pack.hlsl) | Repacks `10` active lanes from `16`-lane rANS into `8B/row` (`16B` BC7/BC5 blocks) | `[numthreads(64,1,1)]`, `t0` (`g_InSyms`), `u0` (`g_OutBlocks`), `b0` (`PackCB`) |
| **AMD Gfx10 Detiler** | [`shaders/gfx10_detile.hlsl`](../shaders/gfx10_detile.hlsl) | Inline tile-to-linear `BCn` compute detile | `[numthreads(64,1,1)]`, `t0` (`g_Src`), `u0` (`g_Dst`), `b0` (`DetileParams`: `0=SW_64KB_D 8B`, `1=SW_4KB_S 8B`, `2=SW_64KB_D 16B`) |
| **Hi-Z Pyramid & Cull** | [`shaders/hiz_build.hlsl`](../shaders/hiz_build.hlsl), [`shaders/hiz_cull.hlsl`](../shaders/hiz_cull.hlsl) | Conservative MAX depth downsample & bounding-sphere Hi-Z cull | `hiz_build`: `8×8`, `t0`→`u0`; `hiz_cull`: `64`, `t0..t8`, `u0`, `b0` |
| **VRS Tier 2 Rate Gen** | [`shaders/aaa_vrs.hlsl`](../shaders/aaa_vrs.hlsl) | `16×16` tile `R8_UINT` shading-rate mask (`1×1` / `2×2` / `4×4`) | `8×8`, `t0` (`MotionVec`), `t1` (`ColorHDR`), `u0` (`g_VrsOut`), `s0`, `b0` (`VrsCB`) |
| **Compute RayQuery RTGI** | [`shaders/aaa_rtgi_cs.hlsl`](../shaders/aaa_rtgi_cs.hlsl) | Cross-vendor Xbox/RDNA2-style tiled Compute `RayQuery` diffuse GI | `8×8` (`VK_KHR_ray_query`), `t0..t6` (`t2`=TLAS), `u0`, `s0`, `b0` (`RtgiCsCB`) |
| **Vulkan 1.3 SPIR-V** | [`shaders/spirv/*.spv`](../shaders/compile_spirv.sh) | 9 precompiled SPIR-V binaries (`dxc v1.9.2609 -spirv -fspv-target-env=vulkan1.3 -O3`) | Ready for Vulkan 1.3/1.4 & Linux/Proton layers |

---

## 2. Critical Engineering Invariants

### A. Zero-Barrier Wave32 rANS & DXC `-O3` Ballot Workaround
Both `bcpack_decode.hlsl` and `bcpack_bc7_rans.hlsl` eliminate all `GroupMemoryBarrierWithGroupSync()` calls inside the hot decode loop by keeping a `64B` stream sliding window in registers (`WaveReadLaneAt`) and transposing output lanes in registers.

**CRITICAL:** Under `dxc.exe -T cs_6_6 -O3`, combining `WavePrefixCountBits(want)` + `WaveActiveCountBits(want)` in the split-half Wave32 byte-allocator loop miscompiles `stream_pos`. Always preserve the explicit ballot mask implementation:
```hlsl
uint4 ballot = WaveActiveBallot(want_byte);
uint mask = ballot.x & half_mask;           // isolate lanes 0..15 or 16..31
uint prefix = countbits(mask & ((1u << ll) - 1u));
uint advance = countbits(mask);
```

### B. C-ABI Memory & Cross-Queue Fence Contract (`bcpack_gpu.h`)
1. `bcpack_gpu_create_from_device(d3d12_device, d3d12_compute_queue, ...)`:
   - Increments COM refcount (`AddRef`) on caller `ID3D12Device*`; caller retains device lifetime ownership.
   - If `d3d12_compute_queue == NULL`, creates an internal `D3D12_COMMAND_LIST_TYPE_COMPUTE` queue.
2. **Resource State Invariant:** `dst_d3d12_resource` (`BcPackDecodeToBuffer`) and `dst_texture2d` (`BcPackDecodeToTexture2D`) MUST belong to `d3d12_device` and enter in **`D3D12_RESOURCE_STATE_COMMON`**. Both functions restore `COMMON` before signaling `*out_fence_value`.
3. **256B Row Pitch Padding (`BcPackDecodeToTexture2D`):** Small mip levels whose uncompressed block row width is `< 256` bytes (e.g., `64×64` BC1 = `128 B/row`) are automatically padded in a scratch upload/copy buffer on the internal `DIRECT` queue before `CopyTextureRegion`, eliminating DX12 Debug Layer alignment faults.
4. **Teardown Safety:** `bcpack_gpu_destroy(ctx)` drains both `compute_fence_value` and `direct_fence_value` before releasing COM heaps.

### C. NVIDIA Driver `621.24` RT-Core Cache Isolation
When mixing Async Compute BCPack `CopyTextureRegion` uploads with DXR 1.1 `RayQuery` (`aaa_rtgi_cs.hlsl`), run `uav_barrier_all()` + an explicit fence boundary around TLAS builds prior to tracing rays to prevent transient `8×8` tile-cache artifacts on Blackwell GPUs.

### D. Vulkan 1.3 Subgroup Size Control
- `bcpack_decode.spv` and `bcpack_bc7_rans.spv` declare `[WaveSize(32)]`. On 64-wide AMD RDNA2/RDNA3 Vulkan drivers, attach `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` (`requiredSubgroupSize = 32`) via `VK_EXT_subgroup_size_control`.
- `aaa_rtgi_cs.spv` uses `[numthreads(8, 8, 1)]` without `[WaveSize(32)]`, executing natively as `2× Wave32` on NVIDIA or `1× Wave64` on AMD RDNA2/RDNA3 under `VK_KHR_ray_query`.
