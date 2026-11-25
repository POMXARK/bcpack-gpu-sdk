# bcpack-gpu-sdk — Hardware GPU BCPack Decoder, Gfx10 Detiler & Compute Kernel SDK

[![DirectX 12](https://img.shields.io/badge/DirectX%2012-SM%206.6%20%2B%20Async%20Compute-2f80ed)](shaders/bcpack_decode.hlsl)
[![Vulkan 1.3 SPIR-V](https://img.shields.io/badge/Vulkan%201.3-SPIR--V%20(9%20Kernels)-e74c3c)](shaders/spirv)
[![C-ABI DLL](https://img.shields.io/badge/C--ABI-bcpack__gpu.dll%20%2B%20bcpack__gpu.h-blue)](include/bcpack_gpu.h)
[![Verified 19/19 PASS](https://img.shields.io/badge/RTX%205080-19%2F19%20Bit--Exact%20PASS-brightgreen)](#verified-hardware-benchmarks-nvidia-rtx-5080)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-green)](LICENSE-MIT)

**Standalone, bloat-free clean-room hardware GPU compute SDK (`MIT OR Apache-2.0`) for Xbox Series X|S translation layers, DirectStorage `IDStorageQueueX` hooks, Vulkan 1.3 Linux layers (`OpenScarlett` / Proton), and GPU backends for [`Ahmet-Dedeler/bcpack-rs`](https://github.com/Ahmet-Dedeler/bcpack-rs).**

No game assets, no demo media bloat, zero copyrighted data — only the production C-ABI DLL (`bcpack_gpu.dll`), Rust crate (`bcpack-gpu-sdk`), `include/bcpack_gpu.h`, **10 HLSL SM 6.6 + DXIL compute kernels**, and **9 precompiled Vulkan 1.3 SPIR-V (`.spv`) kernels**.

> Looking for the full 4K streaming benchmark, 12.41M Mesh-Shader showcase, MP4/GIF captures, and 19-test verification harness? See the companion showcase repository: **[`POMXARK/bcpack-gpu-hlsl`](https://github.com/POMXARK/bcpack-gpu-hlsl)** (which links this SDK via Git submodule at `sdk/`).
> AI coding agents & crawlers: read **[`llms.txt`](llms.txt)** and **[`docs/AI_ARCHITECTURE.md`](docs/AI_ARCHITECTURE.md)**.

---

## Core Capabilities

1. **Zero-Barrier Wave32 BCPack MDU rANS Decoders (`11.33 GB/s` BC1, `6.02 GB/s` BC7 rANS):**
   - Dual-chunk warp packing (`2 × 12` lanes for BC1 = `24/32` active; `2 × 16` lanes for BC7 = **`32/32` 100% Wave32 utilization**).
   - Register-resident `64B` stream sliding window with `WaveReadLaneAt` symbol transpose and 4-step branchless binary search — **zero `GroupMemoryBarrierWithGroupSync()` inside hot loops**.
2. **GPU 16B Block Packer (`bc16_pack.hlsl`) & Fused AMD Gfx10 Swizzle Detiler (`432.6 GB/s`):**
   - Collapses 16-lane BC7/BC5 symbol rows into packed `16B` blocks (`Store2`) on GPU VRAM.
   - Inline RDNA2 tile bit-permutation (`SW_64KB_D` 8B/16B, `SW_4KB_S` 8B) executed on `D3D12_COMMAND_LIST_TYPE_COMPUTE` before `CopyTextureRegion`.
3. **Zero-Copy Shared-Device C-ABI (`bcpack_gpu.dll` + `include/bcpack_gpu.h`):**
   - Attach directly to a caller's `ID3D12Device*` (`bcpack_gpu_create_from_device`) and decode + detile straight into `D3D12_HEAP_TYPE_DEFAULT` buffers (`BcPackDecodeToBuffer`) or `DXGI_FORMAT_BC1_UNORM` 2D subresources (`BcPackDecodeToTexture2D`, automatic `256B` row-pitch padding) synchronized via cross-queue `ID3D12Fence` (`0` CPU readback).
4. **SM 6.6 Virtual `ResourceDescriptorHeap[]` Pager (`1.25M → 262K` slots in `0.064 ms`):**
   - Solves NVIDIA DX12 `~1M` shader-visible descriptor heap `E_OUTOFMEMORY` crashes via `CopyDescriptorsSimple` LRU paging + `ResourceDescriptorHeap[NonUniformResourceIndex(g_VirtToPhys[id])]`.
5. **Cross-Vendor Console Compute Helpers (DXIL + Vulkan 1.3 SPIR-V):**
   - Conservative 6-mip Hi-Z pyramid (`hiz_build`, `hiz_cull`), 4-bone GPU Crowd Skinning (`crowd_skin`), Hardware VRS Tier 2 tile classifier (`aaa_vrs`), and Xbox/RDNA2-style Compute `RayQuery` Diffuse RTGI (`aaa_rtgi_cs`, `VK_KHR_ray_query`).

---

## Quick Start

### 1. C / C++ Zero-Copy DirectStorage / DX12 Hook (`bcpack_gpu.dll`)

Download `bcpack-gpu-sdk-v0.11.0-win64-spirv.zip` from [Releases](https://github.com/POMXARK/bcpack-gpu-sdk/releases) (or build with `cargo build --release`), then link `bcpack_gpu.dll.lib`:

```c
#include "bcpack_gpu.h"

// Attach to caller's existing ID3D12Device* (zero ownership transfer):
BcPackContext* ctx = bcpack_gpu_create_from_device(
    d3d12_device,
    NULL,           // internal D3D12_COMMAND_LIST_TYPE_COMPUTE queue
    2048,           // max chunks per batch
    8388608,        // max BC blocks
    96u * 1024 * 1024);

// Decode + Gfx10 SW_64KB_D detile directly into caller Texture2D subresource (zero CPU copy):
uint64_t fence = 0;
BcPackDecodeToTexture2D(
    ctx,
    compressed_bytes, compressed_size,
    width_blocks * height_blocks,
    BCPACK_FORMAT_BC1,
    BCPACK_SWIZZLE_GFX10_64KB_D,
    width_blocks, height_blocks,
    dst_texture2d, subresource_idx, 0, 0, &fence);

// Pure GPU-side handoff before sampling in graphics shaders:
direct_queue->Wait((ID3D12Fence*)BcPackGetDirectFence(ctx), fence);
```

See full working wrapper in [`examples/directstorage_hook.cpp`](examples/directstorage_hook.cpp) and step-by-step integration patterns in [`docs/INTEGRATION_GUIDE.md`](docs/INTEGRATION_GUIDE.md).

### 2. Vulkan 1.3 / Linux Translation Layers (`shaders/spirv/*.spv`)

All 9 core compute kernels are precompiled with official Microsoft `DirectXShaderCompiler v1.9.2609` (`-spirv -fspv-target-env=vulkan1.3 -T cs_6_6 -E CSMain -O3`):

- Enable **`VK_EXT_subgroup_size_control`** (`requiredSubgroupSize = 32`) when dispatching `bcpack_decode.spv` or `bcpack_bc7_rans.spv` on 64-wide RDNA2/RDNA3 GPUs.
- Enable **`VK_KHR_ray_query`** when loading `aaa_rtgi_cs.spv` (`[numthreads(8,8,1)]` = `2× Wave32` NVIDIA / `1× Wave64` AMD).
- Recompile anytime on Linux/Windows via [`shaders/compile_spirv.sh`](shaders/compile_spirv.sh).

### 3. Rust Crate (`bcpack-gpu-sdk` alongside `bcpack-rs`)

```toml
[dependencies]
bcpack-gpu-sdk = { git = "https://github.com/POMXARK/bcpack-gpu-sdk.git", tag = "v0.11.0" }
```

```rust
use bcpack_gpu::gpu_decoder::{pack_batch, BcpackGpuDecoder, GpuContext};
use bcpack_gpu::descriptor_paging::VirtualDescriptorPager;
```

Or add this repository as a Git submodule:
```bash
git submodule add https://github.com/POMXARK/bcpack-gpu-sdk.git sdk
```

---

## Kernel & Module Inventory

| Kernel / Module | Targets | Wave / Threadgroup | Key Register Bindings | Role |
|---|---|---|---|---|
| `bcpack_decode.{hlsl,dxil,spv}` | SM 6.6, VK 1.3 | `[WaveSize(32)]` `32` | `t0..t4`, `u0` (`uint2`), `u1` | Dual-chunk BC1 rANS + predictor + quad reconstruction |
| `bcpack_bc7_rans.{hlsl,dxil,spv}` | SM 6.6, VK 1.3 | `[WaveSize(32)]` `32` | `t0..t2`, `u0` (`uint`), `u1` | `32/32` 16-lane BC7 rANS entropy stage (`64B`/block syms) |
| `bc16_pack.{hlsl,dxil,spv}` | SM 6.6, VK 1.3 | `64` | `t0`, `u0` (`Store2`), `b0` | Collapses 16-lane symbol rows into `16B` BC7/BC5 blocks |
| `gfx10_detile.{hlsl,dxil,spv}` | SM 6.6, VK 1.3 | `64` | `t0`, `u0`, `b0` | AMD Gfx10 `SW_64KB_D` (8B/16B) & `SW_4KB_S` detiler |
| `bindless_paging.{hlsl,dxil,spv}` | SM 6.6, VK 1.3 | `64` | `t0`, `t1`, `u0`, `b0` | `ResourceDescriptorHeap[NonUniformResourceIndex(phys)]` pager |
| `crowd_skin.{hlsl,dxil,spv}` | SM 6.6, VK 1.3 | `64` | `t0`, `t1`, `u0`, `b0` | 4-bone Linear Blend Skinning for 500+ NPCs |
| `hiz_build.{hlsl,dxil,spv}` | SM 6.6, VK 1.3 | `8×8` | `t0`, `u0`, `b0` | 6-mip conservative MAX depth pyramid |
| `hiz_cull.{hlsl,dxil}` | SM 6.6 | `64` | `t0..t8`, `u0`, `b0` | Frustum + 4-texel Hi-Z bounding sphere occlusion test |
| `aaa_vrs.{hlsl,dxil,spv}` | SM 6.6, VK 1.3 | `8×8` | `t0`, `t1`, `u0` (`R8_UINT`), `b0` | VRS Tier 2 (`16×16`) motion/luma/sky rate map |
| `aaa_rtgi_cs.{hlsl,dxil,spv}` | SM 6.6, VK 1.3 | `8×8` (`VK_KHR_ray_query`) | `t0..t6` (`t2`=TLAS), `u0`, `b0` | Coalesced Compute `RayQuery` diffuse GI |
| `src/ffi.rs` + `include/bcpack_gpu.h` | `cdylib` (`Win64`) | Async Compute + Direct | C-ABI opaque `BcPackContext*` | `bcpack_gpu_create_from_device`, `BcPackDecodeToTexture2D`, `BcPackDecodeToBuffer` |

---

## Verified Hardware Benchmarks (NVIDIA RTX 5080)

All measurements taken with **DX12 Debug Layer (`0` errors)** verified against [`Ahmet-Dedeler/bcpack-rs`](https://github.com/Ahmet-Dedeler/bcpack-rs) CPU reference oracle (**`19/19 PASS`**, `0` byte/symbol mismatches):

| Workload / Kernel | Output / Scale | CPU 1-Core Baseline | `bcpack-gpu-sdk` (RTX 5080) | Speedup / Notes |
|---|---|---:|---:|---|
| **BC1 4K-Tiled Decode** | `64.0 MiB` (`2,048` tiles) | `1,576.11 ms` (`40.6 MB/s`) | **`5.650 ms` (`11,327.6 MB/s`)** | **`279.0×`** (`0` mismatches) |
| **BC7 16-Lane rANS** (`32/32` Wave32) | `32.0 MiB` equiv (`1,024` chunks) | `1,312.73 ms` (`24.4 MB/s`) | **`5.312 ms` (`6,024.0 MB/s`)** | **`247.1×`** (`0` mismatches) |
| **AMD Gfx10 `SW_64KB_D` Detile** | `64.0 MiB` (`512 × 512×512`) | — | **`0.144 ms` (`432.6 GB/s`)** | Fused on `COMPUTE` queue |
| **Virtual Descriptor Paging** | `1.25M → 262,144` physical slots | — | **`0.064 ms`** (`65,536` px) | Fixes NVIDIA ~1M `E_OUTOFMEMORY` |
| **GPU 4-Bone Crowd Skinning** | `500` NPCs (`512k` verts, `32k` bones) | — | **`0.075 ms`** (`max_err=7.63e-6`) | Per-limb motion vectors |
| **VRS Tier 2 + Compute `RayQuery` RTGI** | `1920×1080` HDR (`ultra` preset) | `155.66 FPS` (PS RTGI, VRS OFF) | **`196.77 FPS`** (`+VRS`: **`258.91 FPS`**) | Eliminates `2×2` helper ray waste |

---

## Hardware & Compiler Workarounds Documented in Code

1. **DXC `-O3` Wave-Intrinsics Divergence Bug (`bcpack_decode.hlsl`, `bcpack_bc7_rans.hlsl`):**
   Combining `WavePrefixCountBits` + `WaveActiveCountBits` in a split-half Wave32 byte allocator miscompiles `stream_pos` under `dxc -O3`. Resolved via explicit `WaveActiveBallot` + `countbits(mask & ((1u << ll) - 1u))`.
2. **NVIDIA DX12 `~1M` Descriptor Heap Limit (`descriptor_paging.rs`):**
   Prevents `E_OUTOFMEMORY` when translation layers allocate `> 1,000,000` `SHADER_VISIBLE` descriptors by paging staging SRVs through `262,144` physical Tier-3 slots with `CopyDescriptorsSimple`.
3. **NVIDIA RTX 5080 Driver `621.24` RT-Cache Flicker (`aaa_rtgi_cs.hlsl`):**
   Isolate TLAS builds with `uav_barrier_all()` + explicit fence flush before mixing Async Compute texture uploads with `RayQuery` dispatches.
4. **RDNA2 / Vulkan Subgroup Dispatch:**
   Use `VK_EXT_subgroup_size_control` (`Wave32`) for `bcpack_decode.spv` / `bcpack_bc7_rans.spv`, whereas `[numthreads(8,8,1)]` in `aaa_rtgi_cs.hlsl` maps natively to `2× Wave32` (NVIDIA) or `1× Wave64` (AMD RDNA2+) with zero changes.

---

## License & Attribution

Licensed under **`MIT OR Apache-2.0`** ([`LICENSE-MIT`](LICENSE-MIT), [`LICENSE-APACHE`](LICENSE-APACHE)).
100% clean-room implementation building on the public format research in [`Ahmet-Dedeler/bcpack-rs`](https://github.com/Ahmet-Dedeler/bcpack-rs) and Microsoft's MIT-licensed [`microsoft/silicon-rans`](https://github.com/microsoft/silicon-rans).
