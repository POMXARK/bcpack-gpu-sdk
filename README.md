# bcpack-gpu-sdk — Hardware GPU BCPack Decoder, Gfx10 Detiler & Compute Kernel SDK

[![DirectX 12](https://img.shields.io/badge/DirectX%2012-SM%206.6%20%2B%20Async%20Compute-2f80ed)](shaders/bcpack_decode.hlsl)
[![Vulkan 1.3 SPIR-V](https://img.shields.io/badge/Vulkan%201.3-SPIR--V%20(14%20Kernels)-e74c3c)](shaders/spirv)
[![C-ABI DLL](https://img.shields.io/badge/C--ABI-bcpack__gpu.dll%20%2B%20bcpack__gpu.h-blue)](include/bcpack_gpu.h)
[![Verified 21/21 PASS](https://img.shields.io/badge/RTX%205080-21%2F21%20Bit--Exact%20PASS-brightgreen)](#verified-hardware-benchmarks-nvidia-rtx-5080)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-green)](LICENSE-MIT)

**Standalone, bloat-free clean-room hardware GPU compute SDK (`MIT OR Apache-2.0`) for Xbox Series X|S translation layers, DirectStorage `IDStorageQueueX` hooks, Vulkan 1.3 Linux layers (`OpenScarlett` / Proton), and GPU backends for [`Ahmet-Dedeler/bcpack-rs`](https://github.com/Ahmet-Dedeler/bcpack-rs).**

No game assets, no demo media bloat, zero copyrighted data — only the production C-ABI DLL (`bcpack_gpu.dll`), Rust crate (`bcpack-gpu-sdk` `v0.15.0`), `include/bcpack_gpu.h`, **15 HLSL SM 6.6 + DXIL compute kernels**, and **14 precompiled Vulkan 1.3 SPIR-V (`.spv`) kernels** supporting **all 6 BCPack formats: Full `BC1`, `BC3`, `BC4`, `BC5`, `BC6H` (Modes 1–14), and Multi-Mode `BC7` (Modes 0–7)** with mixed-format GPU batching (`BcPackDecodeBatch`), batched zero-copy 2D texture streaming (`BcPackDecodeBatchToTextures`), and optional caller-supplied runtime table uploads.

> Looking for the full 4K streaming benchmark, 12.41M Mesh-Shader showcase, MP4/GIF captures, and 19-test verification harness? See the companion showcase repository: **[`POMXARK/bcpack-gpu-hlsl`](https://github.com/POMXARK/bcpack-gpu-hlsl)** (which links this SDK via Git submodule at `sdk/`).
> AI coding agents & crawlers: read **[`llms.txt`](llms.txt)** and **[`docs/AI_ARCHITECTURE.md`](docs/AI_ARCHITECTURE.md)**.

---

## Core Capabilities

1. **Zero-Barrier Wave32 BCPack MDU rANS Decoders (`11.33 GB/s` BC1, `6.02 GB/s` BC7 rANS):**
   - Dual-chunk warp packing (`2 × 12` lanes for BC1 = `24/32` active; `2 × 16` lanes for BC7 = **`32/32` 100% Wave32 utilization**).
   - Register-resident `64B` stream sliding window with `WaveReadLaneAt` symbol transpose and 4-step branchless binary search — **zero `GroupMemoryBarrierWithGroupSync()` inside hot loops**.
2. **Full 6-Format Block Rebuilders (`BC1` / `BC3` / `BC4` / `BC5` / `BC6H` Modes 1–14 / `BC7` Modes 0–7):**
   - [`shaders/bcpack_bc7_rebuild.hlsl`](shaders/bcpack_bc7_rebuild.hlsl) (`Bc7FullGpuDecoder` / `BCPACK_FORMAT_BC7`): complete 128-bit BC7 block assembly across all 8 modes (`0..7`), partitions, Mode 4/5 channel rotations & index-selection, p-bits, anchor omissions, 255-entry dictionary, and 256-block history ring.
   - [`shaders/bcpack_bc6h_rans.hlsl`](shaders/bcpack_bc6h_rans.hlsl) + [`shaders/bcpack_bc6h_rebuild.hlsl`](shaders/bcpack_bc6h_rebuild.hlsl) (`Bc6hFullGpuDecoder` / `BCPACK_FORMAT_BC6H`): 16-lane Wave32 rANS + all 14 BC6H HDR modes (`1..14`), ascending high base bits (`13..14`), signed index deltas, 255-entry dictionary, and 256-block history ring.
   - [`shaders/bcpack_bcn_rans.hlsl`](shaders/bcpack_bcn_rans.hlsl) + [`shaders/bcpack_bcn_rebuild.hlsl`](shaders/bcpack_bcn_rebuild.hlsl) (`BcnGpuDecoder` / `BCPACK_FORMAT_BC3`, `BC4`, `BC5`, and full-table `BC1`): generalized 12-lane Wave32 rANS + pair/quad rank tables + dictionary/history rebuild.
   - [`shaders/bc16_pack.hlsl`](shaders/bc16_pack.hlsl) + [`shaders/gfx10_detile.hlsl`](shaders/gfx10_detile.hlsl): inline RDNA2 bit-permutation detile (`SW_64KB_D` 8B/16B, `SW_4KB_S` 8B at **`432.6 GB/s`**) on `D3D12_COMMAND_LIST_TYPE_COMPUTE` before `CopyTextureRegion`.
3. **Zero-Copy & Batched Shared-Device C-ABI (`bcpack_gpu.dll` + `include/bcpack_gpu.h`):**
   - Attach directly to a caller's `ID3D12Device*` (`bcpack_gpu_create_from_device`) and decode + detile straight into `D3D12_HEAP_TYPE_DEFAULT` buffers (`BcPackDecodeToBuffer`), `DXGI_FORMAT_BC{1,3,4,5,6H_UF16,7}_UNORM` 2D subresources (`BcPackDecodeToTexture2D`), or batched zero-copy multi-texture targets (`BcPackDecodeBatchToTextures`: automatic Gfx10 detile + `256B` row-pitch padding, single `queue->Wait(compute_fence)` -> single `CopyTextureRegion` command list -> single direct fence with zero per-item CPU waits; verified on RTX 5080 across `10/11` mixed 6-format/Gfx10/multi-mip items + `1` isolated error, `0` byte mismatches, **`16.0 MiB` (`128×512×512 BC1`) in `40.58 ms` (`394.3 MB/s`)**).
   - Submit heterogeneous multi-chunk batches via `BcPackDecodeBatch` (`BcPackBatchItem` array across `BC1/BC3/BC4/BC5/BC6H/BC7`): groups items by pipeline family (`Bc1Fast`, `Bcn`, `Bc6h`, `Bc7`), auto sub-batches within context caps, isolates invalid items per `item.status`, and eliminates per-chunk dispatch overhead (**`41.7×` faster** than looping `BcPackDecodeAsync`).
   - Callers with external Xbox BCPack tables can upload them cleanly at runtime via `bcpack_gpu_set_bc1_tables`, `bcpack_gpu_set_bc6h_tables`, and `bcpack_gpu_set_bc7_tables` (`0` proprietary constants embedded).
4. **SM 6.6 Virtual `ResourceDescriptorHeap[]` Pager (`1.25M → 262K` slots in `0.064 ms`):**
   - Solves NVIDIA DX12 `~1M` shader-visible descriptor heap `E_OUTOFMEMORY` crashes via `CopyDescriptorsSimple` LRU paging + `ResourceDescriptorHeap[NonUniformResourceIndex(g_VirtToPhys[id])]`.
5. **Cross-Vendor Console Compute Helpers (DXIL + Vulkan 1.3 SPIR-V):**
   - Conservative 6-mip Hi-Z pyramid (`hiz_build`, `hiz_cull`), 4-bone GPU Crowd Skinning (`crowd_skin`), Hardware VRS Tier 2 tile classifier (`aaa_vrs`), and Xbox/RDNA2-style Compute `RayQuery` Diffuse RTGI (`aaa_rtgi_cs`, `VK_KHR_ray_query`).

---

## Quick Start

### 1. C / C++ Zero-Copy DirectStorage / DX12 Hook (`bcpack_gpu.dll`)

Download `bcpack_gpu.dll`, `bcpack_gpu.dll.lib`, and `bcpack_gpu.h` from [Releases](https://github.com/POMXARK/bcpack-gpu-sdk/releases) (or build with `cargo build --release`), then link `bcpack_gpu.dll.lib`:

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
bcpack-gpu-sdk = { git = "https://github.com/POMXARK/bcpack-gpu-sdk.git", tag = "v0.14.1" }
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
| `bcpack_bcn_rans.{hlsl,dxil,spv}` | SM 6.6, VK 1.3 | `[WaveSize(32)]` `32` | `t0..t2`, `u0` (`uint` ×6/row), `u1` | Generalized 12-lane Wave32 rANS (`BC1` / `BC3` / `BC4` / `BC5`) |
| `bcpack_bcn_rebuild.{hlsl,dxil,spv}` | SM 6.6, VK 1.3 | `64` | `b0`, `t0..t4`, `u0`, `u1` | `BC1` / `BC3` / `BC4` / `BC5` pair/quad/dict/history block rebuilder |
| `bcpack_bc6h_rans.{hlsl,dxil,spv}` | SM 6.6, VK 1.3 | `[WaveSize(32)]` `32` | `t0..t2`, `u0` (`uint` ×8/row), `u1` | `32/32` 16-lane BC6H rANS entropy stage (`64B`/block syms) |
| `bcpack_bc6h_rebuild.{hlsl,dxil,spv}` | SM 6.6, VK 1.3 | `64` | `b0`, `t0..t2`, `u0` (`uint4`), `u1` | Full **BC6H (Modes 1–14)** dict/history block rebuilder |
| `bcpack_bc7_rans.{hlsl,dxil,spv}` | SM 6.6, VK 1.3 | `[WaveSize(32)]` `32` | `t0..t2`, `u0` (`uint` ×8/row), `u1` | `32/32` 16-lane BC7 rANS entropy stage (`64B`/block syms) |
| `bcpack_bc7_rebuild.{hlsl,dxil,spv}` | SM 6.6, VK 1.3 | `64` | `b0`, `t0..t3`, `u0` (`uint4`), `u1` | Full Multi-Mode **BC7 (Modes 0–7)** dict/history block rebuilder |
| `bc16_pack.{hlsl,dxil,spv}` | SM 6.6, VK 1.3 | `64` | `t0`, `u0` (`Store2`), `b0` | Collapses 16-lane symbol rows into `16B` BC7/BC5 blocks |
| `gfx10_detile.{hlsl,dxil,spv}` | SM 6.6, VK 1.3 | `64` | `t0`, `u0`, `b0` | AMD Gfx10 `SW_64KB_D` (8B/16B) & `SW_4KB_S` detiler |
| `bindless_paging.{hlsl,dxil,spv}` | SM 6.6, VK 1.3 | `64` | `t0`, `t1`, `u0`, `b0` | `ResourceDescriptorHeap[NonUniformResourceIndex(phys)]` pager |
| `crowd_skin.{hlsl,dxil,spv}` | SM 6.6, VK 1.3 | `64` | `t0`, `t1`, `u0`, `b0` | 4-bone Linear Blend Skinning for 500+ NPCs |
| `hiz_build.{hlsl,dxil,spv}` | SM 6.6, VK 1.3 | `8×8` | `t0`, `u0`, `b0` | 6-mip conservative MAX depth pyramid |
| `hiz_cull.{hlsl,dxil}` | SM 6.6 | `64` | `t0..t8`, `u0`, `b0` | Frustum + 4-texel Hi-Z bounding sphere occlusion test |
| `aaa_vrs.{hlsl,dxil,spv}` | SM 6.6, VK 1.3 | `8×8` | `t0`, `t1`, `u0` (`R8_UINT`), `b0` | VRS Tier 2 (`16×16`) motion/luma/sky rate map |
| `aaa_rtgi_cs.{hlsl,dxil,spv}` | SM 6.6, VK 1.3 | `8×8` (`VK_KHR_ray_query`) | `t0..t6` (`t2`=TLAS), `u0`, `b0` | Coalesced Compute `RayQuery` diffuse GI |
| `src/ffi.rs` + `include/bcpack_gpu.h` | `cdylib` (`Win64`) | Async Compute + Direct | C-ABI opaque `BcPackContext*` | `bcpack_gpu_create_from_device`, `BcPackDecodeBatch`, `BcPackDecodeToTexture2D`, `BcPackDecodeToBuffer` |

---

## Verified Hardware Benchmarks (NVIDIA RTX 5080)

All measurements verified against [`Ahmet-Dedeler/bcpack-rs`](https://github.com/Ahmet-Dedeler/bcpack-rs) CPU reference oracle (**`0` byte/symbol mismatches**):

### `v0.14.1` C-ABI Mixed-Format Batch, Texture2D & Oracle Parity (`tests/v14_batch_and_tex2d.rs`)

- **Correctness (`5/5 PASS`, `0` byte mismatches against `bcpack-rs` `7b2b421`):**
  - `bc1_lane1_index_reference_bit_exact`: `512` BC1 blocks with non-zero `lane 1` index references via branchless 16-pixel SWAR (`codes_to_lines`/`add_lines`/`lines_to_codes` + `ring_line[2][256]` in `bcpack_decode.hlsl`) verified `0` mismatches (`kernel=0.593 ms`).
  - `bc7_rotation_ignored_outside_modes_4_5`: restricts BC7 `rotation` strictly to modes `4` and `5` (`bcpack_bc7_rebuild.hlsl`, `Bc7FullGpuDecoder::update_tables`), verified `0` mismatches (`1,024` bytes).
  - `BcPackDecodeToTexture2D`: all **6 formats (`BC1/BC3/BC4/BC5/BC6H/BC7`)** at `256×256 px` and unaligned `12×12 px` (`row_pad_default`), plus `>16 MiB` row-padded `2052×8192 px` BC7 (`16.03 MiB` raw / `16.50 MiB` padded in 64 strips), `4096×4096 px` BC7 (`16.00 MiB` in 64 strips via `dst_y_blocks`), and pre-submission `-5` bounds guard (`>128 MiB`).
  - `BcPackDecodeBatch`: `62/66` mixed `BC1/BC3/BC4/BC5/BC6H/BC7` chunks auto-sub-batched (`max_chunks=16`) + `4` isolated error items (`-2`, `-4`, `-4`, `-1`) with `0` byte mismatches (`kernel=7.478 ms`).
- **Throughput (`384` realistic `64 KiB` chunks = `24.0 MiB` mixed 6-fmt; CPU subset = `320` chunks / `20.0 MiB` `BC1/3/4/5/7` on Ryzen 7 5700X):**

| Path / API | Scale | Latency | Throughput | Notes |
|---|---|---:|---:|---|
| CPU `bcpack-rs` (`1` thread, `#8`) | `20.0 MiB` (`BC1/3/4/5/7`) | `365.87 ms` | `0.06 GB/s` | Baseline CPU (`7b2b421`) |
| CPU `bcpack-rs` (`16` threads, `#8`) | `20.0 MiB` (`BC1/3/4/5/7`) | `62.61 ms` | `0.33 GB/s` | Full 8C/16T Ryzen 7 5700X |
| GPU `BcPackDecodeAsync` loop | `24.0 MiB` (`384` calls) | `6,681.02 ms` | `< 0.01 GB/s` | 1-chunk submit/sync overhead |
| **GPU `BcPackDecodeBatch` (wall)** | `24.0 MiB` (`384` chunks) | **`156.29 ms`** | **`0.16 GB/s`** | **`42.7×` faster** than async loop, `2.3×` vs 1T CPU |
| **GPU `BcPackDecodeBatch` (kernel)** | `24.0 MiB` (`384` chunks) | **`118.13 ms`** | **`0.21 GB/s`** | Cumulative DX12 GPU timestamps |
| GPU `BcPackDecodeToTexture2D` | `8.0 MiB` (`32 × 512px` BC7) | `4,451.95 ms` | `< 0.01 GB/s` | Zero-copy per-texture DX12 copy |

### Core Kernel Suite (`DX12 Debug Layer: 0 Errors`, `21/21 PASS`)

| Workload / Kernel | Output / Scale | CPU 1-Core Baseline | `bcpack-gpu-sdk` (RTX 5080) | Speedup / Notes |
|---|---|---:|---:|---|
| **BC1 4K-Tiled Decode** | `64.0 MiB` (`2,048` tiles) | `1,576.11 ms` (`40.6 MB/s`) | **`5.650 ms` (`11,327.6 MB/s`)** | **`279.0×`** (`0` mismatches) |
| **BC7 16-Lane rANS** (`32/32` Wave32) | `32.0 MiB` equiv (`1,024` chunks) | `1,312.73 ms` (`24.4 MB/s`) | **`5.312 ms` (`6,024.0 MB/s`)** | **`247.1×`** (`0` mismatches) |
| **Full Multi-Mode BC7 (`0..7` + `SW_64KB_D`)** | `5` chunks (`1,984` blks + Tex2D) | — | **`9.163 ms`** (`0` mismatches) | Modes 0–7, rot, sel, p-bits, dict |
| **Mixed `BC3` + `BC4` + `BC5` Batch** | `1,152` blocks (C-ABI + GPU) | — | **`1.289 ms`** (`0` mismatches) | Pair/quad rank + dict + 256-hist |
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
