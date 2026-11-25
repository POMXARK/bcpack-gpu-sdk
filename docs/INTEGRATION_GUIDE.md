# Integration Guide (`bcpack-gpu-sdk`)

Step-by-step drop-in recipes for clean-room Rust, DX12 C++ translation layers, and Vulkan 1.3 Linux layers.

---

## 1. Hook into DirectX 12 Translation Layers (`IDStorageQueueX` / `xcl`)

`bcpack_gpu.dll` is designed to sit directly behind an emulated `IDStorageQueueX::EnqueueRequest` decompression handler on Windows DX12:

1. Obtain `include/bcpack_gpu.h` and `bcpack_gpu.dll` + `bcpack_gpu.dll.lib` (from [GitHub Releases](https://github.com/POMXARK/bcpack-gpu-sdk/releases) or `cargo build --release`).
2. When the guest creates its `ID3D12Device`, attach `BcPackContext*`:
   ```cpp
   BcPackContext* g_bcpack = bcpack_gpu_create_from_device(
       guest_d3d12_device,
       nullptr, // or pass guest compute queue
       2048,
       8388608,
       96u * 1024u * 1024u);
   ```
3. On compressed texture requests targeting a guest `DXGI_FORMAT_BC{1,3,4,5,7}_UNORM` subresource (even Gfx10-swizzled):
   ```cpp
   uint64_t fence = 0;
   BcPackDecodeToTexture2D(
       g_bcpack,
       src_bytes, src_len,
       width_blocks * height_blocks,
       BCPACK_FORMAT_BC7, // or BC1, BC3, BC4, BC5
       BCPACK_SWIZZLE_GFX10_64KB_D,
       width_blocks, height_blocks,
       dst_tex2d, subresource_index,
       0, 0, &fence);
   guest_direct_queue->Wait(
       static_cast<ID3D12Fence*>(BcPackGetDirectFence(g_bcpack)),
       fence);
   ```
4. For raw linear buffer staging (`IDStorageQueueX` buffer destination), use `BcPackDecodeToBuffer(...)` and wait on `BcPackGetComputeFence(g_bcpack)`.
5. Under Vulkan 1.3 (`OpenScarlett`), chain `bcpack_bcn_rans.spv -> bcpack_bcn_rebuild.spv` for `BC3`/`BC4`/`BC5` and `bcpack_bc7_rans.spv -> bcpack_bc7_rebuild.spv` for full Modes 0–7 `BC7` (total: **12 `.spv` kernels** in `shaders/spirv/`).

---

## 2. Integrate Vulkan 1.3 SPIR-V Kernels (`OpenScarlett` / Linux Layers)

Prebuilt `.spv` compute kernels in `shaders/spirv/` require **no DXC toolchain at runtime**:

| Stage | SPIR-V File | Required Vulkan Extensions / Features |
|---|---|---|
| BC1 Decode | `shaders/spirv/bcpack_decode.spv` | Vulkan 1.3 + `VK_EXT_subgroup_size_control` (`requiredSubgroupSize = 32`) |
| BC7 16-lane rANS | `shaders/spirv/bcpack_bc7_rans.spv` | Vulkan 1.3 + `VK_EXT_subgroup_size_control` (`requiredSubgroupSize = 32`) |
| 16B Block Packer | `shaders/spirv/bc16_pack.spv` | Vulkan 1.3 (`64`-thread workgroup) |
| Gfx10 Swizzle Detile | `shaders/spirv/gfx10_detile.spv` | Vulkan 1.3 (`64`-thread workgroup) |
| Hi-Z Depth Pyramid | `shaders/spirv/hiz_build.spv` | Vulkan 1.3 (`8×8` workgroup) |
| Bindless Pager | `shaders/spirv/bindless_paging.spv` | Vulkan 1.3 descriptor indexing (`runtimeDescriptorArray`, `nonUniformIndexing`) |
| Crowd Skinning | `shaders/spirv/crowd_skin.spv` | Vulkan 1.3 (`64`-thread workgroup) |
| VRS Rate Image Gen | `shaders/spirv/aaa_vrs.spv` | Vulkan 1.3 (`VK_KHR_fragment_shading_rate` / compute `R8_UINT` target) |
| Compute RTGI | `shaders/spirv/aaa_rtgi_cs.spv` | Vulkan 1.3 + `VK_KHR_ray_query` + `VK_KHR_acceleration_structure` |

Dispatch `bcpack_decode.spv` with **`ceil(num_chunks / 2.0)`** workgroups (`1` Wave32 workgroup decodes `2` chunks in parallel). Dispatch `gfx10_detile.spv` with **`ceil(num_blocks / 64.0)`** workgroups (`detile_mode`: `0 = SW_64KB_D 8B`, `1 = SW_4KB_S 8B`, `2 = SW_64KB_D 16B`).

---

## 3. Pair with `Ahmet-Dedeler/bcpack-rs` in Rust

`bcpack-gpu-sdk` already builds its lookup tables directly from `bcpack-rs`:
- `bcpack::bc1::{QuadTable, Window}` -> `t1` (`g_Window`), `t2` (`g_QuadMasks`)
- `bcpack::tables::{BASE_TABLE, FREQ_TABLE, RF_TABLE, RN_TABLE, RS_TABLE}` -> `pack_model_data()` / `ModelDataGpu` (`t0`)

In your Rust engine or decoder crate:
```rust
use bcpack_gpu::gpu_decoder::{pack_batch, BcpackGpuDecoder, GpuContext};
use bcpack::bc1::{QuadTable, Window};

let mut ctx = GpuContext::new(false);
let decoder = BcpackGpuDecoder::new(&ctx.device, &Window::default_table(), &QuadTable::default_table(), 2048, 8_388_608, 96 * 1024 * 1024);
```

See [`docs/AI_ARCHITECTURE.md`](AI_ARCHITECTURE.md) for register numbers and buffer byte layouts.
