//! Clean-room hardware GPU decompression & compute SDK (`bcpack_gpu`).
//!
//! Provides:
//! - [`gpu_decoder`]: Wave32 HLSL SM 6.6 BCPack (BC1 + 16-lane BC7 rANS) + `bc16_pack` + AMD Gfx10 detile on Async Compute.
//! - [`ffi`]: Zero-copy C-ABI exports matching `include/bcpack_gpu.h` (`bcpack_gpu.dll`).
//! - [`descriptor_paging`]: SM 6.6 `ResourceDescriptorHeap[]` virtual descriptor pager (`1.25M -> 262K`).
//! - [`crowd_skinning`]: GPU 4-bone Linear Blend Skinning compute kernel (`500` NPCs).

pub mod crowd_skinning;
pub mod descriptor_paging;
pub mod ffi;
pub mod gpu_decoder;
pub mod primitives;
