//! Clean-room GPU engine primitives (`MIT OR Apache-2.0`).
//!
//! Self-contained, game-agnostic building blocks designed for high-frequency CPU/GPU buffer
//! synchronization and Wave64 -> Wave32 shader safety on discrete DirectX 12 hardware:
//! - [`HierarchicalDirtyBitmap`] + [`diff_dirty_pages_64b`]: 3-level `u64` dirty-page bitmap
//!   with AVX2 `_mm256_cmpeq_epi8` 64-byte slot diffing (scalar fallback on non-AVX2 CPUs).
//! - [`GpuScatterPatcher`]: Coalesced SM 6.6 Compute kernel (`scatter_patch.dxil`) that applies
//!   thousands of scattered 64-byte buffer updates in a single GPU dispatch.
//! - [`Wave32NanGuard`]: IEEE-754 `NaN -> 0` / `±Inf -> ±65504` sanitizer for CPU and HLSL SM 6.6
//!   plus safe logical Wave64 -> Wave32 cross-lane read helpers (`wave32_nan_guard.hlsli`).

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;
use std::ffi::c_void;
use std::ptr::copy_nonoverlapping;
use std::time::Instant;
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::gpu_decoder::{
    cmd_barriers, create_committed_buffer, create_compute_pso, create_root_sig_from_params,
    transition_barrier, uav_barrier, GpuContext,
};

pub const DIRTY_PAGE_BYTES: usize = 4096;
pub const SCATTER_SLOT_BYTES: usize = 64;

/// Compiled SM 6.6 Compute Shader for [`GpuScatterPatcher`].
pub const SCATTER_PATCH_CS_DXIL: &[u8] = include_bytes!("../shaders/scatter_patch.dxil");

// ---------------------------------------------------------------------------
// 1. 3-Level Hierarchical u64 Dirty Bitmap + AVX2 64B Slot Diff
// ---------------------------------------------------------------------------

/// 3-level `u64` bit-tree tracking dirty 4 KiB pages with `O(dirty_pages)` `trailing_zeros` drain.
pub struct HierarchicalDirtyBitmap {
    l0: Vec<u64>, // 1 bit per 4 KiB page
    l1: Vec<u64>, // 1 bit per 64 L0 words (4,096 pages)
    l2: u64,      // 1 bit per L1 word
    num_pages: usize,
}

impl HierarchicalDirtyBitmap {
    pub fn new(num_pages: usize) -> Self {
        let l0_len = num_pages.div_ceil(64);
        let l1_len = l0_len.div_ceil(64);
        assert!(
            l1_len <= 64,
            "HierarchicalDirtyBitmap supports up to 262,144 L0 words (~16M 4 KiB pages = 64 GiB)"
        );
        Self {
            l0: vec![0u64; l0_len],
            l1: vec![0u64; l1_len],
            l2: 0,
            num_pages,
        }
    }

    #[inline]
    pub fn page_count(&self) -> usize {
        self.num_pages
    }

    pub fn clear(&mut self) {
        self.l0.fill(0);
        self.l1.fill(0);
        self.l2 = 0;
    }

    #[inline]
    pub fn mark_page(&mut self, page_idx: usize) {
        let w0 = page_idx >> 6;
        let b0 = page_idx & 63;
        if w0 < self.l0.len() && page_idx < self.num_pages {
            self.l0[w0] |= 1u64 << b0;
            let w1 = w0 >> 6;
            let b1 = w0 & 63;
            self.l1[w1] |= 1u64 << b1;
            self.l2 |= 1u64 << w1;
        }
    }

    /// Marks every 4 KiB page touched by `[byte_offset .. byte_offset + byte_len)`.
    #[inline]
    pub fn mark_byte_range(&mut self, byte_offset: usize, byte_len: usize) {
        if byte_len == 0 || self.num_pages == 0 {
            return;
        }
        let first = byte_offset / DIRTY_PAGE_BYTES;
        let last = byte_offset.saturating_add(byte_len - 1) / DIRTY_PAGE_BYTES;
        let clamped_last = last.min(self.num_pages - 1);
        for p in first..=clamped_last {
            self.mark_page(p);
        }
    }

    /// Invokes `f(page_idx)` for each marked page and clears the bitmap in one pass.
    pub fn drain_dirty_pages<F: FnMut(usize)>(&mut self, mut f: F) {
        let mut m2 = self.l2;
        self.l2 = 0;
        while m2 != 0 {
            let w1 = m2.trailing_zeros() as usize;
            m2 &= m2 - 1;
            let mut m1 = self.l1[w1];
            self.l1[w1] = 0;
            while m1 != 0 {
                let b1 = m1.trailing_zeros() as usize;
                m1 &= m1 - 1;
                let w0 = (w1 << 6) | b1;
                let mut m0 = self.l0[w0];
                self.l0[w0] = 0;
                while m0 != 0 {
                    let b0 = m0.trailing_zeros() as usize;
                    m0 &= m0 - 1;
                    f((w0 << 6) | b0);
                }
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn diff_4kb_page_avx2(
    page_idx: usize,
    src_base: *const u8,
    shadow_base: *mut u8,
    max_slots: usize,
    out_indices: &mut Vec<u32>,
    out_data: &mut Vec<u8>,
) {
    let first_slot = page_idx * (DIRTY_PAGE_BYTES / SCATTER_SLOT_BYTES);
    let end_slot = (first_slot + (DIRTY_PAGE_BYTES / SCATTER_SLOT_BYTES)).min(max_slots);
    for slot in first_slot..end_slot {
        let byte_off = slot * SCATTER_SLOT_BYTES;
        let p_src = src_base.add(byte_off);
        let p_shd = shadow_base.add(byte_off);

        let a0 = _mm256_loadu_si256(p_src as *const __m256i);
        let b0 = _mm256_loadu_si256(p_shd as *const __m256i);
        let eq0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(a0, b0));

        let a1 = _mm256_loadu_si256(p_src.add(32) as *const __m256i);
        let b1 = _mm256_loadu_si256(p_shd.add(32) as *const __m256i);
        let eq1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(a1, b1));

        if (eq0 & eq1) != -1 {
            _mm256_storeu_si256(p_shd as *mut __m256i, a0);
            _mm256_storeu_si256(p_shd.add(32) as *mut __m256i, a1);
            out_indices.push(slot as u32);
            let old_len = out_data.len();
            out_data.resize(old_len + SCATTER_SLOT_BYTES, 0);
            copy_nonoverlapping(p_src, out_data.as_mut_ptr().add(old_len), SCATTER_SLOT_BYTES);
        }
    }
}

fn diff_4kb_page_scalar(
    page_idx: usize,
    src: &[u8],
    shadow: &mut [u8],
    max_slots: usize,
    out_indices: &mut Vec<u32>,
    out_data: &mut Vec<u8>,
) {
    let first_slot = page_idx * (DIRTY_PAGE_BYTES / SCATTER_SLOT_BYTES);
    let end_slot = (first_slot + (DIRTY_PAGE_BYTES / SCATTER_SLOT_BYTES)).min(max_slots);
    for slot in first_slot..end_slot {
        let off = slot * SCATTER_SLOT_BYTES;
        let s = &src[off..off + SCATTER_SLOT_BYTES];
        let sh = &mut shadow[off..off + SCATTER_SLOT_BYTES];
        if s != sh {
            sh.copy_from_slice(s);
            out_indices.push(slot as u32);
            out_data.extend_from_slice(s);
        }
    }
}

/// Scans only marked 4 KiB pages in `src` against `shadow`, updating `shadow` in place and
/// emitting changed 64-byte slot indices + payloads.
pub fn diff_dirty_pages_64b(
    bitmap: &mut HierarchicalDirtyBitmap,
    src: &[u8],
    shadow: &mut [u8],
    max_slots: usize,
    out_indices: &mut Vec<u32>,
    out_data: &mut Vec<u8>,
) {
    out_indices.clear();
    out_data.clear();
    let max_valid_slots = max_slots.min(src.len() / SCATTER_SLOT_BYTES).min(shadow.len() / SCATTER_SLOT_BYTES);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_avx2 = is_x86_feature_detected!("avx2");
    bitmap.drain_dirty_pages(|page_idx| {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if use_avx2 {
            unsafe {
                diff_4kb_page_avx2(
                    page_idx,
                    src.as_ptr(),
                    shadow.as_mut_ptr(),
                    max_valid_slots,
                    out_indices,
                    out_data,
                );
            }
            return;
        }
        diff_4kb_page_scalar(
            page_idx,
            src,
            shadow,
            max_valid_slots,
            out_indices,
            out_data,
        );
    });
}

// ---------------------------------------------------------------------------
// 2. Coalesced GPU Scatter-Patcher (D3D12 Compute SM 6.6)
// ---------------------------------------------------------------------------

/// Reusable D3D12 Compute patcher that writes scattered 64-byte slots into a target GPU buffer.
pub struct GpuScatterPatcher {
    root_sig: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    max_patches: usize,
    src_data_buf: ID3D12Resource,
    src_indices_buf: ID3D12Resource,
    src_data_ptr: *mut u8,
    src_indices_ptr: *mut u32,
}

unsafe impl Send for GpuScatterPatcher {}

impl GpuScatterPatcher {
    pub fn new(device: &ID3D12Device, max_patches: usize) -> Result<Self, String> {
        let cap = max_patches.max(64);
        let root_params = [
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: 4,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: 1,
                        RegisterSpace: 0,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];

        let root_sig = unsafe { create_root_sig_from_params(device, &root_params) };
        let pso = unsafe {
            create_compute_pso(device, &root_sig, SCATTER_PATCH_CS_DXIL, "scatter_patch")
        };

        let src_data_buf = create_committed_buffer(
            device,
            D3D12_HEAP_TYPE_UPLOAD,
            (cap * SCATTER_SLOT_BYTES) as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        let src_indices_buf = create_committed_buffer(
            device,
            D3D12_HEAP_TYPE_UPLOAD,
            (cap * std::mem::size_of::<u32>()) as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );

        let empty_range = D3D12_RANGE { Begin: 0, End: 0 };
        let mut data_ptr: *mut c_void = std::ptr::null_mut();
        let mut idx_ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            src_data_buf
                .Map(0, Some(&empty_range), Some(&mut data_ptr))
                .map_err(|e| format!("Map src_data_buf failed: {e:?}"))?;
            src_indices_buf
                .Map(0, Some(&empty_range), Some(&mut idx_ptr))
                .map_err(|e| format!("Map src_indices_buf failed: {e:?}"))?;
        }

        Ok(Self {
            root_sig,
            pso,
            max_patches: cap,
            src_data_buf,
            src_indices_buf,
            src_data_ptr: data_ptr as *mut u8,
            src_indices_ptr: idx_ptr as *mut u32,
        })
    }

    #[inline]
    pub fn max_patches(&self) -> usize {
        self.max_patches
    }

    /// Uploads `slot_indices` + `slot_data_64b` and records a Compute dispatch onto `list`.
    /// Caller ensures `dst_heap` is in `D3D12_RESOURCE_STATE_UNORDERED_ACCESS` and synchronization
    /// across frames guarantees the upload buffers are idle.
    pub fn record_patch(
        &self,
        list: &ID3D12GraphicsCommandList,
        dst_heap: &ID3D12Resource,
        slot_indices: &[u32],
        slot_data_64b: &[u8],
    ) -> Result<u32, String> {
        let count = slot_indices.len();
        if count == 0 {
            return Ok(0);
        }
        if count > self.max_patches {
            return Err(format!(
                "patch count {count} exceeds max_patches {}",
                self.max_patches
            ));
        }
        let expected_bytes = count * SCATTER_SLOT_BYTES;
        if slot_data_64b.len() < expected_bytes {
            return Err(format!(
                "slot_data_64b len {} < required {expected_bytes}",
                slot_data_64b.len()
            ));
        }
        unsafe {
            copy_nonoverlapping(slot_data_64b.as_ptr(), self.src_data_ptr, expected_bytes);
            copy_nonoverlapping(slot_indices.as_ptr(), self.src_indices_ptr, count);
            let cb = [count as u32, 0u32, 0u32, 0u32];
            list.SetComputeRootSignature(&self.root_sig);
            list.SetPipelineState(&self.pso);
            list.SetComputeRoot32BitConstants(0, 4, cb.as_ptr() as *const c_void, 0);
            list.SetComputeRootShaderResourceView(1, self.src_data_buf.GetGPUVirtualAddress());
            list.SetComputeRootShaderResourceView(2, self.src_indices_buf.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(3, dst_heap.GetGPUVirtualAddress());
            list.Dispatch((count as u32).div_ceil(64), 1, 1);
        }
        Ok(count as u32)
    }

    /// Standalone convenience helper: transitions `dst_heap` from `COMMON -> UAV`, records patch,
    /// executes on `ctx.queue`, waits for completion, and transitions `UAV -> COMMON`.
    pub fn dispatch_standalone(
        &mut self,
        ctx: &mut GpuContext,
        dst_heap: &ID3D12Resource,
        slot_indices: &[u32],
        slot_data_64b: &[u8],
    ) -> Result<f64, String> {
        if slot_indices.is_empty() {
            return Ok(0.0);
        }
        let alloc: ID3D12CommandAllocator = unsafe {
            ctx.device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(|e| format!("CreateCommandAllocator: {e:?}"))?
        };
        let list: ID3D12GraphicsCommandList = unsafe {
            ctx.device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                .map_err(|e| format!("CreateCommandList: {e:?}"))?
        };
        let t0 = Instant::now();
        unsafe {
            cmd_barriers(
                &list,
                vec![transition_barrier(
                    dst_heap,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )],
            );
        }
        self.record_patch(&list, dst_heap, slot_indices, slot_data_64b)?;
        unsafe {
            cmd_barriers(
                &list,
                vec![
                    uav_barrier(dst_heap),
                    transition_barrier(
                        dst_heap,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COMMON,
                    ),
                ],
            );
            list.Close().map_err(|e| format!("Close: {e:?}"))?;
            let cmd: ID3D12CommandList = list.cast().unwrap();
            ctx.queue.ExecuteCommandLists(&[Some(cmd)]);
        }
        ctx.signal_and_wait();
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    }
}

impl Drop for GpuScatterPatcher {
    fn drop(&mut self) {
        unsafe {
            self.src_data_buf.Unmap(0, None);
            self.src_indices_buf.Unmap(0, None);
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Wave32 NaN/Inf Guard (CPU Parity + Reusable SM 6.6 HLSL Header)
// ---------------------------------------------------------------------------

/// Clean-room IEEE-754 `NaN -> 0.0` / `±Inf -> ±65504.0` guard and HLSL SM 6.6 macro header.
pub struct Wave32NanGuard;

impl Wave32NanGuard {
    /// Embedded HLSL SM 6.6 header (`Wave32SanitizeFloat*`, `Wave32SafeReadLane64`).
    pub const HLSL_HEADER: &'static str = include_str!("../shaders/wave32_nan_guard.hlsli");

    /// Sanitizes an IEEE-754 `f32`:
    /// - `strict = true` (float render targets): any `NaN` becomes `0.0`, `±Inf` becomes `±65504.0`.
    /// - `strict = false` (typed UAV stores): only canonical quiet NaN (`0x7FC0_0000` with either
    ///   sign) becomes `0.0` and `±Inf` becomes `±65504.0`, preserving integer bit patterns such as
    ///   `0xFFFF_FFFF`.
    #[inline]
    pub fn sanitize_f32_cpu(f: f32, strict: bool) -> f32 {
        let bits = f.to_bits();
        let abs_bits = bits & 0x7FFF_FFFF;
        let inf_bits = 0x7F80_0000u32;
        let is_inf = abs_bits == inf_bits;
        let is_nan = if strict {
            abs_bits > inf_bits
        } else {
            abs_bits == 0x7FC0_0000u32
        };
        let sign = bits & 0x8000_0000;
        let clamped = sign | 65504.0f32.to_bits();
        let mut out = if is_inf { clamped } else { bits };
        if is_nan {
            out = 0;
        }
        f32::from_bits(out)
    }

    #[inline]
    pub fn sanitize_rgba_cpu(rgba: [f32; 4], strict: bool) -> [f32; 4] {
        [
            Self::sanitize_f32_cpu(rgba[0], strict),
            Self::sanitize_f32_cpu(rgba[1], strict),
            Self::sanitize_f32_cpu(rgba[2], strict),
            Self::sanitize_f32_cpu(rgba[3], strict),
        ]
    }

    /// Prepends [`Wave32NanGuard::HLSL_HEADER`] and wraps `return <expr>;` inside `PSMain` with
    /// `Wave32SanitizeFloat4(..., true)` for clean-room shader instrumentation.
    pub fn inject_hlsl_pixel_guard(hlsl_src: &str) -> String {
        let mut out = String::with_capacity(Self::HLSL_HEADER.len() + hlsl_src.len() + 128);
        out.push_str(Self::HLSL_HEADER);
        out.push('\n');
        let mut in_ps = false;
        let mut brace_depth = 0i32;
        for line in hlsl_src.lines() {
            if line.contains("PSMain") {
                in_ps = true;
            }
            if in_ps {
                for ch in line.chars() {
                    match ch {
                        '{' => brace_depth += 1,
                        '}' => brace_depth -= 1,
                        _ => {}
                    }
                }
                if let Some(ret_idx) = line.find("return ") {
                    let prev_ok = ret_idx == 0
                        || !line.as_bytes()[ret_idx - 1].is_ascii_alphanumeric()
                            && line.as_bytes()[ret_idx - 1] != b'_';
                    let after_ret = &line[ret_idx + "return ".len()..];
                    if prev_ok {
                        if let Some(semi_rel) = after_ret.find(';') {
                            let expr = after_ret[..semi_rel].trim();
                            let prefix = &line[..ret_idx];
                            let suffix = &after_ret[semi_rel + 1..];
                            out.push_str(prefix);
                            out.push_str("return Wave32SanitizeFloat4(");
                            out.push_str(expr);
                            out.push_str(", true);");
                            out.push_str(suffix);
                            out.push('\n');
                            if brace_depth <= 0 && line.contains('}') {
                                in_ps = false;
                            }
                            continue;
                        }
                    }
                }
                if brace_depth <= 0 && line.contains('}') {
                    in_ps = false;
                }
            }
            out.push_str(line);
            out.push('\n');
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_decoder::create_committed_buffer;

    #[test]
    fn dirty_bitmap_marks_and_drains_exact_64b_diffs() {
        const SLOTS: usize = 1_000_000;
        const HEAP_BYTES: usize = SLOTS * SCATTER_SLOT_BYTES;
        let pages = HEAP_BYTES.div_ceil(DIRTY_PAGE_BYTES);
        let mut bmp = HierarchicalDirtyBitmap::new(pages);
        assert_eq!(bmp.page_count(), pages);

        let mut src = vec![0u8; HEAP_BYTES];
        let mut shadow = vec![0u8; HEAP_BYTES];
        let dirty_slots = [0usize, 63, 64, 4097, 131_072, 500_001, 999_999];
        for (k, &s) in dirty_slots.iter().enumerate() {
            let off = s * SCATTER_SLOT_BYTES;
            let tag = (0xA000_0000u32 + k as u32).to_le_bytes();
            src[off..off + 4].copy_from_slice(&tag);
            src[off + 60..off + 64].copy_from_slice(&tag);
            bmp.mark_byte_range(off, SCATTER_SLOT_BYTES);
        }

        let mut indices = Vec::new();
        let mut data = Vec::new();
        diff_dirty_pages_64b(&mut bmp, &src, &mut shadow, SLOTS, &mut indices, &mut data);
        assert_eq!(indices, dirty_slots.iter().map(|&s| s as u32).collect::<Vec<_>>());
        assert_eq!(data.len(), dirty_slots.len() * SCATTER_SLOT_BYTES);
        assert_eq!(src, shadow);

        // Second pass on untouched bitmap must return zero patches.
        diff_dirty_pages_64b(&mut bmp, &src, &mut shadow, SLOTS, &mut indices, &mut data);
        assert!(indices.is_empty());
        assert!(data.is_empty());
    }

    #[test]
    fn wave32_nan_guard_handles_nan_inf_and_int_bitpatterns() {
        let qnan = f32::from_bits(0x7FC0_0000);
        let neg_qnan = f32::from_bits(0xFFC0_0000);
        let snan = f32::from_bits(0x7F80_0001);
        let all_ones = f32::from_bits(0xFFFF_FFFF);

        assert_eq!(Wave32NanGuard::sanitize_f32_cpu(qnan, true), 0.0);
        assert_eq!(Wave32NanGuard::sanitize_f32_cpu(neg_qnan, true), 0.0);
        assert_eq!(Wave32NanGuard::sanitize_f32_cpu(snan, true), 0.0);
        assert_eq!(Wave32NanGuard::sanitize_f32_cpu(all_ones, true), 0.0);

        // Non-strict preserves integer payload bit patterns (e.g. 0xFFFF_FFFF) while zeroing qNaN.
        assert_eq!(Wave32NanGuard::sanitize_f32_cpu(qnan, false), 0.0);
        assert_eq!(Wave32NanGuard::sanitize_f32_cpu(neg_qnan, false), 0.0);
        assert_eq!(Wave32NanGuard::sanitize_f32_cpu(all_ones, false).to_bits(), 0xFFFF_FFFF);

        // +/-Inf clamped to +/-65504.0 (fp16 finite ceiling)
        assert_eq!(Wave32NanGuard::sanitize_f32_cpu(f32::INFINITY, true), 65504.0);
        assert_eq!(Wave32NanGuard::sanitize_f32_cpu(f32::NEG_INFINITY, true), -65504.0);
        assert_eq!(Wave32NanGuard::sanitize_f32_cpu(1.25, true), 1.25);

        let wrapped = Wave32NanGuard::inject_hlsl_pixel_guard(
            "float4 PSMain() : SV_Target { return float4(1, 0, 0, 1); }",
        );
        assert!(wrapped.contains("Wave32SanitizeFloat4(float4(1, 0, 0, 1), true)"));
    }

    #[test]
    fn gpu_scatter_patcher_dx12_debug_layer_roundtrip() {
        let mut ctx = GpuContext::new(true);
        const HEAP_SLOTS: usize = 65_536; // 4 MiB
        const HEAP_BYTES: usize = HEAP_SLOTS * SCATTER_SLOT_BYTES;
        const PATCHES: usize = 1_500;

        let dst_heap = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_DEFAULT,
            HEAP_BYTES as u64,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );
        let readback = create_committed_buffer(
            &ctx.device,
            D3D12_HEAP_TYPE_READBACK,
            HEAP_BYTES as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );

        let mut indices = Vec::with_capacity(PATCHES);
        let mut payload = vec![0u8; PATCHES * SCATTER_SLOT_BYTES];
        for i in 0..PATCHES {
            let slot = ((i * 37 + 13) % HEAP_SLOTS) as u32;
            indices.push(slot);
            let base = i * SCATTER_SLOT_BYTES;
            for q in 0..16 {
                let word = ((i as u32) << 16) | (q as u32) | 0xA500;
                payload[base + q * 4..base + q * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
        }

        let mut patcher = GpuScatterPatcher::new(&ctx.device, PATCHES).expect("GpuScatterPatcher::new");
        let _gpu_ms = patcher
            .dispatch_standalone(&mut ctx, &dst_heap, &indices, &payload)
            .expect("dispatch_standalone");

        // Copy back full 4 MiB heap and check every patched slot byte-for-byte.
        let alloc: ID3D12CommandAllocator = unsafe {
            ctx.device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .unwrap()
        };
        let list: ID3D12GraphicsCommandList = unsafe {
            ctx.device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                .unwrap()
        };
        unsafe {
            cmd_barriers(
                &list,
                vec![transition_barrier(
                    &dst_heap,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )],
            );
            list.CopyResource(&readback, &dst_heap);
            cmd_barriers(
                &list,
                vec![transition_barrier(
                    &dst_heap,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                )],
            );
            list.Close().unwrap();
            let cl: ID3D12CommandList = list.cast().unwrap();
            ctx.queue.ExecuteCommandLists(&[Some(cl)]);
        }
        ctx.signal_and_wait();

        let mut rb_ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            readback.Map(0, None, Some(&mut rb_ptr)).unwrap();
            let got = std::slice::from_raw_parts(rb_ptr as *const u8, HEAP_BYTES);
            for (i, &slot) in indices.iter().enumerate() {
                let g_off = (slot as usize) * SCATTER_SLOT_BYTES;
                let p_off = i * SCATTER_SLOT_BYTES;
                assert_eq!(
                    &got[g_off..g_off + SCATTER_SLOT_BYTES],
                    &payload[p_off..p_off + SCATTER_SLOT_BYTES],
                    "mismatch at patch {i} slot {slot}"
                );
            }
            readback.Unmap(0, None);
            assert!(ctx.device.GetDeviceRemovedReason().is_ok());
            if let Ok(iq) = ctx.device.cast::<ID3D12InfoQueue>() {
                let err_cnt = (0..iq.GetNumStoredMessagesAllowedByRetrievalFilter())
                    .filter(|&mi| {
                        let mut len = 0usize;
                        let _ = iq.GetMessage(mi, None, &mut len);
                        if len == 0 {
                            return false;
                        }
                        let mut buf = vec![0u8; len];
                        let ptr = buf.as_mut_ptr() as *mut D3D12_MESSAGE;
                        iq.GetMessage(mi, Some(ptr), &mut len).is_ok()
                            && matches!(
                                (*ptr).Severity,
                                D3D12_MESSAGE_SEVERITY_CORRUPTION | D3D12_MESSAGE_SEVERITY_ERROR
                            )
                    })
                    .count();
                assert_eq!(err_cnt, 0, "DX12 Debug Layer reported {err_cnt} errors");
            }
        }
    }
}
