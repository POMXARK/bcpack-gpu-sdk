// Clean-room C-ABI interface for bcpack_gpu.dll
// Designed for integration into Xbox translation layers / IDStorageQueueX hooks (e.g., XCL / AnyXboxX).

#ifndef BCPACK_GPU_H
#define BCPACK_GPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && !defined(BCPACK_STATIC)
#  ifdef BCPACK_EXPORTS
#    define BCPACK_API __declspec(dllexport)
#  else
#    define BCPACK_API __declspec(dllimport)
#  endif
#else
#  define BCPACK_API
#endif

typedef struct BcPackContext BcPackContext;

typedef enum BcPackFormat {
    BCPACK_FORMAT_BC1      = 0, // Full 12-lane rANS + predictor + quad reconstruction -> raw BC1 blocks (8B/block)
    BCPACK_FORMAT_BC7_RANS = 1, // 16-lane rANS entropy stage (100% Wave32) -> uint16[16] symbols/row (32B/row)
    BCPACK_FORMAT_BC3      = 2, // Full 12-lane rANS + pair/quad/dict/history rebuild -> raw BC3 blocks (16B/block)
    BCPACK_FORMAT_BC4      = 3, // Full 12-lane rANS + pair/dict/history rebuild -> raw BC4 blocks (8B/block)
    BCPACK_FORMAT_BC5      = 4, // Full 12-lane rANS + pair/dict/history rebuild -> raw BC5 blocks (16B/block)
    BCPACK_FORMAT_BC7      = 5, // Full 16-lane rANS + Modes 0..7 dict/history rebuild -> raw BC7 blocks (16B/block)
    BCPACK_FORMAT_BC6H     = 6, // Full 16-lane rANS + Modes 1..14 dict/history rebuild -> raw BC6H blocks (16B/block)
} BcPackFormat;

typedef enum BcPackSwizzle {
    BCPACK_SWIZZLE_LINEAR        = 0, // Linear block order
    BCPACK_SWIZZLE_GFX10_64KB_D  = 1, // AMD Gfx10 64KB_D ("xyxxyyxyxyxyx") -> detiled to linear on GPU
    BCPACK_SWIZZLE_GFX10_4KB_S   = 2, // AMD Gfx10 4KB_S  ("xyyxxyxyx")     -> detiled to linear on GPU
} BcPackSwizzle;

// Creates a DX12 Async Compute GPU decompression context on the high-performance adapter.
// Pass 0 for defaults (max_chunks=2048, max_blocks=8_388_608, max_stream_bytes=96 MiB).
BCPACK_API BcPackContext* bcpack_gpu_create(
    uint32_t max_chunks,
    uint32_t max_blocks,
    uint32_t max_stream_bytes);

// Creates a zero-copy GPU decompression context sharing an existing caller ID3D12Device*.
// - d3d12_device: non-null ID3D12Device* (COM refcount incremented via AddRef; caller keeps ownership).
// - d3d12_compute_queue: optional ID3D12CommandQueue* (type COMPUTE); pass NULL to create an internal compute queue on d3d12_device.
BCPACK_API BcPackContext* bcpack_gpu_create_from_device(
    void* d3d12_device,
    void* d3d12_compute_queue,
    uint32_t max_chunks,
    uint32_t max_blocks,
    uint32_t max_stream_bytes);

// Destroys the DX12 context and frees GPU heaps/queues.
BCPACK_API void bcpack_gpu_destroy(BcPackContext* ctx);

// Uploads caller-supplied full BC1 rank1 (256B) + dictionary (255 x 6B ep + 255 x 16B ix) tables and switches BCPACK_FORMAT_BC1 to full history/ix_ref/dictionary rebuild. Returns 0 on success.
BCPACK_API int32_t bcpack_gpu_set_bc1_tables(
    BcPackContext* ctx,
    const uint8_t* rank1_256,
    const uint8_t* dict_ep_255x6,
    const uint8_t* dict_ix_255x16);

// Uploads caller-supplied BC7 control words (512 x uint16) and/or 255-entry top-aligned dictionary (subsets[255], ep[255*24], ix[255*32]). Returns 0 on success.
BCPACK_API int32_t bcpack_gpu_set_bc7_tables(
    BcPackContext* ctx,
    const uint16_t* ctrl7_512,
    const uint8_t* dict_subsets_255,
    const uint8_t* dict_ep8_255x24,
    const uint8_t* dict_ix4_255x32);

// Uploads caller-supplied BC6H 255-entry dictionary (ep[255*12] uint16 left-aligned, ix_mode[255] uint8, ix[255*16] int16). Returns 0 on success.
BCPACK_API int32_t bcpack_gpu_set_bc6h_tables(
    BcPackContext* ctx,
    const uint16_t* dict_ep16_255x12,
    const uint8_t* dict_ix_mode_255,
    const int16_t* dict_ix16_255x16);

// Returns the underlying ID3D12Fence* signaled by the compute queue after BcPackDecodeToBuffer / BcPackDecodeAsync
// so callers can execute GPU-side direct_queue->Wait((ID3D12Fence*)BcPackGetComputeFence(ctx), fence_value) without CPU stalls.
BCPACK_API void* BcPackGetComputeFence(const BcPackContext* ctx);

// Returns the internal DIRECT queue ID3D12Fence* signaled after BcPackDecodeToTexture2D completes CopyTextureRegion.
BCPACK_API void* BcPackGetDirectFence(const BcPackContext* ctx);

// Waits on CPU for a DIRECT fence value returned by BcPackDecodeToTexture2D (0 = signaled, 1 = timeout, <0 = error).
BCPACK_API int32_t BcPackWaitDirectFence(
    BcPackContext* ctx,
    uint64_t fence_value,
    uint32_t timeout_ms);

// Zero-copy async decode directly into a caller-owned ID3D12Resource* buffer on the shared device (zero CPU readback).
// - dst_d3d12_resource MUST live on the same ID3D12Device passed to bcpack_gpu_create_from_device and be in D3D12_RESOURCE_STATE_COMMON on entry (restored to COMMON on exit).
// - Signals *out_fence_value on BcPackGetComputeFence(ctx). Returns 0 on success, < 0 on error.
BCPACK_API int32_t BcPackDecodeToBuffer(
    BcPackContext* ctx,
    const uint8_t* compressed_data,
    uint32_t compressed_size,
    uint32_t num_blocks,
    uint32_t format,
    uint32_t swizzle_mode,
    uint32_t width_blocks,
    uint32_t height_blocks,
    void* dst_d3d12_resource,
    uint64_t dst_byte_offset,
    uint64_t* out_fence_value);

// Zero-copy async decode (+ optional Gfx10 detile) directly into a 2D DXGI_FORMAT_BCn_UNORM ID3D12Resource* subresource.
// - Executes BCPack decode (+ detile) on the Async Compute queue, performs GPU Queue::Wait onto DIRECT, and issues CopyTextureRegion
//   (automatically padding row pitches to 256B for small mip levels).
// - dst_texture2d MUST live on the shared device and enter in D3D12_RESOURCE_STATE_COMMON (restored to COMMON on exit).
// - Signals *out_fence_value on BcPackGetDirectFence(ctx). Supports BCPACK_FORMAT_BC1, BC3, BC4, BC5, and BC7.
BCPACK_API int32_t BcPackDecodeToTexture2D(
    BcPackContext* ctx,
    const uint8_t* compressed_data,
    uint32_t compressed_size,
    uint32_t num_blocks,
    uint32_t format,
    uint32_t swizzle_mode,
    uint32_t width_blocks,
    uint32_t height_blocks,
    void* dst_texture2d,
    uint32_t subresource_index,
    uint32_t dst_x_blocks,
    uint32_t dst_y_blocks,
    uint64_t* out_fence_value);

// Submits a BCPack chunk onto the DX12 D3D12_COMMAND_LIST_TYPE_COMPUTE Async Compute Queue.
// - For BCPACK_FORMAT_BC1: num_blocks is BC1 block count (width_blocks * height_blocks), output = num_blocks * 8 bytes.
//   If swizzle_mode != BCPACK_SWIZZLE_LINEAR, chains Gfx10 Swizzle -> Linear compute pass in the same Async Compute list.
// - For BCPACK_FORMAT_BC7_RANS: num_blocks is BC7 block count (emits 2 symbol rows per block, total 2*num_blocks*32 bytes).
// - If out_host_buffer != NULL: waits for the compute fence and copies decoded bytes into out_host_buffer.
// - If out_host_buffer == NULL: returns immediately after signaling *out_fence_value (zero-stall async mode).
// Returns 0 on success, < 0 on error.
BCPACK_API int32_t BcPackDecodeAsync(
    BcPackContext* ctx,
    const uint8_t* compressed_data,
    uint32_t compressed_size,
    uint32_t num_blocks,
    uint32_t format,
    uint32_t swizzle_mode,
    uint32_t width_blocks,
    uint32_t height_blocks,
    uint8_t* out_host_buffer,
    uint32_t out_capacity,
    uint64_t* out_fence_value);

// Waits on CPU for a previously returned Async Compute fence value.
// Returns 0 if signaled, 1 on timeout, < 0 on error.
BCPACK_API int32_t BcPackWaitFence(
    BcPackContext* ctx,
    uint64_t fence_value,
    uint32_t timeout_ms);

// Reads the GPU compute kernel timestamp duration (in milliseconds) of the last completed submission.
// Returns 0 on success, < 0 on error.
BCPACK_API int32_t BcPackGetLastTimingMs(
    const BcPackContext* ctx,
    double* out_kernel_ms);

#ifdef __cplusplus
}
#endif

#endif // BCPACK_GPU_H
