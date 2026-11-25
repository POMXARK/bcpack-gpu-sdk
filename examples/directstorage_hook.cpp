// Clean-room example: Hooking an existing ID3D12Device / DirectStorage queue to
// zero-copy GPU BCPack decompression + AMD Gfx10 detile via bcpack_gpu.dll.
//
// Compile with MSVC (link bcpack_gpu.dll.lib + d3d12.lib):
//   cl /EHsc /I..\include directstorage_hook.cpp bcpack_gpu.dll.lib d3d12.lib

#include <d3d12.h>
#include <cstdint>
#include <cstdio>
#include "../include/bcpack_gpu.h"

struct BcPackTextureUploadRequest {
    const uint8_t* compressed_bytes;
    uint32_t compressed_size;
    uint32_t width_blocks;   // texture_width / 4
    uint32_t height_blocks;  // texture_height / 4
    uint32_t swizzle_mode;   // BCPACK_SWIZZLE_GFX10_64KB_D / BCPACK_SWIZZLE_GFX10_4KB_S / BCPACK_SWIZZLE_LINEAR
    ID3D12Resource* dst_texture2d; // Must be in D3D12_RESOURCE_STATE_COMMON
    uint32_t subresource_index;
};

int32_t SubmitZeroCopyBcPackUpload(
    BcPackContext* ctx,
    ID3D12CommandQueue* caller_direct_queue,
    const BcPackTextureUploadRequest& req)
{
    const uint32_t num_blocks = req.width_blocks * req.height_blocks;
    uint64_t direct_fence_val = 0;

    // 1. Decompresses BCPack stream + runs optional AMD Gfx10 detile on Async Compute,
    //    inserts GPU Compute -> Direct fence wait, and executes CopyTextureRegion
    //    (auto-padding row pitch to 256B for small mip levels).
    int32_t rc = BcPackDecodeToTexture2D(
        ctx,
        req.compressed_bytes,
        req.compressed_size,
        num_blocks,
        BCPACK_FORMAT_BC1,
        req.swizzle_mode,
        req.width_blocks,
        req.height_blocks,
        req.dst_texture2d,
        req.subresource_index,
        0, 0,
        &direct_fence_val);
    if (rc != 0) {
        std::fprintf(stderr, "BcPackDecodeToTexture2D failed: %d\n", rc);
        return rc;
    }

    // 2. Make caller's Direct queue wait purely on GPU (zero CPU stall).
    ID3D12Fence* direct_fence = static_cast<ID3D12Fence*>(BcPackGetDirectFence(ctx));
    caller_direct_queue->Wait(direct_fence, direct_fence_val);
    return 0;
}

int main() {
    std::puts("Include <bcpack_gpu.h> and call bcpack_gpu_create_from_device(device, compute_queue, 0, 0, 0) "
              "followed by SubmitZeroCopyBcPackUpload().");
    return 0;
}
