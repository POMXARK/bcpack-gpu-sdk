// Clean-room AMD Gfx10 (RDNA2) Console Texture Swizzle -> Linear BCn Compute Shader
// Patterns documented in Ahmet-Dedeler/bcpack-rs (src/ttpf.rs):
// - tile_mode 0: SW_64KB_D (8B BC1/BC4): "xyxxyyxyxyxyx" (13 bits, 128x64 blocks)
// - tile_mode 1: SW_4KB_S  (8B BC1/BC4): "xyyxxyxyx"     (9 bits,  32x16 blocks)
// - tile_mode 2: SW_64KB_D (16B BC5/BC7): "xyxxyyxyxyxy" (12 bits, 64x64 blocks)

cbuffer DetileParams : register(b0) {
    uint width_blocks;
    uint height_blocks;
    uint tile_mode;
    uint block_bytes;
    uint src_byte_offset;
    uint dst_byte_offset;
    uint total_swizzled_blocks;
    uint pad0;
};

ByteAddressBuffer   g_Src : register(t0);
RWByteAddressBuffer g_Dst : register(u0);

[numthreads(64, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint i = dtid.x;
    if (i >= total_swizzled_blocks) {
        return;
    }

    uint lx = 0u;
    uint ly = 0u;
    uint tw = 128u;
    uint th = 64u;
    uint tl = 13u;

    if (tile_mode == 0u) {
        // SW_64KB_D (8B): "xyxxyyxyxyxyx"
        lx = ((i >> 0u) & 1u)
           | (((i >> 2u) & 1u) << 1u)
           | (((i >> 3u) & 1u) << 2u)
           | (((i >> 6u) & 1u) << 3u)
           | (((i >> 8u) & 1u) << 4u)
           | (((i >> 10u) & 1u) << 5u)
           | (((i >> 12u) & 1u) << 6u);
        ly = ((i >> 1u) & 1u)
           | (((i >> 4u) & 1u) << 1u)
           | (((i >> 5u) & 1u) << 2u)
           | (((i >> 7u) & 1u) << 3u)
           | (((i >> 9u) & 1u) << 4u)
           | (((i >> 11u) & 1u) << 5u);
        tw = 128u;
        th = 64u;
        tl = 13u;
    } else if (tile_mode == 1u) {
        // SW_4KB_S (8B): "xyyxxyxyx"
        lx = ((i >> 0u) & 1u)
           | (((i >> 3u) & 1u) << 1u)
           | (((i >> 4u) & 1u) << 2u)
           | (((i >> 6u) & 1u) << 3u)
           | (((i >> 8u) & 1u) << 4u);
        ly = ((i >> 1u) & 1u)
           | (((i >> 2u) & 1u) << 1u)
           | (((i >> 5u) & 1u) << 2u)
           | (((i >> 7u) & 1u) << 3u);
        tw = 32u;
        th = 16u;
        tl = 9u;
    } else {
        // SW_64KB_D (16B): "xyxxyyxyxyxy"
        lx = ((i >> 0u) & 1u)
           | (((i >> 2u) & 1u) << 1u)
           | (((i >> 3u) & 1u) << 2u)
           | (((i >> 6u) & 1u) << 3u)
           | (((i >> 8u) & 1u) << 4u)
           | (((i >> 10u) & 1u) << 5u);
        ly = ((i >> 1u) & 1u)
           | (((i >> 4u) & 1u) << 1u)
           | (((i >> 5u) & 1u) << 2u)
           | (((i >> 7u) & 1u) << 3u)
           | (((i >> 9u) & 1u) << 4u)
           | (((i >> 11u) & 1u) << 5u);
        tw = 64u;
        th = 64u;
        tl = 12u;
    }

    uint k = i >> tl;
    uint tpr = max((width_blocks + tw - 1u) / tw, 1u);
    uint x = (k % tpr) * tw + lx;
    uint y = (k / tpr) * th + ly;

    if (x < width_blocks && y < height_blocks) {
        uint linear_idx = y * width_blocks + x;
        if (block_bytes == 8u) {
            uint2 v = g_Src.Load2(src_byte_offset + i * 8u);
            g_Dst.Store2(dst_byte_offset + linear_idx * 8u, v);
        } else {
            uint4 v = g_Src.Load4(src_byte_offset + i * 16u);
            g_Dst.Store4(dst_byte_offset + linear_idx * 16u, v);
        }
    }
}
