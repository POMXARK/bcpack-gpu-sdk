// GPU Compute 16-Byte (BC7 / BC5) Block Packer for 16-lane BCPack rANS streams (HLSL SM 6.6)
// Reads 8 uints (16 x uint16 symbols) per row emitted by bcpack_bc7_rans.hlsl
// and repacks the 10 active lanes into a 64-bit word pair (8 bytes/row = 16 bytes/block).

cbuffer PackCB : register(b0) {
    uint  g_TotalRows;
    uint3 _pad;
};

StructuredBuffer<uint> g_InSyms    : register(t0); // 8 uints per symbol-row
RWByteAddressBuffer    g_OutBlocks : register(u0); // 8 bytes per symbol-row (2 rows = 16B BC7/BC5 block)

[numthreads(64, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint row = dtid.x;
    if (row >= g_TotalRows) {
        return;
    }

    uint base = row * 8u;
    uint u1 = g_InSyms[base + 1u]; // L2  | (L3  << 16)
    uint u2 = g_InSyms[base + 2u]; // L4  | (L5  << 16)
    uint u3 = g_InSyms[base + 3u]; // L6  | (L7  << 16)
    uint u4 = g_InSyms[base + 4u]; // L8  | (L9  << 16)
    uint u5 = g_InSyms[base + 5u]; // L10 | (L11 << 16)
    uint u6 = g_InSyms[base + 6u]; // L12 | (L13 << 16)
    uint u7 = g_InSyms[base + 7u]; // L14 | (L15 << 16)

    uint w0 = (u1 & 0x7Fu)
            | (((u1 >> 16u) & 0x7Fu) << 7u)
            | ((u2 & 0x7Fu) << 14u)
            | (((u2 >> 16u) & 0x7Fu) << 21u)
            | (((u4 >> 16u) & 0x0Fu) << 28u);

    uint w1 = (u3 & 0x7Fu)
            | (((u5 >> 16u) & 0x7Fu) << 7u)
            | ((u6 & 0x7Fu) << 14u)
            | (((u6 >> 16u) & 0x7Fu) << 21u)
            | (((u7 >> 16u) & 0x0Fu) << 28u);

    g_OutBlocks.Store2(row * 8u, uint2(w0, w1));
}
