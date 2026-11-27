// Clean-room SDK Primitive: Coalesced 64-Byte GPU Scatter-Patch Compute Shader (SM 6.6)
// Copyright (c) 2026 POMXARK — MIT OR Apache-2.0
// Replaces thousands of small CPU->GPU buffer/descriptor CopyBufferRegion calls with one Dispatch.

cbuffer ScatterPatchCB : register(b0) {
    uint g_PatchCount;
    uint g_Pad0;
    uint g_Pad1;
    uint g_Pad2;
};

StructuredBuffer<uint4> g_SrcData        : register(t0); // 4 x uint4 (64 bytes) per patch
StructuredBuffer<uint>  g_DstSlotIndices : register(t1); // destination 64-byte slot index per patch
RWByteAddressBuffer     g_DstHeap        : register(u0); // 64-byte aligned target GPU heap/buffer

[numthreads(64, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    if (idx >= g_PatchCount) {
        return;
    }
    uint dst_slot = g_DstSlotIndices[idx];
    uint src_base = idx * 4u;
    uint dst_byte = dst_slot * 64u;
    [unroll]
    for (uint q = 0u; q < 4u; ++q) {
        g_DstHeap.Store4(dst_byte + q * 16u, g_SrcData[src_base + q]);
    }
}
