// SM 6.6 Dynamic Resource Paging Verify Shader: 1.25M Virtual Descriptor IDs -> 262,144 Physical ResourceDescriptorHeap[] Slots
cbuffer PageCB : register(b0) {
    uint g_Count;
    uint g_VirtCap;
    uint g_PhysCap;
    uint g_Pad;
};

StructuredBuffer<uint>   g_ReqVirtIds : register(t0);
StructuredBuffer<uint>   g_VirtToPhys : register(t1);
RWStructuredBuffer<uint> g_OutPixels  : register(u0);

uint pack_rgba8(float4 c) {
    uint r = (uint)clamp(round(c.r * 255.0), 0.0, 255.0);
    uint g = (uint)clamp(round(c.g * 255.0), 0.0, 255.0);
    uint b = (uint)clamp(round(c.b * 255.0), 0.0, 255.0);
    uint a = (uint)clamp(round(c.a * 255.0), 0.0, 255.0);
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

[numthreads(64, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint tid = dtid.x;
    if (tid >= g_Count) {
        return;
    }

    uint virtId = g_ReqVirtIds[tid];
    if (virtId >= g_VirtCap) {
        g_OutPixels[tid] = 0xDEADBEEFu;
        return;
    }

    uint phys = g_VirtToPhys[virtId];
    if (phys >= g_PhysCap) {
        g_OutPixels[tid] = 0xDEADBEEFu;
        return;
    }

    Texture2D<float4> tex = ResourceDescriptorHeap[NonUniformResourceIndex(phys)];
    float4 c = tex.Load(int3(0, 0, 0));
    g_OutPixels[tid] = pack_rgba8(c);
}
