// SM 6.6 Hierarchical-Z Depth Pyramid Builder (Max-Reduction Conservative Depth)
cbuffer HiZBuildCB : register(b0) {
    uint g_DstW;
    uint g_DstH;
    uint g_Pass; // 0 = copy D32->HiZ0, 1 = 2x2 max reduction
    uint g_Pad;
};

Texture2D<float>   g_Src : register(t0);
RWTexture2D<float> g_Dst : register(u0);

[numthreads(8, 8, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= g_DstW || dtid.y >= g_DstH) {
        return;
    }
    int2 px = int2(dtid.xy);
    if (g_Pass == 0u) {
        g_Dst[px] = g_Src.Load(int3(px, 0));
        return;
    }
    int2 srcMax = int2((int)g_DstW * 2 - 1, (int)g_DstH * 2 - 1);
    int2 base = px * 2;
    float d00 = g_Src.Load(int3(clamp(base + int2(0, 0), int2(0, 0), srcMax), 0));
    float d10 = g_Src.Load(int3(clamp(base + int2(1, 0), int2(0, 0), srcMax), 0));
    float d01 = g_Src.Load(int3(clamp(base + int2(0, 1), int2(0, 0), srcMax), 0));
    float d11 = g_Src.Load(int3(clamp(base + int2(1, 1), int2(0, 0), srcMax), 0));
    g_Dst[px] = max(max(d00, d10), max(d01, d11));
}
