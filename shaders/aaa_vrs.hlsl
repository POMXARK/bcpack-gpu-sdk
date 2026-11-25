// Hardware Variable Rate Shading (VRS Tier 2) Tile Rate Generator (SM 6.6)
// Generates a screen-space R8_UINT shading rate image from TAA motion vectors & perceptual luma contrast.
// D3D12 rate encoding: 0 = 1x1 (D3D12_SHADING_RATE_1X1), 5 = 2x2 (D3D12_SHADING_RATE_2X2), 10 = 4x4 (D3D12_SHADING_RATE_4X4).
cbuffer VrsCB : register(b0) {
    uint g_VrsWidth;
    uint g_VrsHeight;
    uint g_RenderWidth;
    uint g_RenderHeight;
    uint g_TileSize;
    uint g_Allow4x4;
    uint g_Aggressive;
    uint g_Pad;
};

Texture2D<float2>  g_MotionVec : register(t0);
Texture2D<float4>  g_ColorHDR  : register(t1);
RWTexture2D<uint>  g_VrsOut    : register(u0);
SamplerState       g_LinClamp  : register(s0);

float perceptual_luma(float3 hdr) {
    float lum = dot(max(hdr, 0.0), float3(0.2126, 0.7152, 0.0722));
    return lum / (1.0 + lum);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= g_VrsWidth || dtid.y >= g_VrsHeight) {
        return;
    }

    float2 renderSize = float2((float)g_RenderWidth, (float)g_RenderHeight);
    float2 tileMinPx = float2(dtid.xy * g_TileSize);
    float2 tileSizePx = float2((float)g_TileSize, (float)g_TileSize);

    float2 uvC  = saturate((tileMinPx + tileSizePx * 0.50) / renderSize);
    float2 uvTL = saturate((tileMinPx + tileSizePx * 0.15) / renderSize);
    float2 uvTR = saturate((tileMinPx + float2(tileSizePx.x * 0.85, tileSizePx.y * 0.15)) / renderSize);
    float2 uvBL = saturate((tileMinPx + float2(tileSizePx.x * 0.15, tileSizePx.y * 0.85)) / renderSize);
    float2 uvBR = saturate((tileMinPx + tileSizePx * 0.85) / renderSize);

    float lC  = perceptual_luma(g_ColorHDR.SampleLevel(g_LinClamp, uvC,  0.0).rgb);
    float lTL = perceptual_luma(g_ColorHDR.SampleLevel(g_LinClamp, uvTL, 0.0).rgb);
    float lTR = perceptual_luma(g_ColorHDR.SampleLevel(g_LinClamp, uvTR, 0.0).rgb);
    float lBL = perceptual_luma(g_ColorHDR.SampleLevel(g_LinClamp, uvBL, 0.0).rgb);
    float lBR = perceptual_luma(g_ColorHDR.SampleLevel(g_LinClamp, uvBR, 0.0).rgb);

    float minL = min(lC, min(min(lTL, lTR), min(lBL, lBR)));
    float maxL = max(lC, max(max(lTL, lTR), max(lBL, lBR)));
    float lumaContrast = maxL - minL;

    float v0 = length(g_MotionVec.SampleLevel(g_LinClamp, uvC,  0.0));
    float v1 = length(g_MotionVec.SampleLevel(g_LinClamp, uvTL, 0.0));
    float v2 = length(g_MotionVec.SampleLevel(g_LinClamp, uvBR, 0.0));
    float maxVel = max(v0, max(v1, v2));

    // Preserve 1x1 detail around center hero vehicle & UI HUD corner
    float centerDist = length((uvC - float2(0.5, 0.54)) * float2(1.0, 1.35));
    float contrastHi = (g_Aggressive != 0u) ? 0.14 : 0.10;
    float contrastLo = (g_Aggressive != 0u) ? 0.08 : 0.05;

    uint rate = 0u; // 1x1
    if (centerDist < 0.24 || lumaContrast > contrastHi) {
        rate = 0u;
    } else if (g_Allow4x4 != 0u && g_Aggressive != 0u && uvC.y < 0.45 && centerDist > 0.48 && lumaContrast < 0.015) {
        rate = 10u; // 4x4 (D3D12_SHADING_RATE_4X4) on flat upper sky/horizon only
    } else if (maxVel > 0.004 || lumaContrast < contrastLo || centerDist > 0.36) {
        rate = 5u;  // 2x2 (D3D12_SHADING_RATE_2X2)
    }

    g_VrsOut[dtid.xy] = rate;
}
