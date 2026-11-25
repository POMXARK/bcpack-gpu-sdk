// SM 6.6 Two-Pass Wave32 Frustum + Hierarchical-Z Occlusion Culling across 12M+ Triangle Meshlet Worlds
cbuffer HiZCullCB : register(b0) {
    float4x4 g_ViewProj;
    float4   g_FrustumPlanes[6];
    float4   g_CameraPos;
    uint     g_TotalWorkItems;
    float    g_MaxCullDist;
    uint     g_HiZWidth;
    uint     g_HiZHeight;
};

struct MeshletDesc {
    uint   vertOffset;
    uint   vertCount;
    uint   primOffset;
    uint   primCount;
    float3 boundCenter;
    float  boundRadius;
};

struct MeshletInstance {
    float3 pos;
    float  rotY;
    float3 scale;
    uint   texIndex;
    float  emissive;
    float  uvScale;
    float  specPower;
    float  pad;
};

StructuredBuffer<MeshletDesc>     g_Meshlets  : register(t0);
StructuredBuffer<MeshletInstance> g_Instances : register(t1);
StructuredBuffer<uint2>           g_WorkItems : register(t2);

Texture2D<float> g_HiZ0 : register(t3);
Texture2D<float> g_HiZ1 : register(t4);
Texture2D<float> g_HiZ2 : register(t5);
Texture2D<float> g_HiZ3 : register(t6);
Texture2D<float> g_HiZ4 : register(t7);
Texture2D<float> g_HiZ5 : register(t8);

// Byte offsets: 0=visMeshlets, 4=visTris, 8=frustumCulled, 12=hizOccluded, 16=hizOccludedTris
RWByteAddressBuffer g_Stats : register(u0);

float sample_hiz(uint mip, int2 px) {
    if (mip == 0u) return g_HiZ0.Load(int3(px, 0));
    if (mip == 1u) return g_HiZ1.Load(int3(px, 0));
    if (mip == 2u) return g_HiZ2.Load(int3(px, 0));
    if (mip == 3u) return g_HiZ3.Load(int3(px, 0));
    if (mip == 4u) return g_HiZ4.Load(int3(px, 0));
    return g_HiZ5.Load(int3(px, 0));
}

[numthreads(64, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint gid = dtid.x;
    bool inBounds = (gid < g_TotalWorkItems);
    bool frustumPass = false;
    bool hizOccluded = false;
    uint primCount = 0u;

    if (inBounds) {
        uint2 work = g_WorkItems[gid];
        MeshletInstance inst = g_Instances[work.x];
        MeshletDesc m = g_Meshlets[work.y];
        primCount = m.primCount;

        float s, c;
        sincos(inst.rotY, s, c);
        float3 sp = m.boundCenter * inst.scale;
        float3 worldCenter = float3(
            sp.x * c + sp.z * s + inst.pos.x,
            sp.y + inst.pos.y,
            -sp.x * s + sp.z * c + inst.pos.z
        );
        float maxScale = max(inst.scale.x, max(inst.scale.y, inst.scale.z));
        float worldRadius = m.boundRadius * maxScale;

        float3 toCenter = worldCenter - g_CameraPos.xyz;
        float dist = length(toCenter);
        frustumPass = (dist <= g_MaxCullDist + worldRadius);

        if (frustumPass) {
            [unroll]
            for (uint i = 0u; i < 6u; ++i) {
                float d = dot(g_FrustumPlanes[i].xyz, worldCenter) + g_FrustumPlanes[i].w;
                if (d < -worldRadius) {
                    frustumPass = false;
                    break;
                }
            }
        }

        if (frustumPass && dist > worldRadius + 1.0) {
            float3 viewDir = toCenter / max(dist, 1e-4);
            float3 nearPt = worldCenter - viewDir * worldRadius;
            float4 clipNear = mul(g_ViewProj, float4(nearPt, 1.0));
            if (clipNear.w > 0.05) {
                float sphereMinZ = clipNear.z / clipNear.w;
                // Compute conservative screen-space AABB around the world bounding sphere
                float2 minUv = float2(1.0, 1.0);
                float2 maxUv = float2(0.0, 0.0);
                bool validProj = true;
                [unroll]
                for (int cx = -1; cx <= 1; cx += 2) {
                    [unroll]
                    for (int cy = -1; cy <= 1; cy += 2) {
                        [unroll]
                        for (int cz = -1; cz <= 1; cz += 2) {
                            float3 corner = worldCenter + float3((float)cx, (float)cy, (float)cz) * worldRadius;
                            float4 clip = mul(g_ViewProj, float4(corner, 1.0));
                            if (clip.w <= 0.05) {
                                validProj = false;
                            } else {
                                float2 ndc = clip.xy / clip.w;
                                float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
                                minUv = min(minUv, uv);
                                maxUv = max(maxUv, uv);
                            }
                        }
                    }
                }
                if (validProj && maxUv.x >= 0.0 && minUv.x <= 1.0 && maxUv.y >= 0.0 && minUv.y <= 1.0) {
                    minUv = saturate(minUv);
                    maxUv = saturate(maxUv);
                    float rectW = (maxUv.x - minUv.x) * (float)g_HiZWidth;
                    float rectH = (maxUv.y - minUv.y) * (float)g_HiZHeight;
                    float maxDim = max(max(rectW, rectH), 1.0);
                    uint mip = (uint)clamp((int)ceil(log2(maxDim)), 0, 5);

                    int mipW = max((int)(g_HiZWidth >> mip), 1);
                    int mipH = max((int)(g_HiZHeight >> mip), 1);
                    int2 p0 = clamp(int2(minUv * float2((float)mipW, (float)mipH)), int2(0, 0), int2(mipW - 1, mipH - 1));
                    int2 p1 = clamp(int2(maxUv * float2((float)mipW, (float)mipH)), int2(0, 0), int2(mipW - 1, mipH - 1));

                    float d00 = sample_hiz(mip, int2(p0.x, p0.y));
                    float d10 = sample_hiz(mip, int2(p1.x, p0.y));
                    float d01 = sample_hiz(mip, int2(p0.x, p1.y));
                    float d11 = sample_hiz(mip, int2(p1.x, p1.y));
                    float maxZ = max(max(d00, d10), max(d01, d11));
                    if (sphereMinZ > maxZ + 0.0005) {
                        hizOccluded = true;
                    }
                }
            }
        }
    }

    bool visible = inBounds && frustumPass && !hizOccluded;
    bool frustumCulled = inBounds && !frustumPass;
    bool occluded = inBounds && frustumPass && hizOccluded;

    uint visCount = WaveActiveCountBits(visible);
    uint visTris = WaveActiveSum(visible ? primCount : 0u);
    uint fCullCount = WaveActiveCountBits(frustumCulled);
    uint hCullCount = WaveActiveCountBits(occluded);
    uint hCullTris = WaveActiveSum(occluded ? primCount : 0u);

    if (WaveIsFirstLane()) {
        if (visCount > 0u) {
            g_Stats.InterlockedAdd(0, visCount);
            g_Stats.InterlockedAdd(4, visTris);
        }
        if (fCullCount > 0u) {
            g_Stats.InterlockedAdd(8, fCullCount);
        }
        if (hCullCount > 0u) {
            g_Stats.InterlockedAdd(12, hCullCount);
            g_Stats.InterlockedAdd(16, hCullTris);
        }
    }
}
