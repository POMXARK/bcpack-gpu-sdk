// Cross-Vendor Xbox Series X|S (RDNA2)-Style Compute RayQuery RTGI Pass (SM 6.6 / Vulkan 1.3 VK_KHR_ray_query)
// Traces coherent 8x8 tile rays (2x Wave32 on NVIDIA RTX / 1x Wave64 on AMD Radeon) over the 4-MRT G-Buffer
// instead of per-fragment pixel-shader ray loops, eliminating 2x2 quad helper ray waste and overdraw rays.

static const float PI = 3.14159265359;

cbuffer RtgiCsCB : register(b0) {
    float4x4 g_InvVP;
    float4   g_CameraPos;  // xyz = camPos, w = maxDist
    float4   g_SunDirTime; // xyz = sunDir, w = time
    uint4    g_Params;     // x = width, y = height, z = frameIdx, w = flags
};

struct RtInstMeta {
    float4 baseColor; // rgb = tint/albedo, a = emissive
    uint4  geom;      // x = baseVert, y = idxOffset, z = texIdx, w = flags
    float4 uvParams;  // x = uvScale
};

Texture2D<float4>               g_GiAlbedoMask : register(t0);
Texture2D<float4>               g_NormalDepth  : register(t1);
RaytracingAccelerationStructure g_SceneTlas    : register(t2);
StructuredBuffer<RtInstMeta>    g_InstMeta     : register(t3);
ByteAddressBuffer               g_SceneVB      : register(t4);
StructuredBuffer<uint>          g_SceneIB      : register(t5);
Texture2D<float4>               g_EnvHdr       : register(t6);
RWTexture2D<float4>             g_OutRTGI      : register(u0);
SamplerState                    g_EnvSampler   : register(s0);

float ign(float2 xy, float frame) {
    float2 p = xy + 5.588238 * fmod(frame, 64.0);
    return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715))));
}

float2 dir_to_equirect_uv(float3 d) {
    float phi = atan2(d.z, d.x);
    float u = frac(phi / 6.28318530718 + 0.5 - 0.115);
    float v = clamp(acos(clamp(d.y, -1.0, 1.0)) / PI, 0.002, 0.998);
    return float2(u, v);
}

float3 sample_env(float3 dir) {
    float3 d = (dot(dir, dir) > 1e-6) ? normalize(dir) : float3(0.0, 1.0, 0.0);
    return g_EnvHdr.SampleLevel(g_EnvSampler, dir_to_equirect_uv(d), 0.0).rgb * 0.50;
}

float3 eval_bounce_hit(
    uint instId,
    uint primIdx,
    float2 bary,
    float3x4 objToWorld,
    float3 sunDir
) {
    RtInstMeta meta = g_InstMeta[instId];
    uint baseIdx = meta.geom.y + primIdx * 3u;
    uint i0 = meta.geom.x + g_SceneIB[baseIdx + 0u];
    uint i1 = meta.geom.x + g_SceneIB[baseIdx + 1u];
    uint i2 = meta.geom.x + g_SceneIB[baseIdx + 2u];

    float3 n0 = asfloat(g_SceneVB.Load3(i0 * 48u + 12u));
    float3 n1 = asfloat(g_SceneVB.Load3(i1 * 48u + 12u));
    float3 n2 = asfloat(g_SceneVB.Load3(i2 * 48u + 12u));

    float w0 = 1.0 - bary.x - bary.y;
    float3 localN = n0 * w0 + n1 * bary.x + n2 * bary.y;
    float3 worldN = mul(objToWorld, float4(localN, 0.0));
    worldN = (dot(worldN, worldN) > 1e-6) ? normalize(worldN) : float3(0.0, 1.0, 0.0);

    float3 albedo = meta.baseColor.rgb;
    float hitNdl = max(dot(worldN, sunDir), 0.0);
    float3 skyHit = sample_env(worldN) * 0.65;
    float3 directHit = albedo * float3(1.55, 0.94, 0.50) * hitNdl;
    float3 lit = albedo * skyHit + directHit + albedo * meta.baseColor.a * 0.45;
    return min(lit, float3(4.0, 4.0, 4.0));
}

[numthreads(8, 8, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= g_Params.x || dtid.y >= g_Params.y) {
        return;
    }

    int2 px = int2(dtid.xy);
    float4 giAlb = g_GiAlbedoMask[px];
    float4 nd = g_NormalDepth[px];
    if (giAlb.a < 0.5 || nd.w >= g_CameraPos.w || dot(nd.xyz, nd.xyz) < 1e-4) {
        g_OutRTGI[px] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float2 uv = (float2(px) + 0.5) / float2((float)g_Params.x, (float)g_Params.y);
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float4 worldPt = mul(g_InvVP, float4(ndc, 1.0, 1.0));
    float3 rawDir = worldPt.xyz / worldPt.w - g_CameraPos.xyz;
    float3 viewDir = (dot(rawDir, rawDir) > 1e-6) ? normalize(rawDir) : float3(0.0, 0.0, 1.0);
    float3 worldPos = g_CameraPos.xyz + viewDir * nd.w;
    float3 N = normalize(nd.xyz);
    float3 sunDir = normalize(g_SunDirTime.xyz);

    float3 upVec = (abs(N.z) < 0.999) ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 T = normalize(cross(upVec, N));
    float3 B = cross(N, T);

    float framePhase = (float)(g_Params.z & 15u);
    float r1 = ign(float2(px), framePhase);
    float r2 = ign(float2(px) + float2(17.0, 31.0), framePhase);
    float phi = 2.0 * PI * r1;
    float cosTheta = sqrt(r2);
    float sinTheta = sqrt(max(1.0 - r2, 0.0));
    float3 giRaw = T * (cos(phi) * sinTheta) + B * (sin(phi) * sinTheta) + N * cosTheta;
    float3 giDir = (dot(giRaw, giRaw) > 1e-6) ? normalize(giRaw) : float3(0.0, 1.0, 0.0);

    RayDesc gRay;
    gRay.Origin = worldPos + N * 0.04;
    gRay.Direction = giDir;
    gRay.TMin = 0.04;
    gRay.TMax = 45.0;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> gq;
    gq.TraceRayInline(g_SceneTlas, RAY_FLAG_NONE, 0xFFu, gRay);
    while (gq.Proceed()) {}

    float3 bounce;
    if (gq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        float fade = saturate(1.0 - gq.CommittedRayT() / 45.0);
        bounce = eval_bounce_hit(
            gq.CommittedInstanceID(),
            gq.CommittedPrimitiveIndex(),
            gq.CommittedTriangleBarycentrics(),
            gq.CommittedObjectToWorld3x4(),
            sunDir
        ) * fade * 1.15;
    } else {
        bounce = sample_env(giDir) * 0.45;
    }

    g_OutRTGI[px] = float4(min(bounce, float3(4.0, 4.0, 4.0)) * giAlb.rgb, 1.0);
}
