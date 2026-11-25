// SM 6.6 500 GPU-Skinned NPC Crowd Compute Kernel (4-Bone Linear Blend Skinning)
cbuffer CrowdCB : register(b0) {
    uint g_NpcCount;
    uint g_VertsPerNpc;
    uint g_BonesPerNpc;
    uint g_Pad;
};

struct BaseSkinnedVertex {
    float4 pos;     // xyz = bind pose pos
    float4 normal;  // xyz = bind pose normal
    uint4  boneIdx; // 4 bone indices in [0, g_BonesPerNpc - 1]
    float4 boneWt;  // 4 weights summing to 1.0
};

struct BoneMat {
    float4 c0;
    float4 c1;
    float4 c2;
    float4 c3;
};

struct OutSkinnedVertex {
    float4 pos;
    float4 normal;
};

StructuredBuffer<BaseSkinnedVertex>  g_BaseVerts : register(t0);
StructuredBuffer<BoneMat>            g_BoneMats  : register(t1);
RWStructuredBuffer<OutSkinnedVertex> g_OutVerts  : register(u0);

float3 xform_point(BoneMat m, float3 p) {
    return (m.c0 * p.x + m.c1 * p.y + m.c2 * p.z + m.c3).xyz;
}

float3 xform_vec(BoneMat m, float3 n) {
    return (m.c0.xyz * n.x + m.c1.xyz * n.y + m.c2.xyz * n.z);
}

[numthreads(64, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint totalVerts = g_NpcCount * g_VertsPerNpc;
    uint idx = dtid.x;
    if (idx >= totalVerts) {
        return;
    }

    uint npcId = idx / g_VertsPerNpc;
    uint localVert = idx - npcId * g_VertsPerNpc;
    BaseSkinnedVertex bv = g_BaseVerts[localVert];
    uint boneBase = npcId * g_BonesPerNpc;

    float3 skinnedPos = float3(0.0, 0.0, 0.0);
    float3 skinnedNrm = float3(0.0, 0.0, 0.0);

    BoneMat m0 = g_BoneMats[boneBase + bv.boneIdx.x];
    BoneMat m1 = g_BoneMats[boneBase + bv.boneIdx.y];
    BoneMat m2 = g_BoneMats[boneBase + bv.boneIdx.z];
    BoneMat m3 = g_BoneMats[boneBase + bv.boneIdx.w];

    skinnedPos += xform_point(m0, bv.pos.xyz) * bv.boneWt.x;
    skinnedPos += xform_point(m1, bv.pos.xyz) * bv.boneWt.y;
    skinnedPos += xform_point(m2, bv.pos.xyz) * bv.boneWt.z;
    skinnedPos += xform_point(m3, bv.pos.xyz) * bv.boneWt.w;

    skinnedNrm += xform_vec(m0, bv.normal.xyz) * bv.boneWt.x;
    skinnedNrm += xform_vec(m1, bv.normal.xyz) * bv.boneWt.y;
    skinnedNrm += xform_vec(m2, bv.normal.xyz) * bv.boneWt.z;
    skinnedNrm += xform_vec(m3, bv.normal.xyz) * bv.boneWt.w;

    float nLenSq = dot(skinnedNrm, skinnedNrm);
    float3 nOut = (nLenSq > 1e-8) ? (skinnedNrm * rsqrt(nLenSq)) : float3(0.0, 1.0, 0.0);

    OutSkinnedVertex ov;
    ov.pos = float4(skinnedPos, 1.0);
    ov.normal = float4(nOut, 0.0);
    g_OutVerts[idx] = ov;
}
