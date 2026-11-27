// Clean-room SDK Primitive: Wave32 NaN/Inf Sanitizer & Safe Wave64->Wave32 Cross-Lane Guard (SM 6.6)
// Copyright (c) 2026 POMXARK — MIT OR Apache-2.0
//
// Prevents single-pixel NaN/Inf propagation in HDR render targets, bloom/TAA histories, and typed UAV stores
// when translating wide-wave (Wave64) multi-layer shaders onto Wave32 desktop GPUs.

#ifndef BCPACK_WAVE32_NAN_GUARD_HLSLI
#define BCPACK_WAVE32_NAN_GUARD_HLSLI

inline float Wave32SanitizeFloat(float f, bool strict_nan) {
    uint bits = asuint(f);
    uint abs_bits = bits & 0x7FFFFFFFu;
    bool is_inf = (abs_bits == 0x7F800000u);
    bool is_nan = strict_nan ? (abs_bits > 0x7F800000u) : (abs_bits == 0x7FC00000u);
    uint sign_bit = bits & 0x80000000u;
    uint clamped_inf = sign_bit | 0x477FE000u; // +-65504.0f (max finite IEEE-754 fp16)
    uint out_bits = is_inf ? clamped_inf : bits;
    out_bits = is_nan ? 0u : out_bits;
    return asfloat(out_bits);
}

inline float3 Wave32SanitizeFloat3(float3 v, bool strict_nan) {
    return float3(
        Wave32SanitizeFloat(v.x, strict_nan),
        Wave32SanitizeFloat(v.y, strict_nan),
        Wave32SanitizeFloat(v.z, strict_nan)
    );
}

inline float4 Wave32SanitizeFloat4(float4 v, bool strict_nan) {
    return float4(
        Wave32SanitizeFloat(v.x, strict_nan),
        Wave32SanitizeFloat(v.y, strict_nan),
        Wave32SanitizeFloat(v.z, strict_nan),
        Wave32SanitizeFloat(v.w, strict_nan)
    );
}

inline float Wave32SafeReadLane64(float val, uint logical_lane64) {
    uint hw_lanes = max(WaveGetLaneCount(), 1u);
    uint safe_lane = logical_lane64 & (hw_lanes - 1u);
    return WaveReadLaneAt(val, safe_lane);
}

#endif // BCPACK_WAVE32_NAN_GUARD_HLSLI
