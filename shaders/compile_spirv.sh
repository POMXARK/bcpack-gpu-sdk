#!/usr/bin/env bash
# Compiles core clean-room Compute Shaders to Vulkan 1.3 SPIR-V using DXC with SPIR-V CodeGen enabled.
# Requires official DirectXShaderCompiler (e.g., microsoft/DirectXShaderCompiler >= v1.9.2609).
set -euo pipefail

DXC="${DXC:-dxc}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/spirv"
mkdir -p "$OUT_DIR"

SHADERS=(
  bcpack_decode
  bcpack_bcn_rans
  bcpack_bcn_rebuild
  bcpack_bc7_rans
  bcpack_bc7_rebuild
  bc16_pack
  gfx10_detile
  hiz_build
  crowd_skin
  bindless_paging
  aaa_vrs
  aaa_rtgi_cs
)

for s in "${SHADERS[@]}"; do
  echo "[dxc-spirv] $s.hlsl -> spirv/$s.spv"
  "$DXC" -spirv -fspv-target-env=vulkan1.3 -T cs_6_6 -E CSMain -O3 \
    -Fo "$OUT_DIR/$s.spv" "$SCRIPT_DIR/$s.hlsl"
done

echo "[dxc-spirv] Compiled ${#SHADERS[@]} Vulkan 1.3 SPIR-V compute kernels into $OUT_DIR"
