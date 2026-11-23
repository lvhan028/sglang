#!/usr/bin/env bash
# Phase 0 / Phase 1a frozen launch script for SGLang Tiered Persistent Engine (v0.5.23).
# Locks down single-GPU, Prefill Eager + Full Decode CUDA Graph, no overlap,
# no unadapted plugins, and post-capture KV sizing disabled at config time.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:?Please set MODEL_PATH (e.g. /path/to/Qwen3-8B)}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.78}"
TPE_CACHE_ROOT="${TPE_CACHE_ROOT:-/tmp/sglang_tpe_cache}"

# Canonical SGLang binary/JIT cache paths (set before Python import)
export SGLANG_CACHE_DIR="${TPE_CACHE_ROOT}/sglang"
export SGLANG_DG_CACHE_DIR="${SGLANG_CACHE_DIR}/deep_gemm"
export SGLANG_JIT_CACHE_DIR="${SGLANG_CACHE_DIR}/jit"
export SGLANG_CUTE_AOT_CACHE_DIR="${SGLANG_CACHE_DIR}/cute_aot"
export TORCH_EXTENSIONS_DIR="${TPE_CACHE_ROOT}/torch_ext"
mkdir -p \
  "${SGLANG_CACHE_DIR}" \
  "${SGLANG_DG_CACHE_DIR}" \
  "${SGLANG_JIT_CACHE_DIR}" \
  "${SGLANG_CUTE_AOT_CACHE_DIR}" \
  "${TORCH_EXTENSIONS_DIR}"

# Explicitly disable post-capture KV sizing before pool construction (F6)
export SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=0

exec python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp-size 1 \
  --disable-overlap-schedule \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-bs 1 4 8 16 \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --disable-radix-cache \
  "$@"
