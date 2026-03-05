#!/usr/bin/env bash
set -euo pipefail

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm is not installed or not on PATH" >&2
  exit 1
fi

MODEL="${QWEN_MODEL:-Qwen/Qwen3.5-4B}"
DTYPE="${QWEN_DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-32768}"
REASONING_PARSER="${QWEN_REASONING_PARSER:-qwen3}"

exec vllm serve "${MODEL}" \
  --dtype "${DTYPE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --reasoning-parser "${REASONING_PARSER}"
