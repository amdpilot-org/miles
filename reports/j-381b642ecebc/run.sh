#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
model_dir="${script_dir}/tiny-qwen3-moe"
cache_root="${MILES_J381B_CACHE_ROOT:-/job/.cache/j-381b642ecebc}"
aiter_config="${cache_root}/aiter/bf16_tuned_gemm.csv"

mkdir -p "${cache_root}/aiter"
cp "/sgl-workspace/aiter/aiter/configs/bf16_tuned_gemm.csv" "${aiter_config}"

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export AITER_CONFIG_GEMM_BF16="${aiter_config}"
export SGLANG_DISABLE_MULTIMEM_AG=0
export OMP_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

timeout 420s /opt/venv/bin/torchrun \
  --standalone --nnodes=1 --nproc-per-node=2 \
  "${script_dir}/reduced_moe_p2p.py" \
  --check same-node --model-dir "${model_dir}" --timeout-seconds 30

timeout 240s /opt/venv/bin/torchrun \
  --standalone --nnodes=1 --nproc-per-node=2 \
  "${script_dir}/reduced_moe_p2p.py" \
  --check issue-boundary --model-dir "${model_dir}" --timeout-seconds 30
