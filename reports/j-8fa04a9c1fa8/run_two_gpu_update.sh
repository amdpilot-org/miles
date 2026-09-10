#!/usr/bin/env bash
set -euo pipefail

mode="${1:-full}"
case "$mode" in
  smoke) num_rollout=1 ;;
  full) num_rollout=64 ;;
  *) echo "usage: $0 [smoke|full]" >&2; exit 2 ;;
esac

report_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$report_dir/../.." && pwd)"
artifacts_dir="${MILES_ARTIFACTS:-/job/miles-artifacts}"
model_dir="$artifacts_dir/model"
python_bin="${MILES_PYTHON:-/opt/venv/bin/python}"
run_id="driver-$mode-$(date +%s%N | tail -c 10)"
output_dir="$artifacts_dir/run/$run_id"
log_path="$artifacts_dir/run/$run_id.log"
ray_tmpdir="/tmp/ray-j8fa-$RANDOM"
run_uuid="$(printf '%016x' "$RANDOM$RANDOM")"

mkdir -p "$artifacts_dir/run" "$artifacts_dir/checkpoints-$mode" "$ray_tmpdir"
if [ ! -f "$model_dir/model.safetensors" ]; then
  mkdir -p "$model_dir"
  PYTHONPATH="$repo_dir" "$python_bin" "$report_dir/make_local_model.py" "$model_dir"
fi

export PYTHONPATH="$repo_dir"
export RAY_TMPDIR="$ray_tmpdir"
export RAY_DEDUP_LOGS=0
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"
export AITER_CONFIG_GEMM_BF16=/sgl-workspace/aiter/aiter/configs/bf16_tuned_gemm.csv
export AITER_LOG_TUNED_CONFIG=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export NCCL_SOCKET_TIMEOUT_MS=600000
export J_INVESTIGATION_OUTPUT_DIR="$output_dir"

mkdir -p "$output_dir"
ln -sfn "$output_dir" "$artifacts_dir/run/latest-driver-$mode"
"$python_bin" "$report_dir/two_gpu_update_driver.py" \
  --run-uuid "$run_uuid" \
  --train-backend fsdp \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 1 \
  --rollout-num-gpus 1 \
  --rollout-num-gpus-per-engine 1 \
  --num-gpus-per-node 2 \
  --hf-checkpoint "$model_dir" \
  --prompt-data "$report_dir/prompts.jsonl" \
  --input-key input \
  --label-key label \
  --custom-rm-path reports.j-8fa04a9c1fa8.deterministic_reward.deterministic_reward \
  --num-rollout "$num_rollout" \
  --rollout-batch-size 2 \
  --n-samples-per-prompt 2 \
  --num-steps-per-rollout 4 \
  --micro-batch-size 1 \
  --rollout-max-prompt-len 16 \
  --rollout-max-response-len 16 \
  --rollout-max-context-len 32 \
  --distributed-timeout-minutes 10 \
  --sglang-dist-timeout 600 \
  --sglang-watchdog-timeout 300 \
  --sglang-mem-fraction-static 0.2 \
  --sglang-attention-backend triton \
  --sglang-disable-cuda-graph \
  --save "$artifacts_dir/checkpoints-$mode" \
  --save-interval 1 \
  2>&1 | tee "$log_path"
status=${PIPESTATUS[0]}
echo "RUN_ID=$run_id STATUS=$status OUTPUT_DIR=$output_dir LOG=$log_path"
exit "$status"
