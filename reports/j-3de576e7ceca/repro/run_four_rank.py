import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL = Path("/job/cache/huggingface/Qwen3-VL-4B-Instruct")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("thd", "bshd"), required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--miles-checkout", default="/job/miles-validation")
    args = parser.parse_args()

    dataset = ROOT / "fixtures" / f"data_mbs{args.micro_batch_size}.jsonl"
    global_batch_size = 4 * args.micro_batch_size
    command = [
        sys.executable,
        str(Path(args.miles_checkout) / "train.py"),
        "--hf-checkpoint",
        str(MODEL),
        "--rollout-function-path",
        "repro_2406_rollout.generate_rollout",
        "--prompt-data",
        str(dataset),
        "--input-key",
        "messages",
        "--apply-chat-template",
        "--multimodal-keys",
        '{"image": "images"}',
        "--num-rollout",
        "2",
        "--rollout-batch-size",
        str(global_batch_size),
        "--n-samples-per-prompt",
        "1",
        "--global-batch-size",
        str(global_batch_size),
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--loss-type",
        "sft_loss",
        "--calculate-per-token-loss",
        "--disable-compute-advantages-and-returns",
        "--debug-train-only",
        "--train-backend",
        "fsdp",
        "--qkv-format",
        args.format,
        "--gradient-checkpointing",
        "--attn-implementation",
        "eager",
        "--distributed-timeout-minutes",
        "1",
        "--optimizer",
        "adam",
        "--lr",
        "1e-6",
        "--lr-decay-style",
        "constant",
        "--weight-decay",
        "0.0",
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        "4",
        "--colocate",
    ]
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": f"{args.miles_checkout}:{ROOT}",
            "PYTHONUNBUFFERED": "1",
            "RAY_TMPDIR": "/job/cache/ray",
            "RAY_DEDUP_LOGS": "0",
            "HF_HOME": "/job/cache/huggingface-home",
            "TRITON_CACHE_DIR": "/job/cache/triton",
            "TORCHINDUCTOR_CACHE_DIR": "/job/cache/inductor",
            "AITER_CONFIG_GEMM_BF16": "/job/cache/aiter/bf16_tuned_gemm.csv",
        }
    )
    return subprocess.call(command, env=env, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
