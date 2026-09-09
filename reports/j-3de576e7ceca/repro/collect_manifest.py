import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import torch


OUTPUT = Path("/job/artifacts/environment_manifest.json")
MODEL = Path("/job/cache/huggingface/Qwen3-VL-4B-Instruct")


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def git_state(path):
    return {
        "path": str(path),
        "head": command("git", "-C", str(path), "rev-parse", "HEAD"),
        "branch": command("git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"),
        "status": command("git", "-C", str(path), "status", "--short"),
        "dirty": bool(command("git", "-C", str(path), "status", "--short")),
    }


def checkpoint_hash():
    digest = hashlib.sha256()
    for path in sorted(item for item in MODEL.rglob("*") if item.is_file()):
        digest.update(path.relative_to(MODEL).as_posix().encode())
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024 * 16):
                digest.update(chunk)
    return digest.hexdigest()


def main():
    manifest = {
        "job_id": "j-3de576e7ceca",
        "original_pr": "https://github.com/amdpilot-org/miles/pull/52",
        "related_issue": "https://github.com/radixark/miles/issues/2406",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "torch_cuda": torch.version.cuda,
        "gpu_architecture": torch.cuda.get_device_capability(0),
        "gpu_architecture_name": "gfx950",
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "visible_gpu_env": {key: os.environ.get(key) for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")},
        "gcc": command("gcc", "--version").splitlines()[0],
        "hipcc": command("hipcc", "--version").splitlines()[0],
        "rocm": command("cat", "/opt/rocm/.info/version"),
        "checkpoint": {
            "repository": "Qwen/Qwen3-VL-4B-Instruct",
            "revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
            "path": str(MODEL),
            "content_sha256": checkpoint_hash(),
        },
        "validation_control": git_state(Path("/job/miles-validation")),
        "validation_candidate": git_state(Path("/job/miles-validation-candidate")),
        "delivery_main": git_state(Path("/job/miles-delivery")),
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
