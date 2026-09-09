import json
import re
from pathlib import Path


LOGS = Path("/job/logs")
OUTPUT = Path("/job/artifacts/run_results.json")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
RANK = re.compile(r"actor_cell0_rank([0-3])")
LOG_GATHER = re.compile(r"phase=end kind=log_gather rank=([0-3]) success=true")
STEP = re.compile(
    r"step ([01]): \{'train/loss': ([^,]+), 'train/grad_norm': ([^,]+)"
)
MESH = re.compile(r"\[Rank ([0-3])\] FSDP mesh shape=\(4,\)")


def split_ranks(text):
    files = {}
    for rank in range(4):
        lines = []
        for line in text.splitlines():
            clean = ANSI.sub("", line)
            if RANK.search(clean) or f"rank={rank} " in clean:
                lines.append(clean + "\n")
        files[f"rank{rank}"] = "".join(lines)
    return files


def summarize(run):
    path = LOGS / run / "combined.log"
    text = ANSI.sub("", path.read_text(errors="replace"))
    rank_files = split_ranks(text)
    for rank, content in rank_files.items():
        output = LOGS / run / f"{rank}.log"
        output.write_text(content)
    steps = {}
    for match in STEP.finditer(text):
        steps[int(match.group(1))] = {
            "loss": float(match.group(2)),
            "grad_norm": float(match.group(3)),
        }
    return {
        "exit_status": int(re.search(r"EXIT_STATUS=([0-9]+)", text).group(1)),
        "fsdp_mesh_ranks": sorted({int(value) for value in MESH.findall(text)}),
        "log_gather_success_counts": {
            str(rank): len([m for m in LOG_GATHER.finditer(text) if int(m.group(1)) == rank])
            for rank in range(4)
        },
        "optimizer_steps": steps,
        "rank_log_line_counts": {
            rank: len(content.splitlines()) for rank, content in rank_files.items()
        },
    }


def main():
    runs = {
        "thd_mbs2_pinned": "thd_mbs2",
        "thd_mbs4_pinned": "thd_mbs4",
        "bshd_mbs2_pinned_rejected": "bshd_mbs2_pinned",
        "bshd_mbs2_candidate": "bshd_mbs2_candidate",
        "bshd_mbs4_candidate": "bshd_mbs4_candidate",
    }
    result = {name: summarize(path) for name, path in runs.items()}
    result["gradient_check"] = json.loads(Path("/job/artifacts/gradient_check.json").read_text())
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
