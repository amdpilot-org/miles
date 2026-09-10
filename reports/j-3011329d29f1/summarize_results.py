#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def timing_stats(values: list[float]) -> dict[str, float]:
    return {
        "minimum_ms": min(values),
        "mean_ms": statistics.fmean(values),
        "maximum_ms": max(values),
    }


def rank_summary(payload: dict[str, Any]) -> dict[str, Any]:
    cycles = payload["cycle_results"]
    gradient_errors = [
        cycle["gradient_comparison"]["maximum_absolute_error"] for cycle in cycles
    ]
    weight_errors = [
        cycle["weight_comparison"]["maximum_absolute_error"]
        for cycle in cycles
        if cycle["weight_comparison"] is not None
    ]
    return {
        "rank": payload["rank"],
        "device_name": payload["device_name"],
        "device_capability": payload["device_capability"],
        "torch_version": payload["torch_version"],
        "hip_version": payload["hip_version"],
        "nccl_version": payload["nccl_version"],
        "commit": payload["commit"],
        "source_paths": payload["source_paths"],
        "config": payload["config"],
        "cycle_count": len(cycles),
        "event_count": len(payload["events"]),
        "gradient_maximum_absolute_error": max(gradient_errors),
        "gradient_maximum_relative_error": max(
            cycle["gradient_comparison"]["maximum_relative_error"] for cycle in cycles
        ),
        "weight_comparison_count": len(weight_errors),
        "weight_maximum_absolute_error": max(weight_errors),
        "weight_maximum_relative_error": max(
            cycle["weight_comparison"]["maximum_relative_error"]
            for cycle in cycles
            if cycle["weight_comparison"] is not None
        ),
        "forward_timing": timing_stats([cycle["forward_ms"] for cycle in cycles]),
        "backward_timing": timing_stats([cycle["backward_ms"] for cycle in cycles]),
        "reference_timing": timing_stats([cycle["reference_ms"] for cycle in cycles]),
        "optimizer_timing": timing_stats([cycle["optimizer_ms"] for cycle in cycles]),
        "first_cycle": cycles[0],
        "last_cycle": cycles[-1],
    }


def collective_order(events: list[dict[str, Any]]) -> list[str]:
    return [event["module"] for event in events if event["operation"] == "all_gather_into_tensor"]



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-dir", type=Path, required=True)
    parser.add_argument("--desync-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank_zero = load_json(args.sync_dir / "rank_0.json")
    rank_one = load_json(args.sync_dir / "rank_1.json")
    combined = load_json(args.sync_dir / "combined.json")
    desync_zero_events = load_jsonl(args.desync_dir / "rank_0_events.jsonl")
    desync_one_events = load_jsonl(args.desync_dir / "rank_1_events.jsonl")
    summary = {
        "synchronized": {
            "rank_0": rank_summary(rank_zero),
            "rank_1": rank_summary(rank_one),
            "collective_order_comparison": combined["collective_order_comparison"],
        },
        "desynchronized": {
            "rank_0_all_gather_order": collective_order(desync_zero_events),
            "rank_1_all_gather_order": collective_order(desync_one_events),
            "rank_0_event_count": len(desync_zero_events),
            "rank_1_event_count": len(desync_one_events),
            "bounded_timeout_seconds": rank_zero["config"]["timeout_seconds"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
