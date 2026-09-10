#!/usr/bin/env python3
"""Validate and summarize the two-GPU Miles lifecycle evidence."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def stats(values: list[float]) -> dict:
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
    }


def baseline_metrics(summary: dict) -> dict:
    return {
        "cycles": len(summary["cycles"]),
        "completed_requests": summary["completed_requests"],
        "failed_requests": summary["failed_requests"],
        "router_admission_cycles": summary["router_admission_cycles"],
        "release_positive_cycles": summary["release_positive_cycles"],
        "version_consistent_cycles": summary["version_consistent_cycles"],
        "checksum_mismatch_cycles": summary["checksum_mismatch_cycles"],
        "all_engine_tensors_checked_cycles": summary["all_engine_tensors_checked_cycles"],
        "expected_engine_checksum_match_cycles": summary["expected_engine_checksum_match_cycles"],
        "output_equal_cycles": summary["output_equal_cycles"],
        "real_weight_update_events": summary["real_weight_update_events"],
        "successful_updates": summary["successful_updates"],
        "unique_train_losses": summary["unique_train_losses"],
    }


def investigation_metrics(summary: dict) -> dict:
    cycles = summary["cycles"]
    release_bytes = [cycle["release"]["released_bytes"] for cycle in cycles]
    output_differences = [
        cycle["output"][engine]["max_abs_log_prob_difference"]
        for cycle in cycles
        for engine in ("a", "b")
    ]
    return {
        **baseline_metrics(summary),
        "cycle_seconds": stats([cycle["cycle_seconds"] for cycle in cycles]),
        "release_bytes": stats(release_bytes),
        "output_max_abs_log_prob_difference": stats(output_differences),
        "failed_wake_cycle": summary["failed_wake_cycle"],
        "wake_recovery_successful": summary["wake_recovery_successful"],
        "failed_update_cycle": summary["failed_update_cycle"],
        "update_recovery_successful": summary["update_recovery_successful"],
        "mixed_weight_version_requests": summary["mixed_weight_version_requests"],
        "timeout_causes": summary["timeout_causes"],
    }


def recovery_details(summary: dict, environment: dict) -> dict:
    failed_wake_cycle = summary["failed_wake_cycle"]
    failed_update_cycle = summary["failed_update_cycle"]
    wake = summary["cycles"][failed_wake_cycle - 1]["wake"]
    next_cycle = summary["cycles"][failed_wake_cycle]
    update_recovery_cycle = summary["cycles"][failed_update_cycle]
    return {
        "failed_wake": {
            "cycle": failed_wake_cycle,
            "failure_type": wake["failure_type"],
            "failure_message": wake["failure_message"],
            "recovery_action": wake["recovery_action"],
            "recovery_attempts": wake["recovery_attempts"],
            "recovery_seconds": wake["recovery_seconds"],
            "recovery_successful": wake["recovery_successful"],
            "update_group_reconnect_seconds": wake["recovery_reconnect_seconds"],
        },
        "wake_next_cycle": {
            "cycle": next_cycle["cycle"],
            "wake_failed": next_cycle["wake"]["failed"],
            "router_admission": next_cycle["router_admission"],
            "version_consistent": next_cycle["version_consistent"],
        },
        "failed_update": {
            "cycle": failed_update_cycle,
            "engine_b_failed": summary["cycles"][failed_update_cycle - 1]["update"]["b"]["failed"],
            "failure_type": summary["cycles"][failed_update_cycle - 1]["update"]["b"]["failure_type"],
            "failure_message": summary["cycles"][failed_update_cycle - 1]["update"]["b"]["failure_message"],
        },
        "update_recovery_cycle": {
            "cycle": update_recovery_cycle["cycle"],
            "engine_b_update_failed": update_recovery_cycle["update"]["b"]["failed"],
            "version_consistent": update_recovery_cycle["version_consistent"],
        },
        "launcher_bounds": environment["wake_recovery"],
    }


def build_summary(baseline: dict, investigation: dict, environment: dict) -> dict:
    cycle_count = len(investigation["cycles"])
    failed_wake_cycle = investigation["failed_wake_cycle"]
    failed_update_cycle = investigation["failed_update_cycle"]
    wake = investigation["cycles"][failed_wake_cycle - 1]["wake"]
    next_wake = investigation["cycles"][failed_wake_cycle]["wake"]
    next_update = investigation["cycles"][failed_update_cycle]["update"]["b"]
    gpu_names = [gpu["name"] for gpu in environment["gpus"]]
    gpu_capabilities = [tuple(gpu["capability"]) for gpu in environment["gpus"]]

    checks = {
        "at_least_32_cycles": cycle_count >= 32,
        "all_requests_completed": investigation["failed_requests"] == 0,
        "router_admission_every_cycle": investigation["router_admission_cycles"] == cycle_count,
        "release_positive_every_cycle": investigation["release_positive_cycles"] == cycle_count,
        "version_consistent_every_cycle": investigation["version_consistent_cycles"] == cycle_count,
        "no_checksum_mismatch": investigation["checksum_mismatch_cycles"] == 0,
        "all_engine_tensors_checked": investigation["all_engine_tensors_checked_cycles"] == 2 * cycle_count,
        "expected_engine_checksum_match": investigation["expected_engine_checksum_match_cycles"] == 2 * cycle_count,
        "output_tokens_equal_both_engines": investigation["output_equal_cycles"] == 2 * cycle_count,
        "real_weight_update_events": investigation["real_weight_update_events"] == 2 * cycle_count,
        "one_injected_update_failure": investigation["failed_updates"] == 1,
        "failed_wake_recorded": failed_wake_cycle is not None,
        "failed_update_recorded": failed_update_cycle is not None,
        "wake_recovery_successful": investigation["wake_recovery_successful"] is True,
        "update_recovery_successful": investigation["update_recovery_successful"] is True,
        "next_wake_not_failed": next_wake["failed"] is False,
        "next_update_not_failed": next_update["failed"] is False,
        "no_mixed_weight_version_requests": investigation["mixed_weight_version_requests"] == 0,
        "no_request_failures": len(investigation["request_failures"]) == 0,
        "no_timeout_causes": len(investigation["timeout_causes"]) == 0,
        "unique_training_losses": investigation["unique_train_losses"] == cycle_count,
        "unique_actor_checksums": investigation["unique_actor_overall_checksums"] == cycle_count,
        "unique_engine_a_checksums": investigation["unique_engine_a_overall_checksums"] == cycle_count,
        "unique_engine_b_checksums": investigation["unique_engine_b_overall_checksums"] == cycle_count,
        "two_assigned_mi350x_gpus": len(gpu_names) == 2 and all(name == "AMD Instinct MI350X" for name in gpu_names),
        "gfx950_capability": all(capability == (9, 5) for capability in gpu_capabilities),
        "downloads_under_8gb": environment["downloaded_bytes"] < 8 * 1024**3,
        "single_bounded_wake_restart": environment["wake_recovery"]["restart_count"] == 1
        and environment["wake_recovery"]["restart_limit"] == 1,
        "wake_recovery_bounded": wake["recovery_seconds"] < environment["wake_recovery"]["health_timeout_seconds"],
    }

    return {
        "requirements_pass": all(checks.values()),
        "checks": checks,
        "baseline": baseline_metrics(baseline),
        "investigation": investigation_metrics(investigation),
        "recovery": recovery_details(investigation, environment),
        "gpu_phase_timings": investigation["gpu_phase_timings"],
        "environment": environment,
        "scope": {
            "selective_lifecycle_operations_supported": True,
            "elastic_scheduler_claimed": False,
            "node_wide_scheduling_changed": False,
            "mi355x_performance_equivalence_claimed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--investigation", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = build_summary(
        load_json(args.baseline),
        load_json(args.investigation),
        load_json(args.environment),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"requirements_pass": result["requirements_pass"], "output": str(args.output)}, indent=2))
    if not result["requirements_pass"]:
        failed_checks = [name for name, passed in result["checks"].items() if not passed]
        raise SystemExit("failed checks: " + ", ".join(failed_checks))


if __name__ == "__main__":
    main()
