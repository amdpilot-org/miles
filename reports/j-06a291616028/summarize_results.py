import argparse
import json
from pathlib import Path


def request_summary(cycle):
    requests = cycle["requests"]
    phases = {}
    version_spans = {}
    timeout_causes = []
    for result in requests:
        phases[result["phase"]] = phases.get(result["phase"], 0) + 1
        span_count = len({span.get("version") for span in result["weight_versions"]})
        version_spans[str(span_count)] = version_spans.get(str(span_count), 0) + 1
        if result["status"] == "failed" and (
            "timeout" in result["error_type"].lower()
            or "timeout" in result["error_message"].lower()
        ):
            timeout_causes.append(
                {
                    "request_id": result["request_id"],
                    "error_type": result["error_type"],
                    "error_message": result["error_message"],
                }
            )
    return {
        "admitted": sum(result["prompt_tokens"] is not None for result in requests),
        "completed": sum(result["status"] == "completed" for result in requests),
        "failed": sum(result["status"] == "failed" for result in requests),
        "phase_counts": phases,
        "weight_version_span_counts": version_spans,
        "timeout_causes": timeout_causes,
    }


def cycle_summary(cycle):
    checksum = cycle["checksum"]
    return {
        "cycle": cycle["cycle"],
        "train": cycle["train"],
        "weight_norm": cycle["weight_norm"],
        "weight_norm_delta": cycle["weight_norm_delta"],
        "update": {
            "version": cycle["update"]["version"],
            "failed": cycle["update"]["failed"],
            "failure_type": cycle["update"]["failure_type"],
            "timings": cycle["update"]["timings"],
        },
        "engine_weight_version": cycle["engine_weight_version"],
        "version_consistent": cycle["version_consistent"],
        "checksum": {
            "actor_tensor_count": checksum["actor_tensor_count"],
            "engine_tensor_count": checksum["engine_tensor_count"],
            "common_tensor_count": checksum["common_tensor_count"],
            "actor_only_tensor_count": checksum["actor_only_tensor_count"],
            "fused_expected_tensor_count": checksum["fused_expected_tensor_count"],
            "unexpected_engine_tensor_count": checksum["unexpected_engine_tensor_count"],
            "mismatch_count": checksum["mismatch_count"],
            "all_engine_tensors_checked": checksum["all_engine_tensors_checked"],
            "expected_engine_overall_checksum": checksum["expected_engine_overall_checksum"],
            "engine_overall_checksum": checksum["engine_overall_checksum"],
        },
        "requests": request_summary(cycle),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    environment = json.loads(Path(args.environment).read_text())
    aggregate = {key: value for key, value in summary.items() if key != "cycles"}
    aggregate["unique_actor_overall_checksums"] = len(
        {cycle["checksum"]["actor_overall_checksum"] for cycle in summary["cycles"]}
    )
    aggregate["unique_engine_overall_checksums"] = len(
        {cycle["checksum"]["engine_overall_checksum"] for cycle in summary["cycles"]}
    )
    aggregate["unique_train_losses"] = len(
        {cycle["train"]["loss"] for cycle in summary["cycles"]}
    )
    output = {
        "environment": environment,
        "aggregate": aggregate,
        "cycles": [cycle_summary(cycle) for cycle in summary["cycles"]],
    }
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
