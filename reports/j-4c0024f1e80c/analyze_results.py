from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np


def r3_summary(path: Path) -> dict:
    payload = json.loads(path.read_text())
    summary = {
        "cycles": payload["cycles"],
        "world_size": payload["world_size"],
        "learning_rate": payload["learning_rate"],
        "compute_dtype": payload["compute_dtype"],
        "parameter_dtype": payload["parameter_dtype"],
        "r3_streams": payload["r3_streams"],
        "phase_seconds": payload["phase_seconds"],
        "total_seconds": payload["total_seconds"],
        "routing_equal_all": all(record["routing"]["equal"] for record in payload["records"]),
        "routing_max_abs": max(record["routing"]["max_abs"] for record in payload["records"]),
    }
    for case in ("r3_on", "r3_off"):
        errors = np.concatenate([record[case]["per_token_abs_error"] for record in payload["records"]])
        gradients = np.array([record[case]["gradient_vs_reference"]["max_abs"] for record in payload["records"]])
        drift = np.array([record[case]["parameter_drift_max"] for record in payload["records"]])
        summary[case] = {
            "token_count": int(errors.size),
            "per_token_abs_error_mean": float(errors.mean()),
            "per_token_abs_error_max": float(errors.max()),
            "per_token_abs_error_p95": float(np.quantile(errors, 0.95)),
            "per_token_abs_error_p99": float(np.quantile(errors, 0.99)),
            "gradient_vs_reference_max_abs": float(gradients.max()),
            "gradient_vs_reference_mean_of_cycle_maxima": float(gradients.mean()),
            "parameter_drift_max": float(drift.max()),
            "all_finite": bool(
                np.isfinite(errors).all() and np.isfinite(gradients).all() and np.isfinite(drift).all()
            ),
        }
    on_off_gradient = np.array([record["r3_on_vs_off"]["gradient"]["max_abs"] for record in payload["records"]])
    on_off_logprob = np.array([record["r3_on_vs_off"]["logprob"]["max_abs"] for record in payload["records"]])
    summary["r3_on_vs_off"] = {
        "gradient_max_abs": float(on_off_gradient.max()),
        "logprob_max_abs": float(on_off_logprob.max()),
        "all_zero": bool((on_off_gradient == 0).all() and (on_off_logprob == 0).all()),
    }
    return summary


def decode_routing(record) -> np.ndarray:
    encoded = record["result"][0]["meta_info"]["routed_experts"]
    flat = np.frombuffer(base64.b64decode(encoded), dtype=np.int32)
    return flat.reshape(-1, 2, 2)


def spec_summary(off_path: Path, on_path: Path) -> dict:
    off = json.loads(off_path.read_text())
    on = json.loads(on_path.read_text())
    assert off["cycles"] == on["cycles"]
    token_mismatches = 0
    exact_routing_cycles = 0
    first16_exact_cycles = 0
    routing_max = 0
    mismatched_rows = []
    for off_record, on_record in zip(off["records"], on["records"], strict=True):
        off_tokens = off_record["result"][0]["output_ids"]
        on_tokens = on_record["result"][0]["output_ids"]
        token_mismatches += sum(left != right for left, right in zip(off_tokens, on_tokens))
        token_mismatches += abs(len(off_tokens) - len(on_tokens))
        off_routing = decode_routing(off_record)
        on_routing = decode_routing(on_record)
        routing_max = max(routing_max, int(np.abs(off_routing - on_routing).max()))
        exact_routing_cycles += int(np.array_equal(off_routing, on_routing))
        first16_exact_cycles += int(np.array_equal(off_routing[:16], on_routing[:16]))
        mismatched_rows.append(int(np.any(off_routing != on_routing, axis=(1, 2)).sum()))
    return {
        "cycles": off["cycles"],
        "token_mismatches": token_mismatches,
        "routing_exact_cycles": exact_routing_cycles,
        "routing_max_abs": routing_max,
        "routing_mismatched_rows_total": sum(mismatched_rows),
        "routing_mismatched_rows_min": min(mismatched_rows),
        "routing_mismatched_rows_max": max(mismatched_rows),
        "first_16_routing_rows_exact_cycles": first16_exact_cycles,
        "spec_off_generate_seconds": off["generate_seconds"],
        "spec_on_generate_seconds": on["generate_seconds"],
        "spec_off_startup_seconds": off["startup_seconds"],
        "spec_on_startup_seconds": on["startup_seconds"],
        "spec_verify_ct": sum(
            record["result"][0]["meta_info"].get("spec_verify_ct", 0) for record in on["records"]
        ),
        "spec_accepted_drafts": sum(
            record["result"][0]["meta_info"].get("spec_accepted_drafts", 0) for record in on["records"]
        ),
        "spec_proposed_drafts": sum(
            record["result"][0]["meta_info"].get("spec_proposed_drafts", 0) for record in on["records"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r3", type=Path, required=True)
    parser.add_argument("--spec-off", type=Path, required=True)
    parser.add_argument("--spec-on", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = {
        "r3": r3_summary(args.r3),
        "speculative": spec_summary(args.spec_off, args.spec_on),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
