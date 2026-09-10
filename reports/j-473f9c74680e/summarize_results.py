#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(Path(__file__).with_name("results_batch_invariant.json")))
    parser.add_argument("--output", default=str(Path(__file__).with_name("summary.json")))
    return parser.parse_args()


def metric(left_values, right_values):
    differences = [
        left - right
        for left, right in zip(left_values, right_values, strict=True)
    ]
    absolute = [abs(value) for value in differences]
    return {
        "count": len(differences),
        "signed_mean": sum(differences) / len(differences),
        "absolute_mean": sum(absolute) / len(absolute),
        "absolute_max": max(absolute),
        "rmse": math.sqrt(sum(value * value for value in differences) / len(differences)),
        "bitwise_equal_count": sum(value == 0.0 for value in differences),
    }


def aggregate(pairs):
    return metric(
        [value for pair in pairs for value in pair[0]],
        [value for pair in pairs for value in pair[1]],
    )


def selected_pairs(left, right, batch_size):
    return [
        (left[str(batch_size)][str(index)], right[str(batch_size)][str(index)])
        for index in range(batch_size)
    ]


def main():
    args = parse_args()
    with open(args.results) as handle:
        results = json.load(handle)

    batch_sizes = results["settings"]["batch_sizes"]
    per_batch = {}
    all_rollout = []
    all_training = {mode: [] for mode in results["training"]}
    all_packed = []
    all_triton = []

    for batch_size in batch_sizes:
        rollout_pairs = selected_pairs(results["rollout"], results["rollout"], batch_size)
        all_rollout.extend(rollout_pairs)
        batch_summary = {
            "rollout_batch_invariance": aggregate(rollout_pairs),
            "training": {},
            "packed_vs_unpacked": aggregate(
                selected_pairs(
                    results["training"]["flash_packed"],
                    results["training"]["flash_unpacked"],
                    batch_size,
                )
            ),
            "triton_vs_flash_unpacked": aggregate(
                selected_pairs(
                    results["training"]["triton_unpacked"],
                    results["training"]["flash_unpacked"],
                    batch_size,
                )
            ),
        }
        for mode, mode_results in results["training"].items():
            invariance_pairs = selected_pairs(mode_results, mode_results, batch_size)
            rollout_pairs = selected_pairs(mode_results, results["rollout"], batch_size)
            all_training[mode].extend(rollout_pairs)
            batch_summary["training"][mode] = {
                "batch_invariance": aggregate(invariance_pairs),
                "rollout_difference": aggregate(rollout_pairs),
            }
        all_packed.extend(
            selected_pairs(
                results["training"]["flash_packed"],
                results["training"]["flash_unpacked"],
                batch_size,
            )
        )
        all_triton.extend(
            selected_pairs(
                results["training"]["triton_unpacked"],
                results["training"]["flash_unpacked"],
                batch_size,
            )
        )
        per_batch[str(batch_size)] = batch_summary

    summary = {
        "schema_version": 1,
        "source_results": str(Path(args.results).resolve()),
        "hardware": results["hardware"],
        "software": results["software"],
        "model": results["model"],
        "settings": results["settings"],
        "per_batch": per_batch,
        "all_sweep_observations": {
            "rollout_batch_invariance": aggregate(all_rollout),
            "training_vs_rollout": {
                mode: aggregate(pairs) for mode, pairs in all_training.items()
            },
            "packed_vs_unpacked": aggregate(all_packed),
            "triton_vs_flash_unpacked": aggregate(all_triton),
        },
    }

    output_path = Path(args.output)
    with open(output_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary["all_sweep_observations"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
