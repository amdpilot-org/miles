from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path

import torch

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss import compute_advantages_and_returns
from miles.backends.training_utils.loss_hub.losses import policy_loss_function
from miles.backends.training_utils.parallel import set_parallel_state
from miles.ray.rollout.train_data_conversion import (
    _compute_rollout_mask_sums,
    _post_process_rewards,
)
from miles.utils.types import Sample
from tests.fast.backends.training_utils.loss.loss_test_utils import GroupInfo, ParallelState, make_args


def _module_record(name: str) -> dict[str, object]:
    module = importlib.import_module(name)
    return {
        "name": name,
        "file": getattr(module, "__file__", None),
        "paths": list(getattr(module, "__path__", [])),
    }


def _single_process_parallel_state() -> None:
    group = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=group,
            intra_dp_cp=group,
            cp=group,
            tp=group,
            pp=group,
            ep=group,
            etp=group,
            indep_dp=group,
            is_pp_last_stage=True,
        )
    )


def _make_samples() -> list[Sample]:
    specifications = [
        (0, 10, 1.0),
        (1, 10, 1.0),
        (2, 11, 0.5),
        (3, 12, 0.25),
    ]
    return [
        Sample(
            group_index=0,
            index=index,
            rollout_id=rollout_id,
            reward=reward,
            tokens=[7, 0, 0],
            response_length=2,
            loss_mask=[1, 1],
            status=Sample.Status.COMPLETED,
        )
        for index, rollout_id, reward in specifications
    ]


def _run_fixture(expectation: str) -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("this fixture requires exactly one CUDA/ROCm GPU")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    _single_process_parallel_state()

    args = make_args(
        advantage_estimator="grpo",
        entropy_coef=0.0,
        kl_coef=0.0,
        observe_training_entropy=False,
        rewards_normalization=True,
        grpo_std_normalization=False,
        use_kl_loss=False,
        use_tis=False,
        get_mismatch_metrics=False,
        use_opsm=False,
        reward_key=None,
        n_samples_per_prompt=3,
        rollout_batch_size=1,
    )
    samples = _make_samples()
    raw_rewards, normalized_rewards = _post_process_rewards(
        args,
        samples,
        custom_reward_post_process_func=None,
        prompt_group_sizes=None,
    )

    rollout_ids = [sample.rollout_id for sample in samples]
    loss_masks = [sample.loss_mask for sample in samples]
    rollout_mask_sums = _compute_rollout_mask_sums(rollout_ids, loss_masks)
    rollout_data = {
        "log_probs": [torch.tensor([-math.log(2.0), -math.log(2.0)], device=device) for _ in samples],
        "rewards": normalized_rewards,
        "response_lengths": [sample.response_length for sample in samples],
        "loss_masks": [torch.tensor(mask, device=device) for mask in loss_masks],
        "total_lengths": [len(sample.tokens) for sample in samples],
        "max_seq_lens": None,
    }
    compute_advantages_and_returns(args, rollout_data)

    response_logits = torch.zeros((sum(rollout_data["response_lengths"]), 2), device=device, requires_grad=True)
    full_logits = torch.zeros((1, sum(rollout_data["total_lengths"]), 2), device=device)
    response_logit_positions = [0, 1, 3, 4, 6, 7, 9, 10]
    full_logits[0, response_logit_positions] = response_logits
    old_log_probs = [
        value.detach()
        for value in torch.log_softmax(response_logits, dim=-1)[:, 0].split(2, dim=0)
    ]
    batch = {
        "unconcat_tokens": [torch.tensor(sample.tokens, device=device) for sample in samples],
        "response_lengths": rollout_data["response_lengths"],
        "total_lengths": rollout_data["total_lengths"],
        "loss_masks": rollout_data["loss_masks"],
        "log_probs": old_log_probs,
        "advantages": rollout_data["advantages"],
        "rollout_mask_sums": torch.tensor(rollout_mask_sums, device=device),
    }
    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        None,
        denominators=batch["rollout_mask_sums"],
    )
    loss, _ = policy_loss_function(args, batch, full_logits, sum_of_sample_mean)
    loss.backward()

    sample_loss_weights = [
        mask_sum / rollout_mask_sum
        for mask_sum, rollout_mask_sum in zip(
            [sum(mask) for mask in loss_masks], rollout_mask_sums, strict=True
        )
    ]
    per_token_loss_weights = [
        mask_value / rollout_mask_sum
        for mask, rollout_mask_sum in zip(loss_masks, rollout_mask_sums, strict=True)
        for mask_value in mask
    ]
    advantages = [value.item() for advantage in rollout_data["advantages"] for value in advantage]
    gradient = response_logits.grad.detach().cpu().tolist()

    if expectation == "current":
        expected_rewards = [5 / 12, 5 / 12, -1 / 12, -1 / 3]
        expected_loss = 0.0
        expected_gradient = [
            [-5 / 96, 5 / 96],
            [-5 / 96, 5 / 96],
            [-5 / 96, 5 / 96],
            [-5 / 96, 5 / 96],
            [1 / 48, -1 / 48],
            [1 / 48, -1 / 48],
            [1 / 12, -1 / 12],
            [1 / 12, -1 / 12],
        ]
    elif expectation == "legacy":
        expected_rewards = [0.3125, 0.3125, -0.1875, -0.4375]
        expected_loss = 0.3125
        expected_gradient = [
            [-5 / 128, 5 / 128],
            [-5 / 128, 5 / 128],
            [-5 / 128, 5 / 128],
            [-5 / 128, 5 / 128],
            [3 / 64, -3 / 64],
            [3 / 64, -3 / 64],
            [7 / 64, -7 / 64],
            [7 / 64, -7 / 64],
        ]
    else:
        raise ValueError(f"unknown expectation: {expectation}")

    torch.testing.assert_close(normalized_rewards, expected_rewards, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(loss.item(), expected_loss, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(gradient, expected_gradient, rtol=0.0, atol=1e-6)

    return {
        "expectation": expectation,
        "gpu": {
            "device": str(device),
            "name": torch.cuda.get_device_name(device),
            "capability": torch.cuda.get_device_capability(device),
            "count": torch.cuda.device_count(),
        },
        "imports": {
            "miles": _module_record("miles"),
            "miles_reward_path": _module_record("miles.ray.rollout.train_data_conversion"),
            "miles_loss_path": _module_record("miles.backends.training_utils.loss"),
            "miles_policy_loss_path": _module_record("miles.backends.training_utils.loss_hub.losses"),
            "sglang": _module_record("sglang"),
            "megatron": _module_record("megatron"),
            "megatron_core": _module_record("megatron.core"),
            "torch": _module_record("torch"),
            "torch_native": torch._C.__file__,
        },
        "torch": {
            "version": torch.__version__,
            "hip": torch.version.hip,
        },
        "logical_rollouts": {
            "ids": [10, 11, 12],
            "rewards": [1.0, 0.5, 0.25],
            "sibling_counts": [2, 1, 1],
        },
        "raw_rewards": raw_rewards,
        "normalized_rewards": normalized_rewards,
        "rollout_ids": rollout_ids,
        "rollout_mask_sums": rollout_mask_sums,
        "sample_loss_weights": sample_loss_weights,
        "per_token_loss_weights": per_token_loss_weights,
        "advantages": advantages,
        "loss": loss.item(),
        "response_logits_gradient": gradient,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expectation", choices=("current", "legacy"), required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = _run_fixture(arguments.expectation)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is None:
        print(rendered)
    else:
        arguments.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
