from argparse import Namespace
from collections.abc import Callable

import torch

from miles.backends.training_utils.loss_hub.logit_processors import _iter_response_chunks
from miles.backends.training_utils.loss_hub.math_utils import calculate_log_probs_and_entropy
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.types import RolloutBatch


def apply_opd_kl_to_advantages(
    args: Namespace,
    rollout_data: RolloutBatch,
    advantages: list[torch.Tensor],
    student_log_probs: list[torch.Tensor] | None,
) -> None:
    """Apply on-policy distillation KL penalty to advantages.

    Computes reverse KL (student_logp - teacher_logp) and adds weighted penalty
    to advantages in-place. This is orthogonal to the base advantage estimator.

    Args:
        args: Configuration containing `use_opd` and `opd_kl_coef`.
        rollout_data: Dict containing "teacher_log_probs".
        advantages: List of advantage tensors to modify in-place.
        student_log_probs: List of old-student log-probability tensors. OPD
            treats these as fixed scoring inputs.

    References:
        https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/train_on_policy.py
    """

    if student_log_probs is None:
        return

    precomputed_reverse_kls = rollout_data.get("opd_reverse_kl")
    if precomputed_reverse_kls is not None:
        if len(advantages) != len(precomputed_reverse_kls):
            raise ValueError(
                f"OPD length mismatch: advantages={len(advantages)}, "
                f"opd_reverse_kl={len(precomputed_reverse_kls)}."
            )

        reverse_kls = []
        for i, adv in enumerate(advantages):
            reverse_kl = precomputed_reverse_kls[i]
            if not torch.is_tensor(reverse_kl):
                reverse_kl = torch.tensor(reverse_kl, dtype=torch.float32)
            # Defensive consumer boundary for direct callers that bypass
            # compute_advantages_and_returns' persistent-data detach.
            reverse_kl = reverse_kl.detach().to(device=adv.device)
            if adv.shape != reverse_kl.shape:
                raise ValueError(
                    f"OPD shape mismatch at sample {i}: advantages={tuple(adv.shape)}, "
                    f"opd_reverse_kl={tuple(reverse_kl.shape)}."
                )
            advantages[i] = adv - args.opd_kl_coef * reverse_kl
            reverse_kls.append(reverse_kl)

        rollout_data["opd_reverse_kl"] = reverse_kls
        return

    teacher_log_probs = rollout_data.get("teacher_log_probs")
    if teacher_log_probs is None:
        raise ValueError(f"OPD with opd_type='{args.opd_type}' requires teacher_log_probs, but it is missing.")

    if not (len(advantages) == len(student_log_probs) == len(teacher_log_probs)):
        raise ValueError(
            f"OPD length mismatch: advantages={len(advantages)}, "
            f"student_log_probs={len(student_log_probs)}, teacher_log_probs={len(teacher_log_probs)}."
        )

    device = student_log_probs[0].device
    detached_teacher_log_probs = [t.detach() for t in teacher_log_probs]
    rollout_data["teacher_log_probs"] = detached_teacher_log_probs
    teacher_log_probs = [t.to(device=device) for t in detached_teacher_log_probs]

    reverse_kls = []
    for i, adv in enumerate(advantages):
        if student_log_probs[i].shape != teacher_log_probs[i].shape:
            raise ValueError(
                f"OPD shape mismatch at sample {i}: student_log_probs={tuple(student_log_probs[i].shape)}, "
                f"teacher_log_probs={tuple(teacher_log_probs[i].shape)}."
            )
        if adv.shape != student_log_probs[i].shape:
            raise ValueError(
                f"OPD shape mismatch at sample {i}: advantages={tuple(adv.shape)}, "
                f"student_log_probs={tuple(student_log_probs[i].shape)}. "
                "OPD expects per-token advantages; broadcast scalar advantages must be expanded before this call."
            )
        old_student_log_prob = student_log_probs[i].detach()
        reverse_kl = old_student_log_prob - teacher_log_probs[i]
        advantages[i] = adv - args.opd_kl_coef * reverse_kl
        reverse_kls.append(reverse_kl)

    # Store reverse KL for logging.
    rollout_data["opd_reverse_kl"] = reverse_kls


def compute_differentiable_topk_reverse_kl(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
    local_loss_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a differentiable top-k reverse-KL loss from stored teacher terms.

    The rollout stores the controlled token set, fixed teacher log-probabilities,
    and fixed reward weights.  This function re-scores those tokens with the
    current policy logits, so the reverse-KL term is differentiable with respect
    to the student while the teacher remains detached training data.
    """

    required_keys = (
        "opd_topk_counts",
        "opd_topk_token_ids",
        "opd_topk_teacher_log_probs",
        "opd_topk_weights",
    )
    missing_keys = [key for key in required_keys if batch.get(key) is None]
    if missing_keys:
        raise ValueError(f"differentiable top-k OPD requires {', '.join(missing_keys)}.")

    for sample_index, response_length in enumerate(batch["response_lengths"]):
        counts = batch["opd_topk_counts"][sample_index]
        if len(counts) != response_length:
            raise ValueError(
                f"differentiable top-k OPD count mismatch at sample {sample_index}: "
                f"counts={len(counts)}, response_length={response_length}."
            )
        selected_count = sum(int(count) for count in counts)
        for key in ("opd_topk_token_ids", "opd_topk_teacher_log_probs", "opd_topk_weights"):
            if len(batch[key][sample_index]) != selected_count:
                raise ValueError(
                    f"differentiable top-k OPD length mismatch at sample {sample_index}, {key}: "
                    f"expected={selected_count}, got={len(batch[key][sample_index])}."
                )

    parallel_state = get_parallel_state()
    reverse_kls: list[torch.Tensor] = []
    response_chunks = _iter_response_chunks(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens", None),
        include_response_indices=True,
    )

    for sample_index, (logits_chunk, _, response_indices) in enumerate(response_chunks):
        response_indices = list(response_indices)
        counts = [int(batch["opd_topk_counts"][sample_index][index]) for index in response_indices]
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        start, end = offsets[0], offsets[-1]
        token_ids = batch["opd_topk_token_ids"][sample_index][start:end]
        teacher_log_probs = batch["opd_topk_teacher_log_probs"][sample_index][start:end]
        weights = batch["opd_topk_weights"][sample_index][start:end]

        if not token_ids:
            empty_reverse_kl = logits_chunk.new_zeros(len(counts), dtype=torch.float32)
            reverse_kls.append(empty_reverse_kl + logits_chunk.sum(dtype=torch.float32) * 0)
            continue

        position_ids = torch.repeat_interleave(
            torch.arange(len(counts), device=logits_chunk.device),
            torch.tensor(counts, device=logits_chunk.device),
        )
        selected_logits = logits_chunk.index_select(0, position_ids)
        selected_token_ids = torch.tensor(token_ids, device=logits_chunk.device, dtype=torch.long)
        current_log_probs, _ = calculate_log_probs_and_entropy(
            selected_logits,
            selected_token_ids,
            parallel_state.tp.group,
            true_on_policy=args.true_on_policy_mode,
            vocab_size=getattr(args, "vocab_size", None),
            temperature=1.0 if args.true_on_policy_mode else args.rollout_temperature,
        )
        current_log_probs = current_log_probs.squeeze(-1)
        teacher = torch.tensor(teacher_log_probs, device=logits_chunk.device, dtype=torch.float32)
        reward_weights = torch.tensor(weights, device=logits_chunk.device, dtype=torch.float32)
        selected_reverse_kl = reward_weights * (current_log_probs - teacher)

        sample_reverse_kl = logits_chunk.new_zeros(len(counts), dtype=torch.float32)
        sample_reverse_kl.index_add_(0, position_ids, selected_reverse_kl.to(torch.float32))
        reverse_kls.append(sample_reverse_kl)

    reverse_kl = torch.cat(reverse_kls, dim=0)
    local_loss_mask = torch.cat(local_loss_masks, dim=0).to(device=reverse_kl.device)
    reverse_kl = torch.where(
        local_loss_mask.bool(),
        torch.nan_to_num(reverse_kl, nan=0.0, posinf=0.0, neginf=0.0),
        reverse_kl.new_zeros(()),
    )
    return args.opd_kl_coef * sum_of_sample_mean(reverse_kl), reverse_kl
