from argparse import Namespace

import torch

from miles.backends.training_utils.loss_hub.logit_processors import get_opd_topk_log_probs
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

    if getattr(args, "opd_log_prob_top_k", 0) > 0:
        return

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


def compute_opd_topk_reverse_kl(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
) -> torch.Tensor:
    """Compute differentiable subset reverse KL from current training logits.

    Both distributions are renormalized over the selected top-k token set, as in
    the Rethinking OPD subset approximation. The stored rollout weights are
    used only as a padding/validity mask; freezing them into training would
    create a snapshot-alignment objective rather than teacher distillation.
    """
    required_fields = (
        "opd_topk_token_ids",
        "opd_topk_teacher_log_probs",
        "opd_topk_weights",
    )
    missing_fields = [field for field in required_fields if batch.get(field) is None]
    if missing_fields:
        raise ValueError(f"Top-k OPD requires {', '.join(missing_fields)} for a differentiable loss.")

    token_ids = batch["opd_topk_token_ids"]
    current_log_probs = get_opd_topk_log_probs(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        token_ids=token_ids,
        max_seq_lens=batch.get("max_seq_lens"),
    )
    teacher_log_probs = batch["opd_topk_teacher_log_probs"]
    weights = batch["opd_topk_weights"]
    reverse_kls = []

    for sample_index, current in enumerate(current_log_probs):
        sample_token_ids = token_ids[sample_index].to(device=current.device, dtype=torch.long)
        sample_teacher = teacher_log_probs[sample_index].detach().to(device=current.device, dtype=torch.float32)
        sample_weights = weights[sample_index].detach().to(device=current.device, dtype=torch.float32)
        if current.shape != sample_teacher.shape or current.shape != sample_weights.shape:
            raise ValueError(
                f"Top-k OPD shape mismatch at sample {sample_index}: "
                f"current={tuple(current.shape)}, teacher={tuple(sample_teacher.shape)}, "
                f"weights={tuple(sample_weights.shape)}."
            )
        if sample_token_ids.shape != current.shape:
            raise ValueError(
                f"Top-k OPD token-id shape mismatch at sample {sample_index}: "
                f"token_ids={tuple(sample_token_ids.shape)}, current={tuple(current.shape)}."
            )
        valid = sample_weights > 0
        if not valid.any():
            reverse_kls.append(current.new_zeros(current.size(0)))
            continue
        current_selected = torch.where(valid, current, current.new_full((), -float("inf")))
        teacher_selected = torch.where(valid, sample_teacher, sample_teacher.new_full((), -float("inf")))
        current_log_probs = torch.log_softmax(current_selected, dim=1)
        teacher_log_probs = torch.log_softmax(teacher_selected, dim=1)
        reverse_kls.append(
            (current_log_probs.exp() * (current_log_probs - teacher_log_probs)).sum(dim=1)
        )

    return torch.cat(reverse_kls, dim=0)
