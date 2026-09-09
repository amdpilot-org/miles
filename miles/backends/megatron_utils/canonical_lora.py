"""Compatibility extensions for Megatron-Bridge CanonicalLoRA."""

from __future__ import annotations

import torch

from megatron.bridge.peft.canonical_lora import CanonicalLoRA as BridgeCanonicalLoRA
from megatron.bridge.peft.canonical_lora import LoRALinearSplitQKV


class GatedLoRALinearSplitQKV(LoRALinearSplitQKV):
    """Split QKV adapter wrapper for attention-output-gated projections.

    Megatron packs gated attention as ``[Q, gate, K, V]`` within each query
    group.  The query adapter's output follows the HuggingFace ``q_proj``
    contract: each query head contributes one Q head followed by one gate head.
    """

    def _interleave_qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if not getattr(self.to_wrap.config, "attention_output_gate", False):
            return super()._interleave_qkv(query, key, value)

        config = self.to_wrap.config
        head_num = getattr(config, "num_attention_heads", None)
        head_size = getattr(config, "kv_channels", None)
        if head_size is None:
            hidden_size = getattr(config, "hidden_size", None)
            if head_num is None or hidden_size is None or hidden_size % head_num != 0:
                raise ValueError("Cannot infer head size for gated CanonicalLoRA.")
            head_size = hidden_size // head_num

        if query.size(-1) % (2 * head_size) != 0:
            raise ValueError("Gated query adapter output must contain Q and gate rows for whole heads.")
        if key.size(-1) % head_size != 0 or value.size(-1) % head_size != 0:
            raise ValueError("Key/value adapter outputs must contain whole heads.")
        if key.size(-1) == 0 or value.size(-1) != key.size(-1):
            raise ValueError("Gated CanonicalLoRA requires matching non-empty K/V adapter outputs.")

        local_head_num = query.size(-1) // (2 * head_size)
        local_query_groups = key.size(-1) // head_size
        if local_head_num % local_query_groups != 0:
            raise ValueError("Local query heads must be divisible by local query groups.")

        heads_per_group = local_head_num // local_query_groups
        leading_shape = query.shape[:-1]
        query_and_gate = query.reshape(-1, local_head_num, 2, head_size)
        query_only = query_and_gate[:, :, 0, :]
        gate = query_and_gate[:, :, 1, :]
        key = key.reshape(-1, local_query_groups, head_size)
        value = value.reshape(-1, local_query_groups, head_size)

        qgv_chunks = []
        for group_index in range(local_query_groups):
            start = group_index * heads_per_group
            stop = start + heads_per_group
            qgv_chunks.extend(
                [
                    query_only[:, start:stop, :],
                    gate[:, start:stop, :],
                    key[:, group_index : group_index + 1, :],
                    value[:, group_index : group_index + 1, :],
                ]
            )

        return torch.cat(qgv_chunks, dim=1).reshape(*leading_shape, -1)


class CanonicalLoRA(BridgeCanonicalLoRA):
    """CanonicalLoRA with attention-output-gate support.

    Megatron-Bridge currently allocates only Q rows for the query adapter.  For
    gated attention, HuggingFace ``q_proj`` contains both Q and gate rows, so
    the adapter must allocate twice as many rows and interleave them with K/V.
    """

    def transform(self, m: torch.nn.Module, name: str | None = None, prefix: str | None = None) -> torch.nn.Module:
        config = getattr(m, "config", None)
        if name != "linear_qkv" or not getattr(config, "attention_output_gate", False):
            return super().transform(m, name=name, prefix=prefix)
        match = self.match(m, name, prefix)
        if match is None:
            return super().transform(m, name=name, prefix=prefix)
        canonical_components = self.canonical_mapping[match[0]]
        if "linear_q" not in canonical_components:
            return super().transform(m, name=name, prefix=prefix)

        original_num_attention_heads = config.num_attention_heads
        config.num_attention_heads = 2 * original_num_attention_heads
        try:
            transformed = super().transform(m, name=name, prefix=prefix)
        finally:
            config.num_attention_heads = original_num_attention_heads

        if isinstance(transformed, LoRALinearSplitQKV):
            return GatedLoRALinearSplitQKV(transformed.to_wrap, transformed.adapter)
        return transformed
