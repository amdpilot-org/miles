"""Compatibility adapter for gated fused attention projections."""

from typing import Any, Optional, Tuple

import torch
from torch import nn

from megatron.bridge.peft.canonical_lora import (
    CanonicalLoRA,
    LoRALinearSplitQKV,
    ModuleDict,
)
from megatron.bridge.peft.utils import (
    ParallelLinearAdapter,
    get_adapter_attributes_from_linear,
)


class GatedLoRALinearSplitQKV(LoRALinearSplitQKV):
    """Pack canonical adapter outputs in Megatron's gated Q/Gate/K/V order."""

    def _interleave_qkv(
        self, query_gate: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        config = self.to_wrap.config
        head_num = config.num_attention_heads
        num_query_groups = config.num_query_groups
        head_size = config.kv_channels

        if head_num is None or num_query_groups is None or head_size is None:
            raise ValueError("Gated fused QKV requires explicit head and group configuration.")
        if head_num % num_query_groups != 0:
            raise ValueError("num_attention_heads must be divisible by num_query_groups.")

        query_size = head_num * head_size
        key_size = num_query_groups * head_size
        if query_gate.size(-1) != 2 * query_size:
            raise ValueError(
                f"Gated query adapter width must be {2 * query_size}, got {query_gate.size(-1)}."
            )
        if key.size(-1) != key_size or value.size(-1) != key_size:
            raise ValueError(f"Key and value adapter width must be {key_size}.")

        leading_shape = query_gate.shape[:-1]
        query_gate = query_gate.reshape(-1, head_num, 2, head_size)
        query = query_gate[:, :, 0]
        gate = query_gate[:, :, 1]
        key = key.reshape(-1, num_query_groups, head_size)
        value = value.reshape(-1, num_query_groups, head_size)

        heads_per_group = head_num // num_query_groups
        qkv_chunks = []
        for group in range(num_query_groups):
            query_start = group * heads_per_group
            query_stop = (group + 1) * heads_per_group
            query_group = query[:, query_start:query_stop, :]
            gate_group = gate[:, query_start:query_stop, :]
            key_group = key[:, group : group + 1, :]
            value_group = value[:, group : group + 1, :]
            qkv_chunks.extend([query_group, gate_group, key_group, value_group])

        qkv = torch.cat(qkv_chunks, dim=1)
        return qkv.reshape(*leading_shape, -1)

    def forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)
        if not self._adapter_enabled:
            return linear_output, bias

        query_gate = self.adapter_forward(
            self.adapter.adapter_q, layernorm_output, *args, **kwargs
        )
        key = self.adapter_forward(self.adapter.adapter_k, layernorm_output, *args, **kwargs)
        value = self.adapter_forward(self.adapter.adapter_v, layernorm_output, *args, **kwargs)
        adapter_output = self._interleave_qkv(query_gate, key, value)
        return linear_output + adapter_output, bias


class GatedCanonicalLoRA(CanonicalLoRA):
    """CanonicalLoRA variant that preserves gated fused QKV output contracts."""

    def _widen_gated_query_adapter(
        self, wrapped: LoRALinearSplitQKV, module: nn.Module
    ) -> GatedLoRALinearSplitQKV:
        old_adapter = wrapped.adapter.adapter_q
        attributes = get_adapter_attributes_from_linear(module)
        query_gate_size = 2 * module.config.kv_channels * module.config.num_attention_heads
        adapter_q = ParallelLinearAdapter(
            attributes.in_features,
            query_gate_size,
            dim=old_adapter.dim,
            base_linear_name=old_adapter.base_linear_name,
            activation="identity",
            column_init_method=self.lora_A_init_method,
            row_init_method=self.lora_B_init_method,
            input_is_parallel=old_adapter.input_is_parallel,
            dropout=self.dropout,
            dropout_position=self.dropout_position,
            model_parallel_config=old_adapter.config,
            alpha=self.alpha,
            is_expert=old_adapter.is_expert,
            disable_tensor_parallel_comm=attributes.disable_tensor_parallel_comm,
            disable_sequence_parallel_comm=old_adapter.disable_sequence_parallel_comm,
            base_linear_is_parallel=old_adapter.base_linear_is_parallel,
        )
        adapters = ModuleDict(
            {
                "adapter_q": adapter_q,
                "adapter_k": wrapped.adapter.adapter_k,
                "adapter_v": wrapped.adapter.adapter_v,
            }
        )
        return GatedLoRALinearSplitQKV(module, adapters)

    def transform(
        self, m: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None
    ) -> nn.Module:
        if name != "linear_qkv" or not getattr(m.config, "attention_output_gate", False):
            return super().transform(m, name, prefix)

        match = self.match(m, name, prefix)
        if match is None:
            return m

        canonical_target, _ = match
        if "linear_q" not in self.canonical_mapping[canonical_target]:
            raise ValueError("Gated fused QKV requires the canonical linear_q target.")

        original_gate = m.config.attention_output_gate
        m.config.attention_output_gate = False
        try:
            wrapped = super().transform(m, name, prefix)
        finally:
            m.config.attention_output_gate = original_gate

        if not isinstance(wrapped, LoRALinearSplitQKV):
            raise TypeError(f"Expected LoRALinearSplitQKV, got {type(wrapped).__name__}.")
        return self._widen_gated_query_adapter(wrapped, m)
