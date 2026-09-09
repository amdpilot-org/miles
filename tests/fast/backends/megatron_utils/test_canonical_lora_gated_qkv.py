from __future__ import annotations

from argparse import Namespace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from miles.backends.training_utils.parallel import ParallelState, set_parallel_state
from miles.backends.megatron_utils.canonical_lora import GatedLoRALinearSplitQKV
from miles.backends.megatron_utils.lora_utils import (
    create_lora_instance,
    load_lora_adapter,
    save_lora_checkpoint,
)
from miles.utils.ft_utils.process_group_utils import GroupInfo


HIDDEN_SIZE = 16
HEAD_NUM = 4
QUERY_GROUPS = 2
HEAD_SIZE = 8
QKV_SIZE = 2 * HEAD_NUM * HEAD_SIZE + 2 * QUERY_GROUPS * HEAD_SIZE
LORA_RANK = 4
LORA_ALPHA = 8


class FusedQKVModel(torch.nn.Module):
    def __init__(self, linear: ColumnParallelLinear) -> None:
        super().__init__()
        self.linear_qkv = linear


@pytest.fixture
def one_gpu_process(tmp_path: Path):
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        pytest.skip("A CUDA/HIP GPU is required for the fused QKV fixture.")

    store = dist.FileStore(str(tmp_path / "torch-dist-store"), 1)
    dist.init_process_group(
        backend="nccl",
        store=store,
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=30),
    )
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    torch.cuda.set_device(0)
    trivial_group = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=trivial_group,
            intra_dp_cp=trivial_group,
            cp=trivial_group,
            tp=trivial_group,
            pp=trivial_group,
            ep=trivial_group,
            etp=trivial_group,
            indep_dp=trivial_group,
            is_pp_last_stage=True,
            vpp_size=1,
        )
    )
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def _make_config() -> TransformerConfig:
    config = TransformerConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=HEAD_NUM,
        num_query_groups=QUERY_GROUPS,
        kv_channels=HEAD_SIZE,
        num_layers=1,
        params_dtype=torch.float32,
    )
    config.attention_output_gate = True
    config.perform_initialization = False
    return config


def _make_model() -> tuple[FusedQKVModel, torch.Tensor]:
    config = _make_config()
    linear = ColumnParallelLinear(
        HIDDEN_SIZE,
        QKV_SIZE,
        config=config,
        init_method=lambda tensor: torch.nn.init.zeros_(tensor),
        bias=False,
        gather_output=False,
        name="linear_qkv",
    ).cuda()
    base_weight = torch.randn_like(linear.weight)
    with torch.no_grad():
        linear.weight.copy_(base_weight)
    return FusedQKVModel(linear), base_weight


def _make_args(hf_checkpoint: str) -> Namespace:
    return Namespace(
        lora_type="canonical_lora",
        target_modules=["q_proj", "k_proj", "v_proj"],
        exclude_modules=None,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        lora_A_init_method="xavier",
        lora_B_init_method="zero",
        experts_shared_outer_loras=False,
        hf_checkpoint=hf_checkpoint,
    )


def _apply_canonical_lora(model: FusedQKVModel, hf_checkpoint: str) -> None:
    model = create_lora_instance(_make_args(hf_checkpoint))(model, training=True)
    assert isinstance(model.linear_qkv, GatedLoRALinearSplitQKV)


def _reference_interleave(
    query_and_gate: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    leading_shape = query_and_gate.shape[:-1]
    query_and_gate = query_and_gate.reshape(-1, HEAD_NUM, 2, HEAD_SIZE)
    query = query_and_gate[:, :, 0, :]
    gate = query_and_gate[:, :, 1, :]
    key = key.reshape(-1, QUERY_GROUPS, HEAD_SIZE)
    value = value.reshape(-1, QUERY_GROUPS, HEAD_SIZE)
    heads_per_group = HEAD_NUM // QUERY_GROUPS

    chunks = []
    for group_index in range(QUERY_GROUPS):
        start = group_index * heads_per_group
        stop = start + heads_per_group
        chunks.extend(
            [
                query[:, start:stop, :],
                gate[:, start:stop, :],
                key[:, group_index : group_index + 1, :],
                value[:, group_index : group_index + 1, :],
            ]
        )
    return torch.cat(chunks, dim=1).reshape(*leading_shape, -1)


def _reference_adapter_output(
    input_tensor: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
) -> torch.Tensor:
    scale = LORA_ALPHA / LORA_RANK
    return scale * ((input_tensor @ lora_a.transpose(0, 1)) @ lora_b.transpose(0, 1))


def _set_adapter_weights(model: FusedQKVModel) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260909)

    def random_tensor(*shape: int) -> torch.Tensor:
        return torch.randn(shape, device="cuda", generator=generator, requires_grad=True)

    reference = {
        "adapter_q": (
            random_tensor(LORA_RANK, HIDDEN_SIZE),
            random_tensor(2 * HEAD_NUM * HEAD_SIZE, LORA_RANK),
        ),
        "adapter_k": (
            random_tensor(LORA_RANK, HIDDEN_SIZE),
            random_tensor(QUERY_GROUPS * HEAD_SIZE, LORA_RANK),
        ),
        "adapter_v": (
            random_tensor(LORA_RANK, HIDDEN_SIZE),
            random_tensor(QUERY_GROUPS * HEAD_SIZE, LORA_RANK),
        ),
    }
    with torch.no_grad():
        for adapter_name, (lora_a, lora_b) in reference.items():
            adapter = model.linear_qkv.adapter[adapter_name]
            adapter.linear_in.weight.copy_(lora_a)
            adapter.linear_out.weight.copy_(lora_b)
    return reference


def _assert_adapter_tensors(
    model: FusedQKVModel,
    reference: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    for adapter_name, (lora_a, lora_b) in reference.items():
        adapter = model.linear_qkv.adapter[adapter_name]
        torch.testing.assert_close(adapter.linear_in.weight, lora_a.detach())
        torch.testing.assert_close(adapter.linear_out.weight, lora_b.detach())


def test_gated_canonical_lora_matches_unfused_reference_and_loads(one_gpu_process, tmp_path: Path):
    model, base_weight = _make_model()
    base_weight_copy = base_weight.detach().clone()
    _apply_canonical_lora(model, str(tmp_path / "missing-hf-checkpoint"))

    assert model.linear_qkv.to_wrap.config.num_attention_heads == HEAD_NUM
    assert model.linear_qkv.adapter["adapter_q"].linear_out.weight.shape == (
        2 * HEAD_NUM * HEAD_SIZE,
        LORA_RANK,
    )
    assert not model.linear_qkv.to_wrap.weight.requires_grad
    assert all(param.requires_grad for param in model.linear_qkv.adapter.parameters())

    reference = _set_adapter_weights(model)
    fused_input = torch.randn(3, HIDDEN_SIZE, device="cuda", requires_grad=True)
    reference_input = fused_input.detach().clone().requires_grad_(True)
    fused_output, fused_bias = model.linear_qkv(fused_input)
    assert fused_bias is None
    assert fused_output.shape == (3, QKV_SIZE)

    reference_output = torch.nn.functional.linear(reference_input, base_weight.detach())
    reference_output = reference_output + _reference_interleave(
        _reference_adapter_output(reference_input, *reference["adapter_q"]),
        _reference_adapter_output(reference_input, *reference["adapter_k"]),
        _reference_adapter_output(reference_input, *reference["adapter_v"]),
    )
    torch.testing.assert_close(fused_output, reference_output)

    output_grad = torch.randn_like(fused_output)
    fused_output.backward(output_grad)
    reference_output.backward(output_grad)
    torch.testing.assert_close(fused_input.grad, reference_input.grad)
    for adapter_name, (lora_a, lora_b) in reference.items():
        adapter = model.linear_qkv.adapter[adapter_name]
        torch.testing.assert_close(adapter.linear_in.weight.grad, lora_a.grad)
        torch.testing.assert_close(adapter.linear_out.weight.grad, lora_b.grad)
    assert model.linear_qkv.to_wrap.weight.grad is None
    torch.testing.assert_close(model.linear_qkv.to_wrap.weight, base_weight_copy)

    checkpoint_dir = tmp_path / "adapter"
    save_lora_checkpoint([model], _make_args(str(tmp_path / "missing-hf-checkpoint")), str(checkpoint_dir))

    loaded_model, loaded_base_weight = _make_model()
    with torch.no_grad():
        loaded_model.linear_qkv.weight.copy_(base_weight_copy)
    _apply_canonical_lora(loaded_model, str(tmp_path / "missing-hf-checkpoint"))
    loaded, iteration = load_lora_adapter([loaded_model], str(checkpoint_dir))
    assert loaded is True
    assert iteration is None
    _assert_adapter_tensors(loaded_model, reference)
    torch.testing.assert_close(loaded_model.linear_qkv.to_wrap.weight, base_weight_copy)

    loaded_input = fused_input.detach().clone().requires_grad_(True)
    loaded_output, loaded_bias = loaded_model.linear_qkv(loaded_input)
    assert loaded_bias is None
    torch.testing.assert_close(loaded_output, fused_output.detach())
