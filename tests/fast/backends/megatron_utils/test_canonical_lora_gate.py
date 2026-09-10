"""GPU regression coverage for gated CanonicalLoRA fused QKV adapters."""

from argparse import Namespace
from datetime import timedelta
import socket

import pytest
import torch
import torch.distributed as dist

from megatron.bridge.peft.canonical_lora import CanonicalLoRA
from megatron.core import parallel_state
from megatron.core.tensor_parallel import ColumnParallelLinear, model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from miles.backends.megatron_utils.lora_utils import create_lora_instance


HEAD_NUM = 6
GROUP_NUM = 2
HEAD_SIZE = 4
HIDDEN_SIZE = 8
RANK = 4
ALPHA = 4


class FusedQKVModel(torch.nn.Module):
    def __init__(self, output_size: int, gated: bool):
        super().__init__()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=HEAD_NUM,
            num_query_groups=GROUP_NUM,
            kv_channels=HEAD_SIZE,
            attention_output_gate=gated,
            params_dtype=torch.float32,
        )
        self.linear_qkv = ColumnParallelLinear(
            HIDDEN_SIZE,
            output_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
        )


def _lora_args() -> Namespace:
    return Namespace(
        lora_type="canonical_lora",
        target_modules=["linear_q", "linear_k", "linear_v"],
        lora_rank=RANK,
        lora_alpha=ALPHA,
        lora_dropout=0.0,
        lora_A_init_method="xavier",
        lora_B_init_method="zero",
    )


def _deterministic(shape: tuple[int, ...], seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).div_(10).to(device)


def _fill_adapter(adapter: torch.nn.Module, seed: int, device: torch.device) -> None:
    with torch.no_grad():
        adapter.linear_in.weight.copy_(_deterministic(adapter.linear_in.weight.shape, seed, device))
        start = seed + 100
        adapter.linear_out.weight.copy_(
            _deterministic(adapter.linear_out.weight.shape, start, device)
        )


def _fill_model(model: torch.nn.Module, seed: int, device: torch.device) -> None:
    wrapped = model.linear_qkv
    with torch.no_grad():
        wrapped.to_wrap.weight.copy_(
            _deterministic(wrapped.to_wrap.weight.shape, seed, device)
        )
    _fill_adapter(wrapped.adapter.adapter_q, seed + 200, device)
    _fill_adapter(wrapped.adapter.adapter_k, seed + 300, device)
    _fill_adapter(wrapped.adapter.adapter_v, seed + 400, device)


def _adapter_delta(
    x: torch.Tensor, adapter: torch.nn.Module, reference: bool = False
) -> torch.Tensor:
    if reference:
        weight_in = adapter[0]
        weight_out = adapter[1]
    else:
        weight_in = adapter.linear_in.weight
        weight_out = adapter.linear_out.weight
    return ((x @ weight_in.t()) @ weight_out.t()) * (ALPHA / RANK)


def _interleave(query_gate: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    leading_shape = query_gate.shape[:-1]
    query_gate = query_gate.reshape(-1, HEAD_NUM, 2, HEAD_SIZE)
    query = query_gate[:, :, 0]
    gate = query_gate[:, :, 1]
    key = key.reshape(-1, GROUP_NUM, HEAD_SIZE)
    value = value.reshape(-1, GROUP_NUM, HEAD_SIZE)

    heads_per_group = HEAD_NUM // GROUP_NUM
    chunks = []
    for group in range(GROUP_NUM):
        start = group * heads_per_group
        stop = (group + 1) * heads_per_group
        chunks.extend(
            [
                query[:, start:stop, :],
                gate[:, start:stop, :],
                key[:, group : group + 1, :],
                value[:, group : group + 1, :],
            ]
        )
    return torch.cat(chunks, dim=1).reshape(*leading_shape, -1)


def _reference_output(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    adapter_q: tuple[torch.Tensor, torch.Tensor],
    adapter_k: tuple[torch.Tensor, torch.Tensor],
    adapter_v: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    query_gate = _adapter_delta(x, adapter_q, reference=True)
    key = _adapter_delta(x, adapter_k, reference=True)
    value = _adapter_delta(x, adapter_v, reference=True)
    return x @ base_weight.t() + _interleave(query_gate, key, value)


@pytest.fixture(scope="module")
def gpu_distributed():
    if not torch.cuda.is_available():
        pytest.skip("This regression requires a CUDA/ROCm GPU.")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    dist.init_process_group(
        "nccl",
        world_size=1,
        rank=0,
        init_method=f"tcp://127.0.0.1:{port}",
        timeout=timedelta(seconds=60),
    )
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    yield torch.device("cuda")
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def test_gated_canonical_lora_matches_unfused_reference(gpu_distributed):
    device = gpu_distributed
    gated_output_size = 2 * HEAD_NUM * HEAD_SIZE + 2 * GROUP_NUM * HEAD_SIZE
    model = FusedQKVModel(gated_output_size, gated=True).to(device)
    create_lora_instance(_lora_args())(model)
    wrapped = model.linear_qkv
    _fill_model(model, seed=1, device=device)
    base_weight = wrapped.to_wrap.weight.detach().clone()

    assert wrapped.adapter.adapter_q.linear_out.weight.shape == (2 * HEAD_NUM * HEAD_SIZE, RANK)
    assert wrapped.adapter.adapter_k.linear_out.weight.shape == (GROUP_NUM * HEAD_SIZE, RANK)
    assert wrapped.adapter.adapter_v.linear_out.weight.shape == (GROUP_NUM * HEAD_SIZE, RANK)

    state = wrapped.state_dict()
    assert "weight" in state
    assert "adapter.adapter_q.linear_in.weight" in state
    assert "adapter.adapter_k.linear_out.weight" in state
    assert torch.equal(state["weight"], base_weight)
    assert wrapped.to_wrap.weight.grad is None

    x = _deterministic((3, HIDDEN_SIZE), seed=10, device=device).requires_grad_()
    output, bias = wrapped(x)
    assert output.shape == (3, gated_output_size)
    assert bias is None

    reference_x = x.detach().clone().requires_grad_()
    reference_q = tuple(
        parameter.detach().clone().requires_grad_()
        for parameter in (wrapped.adapter.adapter_q.linear_in.weight, wrapped.adapter.adapter_q.linear_out.weight)
    )
    reference_k = tuple(
        parameter.detach().clone().requires_grad_()
        for parameter in (wrapped.adapter.adapter_k.linear_in.weight, wrapped.adapter.adapter_k.linear_out.weight)
    )
    reference_v = tuple(
        parameter.detach().clone().requires_grad_()
        for parameter in (wrapped.adapter.adapter_v.linear_in.weight, wrapped.adapter.adapter_v.linear_out.weight)
    )
    reference_output = _reference_output(
        reference_x, base_weight, reference_q, reference_k, reference_v
    )
    actual_adapters = (
        (wrapped.adapter.adapter_q.linear_in.weight, wrapped.adapter.adapter_q.linear_out.weight),
        (wrapped.adapter.adapter_k.linear_in.weight, wrapped.adapter.adapter_k.linear_out.weight),
        (wrapped.adapter.adapter_v.linear_in.weight, wrapped.adapter.adapter_v.linear_out.weight),
    )
    reference_adapters = (reference_q, reference_k, reference_v)
    for actual_pair, reference_pair in zip(actual_adapters, reference_adapters):
        for actual, expected in zip(actual_pair, reference_pair):
            assert torch.equal(actual.detach(), expected.detach())

    torch.testing.assert_close(output, reference_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        output - (x @ base_weight.t()),
        reference_output - (reference_x @ base_weight.t()),
        rtol=1e-5,
        atol=1e-5,
    )

    probe = _deterministic(output.shape, seed=20, device=device)
    (output * probe).sum().backward()
    (reference_output * probe).sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-5, atol=1e-5)
    for actual_pair, reference_pair in zip(actual_adapters, reference_adapters):
        for actual, expected in zip(actual_pair, reference_pair):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-5)


def test_create_lora_instance_preserves_canonical_hf_targets(gpu_distributed):
    args = _lora_args()
    args.target_modules = ["q_proj", "k_proj", "v_proj"]
    peft = create_lora_instance(args)
    assert peft.target_modules == ["linear_q", "linear_k", "linear_v"]



def test_upstream_canonical_lora_rejects_gated_width(gpu_distributed):
    device = gpu_distributed
    gated_output_size = 2 * HEAD_NUM * HEAD_SIZE + 2 * GROUP_NUM * HEAD_SIZE
    model = FusedQKVModel(gated_output_size, gated=True).to(device)
    CanonicalLoRA(
        target_modules=["linear_q", "linear_k", "linear_v"], dim=RANK, alpha=ALPHA
    )(model)
    wrapped = model.linear_qkv
    assert wrapped.adapter.adapter_q.linear_out.weight.shape == (HEAD_NUM * HEAD_SIZE, RANK)

    x = torch.randn((3, HIDDEN_SIZE), device=device)
    with pytest.raises(RuntimeError, match="must match"):
        wrapped(x)


def test_non_gated_canonical_lora_remains_unchanged(gpu_distributed):
    device = gpu_distributed
    output_size = HEAD_NUM * HEAD_SIZE + 2 * GROUP_NUM * HEAD_SIZE
    expected_model = FusedQKVModel(output_size, gated=False).to(device)
    actual_model = FusedQKVModel(output_size, gated=False).to(device)
    CanonicalLoRA(
        target_modules=["linear_q", "linear_k", "linear_v"], dim=RANK, alpha=ALPHA
    )(expected_model)
    create_lora_instance(_lora_args())(actual_model)
    _fill_model(expected_model, seed=2, device=device)
    _fill_model(actual_model, seed=2, device=device)

    x = _deterministic((3, HIDDEN_SIZE), seed=30, device=device)
    expected, _ = expected_model.linear_qkv(x)
    actual, _ = actual_model.linear_qkv(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
