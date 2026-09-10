import json

import pytest


def _write_tiny_qwen3_moe_config(model_path):
    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_size": 16,
        "intermediate_size": 16,
        "moe_intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
        "vocab_size": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "max_position_embeddings": 64,
        "torch_dtype": "bfloat16",
    }
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_cpu_replica_skips_multi_node_topology_probe(tmp_path, monkeypatch):
    aiter_configs = {
        "AITER_CONFIG_GEMM_BF16": "bf16_tuned_gemm.csv",
        "AITER_CONFIG_GEMM_A8W8": "a8w8_tuned_gemm.csv",
        "AITER_CONFIG_GEMM_A4W4": "a4w4_blockscale_tuned_gemm.csv",
        "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE": "a8w8_bpreshuffle_tuned_gemm.csv",
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": "a8w8_blockscale_tuned_gemm.csv",
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": "a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
        "AITER_CONFIG_FMOE": "tuned_fmoe.csv",
        "AITER_CONFIG_GROUPED_FMOE": "tuned_grouped_fmoe.csv",
        "AITER_CONFIG_A8W8_BATCHED_GEMM": "a8w8_tuned_batched_gemm.csv",
        "AITER_CONFIG_BF16_BATCHED_GEMM": "bf16_tuned_batched_gemm.csv",
    }
    for name, filename in aiter_configs.items():
        monkeypatch.setenv(name, f"/sgl-workspace/aiter/aiter/configs/{filename}")

    from sglang.srt import server_args as server_args_module
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.distributed.parallel_state import RankParallelismConfig
    from sglang.srt.runtime_context import get_parallel, publish
    from sglang.srt.server_args import ServerArgs

    from miles.backends.training_utils.weight_update.protocols import p2p as p2p_module
    from miles.backends.training_utils.weight_update.protocols.p2p import UpdateWeightP2P

    model_path = tmp_path / "tiny-qwen3-moe"
    _write_tiny_qwen3_moe_config(model_path)
    server_args = ServerArgs(
        model_path=str(model_path),
        nnodes=2,
        tp_size=2,
        ep_size=2,
        device="cpu",
    )
    publish(server_args, role="scheduler")
    server_args_module.set_global_server_args_for_scheduler(server_args)
    initialize_moe_config()
    initialize_fp8_gemm_config()
    initialize_fp4_gemm_config()

    protocol = UpdateWeightP2P.__new__(UpdateWeightP2P)
    protocol._shared_params_dict = {}
    parallelism_config = RankParallelismConfig(
        tp_size=2,
        tp_rank=0,
        ep_size=2,
        ep_rank=0,
        world_size=2,
        global_rank=0,
        local_rank=0,
    )
    original_get_model = p2p_module.get_model

    def injected_get_model(**kwargs):
        raise RuntimeError("injected")

    p2p_module.get_model = injected_get_model
    try:
        with pytest.raises(RuntimeError, match="injected"):
            protocol._create_cpu_replica(
                parallelism_config,
                str(model_path),
                server_args,
                first_engine_rank=True,
            )
    finally:
        p2p_module.get_model = original_get_model

    assert get_parallel().nnodes == 2
    model = protocol._create_cpu_replica(
        parallelism_config,
        str(model_path),
        server_args,
        first_engine_rank=True,
    )

    assert model.__class__.__name__ == "Qwen3MoeForCausalLM"
    assert get_parallel().nnodes == 2
