import sys
from argparse import Namespace
from types import SimpleNamespace


def test_custom_bridge_layer_spec_replaces_default(monkeypatch):
    from miles.backends.megatron_utils import model_provider

    def custom_spec(args, config, vp_stage):
        assert config.experimental_attention_variant is None
        return args, config, vp_stage

    monkeypatch.setattr(model_provider, "import_module", lambda _path: custom_spec)

    args = Namespace(spec=["custom.module", "custom_spec"])
    provider = SimpleNamespace(
        transformer_layer_spec="default-layer-spec",
        experimental_attention_variant="gated_delta_net",
    )

    model_provider._apply_custom_bridge_layer_spec(provider, args)

    assert provider.transformer_layer_spec(provider, vp_stage=3) == (
        args,
        provider,
        3,
    )
    assert provider.experimental_attention_variant == "gated_delta_net"


def test_static_bridge_layer_spec_replaces_default(monkeypatch):
    from miles.backends.megatron_utils import model_provider

    monkeypatch.setattr(model_provider, "import_module", lambda _path: "static-layer-spec")
    provider = SimpleNamespace(transformer_layer_spec="default-layer-spec")

    model_provider._apply_custom_bridge_layer_spec(provider, Namespace(spec=["custom.module", "custom_spec"]))

    assert provider.transformer_layer_spec == "static-layer-spec"


def test_custom_bridge_layer_spec_honors_caller_variant(monkeypatch):
    from miles.backends.megatron_utils import model_provider

    def custom_spec(args, config, vp_stage):
        return config.experimental_attention_variant, vp_stage

    monkeypatch.setattr(model_provider, "import_module", lambda _path: custom_spec)
    provider = SimpleNamespace(
        transformer_layer_spec="default-layer-spec",
        experimental_attention_variant="bridge-default",
    )
    args = Namespace(
        spec=["custom.module", "custom_spec"],
        experimental_attention_variant="caller-value",
    )

    model_provider._apply_custom_bridge_layer_spec(provider, args)

    assert provider.transformer_layer_spec(provider, vp_stage=2) == ("caller-value", 2)
    assert provider.experimental_attention_variant == "bridge-default"


def test_bridge_provider_applies_custom_layer_spec(monkeypatch):
    from miles.backends.megatron_utils import model_provider

    provider = SimpleNamespace(transformer_layer_spec="default-layer-spec", finalize=lambda: None)

    class FakeBridge:
        def to_megatron_provider(self, load_weights):
            assert load_weights is False
            return provider

    monkeypatch.setattr(
        model_provider,
        "_apply_bridge_runtime_config",
        lambda _provider, _args: None,
    )
    monkeypatch.setattr(
        model_provider,
        "import_module",
        lambda _path: lambda args, config, vp_stage: "custom-layer-spec",
    )
    monkeypatch.setitem(
        sys.modules,
        "megatron.bridge",
        SimpleNamespace(AutoBridge=SimpleNamespace(from_hf_pretrained=lambda *_args, **_kwargs: FakeBridge())),
    )

    model_provider.get_model_provider_func(
        Namespace(
            custom_model_provider_path=None,
            megatron_to_hf_mode="bridge",
            hf_checkpoint="/hf-checkpoint",
            spec=["custom.module", "custom_spec"],
        )
    )

    assert provider.transformer_layer_spec(SimpleNamespace(), vp_stage=None) == "custom-layer-spec"


def test_real_bridge_provider_provide_uses_custom_layer_spec(monkeypatch):
    from megatron.bridge.models import gpt_provider as bridge_provider_module
    from miles.backends.megatron_utils import model_provider

    class FakeGPTModel:
        def __init__(self, _config, **kwargs):
            self.kwargs = kwargs

    provider = bridge_provider_module.GPTModelProvider(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        vocab_size=8,
        seq_length=8,
    )
    monkeypatch.setattr(bridge_provider_module, "MCoreGPTModel", FakeGPTModel)
    monkeypatch.setattr(
        model_provider,
        "import_module",
        lambda _path: lambda _args, _config, _vp_stage: "custom-layer-spec",
    )

    model_provider._apply_custom_bridge_layer_spec(
        provider,
        Namespace(spec=["custom.module", "custom_spec"]),
    )
    model = provider.provide(pre_process=True, post_process=True)

    assert model.kwargs["transformer_layer_spec"] == "custom-layer-spec"
