import sys
from argparse import Namespace
from types import SimpleNamespace


def test_custom_bridge_layer_spec_replaces_default(monkeypatch):
    from miles.backends.megatron_utils import model_provider

    def custom_spec(args, config, vp_stage):
        return args, config, vp_stage

    monkeypatch.setattr(model_provider, "import_module", lambda _path: custom_spec)

    args = Namespace(spec=["custom.module", "custom_spec"])
    provider = SimpleNamespace(transformer_layer_spec="default-layer-spec")

    model_provider._apply_custom_bridge_layer_spec(provider, args)

    bridge_config = SimpleNamespace()
    assert provider.transformer_layer_spec(bridge_config, vp_stage=3) == (
        args,
        bridge_config,
        3,
    )


def test_static_bridge_layer_spec_replaces_default(monkeypatch):
    from miles.backends.megatron_utils import model_provider

    monkeypatch.setattr(model_provider, "import_module", lambda _path: "static-layer-spec")
    provider = SimpleNamespace(transformer_layer_spec="default-layer-spec")

    model_provider._apply_custom_bridge_layer_spec(provider, Namespace(spec=["custom.module", "custom_spec"]))

    assert provider.transformer_layer_spec == "static-layer-spec"


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
