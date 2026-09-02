import dataclasses
from typing import Literal

import pytest

import miles.utils.external_utils.command_utils.legacy as legacy
from miles.utils.external_utils.command_utils import base_backend
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig as CurrentExecuteTrainConfig


class _RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def execute_train(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


class TestExecuteTrainConfig:
    def test_positional_v1_config_is_converted_before_launch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The v1 positional field order and values must reach the current backend unchanged."""
        config = legacy.ExecuteTrainConfig(True, 4, "MY_VAR=value", "/output")
        backend = _RecordingBackend()
        current_configs: list[CurrentExecuteTrainConfig] = []

        def before_ray_job_submit() -> None:
            pass

        def _create_backend(current_config: CurrentExecuteTrainConfig) -> _RecordingBackend:
            current_configs.append(current_config)
            return backend

        monkeypatch.setattr(legacy, "_create_ray_backend", _create_backend)

        legacy.execute_train(
            train_args="--train-backend fsdp",
            num_gpus_per_node=8,
            megatron_model_type=None,
            config=config,
            before_ray_job_submit=before_ray_job_submit,
        )

        assert [field.name for field in dataclasses.fields(legacy.ExecuteTrainConfig)] == [
            "cuda_core_dump",
            "num_nodes",
            "extra_env_vars",
            "output_dir",
        ]
        current_config = current_configs[0]
        assert current_config.cuda_core_dump is True
        assert current_config.num_nodes == 4
        assert current_config.extra_env_vars == "MY_VAR=value"
        assert current_config.output_dir == "/output"
        assert backend.calls[0]["config"] is current_config
        assert backend.calls[0]["before_ray_job_submit"] is before_ray_job_submit


@dataclasses.dataclass
class _LauncherConfig:
    """The shape of a v1 launcher's own config: it carries the --hardware option resolve_hardware reads."""

    hardware: Literal["auto", "h100"] = "h100"


class TestWhatTheV1ModuleReExports:
    def test_a_v1_launcher_can_resolve_its_hardware_through_this_module(self):
        """Every launcher that leaves --hardware on auto calls U.resolve_hardware(self) while building its config."""
        assert legacy.resolve_hardware is base_backend.resolve_hardware

    def test_it_answers_for_a_launcher_config_that_names_its_hardware(self):
        """The v1 api carried this function, so importing it is not enough; it has to work through here."""
        assert legacy.resolve_hardware(_LauncherConfig()) == "h100"

    def test_it_refuses_a_hardware_the_launcher_has_no_profile_for(self):
        """A launcher that reaches this through the v1 module gets the same check as one that does not."""
        with pytest.raises(AssertionError, match="no verified profile"):
            legacy.resolve_hardware(_LauncherConfig(hardware="gb200"))

    def test_every_name_the_module_advertises_is_a_name_it_has(self):
        """__all__ is what a v1 launcher's star import reads, and a name missing from the module is an error."""
        assert [name for name in legacy.__all__ if not hasattr(legacy, name)] == []

    def test_the_re_export_is_advertised_rather_than_only_imported(self):
        """A star import is how the launch script guide tells older ray launchers to reach these."""
        assert "resolve_hardware" in legacy.__all__
