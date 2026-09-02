import dataclasses
from typing import TypeVar

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.helm_backend.naming import RUN_ID_MAX_LENGTH

ScriptArgsT = TypeVar("ScriptArgsT", bound=command_utils.ExecuteTrainConfig)


def script_args_from_environment(script_args_class: type[ScriptArgsT], **recipe: object) -> ScriptArgsT:
    environment = command_utils.default_config()
    launcher_fields = {
        field.name: getattr(environment, field.name)
        for field in dataclasses.fields(command_utils.CommandUtilConfig)
        if field.name not in recipe
    }
    return script_args_class(**launcher_fields, **recipe)


def config_for_launch(
    config: command_utils.ExecuteTrainConfig, *, launch_index: int
) -> command_utils.ExecuteTrainConfig:
    suffix = f"-{launch_index}"
    return dataclasses.replace(config, run_id=f"{config.run_id[: RUN_ID_MAX_LENGTH - len(suffix)]}{suffix}")
