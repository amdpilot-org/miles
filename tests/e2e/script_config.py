import dataclasses
from typing import TypeVar

from miles.utils.external_utils import command_utils

ScriptArgsT = TypeVar("ScriptArgsT", bound=command_utils.ExecuteTrainConfig)


def script_args_from_environment(script_args_class: type[ScriptArgsT], **recipe: object) -> ScriptArgsT:
    environment = command_utils.default_config()
    launcher_fields = {
        field.name: getattr(environment, field.name)
        for field in dataclasses.fields(command_utils.CommandUtilConfig)
        if field.name not in recipe
    }
    return script_args_class(**launcher_fields, **recipe)
