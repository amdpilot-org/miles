import json
import shlex

from tests.e2e.ft.conftest_ft.modes import MODES
from tests.e2e.ft.conftest_ft.scenario_trainer_with_failure import (
    _DIFF_THRESHOLDS,
    _POST_FAULT_DIFF_THRESHOLDS,
    FAULT_ROLLOUT_ID,
    FIRST_INJECTED_ROLLOUT_ID,
    FIRST_POST_FAULT_ROLLOUT_ID,
    _build_target_args,
    _diff_thresholds_for_rollout,
)


def _option_value(args: str, option: str) -> str:
    tokens = shlex.split(args)
    return tokens[tokens.index(option) + 1]


def test_real_rollout_injection_starts_at_fault_rollout() -> None:
    """Real-rollout training data injection must cover the fault rollout itself."""
    args = _build_target_args(
        MODES["kill_train__dp2_cp2"],
        "/tmp/target/phase_b",
        enable_dumper=False,
    )
    actions = json.loads(_option_value(args, "--ci-ft-test-actions"))

    assert FIRST_INJECTED_ROLLOUT_ID == FAULT_ROLLOUT_ID
    assert FIRST_POST_FAULT_ROLLOUT_ID == FAULT_ROLLOUT_ID + 1
    assert int(_option_value(args, "--ci-inject-rollout-data-start-rollout-id")) == FAULT_ROLLOUT_ID
    assert {action["at_rollout"] for action in actions} == {FAULT_ROLLOUT_ID}


def test_fault_rollout_keeps_strict_tensor_thresholds() -> None:
    """The fault rollout must stay strict while measured post-fault floors start later."""
    mode = MODES["kill_train__dp2_cp2"]

    assert _diff_thresholds_for_rollout(mode, FAULT_ROLLOUT_ID) is _DIFF_THRESHOLDS
    assert _diff_thresholds_for_rollout(mode, FIRST_POST_FAULT_ROLLOUT_ID) is _POST_FAULT_DIFF_THRESHOLDS


def test_fake_rollout_does_not_inject_recorded_data() -> None:
    """Fake-rollout scenarios must keep using their generated deterministic fixtures."""
    args = _build_target_args(
        MODES["kill_train__dp2_cp2_pp2__fake_rollout__moe_5layer"],
        "/tmp/target/phase_b",
        enable_dumper=False,
    )

    assert "--ci-inject-rollout-data-start-rollout-id" not in shlex.split(args)
