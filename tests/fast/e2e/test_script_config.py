from tests.e2e.script_config import config_for_launch

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.helm_backend.naming import RUN_ID_MAX_LENGTH, ReleaseName
from miles.utils.workers.types import DeployComponent


class TestConfigForLaunch:
    def test_a_launch_index_reaches_the_run_id(self) -> None:
        """Two launches of one test must not name the same release."""
        config = command_utils.ExecuteTrainConfig(run_id="align")

        assert config_for_launch(config, launch_index=1).run_id == "align-1"

    def test_a_run_id_of_the_greatest_legal_length_stays_legal(self) -> None:
        """A run id that already fills the release-name budget must not overrun it once indexed."""
        config = command_utils.ExecuteTrainConfig(run_id="a" * RUN_ID_MAX_LENGTH)

        run_id = config_for_launch(config, launch_index=7).run_id

        assert run_id.endswith("-7")
        ReleaseName(run_id=run_id, deploy_component=DeployComponent.ALL, deploy_instance_id=None)

    def test_two_launches_of_a_truncated_run_id_stay_apart(self) -> None:
        """Truncating the base id must not collapse two launches onto one release."""
        config = command_utils.ExecuteTrainConfig(run_id="a" * RUN_ID_MAX_LENGTH)

        assert config_for_launch(config, launch_index=1).run_id != config_for_launch(config, launch_index=2).run_id
