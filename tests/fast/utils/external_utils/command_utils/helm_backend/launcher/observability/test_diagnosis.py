import subprocess
from pathlib import Path

import pytest

from miles.utils.external_utils.command_utils.helm_backend.launcher.observability import diagnosis

_NEVER_RESTARTED = 'Error from server (BadRequest): previous terminated container "app" in pod "trainer-0" not found'
_API_SERVER_BLINKED = "Error from server: etcdserver: request timed out"


def _collect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    pods: list[str] | None,
    previous_failure: str | None = None,
) -> diagnosis.Diagnosis:
    def run_process(command: list[str], capture_output: bool, check: bool) -> subprocess.CompletedProcess:
        if "--previous" in command and previous_failure is not None:
            return subprocess.CompletedProcess(args=command, returncode=1, stdout="", stderr=previous_failure)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="captured\n", stderr="")

    monkeypatch.setattr(diagnosis, "run_process", run_process)
    monkeypatch.setattr(diagnosis, "_pod_names", lambda *, namespace, selector: pods)
    return diagnosis.collect_diagnosis(namespace="rl", output_dir=tmp_path)


class TestPreviousLogsOfAPod:
    def test_a_container_that_never_restarted_leaves_the_diagnosis_complete(self, monkeypatch, tmp_path):
        """Most pods of a healthy-looking failure never restarted, and that is not evidence going missing."""
        collected = _collect(monkeypatch, tmp_path, pods=["trainer-0"], previous_failure=_NEVER_RESTARTED)

        assert collected.is_complete
        assert collected.missing == ()

    def test_a_container_that_never_restarted_gets_no_previous_log_file(self, monkeypatch, tmp_path):
        """A file holding nothing but the api server's refusal reads as a crash log that says nothing."""
        collected = _collect(monkeypatch, tmp_path, pods=["trainer-0"], previous_failure=_NEVER_RESTARTED)

        assert not (collected.directory / "trainer-0.previous.log").exists()

    def test_an_api_error_is_reported_rather_than_read_as_a_pod_that_never_restarted(self, monkeypatch, tmp_path):
        """The crash log is the whole point of the diagnosis, and losing it silently is what hid the crash."""
        collected = _collect(monkeypatch, tmp_path, pods=["trainer-0"], previous_failure=_API_SERVER_BLINKED)

        assert not collected.is_complete
        assert "previous logs of trainer-0" in collected.missing

    def test_an_api_error_is_written_down_where_the_crash_log_would_be(self, monkeypatch, tmp_path):
        """Whoever reads the directory has to be able to see why the file is not the log they wanted."""
        collected = _collect(monkeypatch, tmp_path, pods=["trainer-0"], previous_failure=_API_SERVER_BLINKED)

        assert _API_SERVER_BLINKED in (collected.directory / "trainer-0.previous.log").read_text()

    def test_a_collection_that_captured_everything_is_complete(self, monkeypatch, tmp_path):
        """A diagnosis reported as incomplete sends its reader looking for evidence that is right there."""
        collected = _collect(monkeypatch, tmp_path, pods=["trainer-0"])

        assert collected.is_complete
        assert (collected.directory / "trainer-0.previous.log").read_text() == "captured\n"


class TestThePodsACollectionCovers:
    def test_a_run_with_no_pods_left_is_not_a_complete_diagnosis(self, monkeypatch, tmp_path):
        """A directory holding nothing but namespace events answered no question about the failed run."""
        collected = _collect(monkeypatch, tmp_path, pods=[])

        assert not collected.is_complete
        assert "pods of the run in namespace rl" in collected.missing

    def test_a_pod_listing_that_failed_is_reported_as_missing(self, monkeypatch, tmp_path):
        """Not knowing which pods exist is a different gap from knowing there are none."""
        collected = _collect(monkeypatch, tmp_path, pods=None)

        assert "pod listing in namespace rl" in collected.missing
        assert "pods of the run in namespace rl" not in collected.missing

    def test_every_pod_of_the_run_is_captured(self, monkeypatch, tmp_path):
        """A split run fails in one of its deployments, and each pod's log is a candidate answer."""
        collected = _collect(monkeypatch, tmp_path, pods=["trainer-0", "engine-0"])

        assert (collected.directory / "trainer-0.log").exists()
        assert (collected.directory / "engine-0.describe.txt").exists()
        assert collected.is_complete
