import json
from typing import Any

from tests.fast.charts.utils import objects_of_kind, render_run, render_run_error, requires_helm, with_object_names

TRAINER = [
    {
        "name": "trainer-engine-actor",
        "command": ["python", "-m", "miles.utils.workers.process_supervisor"],
        "resources": {"limits": {"nvidia.com/gpu": 4}},
    }
]


def _pod_spec_of_the_only_pool(*args: str) -> dict[str, Any]:
    rendered = render_run("--set-json", f"run.trainerEngines={json.dumps(with_object_names(TRAINER))}", *args)
    pool = objects_of_kind(rendered, "LeaderWorkerSet")[0]
    return pool["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]


def _shm_volume(pod_spec: dict[str, Any]) -> dict[str, Any]:
    [mount] = _shm_mounts(pod_spec)
    [volume] = [volume for volume in pod_spec["volumes"] if volume["name"] == mount["name"]]
    return volume


def _shm_mounts(pod_spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        mount
        for container in pod_spec["containers"]
        for mount in container.get("volumeMounts", [])
        if mount["mountPath"] == "/dev/shm"
    ]


@requires_helm
class TestPoolPodsShareTheHostSharedMemory:
    def test_a_pool_container_mounts_dev_shm(self):
        """Kubernetes' default 64Mi of /dev/shm is less than NCCL asks for per peer it cannot reach over p2p."""
        assert len(_shm_mounts(_pod_spec_of_the_only_pool())) == 1

    def test_that_mount_is_the_host_dev_shm_itself(self):
        """The pods share the host IPC namespace, and a CUDA IPC refcounter only works in the shm they all see."""
        assert _shm_volume(_pod_spec_of_the_only_pool())["hostPath"]["path"] == "/dev/shm"

    def test_the_host_directory_has_to_exist_already(self):
        """DirectoryOrCreate would silently make a plain disk directory on a node whose /dev/shm is missing."""
        assert _shm_volume(_pod_spec_of_the_only_pool())["hostPath"]["type"] == "Directory"

    def test_no_private_volume_of_its_own_shadows_it(self):
        """An emptyDir here is what breaks the sharing, so its absence is the property worth pinning."""
        assert "emptyDir" not in _shm_volume(_pod_spec_of_the_only_pool())

    def test_the_pods_that_mount_it_share_the_host_ipc_namespace(self):
        """Sharing the directory buys nothing unless the pods also share the namespace the handles live in."""
        assert _pod_spec_of_the_only_pool()["hostIPC"] is True

    def test_a_run_cannot_ask_for_a_private_size_any_more(self):
        """The size knob belonged to the emptyDir; leaving it accepted would let a values file break sharing."""
        error = render_run_error("--set", "run.shmSize=8Gi")

        assert "'shmSize' not allowed" in error

    def test_every_mounted_volume_is_declared_by_the_pod_that_mounts_it(self):
        """A container naming a volume the pod does not declare makes the whole manifest invalid."""
        pod_spec = _pod_spec_of_the_only_pool()
        declared = {volume["name"] for volume in pod_spec["volumes"]}
        mounted = {mount["name"] for container in pod_spec["containers"] for mount in container["volumeMounts"]}

        assert mounted <= declared, f"{mounted - declared} is mounted but never declared"

    def test_a_pod_that_runs_no_collective_is_left_alone(self):
        """The orchestrator only talks to the apiserver, so it has no reason to reach into the host's shm."""
        rendered = render_run("--set-json", f"run.trainerEngines={json.dumps(with_object_names(TRAINER))}")
        [orchestrator] = [
            obj for obj in objects_of_kind(rendered, "StatefulSet") if "orchestrator" in obj["metadata"]["name"]
        ]

        assert not _shm_mounts(orchestrator["spec"]["template"]["spec"])
