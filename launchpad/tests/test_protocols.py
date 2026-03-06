from __future__ import annotations

from collections.abc import Iterator

from launchpad.protocols import CloudProvider, SSHExecutor
from launchpad.types import (
    DataCenterId,
    GpuPricing,
    GpuType,
    GpuTypeId,
    PodId,
    PodInfo,
    PodStatus,
    PortMapping,
    SSHConnectionInfo,
    SSHResult,
    VolumeId,
    VolumeInfo,
)

_CONN = SSHConnectionInfo(host="10.0.0.1", port=22, user="root")
_POD_ID = PodId("pod-1")
_GPU_TYPE_ID = GpuTypeId("gpu-a100")
_VOLUME_ID = VolumeId("vol-1")
_DC_ID = DataCenterId("us-east-1")

_POD_INFO = PodInfo(
    id=_POD_ID,
    name="test-pod",
    status=PodStatus.RUNNING,
    cost_per_hr=1.0,
    gpu=_GPU_TYPE_ID,
    public_ip="1.2.3.4",
    ports=(PortMapping(internal_port=22, external_port=22),),
)
_VOLUME_INFO = VolumeInfo(
    id=_VOLUME_ID,
    name="test-vol",
    size_gb=100,
    data_center_id=_DC_ID,
)
_GPU_TYPE = GpuType(
    id=_GPU_TYPE_ID,
    name="A100",
    memory_mb=81920,
    pricing=GpuPricing(min_bid=0.5, uninterruptable=2.0),
)
_SSH_RESULT = SSHResult(exit_code=0, stdout="ok", stderr="")


class FakeProvider:
    def create_pod(
        self,
        name: str,
        gpu_type_id: GpuTypeId,
        image: str,
        *,
        volume_id: VolumeId | None = None,
        data_center_id: DataCenterId | None = None,
        gpu_count: int = 1,
    ) -> PodInfo:
        return _POD_INFO

    def stop_pod(self, pod_id: PodId) -> PodInfo:
        return _POD_INFO

    def resume_pod(self, pod_id: PodId) -> PodInfo:
        return _POD_INFO

    def terminate_pod(self, pod_id: PodId) -> None:
        return None

    def get_pod(self, pod_id: PodId) -> PodInfo:
        return _POD_INFO

    def create_volume(
        self,
        name: str,
        size_gb: int,
        data_center_id: DataCenterId,
    ) -> VolumeInfo:
        return _VOLUME_INFO

    def list_volumes(self) -> list[VolumeInfo]:
        return [_VOLUME_INFO]

    def get_volume(self, volume_id: VolumeId) -> VolumeInfo:
        return _VOLUME_INFO

    def delete_volume(self, volume_id: VolumeId) -> None:
        return None

    def list_gpu_types(self) -> list[GpuType]:
        return [_GPU_TYPE]

    def ssh_info(self, pod_id: PodId) -> SSHConnectionInfo:
        return _CONN


class FakeSSH:
    def run_command(
        self,
        conn: SSHConnectionInfo,
        command: str,
        *,
        timeout: int | None = None,
    ) -> SSHResult:
        return _SSH_RESULT

    def run_background(
        self,
        conn: SSHConnectionInfo,
        command: str,
    ) -> None:
        return None

    def tail_follow(
        self,
        conn: SSHConnectionInfo,
        file_path: str,
    ) -> Iterator[str]:
        yield "line1"

    def file_exists(
        self,
        conn: SSHConnectionInfo,
        remote_path: str,
    ) -> bool:
        return True


class TestCloudProvider:
    def test_fake_satisfies_protocol(self) -> None:
        assert isinstance(FakeProvider(), CloudProvider)

    def test_incomplete_fails_protocol(self) -> None:
        class Incomplete:
            pass

        assert not isinstance(Incomplete(), CloudProvider)

    def test_partial_implementation_fails_protocol(self) -> None:
        class Partial:
            def create_pod(self, name: str, gpu_type_id: GpuTypeId, image: str) -> PodInfo:
                return _POD_INFO

        assert not isinstance(Partial(), CloudProvider)


class TestSSHExecutor:
    def test_fake_satisfies_protocol(self) -> None:
        assert isinstance(FakeSSH(), SSHExecutor)

    def test_incomplete_fails_protocol(self) -> None:
        class Incomplete:
            pass

        assert not isinstance(Incomplete(), SSHExecutor)

    def test_partial_implementation_fails_protocol(self) -> None:
        class Partial:
            def run_command(self, conn: SSHConnectionInfo, command: str) -> SSHResult:
                return _SSH_RESULT

        assert not isinstance(Partial(), SSHExecutor)
