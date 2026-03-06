from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from .types import (
    DataCenterId,
    GpuType,
    GpuTypeId,
    PodId,
    PodInfo,
    SSHConnectionInfo,
    SSHResult,
    VolumeId,
    VolumeInfo,
)


@runtime_checkable
class CloudProvider(Protocol):
    def create_pod(
        self,
        name: str,
        gpu_type_id: GpuTypeId,
        image: str,
        *,
        volume_id: VolumeId | None = None,
        data_center_id: DataCenterId | None = None,
        gpu_count: int = 1,
    ) -> PodInfo: ...

    def stop_pod(self, pod_id: PodId) -> PodInfo: ...

    def resume_pod(self, pod_id: PodId) -> PodInfo: ...

    def terminate_pod(self, pod_id: PodId) -> None: ...

    def get_pod(self, pod_id: PodId) -> PodInfo: ...

    def create_volume(
        self,
        name: str,
        size_gb: int,
        data_center_id: DataCenterId,
    ) -> VolumeInfo: ...

    def list_volumes(self) -> list[VolumeInfo]: ...

    def get_volume(self, volume_id: VolumeId) -> VolumeInfo: ...

    def delete_volume(self, volume_id: VolumeId) -> None: ...

    def list_gpu_types(self) -> list[GpuType]: ...

    def ssh_info(self, pod_id: PodId) -> SSHConnectionInfo: ...


@runtime_checkable
class SSHExecutor(Protocol):
    def run_command(
        self,
        conn: SSHConnectionInfo,
        command: str,
        *,
        timeout: int | None = None,
    ) -> SSHResult: ...

    def run_background(
        self,
        conn: SSHConnectionInfo,
        command: str,
    ) -> None: ...

    def tail_follow(
        self,
        conn: SSHConnectionInfo,
        file_path: str,
    ) -> Iterator[str]: ...

    def file_exists(
        self,
        conn: SSHConnectionInfo,
        remote_path: str,
    ) -> bool: ...
