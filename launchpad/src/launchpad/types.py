from __future__ import annotations

from enum import Enum, auto
from typing import NamedTuple, NewType

PodId = NewType("PodId", str)
VolumeId = NewType("VolumeId", str)
GpuTypeId = NewType("GpuTypeId", str)
DataCenterId = NewType("DataCenterId", str)
RunId = NewType("RunId", str)
SweepId = NewType("SweepId", str)
HfRepoId = NewType("HfRepoId", str)


class PodStatus(Enum):
    CREATED = auto()
    RUNNING = auto()
    EXITED = auto()
    TERMINATED = auto()


class RunStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


class CloudType(Enum):
    COMMUNITY = auto()
    SECURE = auto()


class SweepMode(Enum):
    PARALLEL = auto()
    SEQUENTIAL = auto()


# Validated NamedTuples use a base+subclass pattern because
# Python 3.13 disallows __new__ override directly on NamedTuple classes.


class _SSHConnectionInfoBase(NamedTuple):
    host: str
    port: int
    user: str


class SSHConnectionInfo(_SSHConnectionInfoBase):
    def __new__(cls, host: str, port: int, user: str) -> SSHConnectionInfo:
        if not host:
            raise ValueError("host must be non-empty")
        if not (1 <= port <= 65535):
            raise ValueError(f"port {port} out of range 1-65535")
        if not user:
            raise ValueError("user must be non-empty")
        return super().__new__(cls, host, port, user)


class _RemotePathBase(NamedTuple):
    target: SSHConnectionInfo
    path: str


class RemotePath(_RemotePathBase):
    def __new__(cls, target: SSHConnectionInfo, path: str) -> RemotePath:
        if not path.startswith("/"):
            raise ValueError(f"path must be absolute, got: {path}")
        return super().__new__(cls, target, path)


class _GpuUtilizationBase(NamedTuple):
    percent: float


class GpuUtilization(_GpuUtilizationBase):
    def __new__(cls, percent: float) -> GpuUtilization:
        if not (0 <= percent <= 100):
            raise ValueError(f"percent {percent} out of range 0-100")
        return super().__new__(cls, percent)


class _InactivityConfigBase(NamedTuple):
    timeout_minutes: int
    poll_interval_seconds: int


class InactivityConfig(_InactivityConfigBase):
    def __new__(cls, timeout_minutes: int, poll_interval_seconds: int) -> InactivityConfig:
        if timeout_minutes <= 0:
            raise ValueError(f"timeout_minutes must be > 0, got {timeout_minutes}")
        if poll_interval_seconds <= 0:
            raise ValueError(f"poll_interval_seconds must be > 0, got {poll_interval_seconds}")
        return super().__new__(cls, timeout_minutes, poll_interval_seconds)


class GpuPricing(NamedTuple):
    min_bid: float
    uninterruptable: float


class GpuType(NamedTuple):
    id: GpuTypeId
    name: str
    memory_mb: int
    pricing: GpuPricing


class PortMapping(NamedTuple):
    internal_port: int
    external_port: int


class PodInfo(NamedTuple):
    id: PodId
    name: str
    status: PodStatus
    cost_per_hr: float
    gpu: GpuTypeId
    public_ip: str | None
    ports: tuple[PortMapping, ...]


class VolumeInfo(NamedTuple):
    id: VolumeId
    name: str
    size_gb: int
    data_center_id: DataCenterId


class RunResult(NamedTuple):
    run_id: RunId
    status: RunStatus
    exit_code: int | None
    error_message: str | None


class SSHResult(NamedTuple):
    exit_code: int
    stdout: str
    stderr: str
