from __future__ import annotations

import pytest
from launchpad.types import (
    CloudType,
    GpuUtilization,
    InactivityConfig,
    PodId,
    PodStatus,
    RemotePath,
    RunStatus,
    SSHConnectionInfo,
    SweepMode,
)


class TestSSHConnectionInfo:
    def test_valid(self) -> None:
        info = SSHConnectionInfo(host="example.com", port=22, user="root")
        assert info.host == "example.com"
        assert info.port == 22
        assert info.user == "root"

    def test_empty_host_raises(self) -> None:
        with pytest.raises(ValueError, match="host must be non-empty"):
            SSHConnectionInfo(host="", port=22, user="root")

    def test_port_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="port 0 out of range"):
            SSHConnectionInfo(host="h", port=0, user="root")

    def test_port_too_high_raises(self) -> None:
        with pytest.raises(ValueError, match="port 65536 out of range"):
            SSHConnectionInfo(host="h", port=65536, user="root")

    def test_empty_user_raises(self) -> None:
        with pytest.raises(ValueError, match="user must be non-empty"):
            SSHConnectionInfo(host="h", port=22, user="")

    def test_boundary_ports(self) -> None:
        assert SSHConnectionInfo(host="h", port=1, user="u").port == 1
        assert SSHConnectionInfo(host="h", port=65535, user="u").port == 65535


class TestRemotePath:
    def test_valid(self) -> None:
        target = SSHConnectionInfo(host="h", port=22, user="u")
        rp = RemotePath(target=target, path="/home/user")
        assert rp.path == "/home/user"

    def test_relative_path_raises(self) -> None:
        target = SSHConnectionInfo(host="h", port=22, user="u")
        with pytest.raises(ValueError, match="path must be absolute"):
            RemotePath(target=target, path="relative/path")


class TestGpuUtilization:
    def test_valid(self) -> None:
        assert GpuUtilization(percent=50.0).percent == 50.0

    def test_boundaries(self) -> None:
        assert GpuUtilization(percent=0.0).percent == 0.0
        assert GpuUtilization(percent=100.0).percent == 100.0

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="percent -1.0 out of range"):
            GpuUtilization(percent=-1.0)

    def test_over_100_raises(self) -> None:
        with pytest.raises(ValueError, match="percent 101.0 out of range"):
            GpuUtilization(percent=101.0)


class TestInactivityConfig:
    def test_valid(self) -> None:
        cfg = InactivityConfig(timeout_minutes=10, poll_interval_seconds=30)
        assert cfg.timeout_minutes == 10
        assert cfg.poll_interval_seconds == 30

    def test_zero_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout_minutes must be > 0"):
            InactivityConfig(timeout_minutes=0, poll_interval_seconds=30)

    def test_negative_poll_raises(self) -> None:
        with pytest.raises(ValueError, match="poll_interval_seconds must be > 0"):
            InactivityConfig(timeout_minutes=10, poll_interval_seconds=-1)


class TestEnums:
    def test_pod_status_members(self) -> None:
        assert {s.name for s in PodStatus} == {
            "CREATED",
            "RUNNING",
            "EXITED",
            "TERMINATED",
        }

    def test_run_status_members(self) -> None:
        assert {s.name for s in RunStatus} == {
            "PENDING",
            "RUNNING",
            "COMPLETED",
            "FAILED",
        }

    def test_cloud_type_members(self) -> None:
        assert {c.name for c in CloudType} == {"COMMUNITY", "SECURE"}

    def test_sweep_mode_members(self) -> None:
        assert {m.name for m in SweepMode} == {"PARALLEL", "SEQUENTIAL"}


class TestNewTypes:
    def test_pod_id_callable(self) -> None:
        pid = PodId("abc123")
        assert pid == "abc123"
