from __future__ import annotations

from unittest.mock import patch

import pytest
from launchpad.errors import SshError
from launchpad.runner import check_gpu_utilization, launch_evaluation
from launchpad.types import (
    RunStatus,
    SSHConnectionInfo,
    SSHResult,
)


class FakeSSH:
    def __init__(self, results: dict[str, SSHResult] | None = None) -> None:
        self.commands: list[str] = []
        self._results = results or {}
        self._default = SSHResult(0, "", "")

    def run_command(
        self,
        conn: SSHConnectionInfo,
        command: str,
        *,
        timeout: int | None = None,
    ) -> SSHResult:
        self.commands.append(command)
        for pattern, result in self._results.items():
            if pattern in command:
                return result
        return self._default

    def run_background(self, conn: SSHConnectionInfo, command: str) -> None:
        self.commands.append(command)

    def tail_follow(self, conn: SSHConnectionInfo, file_path: str):  # noqa: ANN201
        return iter([])

    def file_exists(self, conn: SSHConnectionInfo, remote_path: str) -> bool:
        return False


CONN = SSHConnectionInfo(host="1.2.3.4", port=22, user="root")


class TestCheckGpuUtilization:
    def test_parses_nvidia_smi_output(self) -> None:
        ssh = FakeSSH(results={"nvidia-smi": SSHResult(0, "75\n", "")})
        util = check_gpu_utilization(CONN, ssh)
        assert util.percent == 75.0

    def test_averages_multiple_gpus(self) -> None:
        ssh = FakeSSH(results={"nvidia-smi": SSHResult(0, "50\n100\n", "")})
        util = check_gpu_utilization(CONN, ssh)
        assert util.percent == 75.0

    def test_returns_zero_on_empty(self) -> None:
        ssh = FakeSSH(results={"nvidia-smi": SSHResult(0, "", "")})
        util = check_gpu_utilization(CONN, ssh)
        assert util.percent == 0.0

    def test_raises_on_failure(self) -> None:
        ssh = FakeSSH(results={"nvidia-smi": SSHResult(1, "", "error")})
        with pytest.raises(SshError, match="nvidia-smi"):
            check_gpu_utilization(CONN, ssh)

    def test_clamps_above_100(self) -> None:
        ssh = FakeSSH(results={"nvidia-smi": SSHResult(0, "105\n", "")})
        util = check_gpu_utilization(CONN, ssh)
        assert util.percent == 100.0


class TestLaunchEvaluation:
    @patch("launchpad.runner.upload_file")
    def test_returns_completed_on_success(self, _mock_upload: object, tmp_path: object) -> None:
        config = tmp_path / "eval.yaml"  # type: ignore[operator]
        config.write_text("test: true")
        ssh = FakeSSH(results={"coach evaluate": SSHResult(0, "done", "")})
        result = launch_evaluation("run-1", config, CONN, ssh)
        assert result.status == RunStatus.COMPLETED
        assert result.exit_code == 0
        assert result.error_message is None

    @patch("launchpad.runner.upload_file")
    def test_returns_failed_on_error(self, _mock_upload: object, tmp_path: object) -> None:
        config = tmp_path / "eval.yaml"  # type: ignore[operator]
        config.write_text("test: true")
        ssh = FakeSSH(results={"coach evaluate": SSHResult(1, "", "error msg")})
        result = launch_evaluation("run-1", config, CONN, ssh)
        assert result.status == RunStatus.FAILED
        assert result.error_message == "error msg"
