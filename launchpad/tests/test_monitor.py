from __future__ import annotations

from launchpad.monitor import check_process_running, get_run_status, read_metrics_snapshot
from launchpad.types import RunStatus, SSHConnectionInfo, SSHResult


class FakeSSH:
    def __init__(
        self,
        results: dict[str, SSHResult] | None = None,
        existing_files: set[str] | None = None,
    ) -> None:
        self._results = results or {}
        self._default = SSHResult(0, "", "")
        self._existing_files = existing_files or set()

    def run_command(
        self,
        conn: SSHConnectionInfo,
        command: str,
        *,
        timeout: int | None = None,
    ) -> SSHResult:
        for pattern, result in self._results.items():
            if pattern in command:
                return result
        return self._default

    def run_background(self, conn: SSHConnectionInfo, command: str) -> None: ...

    def tail_follow(self, conn: SSHConnectionInfo, file_path: str):  # noqa: ANN201
        return iter([])

    def file_exists(self, conn: SSHConnectionInfo, remote_path: str) -> bool:
        return remote_path in self._existing_files


CONN = SSHConnectionInfo(host="1.2.3.4", port=22, user="root")


class TestCheckProcessRunning:
    def test_true_when_process_found(self) -> None:
        ssh = FakeSSH(results={"coach train": SSHResult(0, "root 123 coach train", "")})
        assert check_process_running(CONN, ssh) is True

    def test_false_when_not_found(self) -> None:
        ssh = FakeSSH(results={"coach train": SSHResult(1, "", "")})
        assert check_process_running(CONN, ssh) is False

    def test_false_when_empty_output(self) -> None:
        ssh = FakeSSH(results={"coach train": SSHResult(0, "", "")})
        assert check_process_running(CONN, ssh) is False


class TestGetRunStatus:
    def test_running_when_process_active(self) -> None:
        ssh = FakeSSH(results={"coach train": SSHResult(0, "root 123 coach train", "")})
        assert get_run_status(CONN, ssh, "run-1") == RunStatus.RUNNING

    def test_completed_when_checkpoint_exists(self) -> None:
        ssh = FakeSSH(
            results={"coach train": SSHResult(1, "", "")},
            existing_files={"/workspace/runs/run-1/checkpoint.pt"},
        )
        assert get_run_status(CONN, ssh, "run-1") == RunStatus.COMPLETED

    def test_failed_when_log_but_no_checkpoint(self) -> None:
        ssh = FakeSSH(
            results={"coach train": SSHResult(1, "", "")},
            existing_files={"/workspace/runs/run-1/training.log"},
        )
        assert get_run_status(CONN, ssh, "run-1") == RunStatus.FAILED

    def test_pending_when_nothing_exists(self) -> None:
        ssh = FakeSSH(results={"coach train": SSHResult(1, "", "")})
        assert get_run_status(CONN, ssh, "run-1") == RunStatus.PENDING


class TestReadMetricsSnapshot:
    def test_parses_json(self) -> None:
        ssh = FakeSSH(results={"cat": SSHResult(0, '[{"loss": 0.5}]', "")})
        metrics = read_metrics_snapshot(CONN, ssh, "run-1")
        assert metrics == [{"loss": 0.5}]

    def test_returns_empty_on_missing(self) -> None:
        ssh = FakeSSH(results={"cat": SSHResult(1, "", "No such file")})
        assert read_metrics_snapshot(CONN, ssh, "run-1") == []

    def test_returns_empty_on_invalid_json(self) -> None:
        ssh = FakeSSH(results={"cat": SSHResult(0, "not json", "")})
        assert read_metrics_snapshot(CONN, ssh, "run-1") == []
