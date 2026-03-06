from __future__ import annotations

from collections.abc import Iterator

from launchpad.types import SSHConnectionInfo, SSHResult
from launchpad.workspace import (
    WORKSPACE_DIR,
    bootstrap_workspace,
    sync_workspace,
)


class FakeSSH:
    def __init__(
        self,
        file_exists_results: dict[str, bool] | None = None,
        command_results: dict[str, SSHResult] | None = None,
    ) -> None:
        self.commands: list[str] = []
        self._file_exists = file_exists_results or {}
        self._command_results = command_results or {}
        self._default_result = SSHResult(exit_code=0, stdout="", stderr="")

    def run_command(
        self,
        conn: SSHConnectionInfo,
        command: str,
        *,
        timeout: int | None = None,
    ) -> SSHResult:
        self.commands.append(command)
        return self._command_results.get(command, self._default_result)

    def run_background(self, conn: SSHConnectionInfo, command: str) -> None:
        self.commands.append(command)

    def tail_follow(self, conn: SSHConnectionInfo, file_path: str) -> Iterator[str]:
        return iter([])

    def file_exists(self, conn: SSHConnectionInfo, remote_path: str) -> bool:
        return self._file_exists.get(remote_path, False)


class TestBootstrapWorkspace:
    def test_clones_when_not_exists(self) -> None:
        conn = SSHConnectionInfo(host="1.2.3.4", port=22, user="root")
        ssh = FakeSSH(
            file_exists_results={f"{WORKSPACE_DIR}/.git": False},
            command_results={"which uv": SSHResult(0, "/usr/bin/uv", "")},
        )
        bootstrap_workspace(conn, ssh)
        assert any("git clone" in cmd for cmd in ssh.commands)

    def test_pulls_when_exists(self) -> None:
        conn = SSHConnectionInfo(host="1.2.3.4", port=22, user="root")
        ssh = FakeSSH(
            file_exists_results={f"{WORKSPACE_DIR}/.git": True},
            command_results={"which uv": SSHResult(0, "/usr/bin/uv", "")},
        )
        bootstrap_workspace(conn, ssh)
        assert any("git pull" in cmd for cmd in ssh.commands)
        assert not any("git clone" in cmd for cmd in ssh.commands)

    def test_installs_uv_when_missing(self) -> None:
        conn = SSHConnectionInfo(host="1.2.3.4", port=22, user="root")
        ssh = FakeSSH(
            file_exists_results={f"{WORKSPACE_DIR}/.git": True},
            command_results={"which uv": SSHResult(1, "", "not found")},
        )
        bootstrap_workspace(conn, ssh)
        assert any("curl" in cmd and "uv" in cmd for cmd in ssh.commands)

    def test_syncs_deps(self) -> None:
        conn = SSHConnectionInfo(host="1.2.3.4", port=22, user="root")
        ssh = FakeSSH(
            file_exists_results={f"{WORKSPACE_DIR}/.git": True},
            command_results={"which uv": SSHResult(0, "/usr/bin/uv", "")},
        )
        bootstrap_workspace(conn, ssh)
        assert any("uv sync" in cmd for cmd in ssh.commands)


class TestSyncWorkspace:
    def test_pulls_and_syncs(self) -> None:
        conn = SSHConnectionInfo(host="1.2.3.4", port=22, user="root")
        ssh = FakeSSH()
        sync_workspace(conn, ssh)
        assert any("git pull" in cmd and "uv sync" in cmd for cmd in ssh.commands)
