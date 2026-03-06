from __future__ import annotations

from pathlib import Path

from launchpad.protocols import SSHExecutor
from launchpad.ssh import SubprocessSSH
from launchpad.types import SSHConnectionInfo


class TestSubprocessSSH:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(SubprocessSSH(), SSHExecutor)

    def test_base_args_without_key(self) -> None:
        ssh = SubprocessSSH()
        conn = SSHConnectionInfo(host="1.2.3.4", port=22, user="root")
        args = ssh._base_args(conn)
        assert args[0] == "ssh"
        assert "-p" in args
        assert "22" in args
        assert "root@1.2.3.4" in args

    def test_base_args_with_key(self) -> None:
        ssh = SubprocessSSH(key_path=Path("/tmp/key"))
        conn = SSHConnectionInfo(host="1.2.3.4", port=22, user="root")
        args = ssh._base_args(conn)
        assert "-i" in args
        assert "/tmp/key" in args

    def test_base_args_strict_host_checking_disabled(self) -> None:
        ssh = SubprocessSSH()
        conn = SSHConnectionInfo(host="example.com", port=2222, user="admin")
        args = ssh._base_args(conn)
        assert "StrictHostKeyChecking=no" in args
        assert "2222" in args
        assert "admin@example.com" in args
