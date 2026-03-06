from __future__ import annotations

from pathlib import Path

from launchpad.sync import DATASET_ROOT, RUNS_ROOT, _rsync_args
from launchpad.types import SSHConnectionInfo


class TestRsyncArgs:
    def test_basic_args(self) -> None:
        conn = SSHConnectionInfo(host="1.2.3.4", port=22, user="root")
        args = _rsync_args(conn)
        assert args[0] == "rsync"
        assert "-avz" in args

    def test_with_key(self) -> None:
        conn = SSHConnectionInfo(host="1.2.3.4", port=22, user="root")
        args = _rsync_args(conn, key_path=Path("/tmp/key"))
        joined = " ".join(args)
        assert "-i /tmp/key" in joined

    def test_includes_port(self) -> None:
        conn = SSHConnectionInfo(host="1.2.3.4", port=2222, user="root")
        args = _rsync_args(conn)
        joined = " ".join(args)
        assert "-p 2222" in joined

    def test_progress_flag(self) -> None:
        conn = SSHConnectionInfo(host="1.2.3.4", port=22, user="root")
        args = _rsync_args(conn)
        assert "--progress" in args


class TestConstants:
    def test_dataset_root(self) -> None:
        assert DATASET_ROOT.startswith("/workspace")

    def test_runs_root(self) -> None:
        assert RUNS_ROOT.startswith("/workspace")
