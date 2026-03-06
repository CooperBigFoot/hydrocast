from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterator
from pathlib import Path

from .errors import SshError
from .types import SSHConnectionInfo, SSHResult

log = logging.getLogger(__name__)


class SubprocessSSH:
    def __init__(self, key_path: Path | None = None) -> None:
        self._key_path = key_path

    def _base_args(self, conn: SSHConnectionInfo) -> list[str]:
        args = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-p",
            str(conn.port),
        ]
        if self._key_path:
            args.extend(["-i", str(self._key_path)])
        args.append(f"{conn.user}@{conn.host}")
        return args

    def run_command(
        self,
        conn: SSHConnectionInfo,
        command: str,
        *,
        timeout: int | None = None,
    ) -> SSHResult:
        args = [*self._base_args(conn), command]
        log.info("SSH run: %s", command)
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise SshError(command, f"Timed out after {timeout}s") from e
        except OSError as e:
            raise SshError(command, str(e)) from e
        return SSHResult(
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def run_background(self, conn: SSHConnectionInfo, command: str) -> None:
        nohup_cmd = f"nohup {command} > /dev/null 2>&1 &"
        args = [*self._base_args(conn), nohup_cmd]
        log.info("SSH background: %s", command)
        try:
            subprocess.run(args, capture_output=True, text=True, timeout=30)
        except (subprocess.TimeoutExpired, OSError) as e:
            raise SshError(command, str(e)) from e

    def tail_follow(self, conn: SSHConnectionInfo, file_path: str) -> Iterator[str]:
        args = [*self._base_args(conn), f"tail -f {file_path}"]
        log.info("SSH tail: %s", file_path)
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                yield line.rstrip("\n")
        finally:
            proc.terminate()
            proc.wait()

    def file_exists(self, conn: SSHConnectionInfo, remote_path: str) -> bool:
        result = self.run_command(conn, f"test -e {remote_path} && echo yes || echo no")
        return result.stdout.strip() == "yes"
