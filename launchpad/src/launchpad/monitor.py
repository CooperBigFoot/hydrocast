from __future__ import annotations

import json
import logging
from collections.abc import Iterator

from .protocols import SSHExecutor
from .sync import RUNS_ROOT
from .types import RunStatus, SSHConnectionInfo

log = logging.getLogger(__name__)


def stream_logs(conn: SSHConnectionInfo, ssh: SSHExecutor, run_name: str) -> Iterator[str]:
    log_path = f"{RUNS_ROOT}/{run_name}/training.log"
    return ssh.tail_follow(conn, log_path)


def check_process_running(conn: SSHConnectionInfo, ssh: SSHExecutor) -> bool:
    result = ssh.run_command(conn, "ps aux | grep 'coach train' | grep -v grep")
    return result.exit_code == 0 and bool(result.stdout.strip())


def read_metrics_snapshot(conn: SSHConnectionInfo, ssh: SSHExecutor, run_name: str) -> list[dict]:
    metrics_path = f"{RUNS_ROOT}/{run_name}/metrics.json"
    result = ssh.run_command(conn, f"cat {metrics_path}")
    if result.exit_code != 0:
        return []
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return []


def get_run_status(conn: SSHConnectionInfo, ssh: SSHExecutor, run_name: str) -> RunStatus:
    if check_process_running(conn, ssh):
        return RunStatus.RUNNING

    checkpoint_path = f"{RUNS_ROOT}/{run_name}/checkpoint.pt"
    if ssh.file_exists(conn, checkpoint_path):
        return RunStatus.COMPLETED

    log_path = f"{RUNS_ROOT}/{run_name}/training.log"
    if ssh.file_exists(conn, log_path):
        return RunStatus.FAILED

    return RunStatus.PENDING
