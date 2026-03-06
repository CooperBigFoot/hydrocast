from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .errors import SshError
from .types import SSHConnectionInfo

log = logging.getLogger(__name__)

DATASET_ROOT = "/workspace/datasets"
RUNS_ROOT = "/workspace/runs"


def _rsync_args(conn: SSHConnectionInfo, key_path: Path | None = None) -> list[str]:
    ssh_cmd = f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {conn.port}"
    if key_path:
        ssh_cmd += f" -i {key_path}"
    return ["rsync", "-avz", "--progress", "-e", ssh_cmd]


def upload_dataset(
    local_path: Path,
    dataset_name: str,
    conn: SSHConnectionInfo,
    *,
    key_path: Path | None = None,
) -> str:
    remote_dest = f"{DATASET_ROOT}/{dataset_name}/"
    remote = f"{conn.user}@{conn.host}:{remote_dest}"
    args = [*_rsync_args(conn, key_path), str(local_path) + "/", remote]
    log.info("Uploading dataset to %s", remote_dest)
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise SshError("rsync upload", result.stderr)
    return remote_dest


def download_run(
    run_id: str,
    local_dest: Path,
    conn: SSHConnectionInfo,
    *,
    key_path: Path | None = None,
) -> Path:
    remote_src = f"{conn.user}@{conn.host}:{RUNS_ROOT}/{run_id}/"
    local_dest.mkdir(parents=True, exist_ok=True)
    args = [*_rsync_args(conn, key_path), remote_src, str(local_dest) + "/"]
    log.info("Downloading run %s to %s", run_id, local_dest)
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise SshError("rsync download", result.stderr)
    return local_dest


def upload_file(
    local_path: Path,
    remote_path: str,
    conn: SSHConnectionInfo,
    *,
    key_path: Path | None = None,
) -> None:
    remote = f"{conn.user}@{conn.host}:{remote_path}"
    args = [*_rsync_args(conn, key_path), str(local_path), remote]
    log.info("Uploading %s to %s", local_path, remote_path)
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise SshError("rsync upload", result.stderr)
