from __future__ import annotations

import logging

from .errors import SshError
from .protocols import SSHExecutor
from .types import SSHConnectionInfo

log = logging.getLogger(__name__)

WORKSPACE_DIR = "/workspace/hydrocast"
REPO_URL = "https://github.com/CooperBigFoot/hydrocast.git"


def bootstrap_workspace(conn: SSHConnectionInfo, ssh: SSHExecutor) -> None:
    if ssh.file_exists(conn, f"{WORKSPACE_DIR}/.git"):
        log.info("Workspace already exists, pulling latest")
        result = ssh.run_command(conn, f"cd {WORKSPACE_DIR} && git pull")
        if result.exit_code != 0:
            raise SshError("git pull", result.stderr)
    else:
        log.info("Cloning workspace to %s", WORKSPACE_DIR)
        result = ssh.run_command(conn, f"git clone {REPO_URL} {WORKSPACE_DIR}")
        if result.exit_code != 0:
            raise SshError("git clone", result.stderr)

    _ensure_uv(conn, ssh)
    _sync_deps(conn, ssh)


def _ensure_uv(conn: SSHConnectionInfo, ssh: SSHExecutor) -> None:
    result = ssh.run_command(conn, "which uv")
    if result.exit_code != 0:
        log.info("Installing uv on remote")
        result = ssh.run_command(conn, "curl -LsSf https://astral.sh/uv/install.sh | sh")
        if result.exit_code != 0:
            raise SshError("uv install", result.stderr)


def _sync_deps(conn: SSHConnectionInfo, ssh: SSHExecutor) -> None:
    log.info("Syncing dependencies")
    result = ssh.run_command(conn, f"cd {WORKSPACE_DIR} && uv sync", timeout=300)
    if result.exit_code != 0:
        raise SshError("uv sync", result.stderr)


def sync_workspace(conn: SSHConnectionInfo, ssh: SSHExecutor) -> None:
    result = ssh.run_command(conn, f"cd {WORKSPACE_DIR} && git pull && uv sync", timeout=300)
    if result.exit_code != 0:
        raise SshError("sync workspace", result.stderr)
