from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from .errors import SshError
from .protocols import CloudProvider, SSHExecutor
from .sync import RUNS_ROOT, upload_file
from .types import (
    GpuTypeId,
    GpuUtilization,
    InactivityConfig,
    PodId,
    PodInfo,
    RunId,
    RunResult,
    RunStatus,
    SSHConnectionInfo,
    VolumeId,
)
from .workspace import bootstrap_workspace

log = logging.getLogger(__name__)

DEFAULT_INACTIVITY = InactivityConfig(timeout_minutes=30, poll_interval_seconds=60)


def launch_training(
    config_path: Path,
    run_name: str,
    provider: CloudProvider,
    ssh: SSHExecutor,
    *,
    gpu_type: GpuTypeId,
    image: str,
    volume_id: VolumeId,
    inactivity: InactivityConfig = DEFAULT_INACTIVITY,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> tuple[PodInfo, RunId]:
    run_id = RunId(run_name)

    pod = provider.create_pod(
        name=f"train-{run_name}",
        gpu_type_id=gpu_type,
        image=image,
        volume_id=volume_id,
    )
    log.info("Pod %s created, waiting for SSH...", pod.id)
    conn = _wait_for_ssh(pod.id, provider, ssh)

    bootstrap_workspace(conn, ssh)

    remote_config = f"{RUNS_ROOT}/{run_name}/config.yaml"
    ssh.run_command(conn, f"mkdir -p {RUNS_ROOT}/{run_name}")
    upload_file(config_path, remote_config, conn)

    train_cmd = (
        f"cd /workspace/hydrocast && "
        f"uv run coach train {remote_config} {RUNS_ROOT}/{run_name} "
        f"> {RUNS_ROOT}/{run_name}/training.log 2>&1"
    )
    ssh.run_background(conn, train_cmd)
    log.info("Training started for run %s", run_id)

    watchdog = threading.Thread(
        target=_inactivity_watchdog,
        args=(conn, provider, ssh, pod.id, inactivity),
        kwargs={"clock": clock},
        daemon=True,
    )
    watchdog.start()

    return pod, run_id


def launch_evaluation(
    run_name: str,
    eval_config_path: Path,
    conn: SSHConnectionInfo,
    ssh: SSHExecutor,
) -> RunResult:
    remote_eval_config = f"{RUNS_ROOT}/{run_name}/eval_config.yaml"
    upload_file(eval_config_path, remote_eval_config, conn)

    eval_cmd = (
        f"cd /workspace/hydrocast && uv run coach evaluate {RUNS_ROOT}/{run_name} --eval-config {remote_eval_config}"
    )
    result = ssh.run_command(conn, eval_cmd, timeout=3600)

    status = RunStatus.COMPLETED if result.exit_code == 0 else RunStatus.FAILED
    return RunResult(
        run_id=RunId(run_name),
        status=status,
        exit_code=result.exit_code,
        error_message=result.stderr if result.exit_code != 0 else None,
    )


def check_gpu_utilization(conn: SSHConnectionInfo, ssh: SSHExecutor) -> GpuUtilization:
    result = ssh.run_command(
        conn,
        "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits",
    )
    if result.exit_code != 0:
        raise SshError("nvidia-smi", result.stderr)
    lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
    if not lines:
        return GpuUtilization(percent=0.0)
    avg = sum(float(line) for line in lines) / len(lines)
    return GpuUtilization(percent=min(max(avg, 0.0), 100.0))


def terminate_run(pod_id: PodId, provider: CloudProvider) -> None:
    log.info("Terminating pod %s", pod_id)
    provider.terminate_pod(pod_id)


def _wait_for_ssh(
    pod_id: PodId,
    provider: CloudProvider,
    ssh: SSHExecutor,
    *,
    max_attempts: int = 30,
    delay: float = 10.0,
) -> SSHConnectionInfo:
    for attempt in range(max_attempts):
        try:
            conn = provider.ssh_info(pod_id)
            result = ssh.run_command(conn, "echo ok", timeout=10)
            if result.exit_code == 0:
                log.info("SSH connected to pod %s", pod_id)
                return conn
        except Exception:
            pass
        log.info("Waiting for SSH (attempt %d/%d)...", attempt + 1, max_attempts)
        time.sleep(delay)
    raise SshError(
        "ssh connect",
        f"Failed to connect to pod {pod_id} after {max_attempts} attempts",
    )


def _inactivity_watchdog(
    conn: SSHConnectionInfo,
    provider: CloudProvider,
    ssh: SSHExecutor,
    pod_id: PodId,
    config: InactivityConfig,
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> None:
    last_active = clock()
    timeout_seconds = config.timeout_minutes * 60

    while True:
        time.sleep(config.poll_interval_seconds)
        try:
            util = check_gpu_utilization(conn, ssh)
            if util.percent > 5.0:
                last_active = clock()
            elif (clock() - last_active).total_seconds() > timeout_seconds:
                log.warning("Inactivity timeout reached, terminating pod %s", pod_id)
                terminate_run(pod_id, provider)
                return
        except SshError:
            log.warning("SSH error during watchdog check, terminating pod %s", pod_id)
            terminate_run(pod_id, provider)
            return
