from __future__ import annotations

import logging
from pathlib import Path

import typer

from .config import LaunchpadConfig
from .errors import LaunchpadError
from .output import console, print_error, print_pod_info, print_status, print_success, print_volumes_table
from .types import DataCenterId, GpuTypeId, HfRepoId, PodId

log = logging.getLogger(__name__)

app = typer.Typer(name="launchpad", no_args_is_help=True)
volume_app = typer.Typer(name="volume", no_args_is_help=True, help="Manage RunPod network volumes.")
app.add_typer(volume_app, name="volume")


def _find_dotenv() -> None:
    from pathlib import Path

    from dotenv import load_dotenv

    cwd = Path.cwd()
    for directory in [cwd, *cwd.parents]:
        env_file = directory / ".env"
        if env_file.is_file():
            load_dotenv(env_file)
            return


def _load_config() -> LaunchpadConfig:
    try:
        _find_dotenv()
        return LaunchpadConfig.load()
    except LaunchpadError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e


def _make_provider(config: LaunchpadConfig):  # noqa: ANN202
    from .providers.runpod import RunPodProvider

    return RunPodProvider(config)


def _make_ssh(config: LaunchpadConfig):  # noqa: ANN202
    from .ssh import SubprocessSSH

    return SubprocessSSH(key_path=config.ssh_key_path.expanduser())


@volume_app.command("create")
def volume_create(
    name: str = typer.Option(..., help="Volume name"),
    size: int = typer.Option(..., help="Size in GB"),
    region: str = typer.Option(..., help="Data center region"),
) -> None:
    """Create a new network volume."""
    config = _load_config()
    provider = _make_provider(config)
    try:
        vol = provider.create_volume(name, size, DataCenterId(region))
        print_success(f"Volume '{vol.name}' created (ID: {vol.id}, {vol.size_gb}GB)")
    except LaunchpadError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e


@volume_app.command("ls")
def volume_ls(
    volume_name: str | None = typer.Argument(None, help="Filter by volume name"),
) -> None:
    """List network volumes."""
    config = _load_config()
    provider = _make_provider(config)
    try:
        volumes = provider.list_volumes()
        if volume_name:
            volumes = [v for v in volumes if v.name == volume_name]
        print_volumes_table(volumes)
    except LaunchpadError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e


@volume_app.command("upload")
def volume_upload(
    local_path: str = typer.Argument(..., help="Local path to upload"),
    remote: str = typer.Argument(..., help="volume-name:remote-path"),
) -> None:
    """Upload data to a network volume."""
    config = _load_config()
    provider = _make_provider(config)
    try:
        if ":" not in remote:
            print_error("Remote must be in format volume-name:remote-path")
            raise typer.Exit(code=1)
        volume_name, dataset_name = remote.split(":", 1)
        volumes = provider.list_volumes()
        vol = next((v for v in volumes if v.name == volume_name), None)
        if not vol:
            print_error(f"Volume '{volume_name}' not found")
            raise typer.Exit(code=1)

        from .sync import upload_dataset

        pod = provider.create_pod(
            name=f"upload-{volume_name}",
            gpu_type_id=config.default_gpu,
            image=config.default_image,
            volume_id=vol.id,
        )
        conn = provider.ssh_info(pod.id)
        upload_dataset(Path(local_path), dataset_name, conn, key_path=config.ssh_key_path.expanduser())
        provider.terminate_pod(pod.id)
        print_success(f"Uploaded {local_path} to {remote}")
    except LaunchpadError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e


@volume_app.command("delete")
def volume_delete(
    volume_name: str = typer.Argument(..., help="Volume name to delete"),
) -> None:
    """Delete a network volume."""
    config = _load_config()
    provider = _make_provider(config)
    try:
        volumes = provider.list_volumes()
        vol = next((v for v in volumes if v.name == volume_name), None)
        if not vol:
            print_error(f"Volume '{volume_name}' not found")
            raise typer.Exit(code=1)
        provider.delete_volume(vol.id)
        print_success(f"Volume '{volume_name}' deleted")
    except LaunchpadError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e


@volume_app.command("setup")
def volume_setup(
    volume_name: str = typer.Argument(..., help="Volume name to set up"),
) -> None:
    """Clone coach + uv sync on a network volume."""
    config = _load_config()
    provider = _make_provider(config)
    ssh = _make_ssh(config)
    try:
        volumes = provider.list_volumes()
        vol = next((v for v in volumes if v.name == volume_name), None)
        if not vol:
            print_error(f"Volume '{volume_name}' not found")
            raise typer.Exit(code=1)

        pod = provider.create_pod(
            name=f"setup-{volume_name}",
            gpu_type_id=config.default_gpu,
            image=config.default_image,
            volume_id=vol.id,
        )
        conn = provider.ssh_info(pod.id)

        from .workspace import bootstrap_workspace

        bootstrap_workspace(conn, ssh)
        provider.terminate_pod(pod.id)
        print_success(f"Workspace set up on volume '{volume_name}'")
    except LaunchpadError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e


@app.command()
def train(
    config: str = typer.Option(..., help="Path to training config YAML"),
    volume: str = typer.Option(..., help="Network volume name"),
    gpu: str = typer.Option(..., help="GPU type"),
    qty: int = typer.Option(1, help="Number of GPUs"),
) -> None:
    """Launch a training run on a remote GPU pod."""
    cfg = _load_config()
    provider = _make_provider(cfg)
    ssh = _make_ssh(cfg)
    try:
        volumes = provider.list_volumes()
        vol = next((v for v in volumes if v.name == volume), None)
        if not vol:
            print_error(f"Volume '{volume}' not found")
            raise typer.Exit(code=1)

        config_path = Path(config)
        run_name = config_path.stem

        from .runner import launch_training

        pod, run_id = launch_training(
            config_path,
            run_name,
            provider,
            ssh,
            gpu_type=GpuTypeId(gpu),
            image=cfg.default_image,
            volume_id=vol.id,
        )
        print_pod_info(pod)
        print_success(f"Training launched: run_id={run_id}")
    except LaunchpadError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e


@app.command()
def evaluate(
    run_dir: str = typer.Option(..., "--run-dir", help="Run directory name"),
    eval_config: str = typer.Option(..., "--eval-config", help="Evaluation config YAML"),
    volume: str = typer.Option(..., help="Network volume name"),
    gpu: str = typer.Option(..., help="GPU type"),
) -> None:
    """Evaluate a trained model on a remote GPU pod."""
    cfg = _load_config()
    provider = _make_provider(cfg)
    ssh = _make_ssh(cfg)
    try:
        volumes = provider.list_volumes()
        vol = next((v for v in volumes if v.name == volume), None)
        if not vol:
            print_error(f"Volume '{volume}' not found")
            raise typer.Exit(code=1)

        pod = provider.create_pod(
            name=f"eval-{run_dir}",
            gpu_type_id=GpuTypeId(gpu),
            image=cfg.default_image,
            volume_id=vol.id,
        )
        conn = provider.ssh_info(pod.id)

        from .runner import launch_evaluation

        result = launch_evaluation(run_dir, Path(eval_config), conn, ssh)
        provider.terminate_pod(pod.id)

        print_status(str(result.run_id), result.status.name)
        if result.error_message:
            print_error(result.error_message)
    except LaunchpadError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e


@app.command()
def sweep(
    config: str = typer.Option(..., help="Path to sweep config YAML"),
    volume: str = typer.Option(..., help="Network volume name"),
    gpu: str = typer.Option(..., help="GPU type"),
    parallel: bool = typer.Option(False, "--parallel", help="Run sweep in parallel"),
    sequential: bool = typer.Option(False, "--sequential", help="Run sweep sequentially"),
) -> None:
    """Run a hyperparameter sweep."""
    cfg = _load_config()
    provider = _make_provider(cfg)
    ssh = _make_ssh(cfg)
    try:
        volumes = provider.list_volumes()
        vol = next((v for v in volumes if v.name == volume), None)
        if not vol:
            print_error(f"Volume '{volume}' not found")
            raise typer.Exit(code=1)

        import yaml

        from .sweep import SweepConfig, run_sweep_parallel, run_sweep_sequential

        with open(config) as f:
            raw = yaml.safe_load(f)
        sweep_config = SweepConfig.model_validate(raw)

        kwargs = {
            "gpu_type": GpuTypeId(gpu),
            "image": cfg.default_image,
            "volume_id": vol.id,
        }

        if parallel:
            tracker = run_sweep_parallel(sweep_config, provider, ssh, **kwargs)
        else:
            tracker = run_sweep_sequential(sweep_config, provider, ssh, **kwargs)

        for name, st in tracker.statuses.items():
            print_status(name, st.name)
    except LaunchpadError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e


@app.command()
def status(
    run_id: str | None = typer.Argument(None, help="Run ID to check"),
) -> None:
    """Check status of a training run."""
    if not run_id:
        console.print("[yellow]Provide a run ID[/]")
        raise typer.Exit(code=1)

    cfg = _load_config()
    provider = _make_provider(cfg)
    ssh = _make_ssh(cfg)
    try:
        conn = provider.ssh_info(PodId(run_id))

        from .monitor import get_run_status

        st = get_run_status(conn, ssh, run_id)
        print_status(run_id, st.name)
    except LaunchpadError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e


@app.command()
def logs(
    run_id: str = typer.Argument(..., help="Run ID"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
) -> None:
    """Stream logs from a training run."""
    cfg = _load_config()
    provider = _make_provider(cfg)
    try:
        conn = provider.ssh_info(PodId(run_id))
        ssh = _make_ssh(cfg)

        from .monitor import stream_logs

        for line in stream_logs(conn, ssh, run_id):
            console.print(line)
    except LaunchpadError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e


@app.command()
def pull(
    run_id: str = typer.Argument(..., help="Run ID to pull"),
    output: str | None = typer.Option(None, "--output", "-o", help="Local output directory"),
) -> None:
    """Pull run artifacts from a remote pod."""
    cfg = _load_config()
    provider = _make_provider(cfg)
    try:
        conn = provider.ssh_info(PodId(run_id))
        local_dest = Path(output) if output else Path(f"./runs/{run_id}")

        from .sync import download_run

        download_run(run_id, local_dest, conn, key_path=cfg.ssh_key_path.expanduser())
        print_success(f"Downloaded run '{run_id}' to {local_dest}")
    except LaunchpadError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e


@app.command()
def push(
    run_id: str = typer.Argument(..., help="Run ID to push"),
    repo: str = typer.Option(..., help="HuggingFace repo ID"),
) -> None:
    """Push run checkpoint to HuggingFace Hub."""
    _load_config()
    try:
        from .hub import push_checkpoint

        local_dir = Path(f"./runs/{run_id}")
        url = push_checkpoint(local_dir, HfRepoId(repo))
        print_success(f"Pushed to {url}")
    except FileNotFoundError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e
    except LaunchpadError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e


@app.command()
def down(
    run_id: str = typer.Argument(..., help="Run ID to terminate"),
) -> None:
    """Terminate a running pod."""
    cfg = _load_config()
    provider = _make_provider(cfg)
    try:
        from .runner import terminate_run

        terminate_run(PodId(run_id), provider)
        print_success(f"Pod '{run_id}' terminated")
    except LaunchpadError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
