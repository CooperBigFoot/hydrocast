from __future__ import annotations

from rich.console import Console
from rich.table import Table

from .types import PodInfo, VolumeInfo

console = Console()
err_console = Console(stderr=True)


def print_error(message: str) -> None:
    err_console.print(f"[bold red]Error:[/] {message}")


def print_success(message: str) -> None:
    console.print(f"[bold green]OK:[/] {message}")


def print_volumes_table(volumes: list[VolumeInfo]) -> None:
    table = Table(title="Network Volumes")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Size (GB)", justify="right")
    table.add_column("Data Center")
    for v in volumes:
        table.add_row(str(v.id), v.name, str(v.size_gb), str(v.data_center_id))
    console.print(table)


def print_pod_info(pod: PodInfo) -> None:
    table = Table(title="Pod Info")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("ID", str(pod.id))
    table.add_row("Name", pod.name)
    table.add_row("Status", pod.status.name)
    table.add_row("GPU", str(pod.gpu))
    table.add_row("Cost/hr", f"${pod.cost_per_hr:.2f}")
    table.add_row("Public IP", pod.public_ip or "N/A")
    console.print(table)


def print_status(run_id: str, status_name: str) -> None:
    console.print(f"[bold]Run {run_id}:[/] {status_name}")
