from __future__ import annotations

import copy
import itertools
import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from .protocols import CloudProvider, SSHExecutor
from .runner import launch_training
from .types import (
    GpuTypeId,
    InactivityConfig,
    RunStatus,
    SweepId,
    SweepMode,
    VolumeId,
)

log = logging.getLogger(__name__)


class SweepConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    base: Path
    sweep: dict[str, list[Any]]
    mode: SweepMode = SweepMode.SEQUENTIAL


def expand_grid(sweep_config: SweepConfig) -> list[dict[str, Any]]:
    keys = list(sweep_config.sweep.keys())
    values = list(sweep_config.sweep.values())
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


def deep_merge_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        parts = key.split(".")
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result


def generate_variant_configs(
    sweep_config: SweepConfig,
) -> list[tuple[str, dict[str, Any]]]:
    with open(sweep_config.base) as f:
        base_config = yaml.safe_load(f)

    grid = expand_grid(sweep_config)
    variants: list[tuple[str, dict[str, Any]]] = []
    for overrides in grid:
        name_parts = [f"{k.split('.')[-1]}={v}" for k, v in overrides.items()]
        variant_name = "_".join(name_parts)
        merged = deep_merge_overrides(base_config, overrides)
        variants.append((variant_name, merged))
    return variants


class SweepTracker:
    def __init__(self, sweep_id: SweepId, variant_names: list[str]) -> None:
        self.sweep_id = sweep_id
        self._statuses: dict[str, RunStatus] = dict.fromkeys(variant_names, RunStatus.PENDING)

    def update(self, variant_name: str, status: RunStatus) -> None:
        self._statuses[variant_name] = status

    def get(self, variant_name: str) -> RunStatus:
        return self._statuses[variant_name]

    def all_terminal(self) -> bool:
        terminal = {RunStatus.COMPLETED, RunStatus.FAILED}
        return all(s in terminal for s in self._statuses.values())

    @property
    def statuses(self) -> dict[str, RunStatus]:
        return dict(self._statuses)


def _run_sweep_variants(
    sweep_config: SweepConfig,
    provider: CloudProvider,
    ssh: SSHExecutor,
    *,
    gpu_type: GpuTypeId,
    image: str,
    volume_id: VolumeId,
    inactivity: InactivityConfig | None = None,
) -> SweepTracker:
    sweep_id = SweepId(f"sweep-{sweep_config.base.stem}")
    variants = generate_variant_configs(sweep_config)
    tracker = SweepTracker(sweep_id, [name for name, _ in variants])

    for variant_name, config in variants:
        config_path = Path(f"/tmp/sweep_{variant_name}.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        try:
            kwargs: dict[str, Any] = {
                "gpu_type": gpu_type,
                "image": image,
                "volume_id": volume_id,
            }
            if inactivity:
                kwargs["inactivity"] = inactivity
            launch_training(config_path, variant_name, provider, ssh, **kwargs)
            tracker.update(variant_name, RunStatus.RUNNING)
        except Exception:
            log.exception("Failed to launch variant %s", variant_name)
            tracker.update(variant_name, RunStatus.FAILED)

    return tracker


def run_sweep_parallel(
    sweep_config: SweepConfig,
    provider: CloudProvider,
    ssh: SSHExecutor,
    *,
    gpu_type: GpuTypeId,
    image: str,
    volume_id: VolumeId,
    inactivity: InactivityConfig | None = None,
) -> SweepTracker:
    return _run_sweep_variants(
        sweep_config,
        provider,
        ssh,
        gpu_type=gpu_type,
        image=image,
        volume_id=volume_id,
        inactivity=inactivity,
    )


def run_sweep_sequential(
    sweep_config: SweepConfig,
    provider: CloudProvider,
    ssh: SSHExecutor,
    *,
    gpu_type: GpuTypeId,
    image: str,
    volume_id: VolumeId,
    inactivity: InactivityConfig | None = None,
) -> SweepTracker:
    return _run_sweep_variants(
        sweep_config,
        provider,
        ssh,
        gpu_type=gpu_type,
        image=image,
        volume_id=volume_id,
        inactivity=inactivity,
    )
