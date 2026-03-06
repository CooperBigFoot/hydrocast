from __future__ import annotations

from pathlib import Path

import yaml
from launchpad.sweep import (
    SweepConfig,
    SweepTracker,
    deep_merge_overrides,
    expand_grid,
    generate_variant_configs,
)
from launchpad.types import RunStatus, SweepId, SweepMode


class TestExpandGrid:
    def test_cartesian_product(self) -> None:
        config = SweepConfig(
            base=Path("dummy.yaml"),
            sweep={"lr": [0.001, 0.01], "batch_size": [32, 64]},
        )
        grid = expand_grid(config)
        assert len(grid) == 4
        assert {"lr": 0.001, "batch_size": 32} in grid
        assert {"lr": 0.01, "batch_size": 64} in grid

    def test_single_param(self) -> None:
        config = SweepConfig(
            base=Path("dummy.yaml"),
            sweep={"lr": [0.001, 0.01, 0.1]},
        )
        grid = expand_grid(config)
        assert len(grid) == 3

    def test_empty_sweep(self) -> None:
        config = SweepConfig(base=Path("dummy.yaml"), sweep={})
        grid = expand_grid(config)
        assert grid == [{}]

    def test_default_mode_is_sequential(self) -> None:
        config = SweepConfig(base=Path("dummy.yaml"), sweep={})
        assert config.mode == SweepMode.SEQUENTIAL


class TestDeepMergeOverrides:
    def test_flat_override(self) -> None:
        base = {"a": 1, "b": 2}
        result = deep_merge_overrides(base, {"a": 10})
        assert result == {"a": 10, "b": 2}

    def test_dotted_key(self) -> None:
        base = {"training": {"lr": 0.001, "epochs": 100}}
        result = deep_merge_overrides(base, {"training.lr": 0.01})
        assert result["training"]["lr"] == 0.01
        assert result["training"]["epochs"] == 100

    def test_nested_dotted_key(self) -> None:
        base = {"model": {"encoder": {"hidden": 64}}}
        result = deep_merge_overrides(base, {"model.encoder.hidden": 128})
        assert result["model"]["encoder"]["hidden"] == 128

    def test_does_not_mutate_original(self) -> None:
        base = {"a": {"b": 1}}
        deep_merge_overrides(base, {"a.b": 2})
        assert base["a"]["b"] == 1

    def test_creates_intermediate_keys(self) -> None:
        base: dict = {}
        result = deep_merge_overrides(base, {"a.b.c": 42})
        assert result["a"]["b"]["c"] == 42


class TestGenerateVariantConfigs:
    def test_generates_named_variants(self, tmp_path: Path) -> None:
        base_config = {"training": {"lr": 0.001, "batch_size": 32}}
        config_file = tmp_path / "base.yaml"
        config_file.write_text(yaml.dump(base_config))

        sweep_config = SweepConfig(
            base=config_file,
            sweep={"training.lr": [0.001, 0.01]},
        )
        variants = generate_variant_configs(sweep_config)
        assert len(variants) == 2
        names = [name for name, _ in variants]
        assert any("lr=0.001" in n for n in names)
        assert any("lr=0.01" in n for n in names)

    def test_variant_configs_contain_overridden_values(self, tmp_path: Path) -> None:
        base_config = {"training": {"lr": 0.001}}
        config_file = tmp_path / "base.yaml"
        config_file.write_text(yaml.dump(base_config))

        sweep_config = SweepConfig(
            base=config_file,
            sweep={"training.lr": [0.1]},
        )
        variants = generate_variant_configs(sweep_config)
        _, config = variants[0]
        assert config["training"]["lr"] == 0.1


class TestSweepTracker:
    def test_initial_status_pending(self) -> None:
        tracker = SweepTracker(SweepId("sweep-1"), ["v1", "v2"])
        assert tracker.get("v1") == RunStatus.PENDING

    def test_update_and_get(self) -> None:
        tracker = SweepTracker(SweepId("sweep-1"), ["v1", "v2"])
        tracker.update("v1", RunStatus.RUNNING)
        assert tracker.get("v1") == RunStatus.RUNNING

    def test_all_terminal_when_done(self) -> None:
        tracker = SweepTracker(SweepId("sweep-1"), ["v1", "v2"])
        tracker.update("v1", RunStatus.COMPLETED)
        tracker.update("v2", RunStatus.FAILED)
        assert tracker.all_terminal()

    def test_not_terminal_when_running(self) -> None:
        tracker = SweepTracker(SweepId("sweep-1"), ["v1", "v2"])
        tracker.update("v1", RunStatus.COMPLETED)
        tracker.update("v2", RunStatus.RUNNING)
        assert not tracker.all_terminal()

    def test_statuses_returns_copy(self) -> None:
        tracker = SweepTracker(SweepId("sweep-1"), ["v1"])
        statuses = tracker.statuses
        statuses["v1"] = RunStatus.COMPLETED
        assert tracker.get("v1") == RunStatus.PENDING
