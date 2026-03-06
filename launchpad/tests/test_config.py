from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from launchpad.config import (
    DEFAULT_GRAPHQL_URL,
    DEFAULT_IMAGE,
    DEFAULT_REST_URL,
    LaunchpadConfig,
)
from launchpad.types import CloudType, DataCenterId, GpuTypeId
from pydantic import ValidationError


class TestLaunchpadConfigDefaults:
    def test_defaults_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
        monkeypatch.delenv("HF_TOKEN", raising=False)
        cfg = LaunchpadConfig.load(config_path=Path("/nonexistent/path.yaml"))

        assert cfg.api_key.get_secret_value() == "test-key"
        assert cfg.hf_token is None
        assert cfg.default_gpu == GpuTypeId("NVIDIA A40")
        assert cfg.default_cloud_type == CloudType.COMMUNITY
        assert cfg.default_data_center == DataCenterId("US-TX-3")
        assert cfg.default_image == DEFAULT_IMAGE
        assert cfg.rest_base_url == DEFAULT_REST_URL
        assert cfg.graphql_url == DEFAULT_GRAPHQL_URL


class TestLaunchpadConfigFromEnv:
    def test_loads_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RUNPOD_API_KEY", "env-key")
        cfg = LaunchpadConfig.load(config_path=Path("/nonexistent/path.yaml"))
        assert cfg.api_key.get_secret_value() == "env-key"

    def test_loads_hf_token_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RUNPOD_API_KEY", "k")
        monkeypatch.setenv("HF_TOKEN", "hf-token")
        cfg = LaunchpadConfig.load(config_path=Path("/nonexistent/path.yaml"))
        assert cfg.hf_token is not None
        assert cfg.hf_token.get_secret_value() == "hf-token"


class TestLaunchpadConfigFromYaml:
    def test_loads_from_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"api_key": "yaml-key", "default_image": "custom:latest"}))
        cfg = LaunchpadConfig.load(config_path=config_file)
        assert cfg.api_key.get_secret_value() == "yaml-key"
        assert cfg.default_image == "custom:latest"

    def test_env_overrides_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"api_key": "yaml-key"}))
        monkeypatch.setenv("RUNPOD_API_KEY", "env-key")
        cfg = LaunchpadConfig.load(config_path=config_file)
        assert cfg.api_key.get_secret_value() == "env-key"


class TestLaunchpadConfigValidation:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with pytest.raises(Exception, match="api_key"):
            LaunchpadConfig.load(config_path=Path("/nonexistent/path.yaml"))

    def test_frozen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RUNPOD_API_KEY", "k")
        cfg = LaunchpadConfig.load(config_path=Path("/nonexistent/path.yaml"))
        with pytest.raises(ValidationError):
            cfg.api_key = "new"  # type: ignore[assignment]
