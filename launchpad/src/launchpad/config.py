from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from .types import CloudType, DataCenterId, GpuTypeId

DEFAULT_IMAGE = "runpod/pytorch:2.8.0-py3.12-cuda12.8.1-cudnn-devel-ubuntu22.04"
DEFAULT_REST_URL = "https://rest.runpod.io/v1"
DEFAULT_GRAPHQL_URL = "https://api.runpod.io/graphql"


class LaunchpadConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key: SecretStr
    hf_token: SecretStr | None = None
    default_gpu: GpuTypeId = Field(default=GpuTypeId("NVIDIA A40"))
    default_cloud_type: CloudType = Field(default=CloudType.COMMUNITY)
    default_data_center: DataCenterId = Field(default=DataCenterId("US-TX-3"))
    default_image: str = Field(default=DEFAULT_IMAGE)
    ssh_key_path: Path = Field(default=Path("~/.ssh/id_ed25519"))
    rest_base_url: str = Field(default=DEFAULT_REST_URL)
    graphql_url: str = Field(default=DEFAULT_GRAPHQL_URL)

    @classmethod
    def load(cls, config_path: Path | None = None) -> LaunchpadConfig:
        import os

        from dotenv import load_dotenv

        load_dotenv()

        raw: dict[str, Any] = {}
        path = config_path or Path("~/.launchpad/config.yaml").expanduser()
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}

        if api_key := os.environ.get("RUNPOD_API_KEY"):
            raw["api_key"] = api_key
        if hf_token := os.environ.get("HF_TOKEN"):
            raw["hf_token"] = hf_token

        return cls.model_validate(raw)
