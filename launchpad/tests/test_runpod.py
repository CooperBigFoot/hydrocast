from __future__ import annotations

import json

import httpx
import pytest
from launchpad.config import LaunchpadConfig
from launchpad.errors import PodNotFoundError, RunPodApiError
from launchpad.protocols import CloudProvider
from launchpad.providers.runpod import RunPodProvider
from launchpad.types import (
    DataCenterId,
    GpuTypeId,
    PodId,
    PodStatus,
)


def _make_config() -> LaunchpadConfig:
    return LaunchpadConfig(
        api_key="test-key",
        rest_base_url="https://rest.example.com/v1",
        graphql_url="https://graphql.example.com/graphql",
    )


def _mock_transport(handler):
    return httpx.MockTransport(handler)


class TestRunPodProvider:
    def test_satisfies_cloud_provider_protocol(self):
        config = _make_config()
        provider = RunPodProvider(config)
        assert isinstance(provider, CloudProvider)

    def test_create_pod(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "/pods" in str(request.url)
            body = json.loads(request.content)
            assert body["name"] == "test-pod"
            return httpx.Response(
                200,
                json={
                    "id": "pod-123",
                    "name": "test-pod",
                    "desiredStatus": "RUNNING",
                    "costPerHr": 0.5,
                    "gpuTypeId": "NVIDIA A40",
                    "publicIp": "1.2.3.4",
                    "ports": [{"privatePort": 22, "publicPort": 22022}],
                },
            )

        config = _make_config()
        client = httpx.Client(transport=_mock_transport(handler))
        provider = RunPodProvider(config, client=client)
        pod = provider.create_pod("test-pod", GpuTypeId("NVIDIA A40"), "runpod/pytorch:latest")

        assert pod.id == PodId("pod-123")
        assert pod.status == PodStatus.RUNNING
        assert pod.cost_per_hr == 0.5

    def test_get_pod_not_found(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="Not found")

        config = _make_config()
        client = httpx.Client(transport=_mock_transport(handler))
        provider = RunPodProvider(config, client=client)

        with pytest.raises(PodNotFoundError):
            provider.get_pod(PodId("missing"))

    def test_list_gpu_types(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "gpuTypes": [
                            {
                                "id": "NVIDIA A40",
                                "displayName": "A40",
                                "memoryInGb": 48,
                                "communityPrice": 0.39,
                                "securePrice": 0.69,
                            }
                        ]
                    }
                },
            )

        config = _make_config()
        client = httpx.Client(transport=_mock_transport(handler))
        provider = RunPodProvider(config, client=client)
        gpus = provider.list_gpu_types()

        assert len(gpus) == 1
        assert gpus[0].name == "A40"
        assert gpus[0].memory_mb == 48 * 1024

    def test_ssh_info(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "id": "pod-123",
                    "name": "test",
                    "desiredStatus": "RUNNING",
                    "costPerHr": 0.5,
                    "gpuTypeId": "NVIDIA A40",
                    "publicIp": "1.2.3.4",
                    "ports": [{"privatePort": 22, "publicPort": 22022}],
                },
            )

        config = _make_config()
        client = httpx.Client(transport=_mock_transport(handler))
        provider = RunPodProvider(config, client=client)
        info = provider.ssh_info(PodId("pod-123"))

        assert info.host == "1.2.3.4"
        assert info.port == 22022
        assert info.user == "root"

    def test_api_error_raised(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal error")

        config = _make_config()
        client = httpx.Client(transport=_mock_transport(handler))
        provider = RunPodProvider(config, client=client)

        with pytest.raises(RunPodApiError, match="500"):
            provider.get_pod(PodId("any"))

    def test_create_volume(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "id": "vol-123",
                    "name": body["name"],
                    "size": body["size"],
                    "dataCenterId": body["dataCenterId"],
                },
            )

        config = _make_config()
        client = httpx.Client(transport=_mock_transport(handler))
        provider = RunPodProvider(config, client=client)
        vol = provider.create_volume("test-vol", 100, DataCenterId("US-TX-3"))

        assert vol.name == "test-vol"
        assert vol.size_gb == 100

    def test_terminate_pod(self):
        called = {"terminated": False}

        def handler(request: httpx.Request) -> httpx.Response:
            called["terminated"] = True
            return httpx.Response(200, json={})

        config = _make_config()
        client = httpx.Client(transport=_mock_transport(handler))
        provider = RunPodProvider(config, client=client)
        provider.terminate_pod(PodId("pod-123"))
        assert called["terminated"]
