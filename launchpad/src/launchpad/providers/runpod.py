from __future__ import annotations

import logging
from typing import Any

import httpx

from ..config import LaunchpadConfig
from ..errors import PodNotFoundError, RunPodApiError, VolumeNotFoundError
from ..types import (
    CloudType,
    DataCenterId,
    GpuPricing,
    GpuType,
    GpuTypeId,
    PodId,
    PodInfo,
    PodStatus,
    PortMapping,
    SSHConnectionInfo,
    VolumeId,
    VolumeInfo,
)

log = logging.getLogger(__name__)

_STATUS_MAP: dict[str, PodStatus] = {
    "CREATED": PodStatus.CREATED,
    "RUNNING": PodStatus.RUNNING,
    "EXITED": PodStatus.EXITED,
    "TERMINATED": PodStatus.TERMINATED,
}


class RunPodProvider:
    def __init__(self, config: LaunchpadConfig, *, client: httpx.Client | None = None) -> None:
        self._config = config
        self._api_key = config.api_key.get_secret_value()
        self._client = client or httpx.Client(timeout=30)
        self._rest_base = config.rest_base_url.rstrip("/")
        self._graphql_url = config.graphql_url

    def _rest_request(self, method: str, path: str, *, json: dict | None = None) -> dict[str, Any]:
        url = f"{self._rest_base}/{path.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        response = self._client.request(method, url, headers=headers, json=json)
        if response.status_code >= 400:
            raise RunPodApiError(response.status_code, response.text)
        return response.json()

    def _graphql_query(self, query: str, variables: dict | None = None) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables
        response = self._client.post(self._graphql_url, headers=headers, json=payload)
        if response.status_code >= 400:
            raise RunPodApiError(response.status_code, response.text)
        data = response.json()
        if "errors" in data:
            raise RunPodApiError(400, str(data["errors"]))
        return data["data"]

    def _parse_pod(self, raw: dict[str, Any]) -> PodInfo:
        ports = tuple(
            PortMapping(internal_port=p["privatePort"], external_port=p["publicPort"])
            for p in (raw.get("ports") or [])
            if p.get("publicPort")
        )
        status_str = raw.get("desiredStatus") or raw.get("status", "CREATED")
        return PodInfo(
            id=PodId(raw["id"]),
            name=raw.get("name", ""),
            status=_STATUS_MAP.get(status_str.upper(), PodStatus.CREATED),
            cost_per_hr=float(raw.get("costPerHr", 0)),
            gpu=GpuTypeId(raw.get("gpuTypeId", raw.get("machine", {}).get("gpuTypeId", "unknown"))),
            public_ip=raw.get("publicIp"),
            ports=ports,
        )

    def _parse_volume(self, raw: dict[str, Any]) -> VolumeInfo:
        return VolumeInfo(
            id=VolumeId(raw["id"]),
            name=raw.get("name", ""),
            size_gb=int(raw.get("size", 0)),
            data_center_id=DataCenterId(raw.get("dataCenterId", "")),
        )

    def create_pod(
        self,
        name: str,
        gpu_type_id: GpuTypeId,
        image: str,
        *,
        volume_id: VolumeId | None = None,
        data_center_id: DataCenterId | None = None,
        gpu_count: int = 1,
    ) -> PodInfo:
        cloud_type = "COMMUNITY" if self._config.default_cloud_type == CloudType.COMMUNITY else "SECURE"
        body: dict[str, Any] = {
            "name": name,
            "gpuTypeId": str(gpu_type_id),
            "imageName": image,
            "gpuCount": gpu_count,
            "cloudType": cloud_type,
            "supportPublicIp": True,
            "ports": "22/tcp,8888/http",
        }
        if volume_id:
            body["networkVolumeId"] = str(volume_id)
            body["volumeMountPath"] = "/workspace"
        if data_center_id:
            body["dataCenterId"] = str(data_center_id)

        log.info("Creating pod %s with GPU %s", name, gpu_type_id)
        data = self._rest_request("POST", "/pods", json=body)
        return self._parse_pod(data)

    def stop_pod(self, pod_id: PodId) -> PodInfo:
        log.info("Stopping pod %s", pod_id)
        data = self._rest_request("POST", f"/pods/{pod_id}/stop")
        return self._parse_pod(data)

    def resume_pod(self, pod_id: PodId) -> PodInfo:
        log.info("Resuming pod %s", pod_id)
        data = self._rest_request("POST", f"/pods/{pod_id}/start")
        return self._parse_pod(data)

    def terminate_pod(self, pod_id: PodId) -> None:
        log.info("Terminating pod %s", pod_id)
        self._rest_request("DELETE", f"/pods/{pod_id}")

    def get_pod(self, pod_id: PodId) -> PodInfo:
        try:
            data = self._rest_request("GET", f"/pods/{pod_id}")
        except RunPodApiError as e:
            if e.status_code == 404:
                raise PodNotFoundError(str(pod_id)) from e
            raise
        return self._parse_pod(data)

    def create_volume(
        self,
        name: str,
        size_gb: int,
        data_center_id: DataCenterId,
    ) -> VolumeInfo:
        log.info("Creating volume %s (%dGB) in %s", name, size_gb, data_center_id)
        body = {
            "name": name,
            "size": size_gb,
            "dataCenterId": str(data_center_id),
        }
        data = self._rest_request("POST", "/networkvolumes", json=body)
        return self._parse_volume(data)

    def list_volumes(self) -> list[VolumeInfo]:
        data = self._rest_request("GET", "/networkvolumes")
        return [self._parse_volume(v) for v in data]

    def get_volume(self, volume_id: VolumeId) -> VolumeInfo:
        try:
            data = self._rest_request("GET", f"/networkvolumes/{volume_id}")
        except RunPodApiError as e:
            if e.status_code == 404:
                raise VolumeNotFoundError(str(volume_id)) from e
            raise
        return self._parse_volume(data)

    def delete_volume(self, volume_id: VolumeId) -> None:
        log.info("Deleting volume %s", volume_id)
        self._rest_request("DELETE", f"/networkvolumes/{volume_id}")

    def list_gpu_types(self) -> list[GpuType]:
        query = """
        query GpuTypes {
            gpuTypes {
                id
                displayName
                memoryInGb
                communityPrice
                securePrice
            }
        }
        """
        data = self._graphql_query(query)
        return [
            GpuType(
                id=GpuTypeId(g["id"]),
                name=g["displayName"],
                memory_mb=int(g["memoryInGb"]) * 1024,
                pricing=GpuPricing(
                    min_bid=float(g.get("communityPrice") or 0),
                    uninterruptable=float(g.get("securePrice") or 0),
                ),
            )
            for g in data.get("gpuTypes", [])
        ]

    def ssh_info(self, pod_id: PodId) -> SSHConnectionInfo:
        pod = self.get_pod(pod_id)
        if not pod.public_ip:
            raise PodNotFoundError(f"Pod {pod_id} has no public IP")

        ssh_port = 22
        for pm in pod.ports:
            if pm.internal_port == 22:
                ssh_port = pm.external_port
                break

        return SSHConnectionInfo(host=pod.public_ip, port=ssh_port, user="root")
