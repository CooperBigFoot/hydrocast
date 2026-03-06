from __future__ import annotations


class LaunchpadError(Exception): ...


class LaunchpadConfigError(LaunchpadError): ...


class RunPodApiError(LaunchpadError):
    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"RunPod API error {status_code}: {message}")


class PodNotFoundError(LaunchpadError):
    def __init__(self, pod_id: str) -> None:
        self.pod_id = pod_id
        super().__init__(f"Pod not found: {pod_id}")


class VolumeNotFoundError(LaunchpadError):
    def __init__(self, volume_id: str) -> None:
        self.volume_id = volume_id
        super().__init__(f"Volume not found: {volume_id}")


class GpuUnavailableError(LaunchpadError):
    def __init__(self, gpu_type_id: str) -> None:
        self.gpu_type_id = gpu_type_id
        super().__init__(f"GPU unavailable: {gpu_type_id}")


class SshError(LaunchpadError):
    def __init__(self, command: str, message: str) -> None:
        self.command = command
        super().__init__(f"SSH error running '{command}': {message}")
