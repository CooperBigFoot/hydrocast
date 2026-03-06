from __future__ import annotations

from pathlib import Path

import pytest
from launchpad.hub import pull_checkpoint, push_checkpoint
from launchpad.types import HfRepoId


class FakeHfApi:
    def __init__(self) -> None:
        self.created_repos: list[str] = []
        self.uploaded_folders: list[dict[str, str]] = []
        self.downloaded: list[dict[str, str]] = []

    def create_repo(self, repo_id: str, exist_ok: bool = False) -> None:
        self.created_repos.append(repo_id)

    def upload_folder(self, folder_path: str, repo_id: str) -> None:
        self.uploaded_folders.append({"folder_path": folder_path, "repo_id": repo_id})

    def snapshot_download(self, repo_id: str, local_dir: str) -> str:
        self.downloaded.append({"repo_id": repo_id, "local_dir": local_dir})
        return local_dir


class TestPushCheckpoint:
    def test_pushes_existing_directory(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "my-run"
        run_dir.mkdir()
        (run_dir / "checkpoint.pt").write_text("fake")

        api = FakeHfApi()
        url = push_checkpoint(run_dir, HfRepoId("org/model"), api=api)  # type: ignore[arg-type]

        assert url == "https://huggingface.co/org/model"
        assert api.created_repos == ["org/model"]
        assert len(api.uploaded_folders) == 1
        assert api.uploaded_folders[0]["repo_id"] == "org/model"

    def test_raises_on_missing_directory(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError, match="Run directory not found"):
            push_checkpoint(missing, HfRepoId("org/model"))


class TestPullCheckpoint:
    def test_downloads_to_local_dest(self, tmp_path: Path) -> None:
        api = FakeHfApi()
        dest = tmp_path / "downloaded"
        dest.mkdir()

        result = pull_checkpoint(HfRepoId("org/model"), dest, api=api)  # type: ignore[arg-type]

        assert result == dest
        assert len(api.downloaded) == 1
        assert api.downloaded[0]["repo_id"] == "org/model"
