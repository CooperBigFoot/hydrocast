from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import HfApi

from .types import HfRepoId

log = logging.getLogger(__name__)


def push_checkpoint(
    run_dir: Path,
    repo_id: HfRepoId,
    *,
    api: HfApi | None = None,
) -> str:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    api = api or HfApi()

    api.create_repo(repo_id=repo_id, exist_ok=True)

    log.info("Pushing %s to %s", run_dir, repo_id)
    api.upload_folder(
        folder_path=str(run_dir),
        repo_id=repo_id,
    )

    return f"https://huggingface.co/{repo_id}"


def pull_checkpoint(
    repo_id: HfRepoId,
    local_dest: Path,
    *,
    api: HfApi | None = None,
) -> Path:
    api = api or HfApi()

    log.info("Pulling %s to %s", repo_id, local_dest)
    snapshot_path = api.snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dest),
    )

    return Path(snapshot_path)
