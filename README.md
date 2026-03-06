# model-training

A [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) combining three packages for hydrological model training:

| Package | Role |
|---------|------|
| **coach** | Training harness and CLI |
| **data-leak** | Data loading (Parquet, Zarr, spatial) |
| **time-flies** | Model architectures (LSTM, Perceiver) |

## Setup

```bash
git clone --recurse-submodules https://github.com/CooperBigFoot/model-training.git
cd model-training
uv sync
```

## Common Commands

```bash
uv run coach --help          # CLI
uv run pytest                # run all tests
uv run ruff check .          # lint
uv run ruff format --check . # format check
```
