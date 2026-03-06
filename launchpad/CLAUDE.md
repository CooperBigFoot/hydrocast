# Claude Agent Guidelines

## Project Overview

launchpad — GPU training orchestrator for Hydrocast. Automates RunPod pod provisioning, data staging, remote `coach train`/`coach evaluate` execution, log monitoring, hyperparameter sweeps, and checkpoint archival to HuggingFace Hub.

## Package Structure

```
launchpad/src/launchpad/
├── cli.py            # typer CLI entrypoint
├── output.py         # rich output helpers
├── config.py         # LaunchpadConfig (pydantic, .env + YAML)
├── types.py          # Domain types: PodId, VolumeId, RunId, etc.
├── errors.py         # LaunchpadError hierarchy
├── protocols.py      # CloudProvider, SSHExecutor protocols
├── ssh.py            # SubprocessSSH: run commands, tail logs
├── sync.py           # rsync upload/download, dataset staging
├── workspace.py      # bootstrap/sync coach on network volume
├── runner.py         # pod lifecycle + coach execution + watchdog
├── monitor.py        # log streaming, status polling
├── sweep.py          # grid expansion, parallel/sequential dispatch
├── hub.py            # HuggingFace Hub push/pull
└── providers/
    └── runpod.py     # RunPodProvider: REST + GraphQL API wrapper
```

## CLI Reference

All commands are invoked via `uv run launchpad <command>`.

### Volume Management

```bash
# Create a network volume
launchpad volume create --name <name> --size <gb> --region <region>

# List volumes (optionally filter by name)
launchpad volume ls [<volume-name>]

# Upload local data to a volume
launchpad volume upload <local-path> <volume-name>:<dataset-name>

# Delete a volume
launchpad volume delete <volume-name>

# Bootstrap workspace on volume (clone repo + uv sync)
launchpad volume setup <volume-name>
```

### Training

```bash
# Launch a training run
launchpad train --config <yaml> --volume <name> --gpu <type> [--qty N]

# Evaluate a trained model
launchpad evaluate --run-dir <name> --eval-config <yaml> --volume <name> --gpu <type>
```

### Sweeps

```bash
# Run a hyperparameter sweep (parallel or sequential)
launchpad sweep --config <sweep.yaml> --volume <name> --gpu <type> [--parallel|--sequential]
```

Sweep YAML format:
```yaml
base: path/to/base_config.yaml
mode: SEQUENTIAL  # or PARALLEL
sweep:
  training.lr: [0.001, 0.01, 0.1]
  training.batch_size: [32, 64, 128]
```

### Monitoring & Management

```bash
# Check run status
launchpad status <run-id>

# Stream logs (follow mode)
launchpad logs <run-id> [--follow]

# Pull run artifacts to local
launchpad pull <run-id> [--output <local-dir>]

# Push checkpoint to HuggingFace Hub
launchpad push <run-id> --repo <hf-repo>

# Terminate a pod
launchpad down <run-id>
```

## Configuration

Launchpad loads configuration in priority order: **env vars > .env file > ~/.launchpad/config.yaml > defaults**.

### Required

| Variable | Description |
|---|---|
| `RUNPOD_API_KEY` | RunPod API key |

### Optional

| Variable | Description |
|---|---|
| `HF_TOKEN` | HuggingFace token (for push/pull commands) |

### Config file (~/.launchpad/config.yaml)

```yaml
api_key: "your-runpod-key"
hf_token: "your-hf-token"
default_gpu: "NVIDIA A40"
default_cloud_type: "COMMUNITY"
default_data_center: "US-TX-3"
default_image: "runpod/pytorch:2.8.0-py3.12-cuda12.8.1-cudnn-devel-ubuntu22.04"
ssh_key_path: "~/.ssh/id_ed25519"
```

All fields except `api_key` have sensible defaults.

## Key Domain Types

| Type | Kind | Purpose |
|---|---|---|
| `PodId` | NewType(str) | RunPod pod identifier |
| `VolumeId` | NewType(str) | Network volume identifier |
| `GpuTypeId` | NewType(str) | GPU model string (e.g., "NVIDIA A40") |
| `RunId` | NewType(str) | Training run identifier |
| `HfRepoId` | NewType(str) | HuggingFace repo (e.g., "org/model") |
| `PodStatus` | Enum | CREATED, RUNNING, EXITED, TERMINATED |
| `RunStatus` | Enum | PENDING, RUNNING, COMPLETED, FAILED |
| `SweepMode` | Enum | PARALLEL, SEQUENTIAL |

## Remote Layout

On the RunPod network volume (`/workspace/`):
- `/workspace/hydrocast/` — cloned repo with coach + dependencies
- `/workspace/datasets/<name>/` — uploaded datasets
- `/workspace/runs/<run-id>/` — training outputs, logs, checkpoints

## Auto-Shutdown

The inactivity watchdog monitors GPU utilization. If GPU usage stays below 5% for 30 minutes (default), the pod is automatically terminated. This prevents forgotten pods from burning money.

## Testing

```bash
uv run pytest launchpad/tests/ -v          # run all tests
uv run ruff check launchpad/               # lint
uv run ruff format --check launchpad/      # format check
```

## Workspace Conventions

Follow the same conventions as the root CLAUDE.md:
- `uv` for package management
- Type-driven development (NewType, NamedTuple, Enum, Protocol)
- Frozen pydantic models for config
- Fakes over mocks in tests
- No docstrings during prototyping
- Version bump with every commit
