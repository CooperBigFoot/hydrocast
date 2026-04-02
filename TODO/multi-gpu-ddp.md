# Multi-GPU DDP Training for Coach

## Context

Coach is a hand-rolled PyTorch training harness with a single-device training loop (`loop.py`). All models in time-flies are vanilla `nn.Module` — fully DDP-compatible. Data loaders in data-leak use standard PyTorch `DataLoader` with no sampler override. The goal is to add multi-GPU training via PyTorch's `DistributedDataParallel` with minimal changes, keeping the architecture clean.

**Key design decision** (agreed with user): A single `gpus: int = 1` field in `TrainingSection`. No separate distributed section. No FSDP/DeepSpeed. Backend always `nccl`. Launch via `torchrun`.

---

## Phase 1: Config (`coach/src/coach/config.py`)

### 1a. Add `gpus` field to `TrainingSection` (line 57)

```python
gpus: int = Field(default=1, ge=1)
```

`batch_size` means **per-GPU** batch size. Effective batch size = `batch_size * gpus`.

### 1b. Exclude `gpus` from `config_hash()` (lines 79-85)

Changing GPU count is infrastructure, not semantics. Same config on 1 vs 4 GPUs should be considered the same run.

```python
def config_hash(self) -> str:
    data = self.model_dump(mode="json")
    data.get("training", {}).pop("gpus", None)
    canonical = yaml.dump(data, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]
```

---

## Phase 2: Distributed helpers (`coach/src/coach/distributed.py` — NEW)

Small stateless module encapsulating all DDP concerns. Reads from `torchrun` env vars.

```python
import logging, os
import torch
import torch.distributed as dist

def is_distributed() -> bool:
    return "LOCAL_RANK" in os.environ

def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def global_rank() -> int:
    return int(os.environ.get("RANK", "0"))

def is_main_process() -> bool:
    return global_rank() == 0

def setup_ddp() -> None:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank())

def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

def reduce_mean(val: float, device: torch.device) -> float:
    t = torch.tensor([val], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()
```

---

## Phase 3: Data-leak sampler support (`data-leak/src/data_leak/loaders.py`)

Add `sampler: Sampler | None = None` to `make_dataloader` (line 12) and `make_lazy_dataloader` (line 61). When sampler is provided, force `shuffle=False`.

```python
from torch.utils.data import DataLoader, Sampler

def make_dataloader(
    bundle: DatasetBundle,
    batch_size: int,
    shuffle: bool = True,
    sampler: Sampler | None = None,    # NEW
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    input_dropout: InputDropoutConfig | None = None,
    static_dropout: StaticDropoutConfig | None = None,
    is_training: bool = False,
) -> DataLoader[TrainingBatch]:
    dataset = HydroDataset(...)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        ...
    )
```

Same for `make_lazy_dataloader`. **Backward-compatible** — `None` default preserves all existing behavior.

The `DistributedSampler` needs the `Dataset` object. Since the factory creates the dataset internally, coach needs to access it after construction via `loader.dataset` to build the sampler. This creates a chicken-and-egg problem.

**Solution**: Build the sampler from a throwaway length measurement, OR refactor the factory to also return the dataset. Simplest: coach accesses `loader.dataset` from a first non-DDP loader, creates the sampler, then rebuilds the loader with it. Actually the cleanest approach: have data-leak also expose `make_dataset` / `make_lazy_dataset` helpers that return just the dataset. Then coach can:
1. `dataset = make_dataset(bundle, ...)`
2. `sampler = DistributedSampler(dataset, ...)`
3. `loader = DataLoader(dataset, sampler=sampler, collate_fn=collate, ...)`

This avoids the chicken-and-egg entirely. Add to `loaders.py`:

```python
def make_dataset(
    bundle: DatasetBundle,
    input_dropout: InputDropoutConfig | None = None,
    static_dropout: StaticDropoutConfig | None = None,
    is_training: bool = False,
) -> HydroDataset:
    return HydroDataset(bundle=bundle, input_dropout=input_dropout,
                        static_dropout=static_dropout, is_training=is_training)

def make_lazy_dataset(
    bundle: LazyDatasetBundle,
    input_dropout: InputDropoutConfig | None = None,
    static_dropout: StaticDropoutConfig | None = None,
    is_training: bool = False,
) -> LazyHydroDataset:
    return LazyHydroDataset(bundle=bundle, input_dropout=input_dropout,
                            static_dropout=static_dropout, is_training=is_training)
```

Then `make_dataloader` internally calls `make_dataset` (DRY).

---

## Phase 4: CLI torchrun re-launch (`coach/src/coach/cli.py`)

When `gpus > 1` and we're NOT inside `torchrun` (no `LOCAL_RANK` env var), the CLI re-launches itself under `torchrun`.

```python
def _relaunch_under_torchrun(gpus: int) -> None:
    import torch
    available = torch.cuda.device_count()
    if gpus > available:
        raise RuntimeError(f"Config requests {gpus} GPUs but only {available} available")
    args = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", str(gpus),
        "--standalone",
        *sys.argv,
    ]
    logger.info("Re-launching under torchrun: %s", " ".join(args))
    os.execvp(sys.executable, args)
```

In `train` command (line 35), add re-launch gate:

```python
config = RunConfig.from_yaml(config_path)
if config.training.gpus > 1 and "LOCAL_RANK" not in os.environ:
    _relaunch_under_torchrun(config.training.gpus)
    return  # unreachable after execvp
```

Uses `sys.executable -m torch.distributed.run` (not the `torchrun` script) to ensure correct venv Python.

---

## Phase 5: Training loop (`coach/src/coach/loop.py`)

This is the biggest change. The `train()` function (lines 407-530) gains DDP awareness.

### 5a. DDP init/teardown — wrap `train()` body

```python
from .distributed import is_distributed, setup_ddp, cleanup_ddp, is_main_process, local_rank, reduce_mean

def train(config: RunConfig, run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ddp = is_distributed()
    if ddp:
        setup_ddp()

    try:
        return _train_impl(config, run_dir, ddp)
    finally:
        if ddp:
            cleanup_ddp()
```

### 5b. Device selection (replaces line 441)

```python
if ddp:
    device = torch.device("cuda", local_rank())
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 5c. Seed — call BEFORE `setup_ddp()`

Same seed on all ranks ensures identical model initialization. `DistributedSampler` handles per-rank data shuffling.

### 5d. DDP model wrapping (after line 442)

```python
model, model_config_dict, ext_type, ext_config_dict = _build_model(config, device)
if ddp:
    from torch.nn.parallel import DistributedDataParallel as DDP
    model = DDP(model, device_ids=[local_rank()])
```

### 5e. Data loaders with DistributedSampler

Modify `_build_eager_loaders` and `_build_lazy_loaders` to accept `ddp: bool = False`. When DDP:

```python
from torch.utils.data.distributed import DistributedSampler

train_dataset = make_dataset(bundle=train_bundle, is_training=True)
train_sampler = DistributedSampler(train_dataset, shuffle=True) if ddp else None
train_loader = DataLoader(
    dataset=train_dataset, batch_size=config.training.batch_size,
    shuffle=(train_sampler is None), sampler=train_sampler,
    num_workers=config.training.num_workers, pin_memory=True,
    collate_fn=collate,
)
```

Return type changes to include samplers:

```python
def _build_data_loaders(
    config: RunConfig, ddp: bool = False,
) -> tuple[DataLoader, DataLoader, DistributedSampler | None]:
```

### 5f. Epoch loop changes (lines 458-507)

```python
for epoch in range(1, config.training.max_epochs + 1):
    # Sampler epoch sync
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    train_loss = _train_one_epoch(...)
    val_loss = _validate(...)

    # All-reduce val loss for consistent early stopping
    if ddp:
        val_loss = reduce_mean(val_loss, device)

    # ALL ranks track metrics + patience (must stay in sync)
    metrics.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    # Only rank 0 does I/O
    is_main = not ddp or is_main_process()
    if is_main:
        logger.info("Epoch %d/%d — train=%.6f, val=%.6f", ...)
        save_checkpoint(..., path=run_dir / "last.ckpt")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0
        if is_main:
            save_checkpoint(..., path=run_dir / "checkpoint.pt")
    else:
        patience_counter += 1

    if patience_counter >= config.training.early_stopping_patience:
        break
```

### 5g. `_train_one_epoch` and `_validate` — NO changes

These functions are rank-agnostic. DDP averages gradients automatically in `loss.backward()`. `clip_grad_norm_` operates on local (already-averaged) gradients — correct as-is.

---

## Phase 6: Checkpoints (`coach/src/coach/checkpoint.py`)

### 6a. Save unwrapped state_dict (line 57)

Always save `model.module.state_dict()` when DDP-wrapped, so checkpoints are portable:

```python
from torch.nn.parallel import DistributedDataParallel

unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
checkpoint = {
    "state_dict": unwrapped.state_dict(),
    ...
}
```

### 6b. Add `world_size` to training_metadata

```python
def save_checkpoint(
    ...,
    world_size: int = 1,  # NEW
) -> None:
    ...
    "training_metadata": {
        ...,
        "world_size": world_size,
    }
```

### 6c. `load_checkpoint` — NO changes

Unwrapped state_dicts load identically for single-GPU eval and DDP resume.

---

## Phase 7: Evaluate (`coach/src/coach/evaluate.py`) — NO changes

Evaluation always runs single-GPU. Checkpoints with unwrapped state_dicts load correctly.

---

## Phase 8: Tests

### `coach/tests/test_config.py`
- `test_gpus_default_is_one`
- `test_gpus_rejects_zero`
- `test_config_hash_unchanged_by_gpus` — two configs identical except gpus should have same hash

### `coach/tests/test_distributed.py` (NEW)
- `test_is_distributed_false_by_default`
- `test_is_distributed_true_with_local_rank` (monkeypatch env)
- `test_local_rank_reads_env`
- `test_is_main_process_rank_zero` / `_rank_nonzero`

### `coach/tests/test_checkpoint.py`
- `test_save_unwraps_ddp_module` — use a `FakeDDP` wrapper with `.module` attribute
- `test_world_size_recorded_in_metadata`

### `data-leak/tests/test_loaders.py`
- `test_sampler_overrides_shuffle` — pass `SequentialSampler`, verify deterministic order

---

## Files Changed Summary

| File | Change |
|------|--------|
| `coach/src/coach/config.py` | Add `gpus` field, exclude from hash |
| `coach/src/coach/distributed.py` | **NEW** — DDP helpers |
| `coach/src/coach/cli.py` | Add `torchrun` re-launch gate |
| `coach/src/coach/loop.py` | DDP init/teardown, device by rank, DDP wrap, sampler plumbing, rank-0 I/O guards, val loss all-reduce |
| `coach/src/coach/checkpoint.py` | Unwrap DDP model, add `world_size` param |
| `data-leak/src/data_leak/loaders.py` | Add `sampler` param + `make_dataset`/`make_lazy_dataset` helpers |
| `coach/tests/test_config.py` | gpus validation + hash tests |
| `coach/tests/test_distributed.py` | **NEW** — env var helper tests |
| `coach/tests/test_checkpoint.py` | DDP unwrap + world_size tests |
| `data-leak/tests/test_loaders.py` | Sampler override test |
| **time-flies** | **NO CHANGES** |

---

## Gotchas & Risks

1. **`sys.argv` with `uv run coach`**: `os.execvp` passes `sys.argv` to torchrun. Verify that torchrun can invoke the `coach` entry point script. If not, reconstruct as `sys.executable -m coach.cli train ...`.
2. **`num_workers` is per-GPU**: 4 GPUs * 4 workers = 16 total processes. Document this.
3. **No automatic LR scaling**: User manages via `learning_rate_scale` in config. Effective batch scales with `gpus`.
4. **Zarr concurrent reads**: `LazyHydroDataset` opens zarr in read-only mode — safe for multi-process.
5. **`drop_last=True`** recommended for DDP training loader to avoid uneven batch sizes across ranks.
6. **NCCL requires CUDA**: Validate `torch.cuda.is_available()` before DDP init.

---

## Verification

1. **Unit tests**: `uv run pytest coach/tests/test_config.py coach/tests/test_distributed.py coach/tests/test_checkpoint.py`
2. **data-leak tests**: `uv run pytest data-leak/tests/test_loaders.py`
3. **Single-GPU regression**: `uv run coach train config.yaml run_dir` with `gpus: 1` — behavior unchanged
4. **Multi-GPU smoke test** (if hardware available): `uv run coach train config.yaml run_dir` with `gpus: 2` — verify torchrun re-launch, training completes, checkpoint loadable for eval
5. **Eval portability**: Train with `gpus: 4`, evaluate with `coach evaluate run_dir` (single GPU) — verify checkpoint loads cleanly
