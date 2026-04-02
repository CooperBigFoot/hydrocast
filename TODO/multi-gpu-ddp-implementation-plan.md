# Multi-GPU DDP Implementation Plan

## Orchestration Protocol

**CRITICAL: The orchestrator (you) must NEVER write code directly.**

Your responsibilities:
1. **Explore** the codebase before each phase to gather context for agent prompts
2. **Delegate** all implementation work to Sonnet 4.6 general-purpose agents
3. **Coordinate** parallel vs sequential execution based on dependencies
4. **Review** all changes at the end via `git diff`
5. **Iterate** by delegating fixes to subagents if issues found
6. **Commit** only when all code passes

---

## Implementation Units

### Unit A: Config (`gpus` field + hash exclusion + tests)

**Phases**: 1 + 8-config

**Files to read**:
- `coach/src/coach/config.py`
- `coach/tests/test_config.py`

**Files to modify**:
- `coach/src/coach/config.py` — add `gpus: int = Field(default=1, ge=1)` to `TrainingSection` (after line 57); modify `config_hash()` (lines 79-85) to pop `gpus` from the serialized dict before hashing

**Tests to write** (append to `coach/tests/test_config.py`):
- `TestTrainingSection.test_gpus_default_is_one`
- `TestTrainingSection.test_gpus_rejects_zero`
- `TestRunConfig.test_config_hash_unchanged_by_gpus` — two configs identical except `gpus` (1 vs 4) produce the same hash

**Verify**: `uv run pytest coach/tests/test_config.py -v`

**Gotchas**:
- `TrainingSection` has `extra="forbid"` — the field must be explicit
- `config_hash()` uses `model_dump(mode="json")` — pop `gpus` from the nested `training` dict before hashing
- Do NOT bump version (deferred to Unit F)

---

### Unit B: Distributed Helpers (new module + tests)

**Phases**: 2 + 8-distributed

**Files to read**:
- `coach/src/coach/__init__.py`
- `coach/tests/test_config.py` (for test patterns)

**Files to create**:
- `coach/src/coach/distributed.py` — stateless DDP helpers: `is_distributed()`, `local_rank()`, `global_rank()`, `is_main_process()`, `setup_ddp()`, `cleanup_ddp()`, `reduce_mean()`
- `coach/tests/test_distributed.py`

**Tests to write**:
- `test_is_distributed_false_by_default`
- `test_is_distributed_true_with_local_rank` (monkeypatch env)
- `test_local_rank_reads_env`
- `test_global_rank_reads_env`
- `test_is_main_process_rank_zero` / `_rank_nonzero`

**Verify**: `uv run pytest coach/tests/test_distributed.py -v`

**Gotchas**:
- Do NOT test `setup_ddp()`, `cleanup_ddp()`, or `reduce_mean()` — they require NCCL/CUDA
- Use `monkeypatch.setenv` / `monkeypatch.delenv` for env var manipulation
- `ReduceOp.AVG` requires PyTorch >= 1.11
- Do NOT bump version (deferred to Unit F)

---

### Unit C: Data-Leak Sampler Support (loaders + tests)

**Phases**: 3 + 8-loaders

**Files to read**:
- `data-leak/src/data_leak/loaders.py`
- `data-leak/src/data_leak/dataset.py`
- `data-leak/src/data_leak/lazy_dataset.py`
- `data-leak/src/data_leak/__init__.py`
- `data-leak/tests/test_loaders.py`
- `data-leak/conftest.py`

**Files to modify**:
- `data-leak/src/data_leak/loaders.py`:
  1. Add `sampler: Sampler | None = None` to `make_dataloader` and `make_lazy_dataloader`; when sampler provided, force `shuffle=False`
  2. Extract `make_dataset()` and `make_lazy_dataset()` factory functions that return just the dataset
  3. Refactor `make_dataloader` to call `make_dataset` internally (DRY)
- `data-leak/src/data_leak/__init__.py` — export `make_dataset`, `make_lazy_dataset`

**Tests to write** (append to `data-leak/tests/test_loaders.py`):
- `TestMakeDataloader.test_sampler_overrides_shuffle` — pass `SequentialSampler`, verify deterministic order
- `TestMakeDataset.test_returns_hydro_dataset`
- `TestMakeDataset.test_dataset_length_matches_sequences`

**Verify**: `uv run pytest data-leak/tests/test_loaders.py -v`

**Gotchas**:
- Import `Sampler` from `torch.utils.data`, not `torch.utils.data.distributed`
- `make_spatial_dataloader` does NOT need the sampler parameter
- All existing tests must pass unchanged (backward compatible via `sampler=None` default)
- Do NOT bump version (deferred to Unit F)

---

### Unit D: Checkpoint DDP Awareness (unwrap + world_size + tests)

**Phases**: 6 + 8-checkpoint

**Files to read**:
- `coach/src/coach/checkpoint.py`
- `coach/tests/test_checkpoint.py`

**Files to modify**:
- `coach/src/coach/checkpoint.py`:
  1. In `save_checkpoint`, unwrap DDP: `model.module.state_dict()` when `isinstance(model, DistributedDataParallel)`
  2. Add `world_size: int = 1` keyword argument
  3. Include `world_size` in `training_metadata` dict

**Tests to write** (append to `coach/tests/test_checkpoint.py`):
- `TestCheckpointDDP.test_save_unwraps_ddp_module` — use a FakeDDP wrapper with `.module` attribute
- `TestCheckpointDDP.test_world_size_recorded_in_metadata`
- `TestCheckpointDDP.test_world_size_defaults_to_one`

**Verify**: `uv run pytest coach/tests/test_checkpoint.py -v`

**Gotchas**:
- `from torch.nn.parallel import DistributedDataParallel` is safe to import on CPU (it's a Python class)
- For tests: cannot create real DDP without `init_process_group`. Use a FakeDDP approach — a simple `nn.Module` wrapper with `.module` attribute. To make `isinstance` work, either subclass DDP (won't work without NCCL) or use `hasattr(model, 'module')` check instead. Prefer `hasattr` for testability.
- `load_checkpoint` needs NO changes
- All existing tests must pass unchanged
- Do NOT bump version (deferred to Unit F)

---

### Unit E: CLI Re-launch + Training Loop DDP Integration

**Phases**: 4 + 5

**Dependencies**: Units A, B, C, D must all be complete first.

**Files to read**:
- `coach/src/coach/cli.py`
- `coach/src/coach/loop.py`
- `coach/src/coach/distributed.py` (from Unit B)
- `coach/src/coach/checkpoint.py` (updated by Unit D)
- `coach/src/coach/config.py` (updated by Unit A)
- `data-leak/src/data_leak/loaders.py` (updated by Unit C)

**Files to modify**:
- `coach/src/coach/cli.py`:
  1. Add `_relaunch_under_torchrun(gpus: int) -> None` using `os.execvp`
  2. In `train` command, after config parse: if `gpus > 1` and `LOCAL_RANK` not in env, re-launch

- `coach/src/coach/loop.py` (largest change):
  1. **DDP init/teardown**: Wrap `train()` body in `setup_ddp()`/`cleanup_ddp()` try/finally
  2. **Seed before DDP**: `_seed_everything` must run BEFORE `setup_ddp()` for identical model init
  3. **Device selection**: Use `torch.device("cuda", local_rank())` when DDP
  4. **DDP model wrap**: `DDP(model, device_ids=[local_rank()])` after `_build_model`
  5. **Data loaders**: Modify `_build_data_loaders`, `_build_eager_loaders`, `_build_lazy_loaders` to accept `ddp: bool`, create `DistributedSampler` when DDP, return `(train_loader, val_loader, train_sampler | None)`
  6. **Epoch loop**: Call `train_sampler.set_epoch(epoch)` when sampler exists; `reduce_mean(val_loss, device)` when DDP; all ranks track patience identically
  7. **I/O guards**: Only rank 0 writes config.yaml snapshot, checkpoint files, metrics.json, and log messages
  8. **`save_checkpoint` calls**: Pass `world_size=dist.get_world_size()` when DDP
  9. **`drop_last=True`** on DDP training loader for even batch distribution

**Tests**: No new tests needed — this is integration code. Verify single-GPU regression:
- `uv run pytest coach/tests/ -v`

**Gotchas**:
- `_train_one_epoch` and `_validate` need NO changes — DDP averages gradients automatically, `clip_grad_norm_` works on DDP-wrapped model parameters
- `train_loss` from `_train_one_epoch` is per-rank, NOT globally averaged — this is acceptable for logging (val_loss IS all-reduced for early stopping consistency)
- `sys.argv[0]` with `uv run coach` is the entry point script path — torchrun must be able to re-invoke it
- `run_dir.mkdir(parents=True, exist_ok=True)` is safe for all ranks (idempotent)
- Pretrained checkpoint loading (`from_run`) works identically — all ranks load same file
- Import `DistributedSampler` from `torch.utils.data.distributed`
- Do NOT bump version (deferred to Unit F)

---

### Unit F: Integration Verification + Version Bump + Commit

**Dependencies**: Unit E must be complete.

**No files to implement** — verification and housekeeping only.

**Steps**:
1. Run full test suite: `uv run pytest coach/tests/ data-leak/tests/ -v`
2. Lint: `uv run ruff check .`
3. Format: `uv run ruff format --check .`
4. Review: `git diff` — verify all changes are correct
5. Version bumps:
   - `cd coach && uv run bump-my-version bump patch`
   - `cd data-leak && uv run bump-my-version bump patch`
6. Stage and commit with conventional message
7. Tag: `git tag v$(uv run bump-my-version show current_version)`

---

## Dependency Graph

```
Wave 1 (parallel):      Wave 2 (sequential):     Wave 3 (sequential):

  Unit A ──┐
  Unit B ──┤
           ├──────────> Unit E ──────────────────> Unit F
  Unit C ──┤
  Unit D ──┘
```

```json
{
  "units": {
    "A": {
      "name": "Config (gpus field + hash exclusion + tests)",
      "phases": [1, "8-config"],
      "dependencies": [],
      "wave": 1,
      "packages_modified": ["coach"]
    },
    "B": {
      "name": "Distributed helpers (new module + tests)",
      "phases": [2, "8-distributed"],
      "dependencies": [],
      "wave": 1,
      "packages_modified": ["coach"]
    },
    "C": {
      "name": "Data-leak sampler support (loaders + tests)",
      "phases": [3, "8-loaders"],
      "dependencies": [],
      "wave": 1,
      "packages_modified": ["data-leak"]
    },
    "D": {
      "name": "Checkpoint DDP awareness (unwrap + world_size + tests)",
      "phases": [6, "8-checkpoint"],
      "dependencies": [],
      "wave": 1,
      "packages_modified": ["coach"]
    },
    "E": {
      "name": "CLI re-launch + Training loop DDP integration",
      "phases": [4, 5],
      "dependencies": ["A", "B", "C", "D"],
      "wave": 2,
      "packages_modified": ["coach"]
    },
    "F": {
      "name": "Integration verification + version bump + commit",
      "phases": ["verification"],
      "dependencies": ["E"],
      "wave": 3,
      "packages_modified": ["coach", "data-leak"]
    }
  },
  "execution_waves": {
    "wave_1": {
      "units": ["A", "B", "C", "D"],
      "parallel": true,
      "agents": 4
    },
    "wave_2": {
      "units": ["E"],
      "parallel": false,
      "agents": 1
    },
    "wave_3": {
      "units": ["F"],
      "parallel": false,
      "agents": 1
    }
  }
}
```

## Issues Found During Validation

### Resolved in this plan:
1. **Phase 3 inconsistency**: The original plan presented two approaches (sampler param vs make_dataset). This plan uses BOTH — `make_dataset`/`make_lazy_dataset` helpers for DRY + sampler param on existing factories. Unit E uses the dataset helpers to build `DistributedSampler`, then passes it to the factory.
2. **Missing I/O guards**: Original plan didn't guard config.yaml snapshot (line 434) or metrics.json (line 510-511) writes. Added to Unit E.
3. **train_loss all-reduce**: Intentionally NOT all-reduced — per-rank train_loss is acceptable for logging. Only val_loss is all-reduced for early stopping consistency.
4. **Checkpoint isinstance check**: Changed from `isinstance(model, DistributedDataParallel)` to `hasattr(model, 'module')` for testability without NCCL.

### Known risks (from original plan, still apply):
1. `sys.argv` with `uv run coach` — torchrun must be able to re-invoke the entry point
2. `num_workers` is per-GPU — 4 GPUs * 4 workers = 16 processes
3. No automatic LR scaling — user manages via config
4. Zarr concurrent reads are safe (read-only mode)
5. NCCL requires CUDA — validate `torch.cuda.is_available()` before DDP init
