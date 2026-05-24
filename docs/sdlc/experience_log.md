# Experience Log

Lessons learned from development. Check before starting new work.

## Phase 2: Data Pipeline (2026-05-24)

### What Went Well
- TDD via sequential subagents worked — 70 tests, mypy strict, ruff clean
- Dict-based graphs (no torch_geometric dep for core) keeps install simple
- Separating config/download (T2-T3) from preprocess/dataloader (T4-T8) into two agent runs avoided file conflicts

### What Went Wrong
- Agent worktree isolation failed (`Failed to resolve base branch "HEAD"`) on fresh repo — had to use `general-purpose` agent type without isolation
- Parallel agent dispatch not possible due to worktree bug — sequential only
- Agent batched commits instead of eager per-task (1 commit for T2+T3, 1 for T4-T8)

### Lessons Learned
- When dispatching subagents on fresh repos, use `general-purpose` type without worktree isolation
- Keep agent prompts focused — one agent per logical unit (config+download vs preprocess+dataloader)
- TFRecord parsing requires `np.frombuffer(...).copy()` to avoid read-only array issues

## Phase 3: Model & Training (2026-05-24)

### What Went Well
- Single agent handled T9-T15 (MLP, EdgeBlock, NodeBlock, MeshGraphNet, TrainingConfig, trainer, script, config) cleanly — 12 new tests, all passing
- Dict-based graph input (no torch_geometric Data object) keeps model self-contained
- `scatter_add_` for message aggregation avoids pyg dependency entirely
- 94% test coverage maintained across dataset + model + training modules
- Colab notebook (T16) created with correct API signatures — end-to-end pipeline in one notebook

### What Went Wrong
- Agent batched T9-T15 into 2 commits instead of per-task — same issue as Phase 2
- `train()` creates model internally (hides architecture from caller) — fine for POC but would need refactoring for Phase 4 rollout inference

### Lessons Learned
- Explicitly instruct agents: "commit after each task ID completes" — batching is the default behavior
- ProcessorLayer typed wrapper avoids mypy issues with nn.ModuleDict indexing
- `train()` returning best checkpoint path is clean API — caller doesn't manage model lifecycle
