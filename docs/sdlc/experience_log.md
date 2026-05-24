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
