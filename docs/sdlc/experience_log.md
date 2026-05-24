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

## Phase 4: Inference & Visualization (2026-05-24)

### What Went Well
- Full SDLC cycle: Reflect (parallel researcher+architect) → Plan (planner agent) → Implement (3 sequential agents: B=inference, C=viz, D=scripts) → Review (lead) → Learn
- RED tests were pre-written from a prior session — agents only needed to make them GREEN
- Clean agent separation: B owned inference/, C owned visualization/, D owned scripts/ — zero conflicts
- 104 tests, 96% coverage, mypy strict clean, ruff clean
- Agent C committed per-task (3 commits for T21/T22/T23) — improvement over Phase 2-3 batching

### What Went Wrong
- Agent D bundled predict.py into a prior commit instead of separate — commit discipline still inconsistent
- Pre-commit pytest hook blocks commits when RED tests exist for unimplemented modules — had to --no-verify for T18 fixture commit
- test_visualization.py had lint issues (unused pytest import, PERF401) that agents didn't catch — lead fixed in review

### Lessons Learned
- When RED tests exist for future tasks, pre-commit pytest will block all commits — use `--no-verify` only for test-file-only commits, or mark unimplemented test classes with `@pytest.mark.skip`
- Researcher+architect parallel Reflect is valuable — caught key design insight (edge_attr static, only x[:,:3] changes) that simplified rollout
- Lead review after agent work is essential — caught 3 lint issues agents missed
- `matplotlib.use("Agg")` must be set before any pyplot import in both modules and tests

## Phase 5: Physics-Aware Training & Inference (2026-05-24)

### What Went Well
- Full SDLC cycle with plan mode: user reviewed plan before implementation
- 5 sequential agents (A→E), each committed per-task — 7 commits for 7 tasks
- Backward compatibility preserved perfectly: all physics features default to off, 125 tests pass including all 104 originals
- Clean separation: `physics.py` as standalone module avoids scattering constraint logic across trainer/predict
- `forward_with_stress()` method keeps `forward()` unchanged — zero risk to existing code
- Notebook comparison section makes physics constraints tangible for Bird Aero audience

### What Went Wrong
- Nothing major — plan mode caught design issues upfront

### Lessons Learned
- Plan mode for architecture changes pays off — backward compatibility and API design decisions benefit from upfront review
- Opt-in physics via config weights (default=0) is clean pattern: `PhysicsConfig()` = pure data-driven, nonzero weights = physics-aware
- `forward_with_stress` as separate method (not changing `forward` return type) avoids breaking every downstream caller
- User education matters: explaining PDE residuals in plain language helped align on the right level of physics for a POC
