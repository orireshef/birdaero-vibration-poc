---
name: sdlc
description: "Development discipline for research-engineering projects. Applies reflect-plan-implement-review thinking at every scale - from ad hoc tasks to full epics. Pairs Researcher and Architect agents to bridge scientific rigor with engineering quality. Enforces TDD and eager commits."
---

# SDLC: Development Discipline

**Usage**: `/sdlc` - Apply this discipline to whatever work is at hand.

This is not a rigid process. It is a **thinking discipline** that applies at every level of work: a quick bug fix, a complex ad hoc task, a user story, a research investigation, a document, an entire epic. The depth of each step scales with the complexity and risk of the work.

## The Cycle

Every meaningful unit of work benefits from:

```
Reflect -> Plan -> Implement -> Review -> Learn
```

For a 5-minute fix, this might be 30 seconds of thought before acting. For an epic, it's days of structured work with multiple agents. The discipline is the same; the depth scales.

| Work Scale | Reflect | Plan | Implement | Review | Learn |
|-----------|---------|------|-----------|--------|-------|
| Ad hoc fix | Read the code, check git blame | Mental model of change | Make the fix, commit | Sanity check tests pass | Note if recurring |
| Complex task | Read plan, check dependencies | Define acceptance criteria | TDD, eager commits | Self-review or agent review | Update plan status |
| User story | Agent pair assesses state | Agent pair creates task breakdown | TDD per task, ~15-25 commits | Dual agent review | Experience log entry |
| Epic | Full team reflection on prior epics | PRD + solution design + impl plan | Story-by-story SDLC | Cross-story integration review | Retrospective |
| Research question | Literature review, prior findings | Hypothesis + experimental design | Run experiments, document results | Validate methodology, peer review | Update knowledge base |
| Document/design | Read existing docs, check consistency | Outline structure and audience | Write, review for accuracy | Cross-functional review | Note what worked |

## The Two Tracks

This project bridges **research** and **engineering**. Every decision has both a scientific and a technical dimension. The SDLC uses two agent tracks to ensure neither is neglected:

### Researcher Track (Scientific Rigor)

| Role | Agent | Does |
|------|-------|------|
| **Lead Researcher** | `ai-researcher` | Plans research, reviews methodology, validates statistical claims, checks feasibility |
| **Senior Researcher** | `ai-researcher` | Executes literature review, runs analysis, validates assumptions |

The Researcher asks: *Is this correct? Is this sound? Does the data support this? Are we measuring the right thing?*

### Architect Track (Engineering Quality)

| Role | Agent | Does |
|------|-------|------|
| **Lead Architect** | `architect` / `planner` | Designs systems, creates task breakdowns, reviews code quality, manages dependencies |
| **Senior Architect** | `code-reviewer` / `tdd-guide` | Implements code, enforces TDD, refactors, ensures patterns are followed |

The Architect asks: *Is this clean? Is this testable? Is this maintainable? Does this follow our patterns?*

### When to Use Which

- **Both tracks**: Stories, epics, features that touch algorithms + code
- **Architect only**: Pure refactoring, build fixes, dependency updates
- **Researcher only**: Literature review, experimental design, metric definition
- **Lightweight (no agents)**: Trivial fixes, typo corrections, config changes

---

## Reflect

Before acting, understand where you are.

**Ask**:
- What exists already? (code, docs, tests, prior work)
- What did we learn last time? (experience log, memory)
- What are the risks? (dependencies, unknowns, blockers)
- Are we ready to proceed?

**At scale** (stories/epics): Spawn Lead Researcher + Lead Architect to assess state in parallel. Read the experience log at `docs/sdlc/experience_log.md`. Check git log and test health.

**At small scale** (tasks/fixes): Read the relevant code. Check if there are existing tests. Understand the context before changing anything.

**Always**: If you've done similar work before, check what you learned.

---

## Plan

Before building, decide what to build and how.

**Ask**:
- What are the acceptance criteria? (explicit, measurable)
- What's the smallest thing that could work?
- In what order should we build? (dependencies, TDD sequence)
- What could go wrong?

**At scale**: Lead Architect creates task breakdown with IDs, priorities, acceptance criteria. Senior Researcher validates feasibility and challenges assumptions. User approves before any implementation. Plan goes to `docs/epics/epic_N/us_X_Y_plan.md`.

**At small scale**: A mental model of the change. Maybe a 3-line comment describing the approach before writing code.

**For research**: Hypothesis formulation, experimental design, metric selection.

**For documents**: Outline the structure, identify the audience, list key points.

**Always**: If TDD applies, plan the test sequence before writing any tests.

---

## Implement

Build it right, in small steps, with evidence at every step.

### TDD (When Writing Code)

```
RED:      Write a failing test
GREEN:    Write minimal code to pass
REFACTOR: Clean up without changing behavior
```

This is non-negotiable for production code. Tests are specifications, not afterthoughts.

### Running Slow Test Suites

**Never pipe pytest through `tail -N` or `head -N`.** Pytest buffers stdout when not on a tty, and `tail/head` only emits after the upstream command exits — a 15-minute run appears silent until the end, and you cannot see which test is hanging.

**Always run with live DEBUG/INFO logs.** Per-test PASS/FAIL lines alone are too thin — when a test takes minutes, you want to see *what the code is doing inside it*. Use `--log-cli-level=INFO` (or `DEBUG` for deep traces) together with `-v`, `-s`, and `PYTHONUNBUFFERED=1`.

**Correct invocation**:

```bash
# Unique log file per run: include PID + timestamp to avoid collisions
# when multiple agents run tests concurrently.
LOG=/tmp/pytest_$(date +%Y%m%d_%H%M%S)_$$.log

PYTHONUNBUFFERED=1 uv run pytest tests/test_backbone/ \
    -v -s \
    --log-cli-level=INFO \
    --log-cli-format='%(asctime)s %(levelname)-5s %(name)s: %(message)s' \
    --log-cli-date-format='%H:%M:%S' \
    > "$LOG" 2>&1 &

echo "pid=$! log=$LOG"
tail -f "$LOG"
```

**Rules**:

- **Overwrite, never append.** Use `>` not `>>`. Old runs are git history / console scrollback, not log-file content. A stale log mixed with a new run is worse than either alone.
- **Unique log name per run.** Include `$(date +%Y%m%d_%H%M%S)` AND `$$` (PID) in the filename. Two agents hitting `/tmp/run.log` simultaneously will corrupt each other's output and leave everyone confused about which run is which.
- **Print the log path + pid** immediately after launch so you (and any parallel agent) can see which run is yours.
- **Use `--log-cli-level=INFO`** as the default. Escalate to `DEBUG` only when chasing a specific hang — DEBUG output is enormous and slows tests.
- **Force unbuffered**: `PYTHONUNBUFFERED=1` + pytest `-s` disables pytest's output capture so logs flush live.

**Before killing a stalled pytest**: run `ps -o etime,pcpu -p <pid>` first. ~100% CPU = computing (JIT, real DB sweep, stability sweep), not hung. Don't kill live processes.

**Concurrency**: when another agent may be running tests in parallel, NEVER reuse a fixed log path like `/tmp/run.log`. Each agent picks its own timestamped path at launch and the coordinator tracks `{agent_id → log_path}` in the task/notes.

### Eager Commits

**Commit after every meaningful change.** The goal: small, atomic, reversible, reviewable.

**What triggers a commit**:

| Action | Commit Pattern |
|--------|---------------|
| Test scaffolded | `test(<scope>): scaffold tests for <Module>` |
| Failing test added | `test(<scope>): add failing test for <behavior>` |
| Code passes test | `feat(<scope>): implement <feature>` |
| Refactored | `refactor(<scope>): <what changed>` |
| Bug fixed | `fix(<scope>): <what was fixed>` |
| Fixture added | `test(<scope>): add fixture for <purpose>` |
| Doc/plan updated | `docs(<scope>): <what was updated>` |
| Research finding | `docs(<scope>): document <finding>` |

**Commit message format**:
```
<type>(<scope>): <description>

Types: feat, fix, test, refactor, docs, chore, research
Scope: story ID, module name, or topic (e.g., US-1.6, ground-truth, experiment-design)
```

**Cadence rules**:
- Never more than 15 minutes without a commit during active work
- Separate test commits from implementation commits when practical
- Never batch unrelated changes into one commit
- Each commit should leave the codebase in a valid state

### For Non-Code Work

Eager commits apply to documents, plans, and research too:
```bash
git commit -m "docs(epic-2): draft PRD executive summary"
git commit -m "research(embeddings): document literature review findings"
git commit -m "plan(US-1.7): add task breakdown for ablation framework"
```

---

## Review

After building, verify from both perspectives.

**Architect perspective**:
- Does it follow project conventions?
- Is it well-structured and maintainable?
- Are there unnecessary abstractions or missing error handling?
- Do tests cover the important behaviors?

**Researcher perspective**:
- Does it meet acceptance criteria?
- Are edge cases handled?
- Is the methodology sound (for research work)?
- Are there silent correctness issues?

**At scale**: Spawn both Lead agents for formal review. Document findings. Fix issues and commit before marking complete.

**At small scale**: Self-review the diff before pushing. Run the tests. Re-read the code with fresh eyes.

**Always**: If you find something worth noting, add it to the experience log.

---

## Learn

After completing work, capture what you learned so the process improves.

**Location**: `docs/sdlc/experience_log.md`

**What to record**:
```markdown
## <Scope>: <Title> (<Date>)

### What Went Well
- [Specific practices that worked]

### What Went Wrong
- [Issues, rework, surprises]

### Lessons Learned
- [Actionable: "When X, do Y instead of Z"]

### Metrics (for stories/epics)
- Tasks: planned / completed / deferred
- Commits: total count
- Review rounds: N (1 = clean, 2+ = revisions)
```

The experience log is also mirrored in Claude's persistent memory (`memory/sdlc_lessons.md`) for cross-conversation continuity. Before starting any Reflect phase, read both.

**The feedback loop**: Lessons from Review feed into the next Reflect. Over time, common pitfalls become explicit checklist items, estimates get calibrated, and the team gets faster.

---

## Agent Teams (Default for Dev/Research)

**Agent teams are the default mechanism** for all non-trivial dev/research work. The SDLC cycle (Reflect → Plan → Implement → Review → Learn) runs through a coordinated team, not isolated subagents.

### Model & Communication Policy

- **Team lead (main session)**: `opus` — orchestrates, reviews, validates, synthesizes
- **Teammates (default)**: `sonnet` — execute research, coding, initial analysis
- **Teammates (simple tasks)**: `haiku` — lead may assign haiku for routine/mechanical work (formatting, simple lookups, boilerplate generation)
- **Communication**: all teammates use `/caveman ultra` mode

Lead decides model per teammate based on task complexity. Default sonnet; haiku when task is routine.

### Lead Role

The lead (opus) is NOT just a dispatcher. It:
1. **Orchestrates** — creates task list, assigns work, picks teammate models
2. **Validates** — reviews all teammate output, approves/rejects plans
3. **Challenges** — questions teammate assumptions, catches gaps
4. **Synthesizes** — combines findings into coherent decisions
5. **Gates quality** — no phase transition without lead approval

Teammates execute. Lead thinks.

### When to Skip Teams

Only for truly trivial work: typo fixes, config changes, single-line bug fixes. If the work has a Reflect step worth doing, it deserves a team.

### Team Structure by SDLC Phase

#### Reflect
Lead spawns teammates for parallel investigation, then validates their findings:
```
"Create an agent team for <work>. Use /caveman ultra mode.
Spawn two teammates:
- 'researcher' (Sonnet): assess scientific state — prior lessons, research readiness, gaps, experience log
- 'architect' (Sonnet): assess technical state — git state, test health, codebase patterns, technical debt
Report findings to lead. Lead validates and identifies gaps."
```
Lead reviews both reports, challenges weak areas, decides readiness to proceed.

#### Plan
Lead directs planning, then validates before implementation:
```
"Spawn teammates. Use /caveman ultra mode.
- 'planner' (Sonnet): create task breakdown with TDD ordering, acceptance criteria, dependencies
- 'researcher' (Sonnet): validate plan feasibility, challenge assumptions, check coverage
Require plan approval from lead before any implementation begins."
```
Lead reviews plan, rejects if missing test coverage or acceptance criteria. Teammates revise until approved.

#### Implement
Teammates claim tasks from shared task list, implement via TDD. Lead monitors and reviews:
```
"Spawn implementation teammates. Use /caveman ultra mode.
- 'implementer-1' (Sonnet): owns src/module_a/ — TDD, eager commits
- 'implementer-2' (Sonnet): owns src/module_b/ — TDD, eager commits
- 'formatter' (Haiku): runs linting, formatting, boilerplate generation
Each teammate owns distinct files. No overlapping edits.
Commit after every meaningful change. Report to lead on completion."
```
Lead reviews completed work, requests revisions if needed.

#### Review
Lead orchestrates review — spawns review teammates, but lead makes final calls:
```
"Spawn review teammates. Use /caveman ultra mode.
- 'code-reviewer' (Sonnet): patterns, maintainability, CLAUDE.md compliance, security
- 'researcher' (Sonnet): correctness, algorithm invariants, edge cases, dtype choices
Report all findings to lead. Lead makes final accept/reject decision."
```
Lead synthesizes review findings, decides what must be fixed before marking complete.

#### Learn
Lead synthesizes all teammate findings into experience log entry. No team needed — lead does this directly.

### Team Coordination Rules

1. **Lead orchestrates AND validates** — teammates report to lead, lead approves/rejects
2. **No overlapping file ownership** — assign distinct file sets to avoid conflicts
3. **Lead gates phase transitions** — no Implement without Plan approval, no Complete without Review approval
4. **Lead picks teammate models** — sonnet default, haiku for routine tasks
5. **Wait for completion** — lead waits for all teammates before moving to next phase
6. **Clean up** — lead shuts down teammates and cleans up team when done

---

## Teammate Guidelines

### Permission-Aware Setup
Teammates inherit lead's permissions. Pre-approve common operations to reduce friction:
- **Research/review teammates**: primarily Read/Grep/Glob — rarely blocked
- **Implementation teammates**: need Write/Bash — ensure pre-approved for project paths
- If blocked, message teammate directly via Shift+Down with additional instructions

### Teammate Spawn Checklist
Include in every teammate spawn prompt:
1. The specific task ID and acceptance criteria
2. Which files they own (distinct from other teammates)
3. Key project rules (typing, Pydantic, NamedTuple vs Pydantic, etc.)
4. Commit message format: `<type>(<scope>): <description>`
5. What tests to write first (TDD)
6. `"Use Sonnet."` (model policy)
7. `"Use /caveman ultra mode."` (communication policy)

### Review Teammate Focus Areas
- **code-reviewer**: patterns, maintainability, security, CLAUDE.md compliance
- **ai-researcher**: correctness, algorithm invariants, edge cases, pytree structure, dtype choices, cross-field consistency

---

## Anti-Patterns

| Don't | Do Instead |
|-------|-----------|
| Jump straight to code | Reflect first, even briefly |
| Discover architecture while coding | Plan the approach, then build |
| Commit 500 lines at once | One logical change per commit |
| Write tests after implementation | Red-Green-Refactor always |
| Repeat the same mistake | Check and update the experience log |
| Let plans go stale | Update status with every completed task |
| Skip review on "simple" changes | At minimum, self-review the diff |
| Use subagents for story-level work | Use agent teams — subagents only for isolated subtasks |
| Skip lead validation | Lead (opus) must approve every phase transition |
| Use one agent perspective only | Pair researcher + architect teammates for non-trivial work |
| Treat docs as second-class | Same discipline: plan, write, review, commit |
| Modify code without understanding it | Read validation reports/docs first |
| Return dict where NamedTuple expected | Match pytree structure exactly for io_callback |
| Use `Any` in type signatures | Use Protocol or concrete types (mypy strict) |
| Skip cross-field validation | Add @model_validator for related Pydantic fields |
