# Bird Aero Vibration POC — Project Rules

## Communication Style

- Always use `/caveman full` mode by default. Terse, compressed responses — save tokens.

## Git Commits

- Do NOT add `Co-Authored-By` trailers to commit messages.

## Package Management

Use **uv** for all Python dependency management:

```bash
uv add <package>
uv add --dev <package>
uv run python ...
uv run pytest ...
uv sync
# Never use pip directly
```

## Code Style

- **Python**: Pydantic v2 for all data models (not dataclasses)
- **Type hints**: Full annotations, mypy strict mode
- **Testing**: pytest with 80%+ coverage target

## Python Typing Rules (mypy strict)

All code must pass `mypy --strict`. Follow these rules:

### Required Annotations
```python
def process(data: list[str], limit: int = 10) -> dict[str, int]:
    ...
```

### Modern Syntax (Python 3.11+)
```python
list[str]           # not List[str]
dict[str, int]      # not Dict[str, int]
tuple[int, ...]     # not Tuple[int, ...]
set[str]            # not Set[str]
name: str | None = None  # not Optional[str]
value: int | str         # not Union[int, str]
```

### Pydantic Models
```python
from pydantic import BaseModel, Field

class TrainingConfig(BaseModel):
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0, le=1)
    hidden_dim: int = 64
    num_layers: int = 8
```

### Avoid Any
```python
# Use specific types or generics, not Any
from typing import TypeVar
T = TypeVar("T")
def process(data: T) -> T: ...
```

### Import-Only Types
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibration_poc.model.meshgraphnet import MeshGraphNet

def get_model() -> "MeshGraphNet":
    ...
```

## Dev Workflow

Local-only enforcement via pre-commit hooks and Makefile.

### First-Time Setup
```bash
make setup  # installs deps, pre-commit hooks
```

### Common Commands
```bash
make check       # lint + format + typecheck + test
make test        # fast tests only
make test-all    # all tests including slow/integration
make lint        # ruff check with auto-fix
make format      # ruff format
make typecheck   # mypy strict
```

### Pre-commit Hooks (automatic on commit)
- ruff check (auto-fix)
- ruff format
- mypy strict
- pytest (fast tests)

### Test Markers
- `@pytest.mark.slow` — long-running tests
- `@pytest.mark.integration` — requires downloaded dataset
- `@pytest.mark.gpu` — requires GPU

## PyTorch / PhysicsNeMo Patterns

### Device Handling
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### Data Pipeline
- DeepMind deforming plate dataset (COMSOL FEA simulations)
- Graph representation: nodes = mesh vertices, edges = mesh connectivity
- Autoregressive prediction: predict next-step deformation from current state

### Model Architecture
- MeshGraphNet (Graph Neural Network)
- Encoder → Message Passing → Decoder
- PhysicsNeMo implementation preferred; torch-geometric fallback

## Development Discipline (SDLC)

> **MANDATORY**: All non-trivial work follows the `/sdlc` discipline defined in `.claude/skills/sdlc/SKILL.md`.

### Core Cycle

Every unit of work follows **Reflect -> Plan -> Implement -> Review -> Learn**. Depth scales with complexity:

- **Ad hoc task**: Quick reflection, mental plan, implement with commits, self-review
- **Story/feature**: Agent pairs (Researcher + Architect), TDD, eager commits, formal review
- **Epic**: Full PRD -> solution design -> implementation plan -> story-by-story SDLC

### Agent Teams (Default)

All non-trivial work uses **agent teams**, not standalone subagents:

| Role | Model | Responsibility |
|------|-------|---------------|
| **Lead** (main session) | `opus` | Orchestrate, validate, gate phase transitions, synthesize |
| **Teammates** (default) | `sonnet` | Execute research, coding, reviews |
| **Teammates** (simple) | `haiku` | Routine/mechanical tasks (formatting, lookups) |

- ALL teammates use `/caveman ultra` mode
- Lead reviews all output, approves/rejects plans, gates phase transitions

### Eager Commits

Commit after **every meaningful change**. Never batch unrelated changes. Format:
```
<type>(<scope>): <description>
Types: feat, fix, test, refactor, docs, chore, research
```

### TDD (When Writing Code)

Non-negotiable: **Red -> Green -> Refactor**. Tests are specifications, not afterthoughts.

### Experience Log

Lessons learned captured in `docs/sdlc/experience_log.md`. Always check before starting new work.

See `.claude/skills/sdlc/SKILL.md` for full details.

## Epic Implementation Workflow

> **IMPORTANT**: Plans are ALWAYS written to `docs/epics/epic_N/implementation_plan.md`, NOT to `.claude/plan.md`.

### Planning Phase
Each epic MUST have in `docs/epics/epic_N/`:
1. **prd.md** — Product Requirements Document
2. **solution_design.md** — Technical architecture
3. **implementation_plan.md** — Task breakdown with status tracking

### Status Legend
- ⬜ **Pending** — Not started
- 🔄 **In Progress** — Currently being worked on
- ✅ **Complete** — Done and verified
- ❌ **Blocked** — Cannot proceed
- 🔍 **Review** — Awaiting review
