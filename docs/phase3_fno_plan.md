# Phase 3: Fourier Neural Operator (FNO) for Vibration Analysis

## Context

### Where We Are

The POC uses **MeshGraphNet** (a Graph Neural Network) to predict structural vibration. It works end-to-end — data pipeline, training, inference, Streamlit demo — but accuracy is limited:

- Predictions cluster around ~1.2mm regardless of input (model learned an average)
- 2× overshoot on mid-range cases, 4× undershoot on large-deformation cases
- More training data didn't help — the bottleneck is **autoregressive error compounding** (50 sequential predictions, each feeding errors into the next)

### Why FNO

**Fourier Neural Operator** is a fundamentally different architecture that addresses exactly our weaknesses:

1. **Works in frequency domain** — vibration IS frequencies. The network's internal representation matches the physics naturally
2. **No autoregressive rollout needed** — can predict the full displacement field in one shot, eliminating error compounding
3. **Built-in regularization** — truncating high Fourier modes constrains the output to physically-plausible smooth fields
4. **Better data efficiency** — the spectral inductive bias means less data needed to learn the same dynamics

---

## Part 1: Understanding FNO (Conceptual Foundation)

### What is a Neural Operator?

A regular neural network learns a mapping between fixed-size vectors: input vector → output vector. If you change the mesh resolution (say from 1305 to 5000 nodes), you must retrain because the dimensions changed.

A **neural operator** learns a mapping between **functions** (continuous fields). It approximates the solution operator of the underlying PDE:

```
Input: displacement field u(x, y, z) defined over the 3D plate
Output: next displacement field u'(x, y, z) over the same domain
```

The key property is **discretization invariance** — you can train on a coarse grid and evaluate on a fine grid, because the operator learns the continuous relationship, not a discrete vector mapping. Think of it like learning "how displacement fields evolve" rather than "how these 1305 specific numbers change."

**Analogy**: A regular neural network is like memorizing a lookup table. A neural operator is like learning the rule that generates the table — it works for any table size.

### Why Fourier?

When a plate vibrates, the motion decomposes into **vibration modes** — standing wave patterns at specific frequencies:

- **Mode 1** (lowest frequency): the whole plate bows in the middle (fundamental mode)
- **Mode 2**: the plate has one nodal line where displacement = 0, two lobes moving oppositely
- **Mode 3+**: increasingly complex patterns with more nodal lines

These modes are mathematically very close to **Fourier basis functions**. For simple geometries (rectangles, circles), they literally ARE sinusoidal. This means:

- Fourier coefficients directly represent vibration mode amplitudes
- Low-frequency coefficients capture the dominant structural behavior
- High-frequency coefficients are mostly noise/fine detail

A Fourier-based network has a natural inductive bias: it "thinks" in the same basis the physics uses.

**Compare to MeshGraphNet**: The GNN has no frequency-domain awareness. It must learn oscillatory wave behavior from local node-to-node messages, which is fundamentally harder — like trying to understand ocean waves by only looking at neighboring water molecules.

### How an FNO Layer Works

Each FNO layer has two parallel paths that are summed:

```
                    ┌─── FFT ──→ Spectral Conv (R) ──→ iFFT ───┐
Input v(x) ────────┤                                            ├──→ Activation ──→ Output
                    └─── Local Linear (W) ─────────────────────┘
```

**Step by step:**

**1. FFT path (captures global/wave behavior):**
- Apply Fast Fourier Transform to the input field
- Now each coefficient represents a specific spatial frequency
- Apply a learnable weight matrix R to these coefficients — this learns how vibration modes interact with each other ("mode 3 at 150 Hz excites mode 7 at 420 Hz")
- **Truncate**: only keep the lowest k_max modes. Real structures are dominated by ~10 low modes — high modes are noise. This truncation IS physics knowledge baked into the architecture
- Inverse FFT back to physical space

**2. Local path (captures boundary/point effects):**
- A pointwise linear transformation W (like a 1×1 convolution)
- Each point is transformed independently
- Captures local effects the truncated FFT path misses: stress concentrations, boundary layers, loading points

**3. Sum + activation:** Add both paths, apply GeLU nonlinearity.

**Why two paths?** The FFT path is great for global correlations (a wave on one side of the plate affects the other side) but poor at sharp local features (because high frequencies were truncated). The local path fills that gap. Together they approximate the physics solution kernel.

### What "Modes" Means (the double meaning)

"Modes" appears in two related contexts:

**Fourier modes** (in the FNO): the number of frequency coefficients k_max we keep. With k_max=12, the spectral path can represent patterns with up to 12 oscillations across the domain. Higher frequencies are handled only by the local path.

**Vibration modes** (in structural engineering): the natural resonant patterns of the structure. A plate might have 5-10 dominant modes in the frequency range of interest.

**The connection**: if the plate has ~10 dominant vibration modes, keeping k_max=12 Fourier modes captures all the important physics. The truncation of higher modes isn't just computational savings — it's physics-informed regularization. The network is structurally prevented from fitting high-frequency noise.

**Why this fixes our MeshGraphNet problem**: The current model predicts "~1.2mm regardless of input" because it can't reliably predict high-frequency dynamics, so it hedges toward the mean. FNO's mode truncation explicitly says "don't try to predict high-frequency details" — instead, it focuses capacity on the low-frequency modes that actually determine the vibration behavior.

### Single-Shot vs. Autoregressive

**MeshGraphNet (current)**: predict one step → update state → predict next step → repeat 50 times. Error compounds at each step.

**FNO can do both**, but its strength is **single-shot**: predict the entire displacement field at once, or even predict multiple timesteps simultaneously by encoding time as additional channels. No error compounding.

For our problem, we'd start single-step (same as now, validates the pipeline), then upgrade to multi-step where the FNO predicts steps 1-50 in one forward pass.

---

## Part 2: The Irregular Mesh Problem

### The Challenge

FNO needs data on a **regular grid** (uniform spacing, like a 3D image/voxel grid) because FFT requires evenly-spaced samples. Our FEA mesh is irregular tetrahedra with ~1305 nodes at non-uniform positions.

### Three Solutions

#### Approach 1: Grid Interpolation (Recommended to start)

```
Irregular mesh → scipy interpolation → Regular grid → FNO → Regular grid → scipy interpolation → Irregular mesh
```

- Use `scipy.interpolate.LinearNDInterpolator` (mesh → grid) and `scipy.interpolate.RegularGridInterpolator` (grid → mesh)
- Simple, well-understood, off-the-shelf tools
- **Downside**: interpolation is lossy — introduces ~1-5% error before the model even sees the data
- **Good enough for POC**: we can measure the interpolation error and decide if it's acceptable

#### Approach 2: Geo-FNO (Upgrade path)

```
Irregular mesh → Learned deformation φ → Latent regular grid → FNO → Latent grid → φ⁻¹ → Irregular mesh
```

- Published 2023 (Li et al.). A small MLP learns to map irregular coordinates to a regular "computational space"
- The deformation is trained jointly with the FNO — the network learns the best grid for the problem
- Less interpolation error, but more complex to implement and train
- **When to consider**: if Approach 1's interpolation error is too high (>5%)

#### Approach 3: Graph FNO (Research-stage)

- Define the Fourier transform on the graph itself using the mesh Laplacian eigendecomposition
- Theoretically cleanest — no interpolation at all
- **Status**: research-stage, few battle-tested implementations, not recommended for POC

**Plan**: implement Approach 1 first. Measure interpolation quality. If acceptable, ship it. If not, upgrade to Approach 2.

---

## Part 3: Architecture Design

### Grid Resolution

The plate geometry: ~250×488×136mm with ~18mm average edge length.

```
Grid: 32 × 64 × 16  (x, y, z)
      ~8mm   ~8mm  ~5mm   grid spacing
```

This respects the plate's aspect ratio (~1:2:0.5) and resolves features at the mesh edge scale. The z dimension is smaller because the plate is thin.

Alternative: 16×32×8 for fast experimentation (coarser, ~4K points vs ~32K).

### Fourier Modes

```
modes_x = 12, modes_y = 12, modes_z = 8
```

12 modes in x/y captures the dominant vibration patterns. 8 in z (thin dimension). Total spectral parameters per layer: 12 × 12 × 8 = 1,152 complex-valued weights per channel pair.

### Full Architecture

```
Input: [B, 7, 32, 64, 16]
       channels: world_pos(3) + mesh_pos(3) + node_type(1)

Lifting:        Linear(7 → 64)              ← project to hidden dim

FNO Block ×4:   SpectralConv3d(64, 64, modes=(12,12,8))   ← global Fourier path
              + Conv3d(64, 64, kernel=1)                    ← local bypass path
              + BatchNorm3d(64)
              + GeLU

Projection:     Linear(64 → 128) → GeLU → Linear(128 → 3)

Output: [B, 3, 32, 64, 16]
        channels: displacement(3)
```

~500K-800K parameters. Comparable to MeshGraphNet but with spectral inductive bias.

### Occupancy Mask

The plate is an irregularly-shaped thin object inside a rectangular bounding box. Most grid voxels are **empty** (outside the plate). We need a binary mask:

```
occupancy_mask: [B, 1, 32, 64, 16]  — 1.0 inside plate, 0.0 outside
```

Used for:
- **Masking the loss** — only compute loss where the plate exists, otherwise the model would be rewarded for predicting zero in empty space
- **Input channel** — optionally feed as 8th input channel so the model knows the geometry

### Boundary Conditions

Same concept as current, adapted for grid:
- Interpolate the node_type field onto the grid
- Create a BC mask on the grid (grid points near boundary nodes)
- **Soft BC**: loss penalty for non-zero displacement at BC locations (same as now)
- **Hard BC**: multiply output by (1 - BC_mask) to force zero at boundaries

---

## Part 4: Data Pipeline Changes

### Current vs. New

```
Current pipeline:
  TFRecord → graph dict {x[N,4], edge_index[2,E], edge_attr[E,4], y[N,3]} → MeshGraphNet

New pipeline:
  TFRecord → graph dict → INTERPOLATE → grid tensor [C, Gx, Gy, Gz] → FNO3d
                                                                         ↓
                                                          INTERPOLATE BACK → displacement[N,3]
```

### What Changes

| Aspect | MeshGraphNet | FNO |
|--------|-------------|-----|
| Input format | Graph dict (variable-size) | Grid tensor (fixed-size) |
| Batch size | Always 1 (variable graph sizes) | 4-16 (fixed grids, easy to batch) |
| Edge features | Needed (relative position, distance) | **Not needed** — spatial relations captured by FFT |
| Normalization | Per-node-feature mean/std | Per-channel mean/std on grid |
| File size per sample | ~300KB (.pt graph) | ~1.4MB (.pt grid) |
| DataLoader | Custom collate (batch[0]) | Standard PyTorch batching |

### Grid Tensor Format

Each preprocessed sample:
```python
{
    "grid_input":      Tensor [7, 32, 64, 16],   # input channels on grid
    "grid_target":     Tensor [3, 32, 64, 16],   # target displacement on grid
    "occupancy_mask":  Tensor [1, 32, 64, 16],   # where the plate is
    "mesh_pos":        Tensor [N, 3],             # original mesh (for back-interpolation)
    "node_type":       Tensor [N, 1],             # for BC at inference
    "grid_bounds":     Tensor [6],                # bounding box
}
```

### Normalization

No more edge normalization (no edges). Grid channels normalized independently:
```python
class GridNormStats(BaseModel):
    channel_mean: list[float]   # length 7 (input channels)
    channel_std: list[float]    # length 7
    target_mean: list[float]    # length 3
    target_std: list[float]     # length 3
```

---

## Part 5: Training Changes

### Loss Function

**Primary**: Masked MSE — only compute loss where the plate exists:
```python
loss = ((pred - target)² × occupancy_mask).sum() / (mask.sum() × 3)
```

Without masking, the model would get artificially low loss by predicting zero everywhere — most of the bounding box is empty.

### Physics Losses (adapted for grid)

**Grid smoothness** — finite-difference gradient penalty instead of edge-based:
```python
dx = pred[:,:,1:,:,:] - pred[:,:,:-1,:,:]   # spatial gradient in x
# same for y, z
smoothness_loss = (dx² + dy² + dz²).mean()
```

**BC penalty** — same concept, applied to grid points near boundary nodes.

### PDE Residual Loss (stretch goal)

The governing PDE for structural vibration:
```
ρ ∂²u/∂t² = ∇·σ + f     (Newton's 2nd law for continua)
```

On a regular grid, we can compute ∇·σ using finite differences of the predicted displacement. The loss penalizes how badly the prediction violates this equation.

**Caveat**: requires material properties (Young's modulus, Poisson's ratio) not in our dataset. Skip for initial implementation, add later if material properties can be estimated or assumed.

### Training Characteristics

| | MeshGraphNet | FNO |
|---|---|---|
| Batch size | 1 | 8-16 |
| GPU memory/sample | Low | Higher (grid > mesh) |
| Speed | Slow (scatter ops) | Fast (FFT optimized on GPU) |
| Learning rate | 1e-4 | 1e-3 (FNO typically higher) |
| LR scheduler | ExponentialLR | CosineAnnealingLR |
| Convergence | 50-100 epochs | 20-50 epochs |

---

## Part 6: Implementation Tasks

### Phase 3A: Data Pipeline for Grid

**T40: Grid interpolation module** `NEW src/vibration_poc/dataset/grid_interpolation.py`
- `MeshToGridInterpolator(mesh_pos, grid_resolution, padding)` — Delaunay-based interpolation
- `GridToMeshInterpolator(grid_bounds, grid_resolution)` — RegularGridInterpolator
- `compute_occupancy_mask()` — binary mask of plate geometry on grid
- Acceptance: round-trip error (mesh→grid→mesh) < 5% on test samples

**T41: Grid config** `MODIFY src/vibration_poc/dataset/config.py`
- Add `GridConfig(BaseModel)`: resolution, padding_ratio, modes
- Add `GridNormStats(BaseModel)`: channel mean/std, target mean/std
- Acceptance: backward-compatible with existing graph pipeline

**T42: Grid preprocessing** `NEW src/vibration_poc/dataset/preprocess_grid.py`
- `preprocess_grid_split()` — TFRecord → graph → grid tensor → .pt files
- Streaming to disk (same pattern as current preprocessor)
- Compute grid normalization stats
- Acceptance: produces correctly-shaped grid tensors for all splits

**T43: Grid DataLoader** `NEW src/vibration_poc/dataset/grid_dataloader.py`
- `GridDataset(Dataset)` — loads grid .pt files
- Standard batched DataLoader (no custom collate needed — fixed tensor sizes!)
- Acceptance: yields batches [B, C, Gx, Gy, Gz] with B > 1

### Phase 3B: FNO Model

**T44: SpectralConv3d** `NEW src/vibration_poc/model/fno.py`
- Complex-valued weight tensor [in_ch, out_ch, modes_x, modes_y, modes_z]
- Forward: `torch.fft.rfftn` → spectral multiply → `torch.fft.irfftn`
- Handle real-FFT output shape (last dim is Gz//2 + 1)
- Acceptance: correct output shape, gradient flows, matches expected spectral convolution on simple inputs

**T45: FNO3d model** `SAME FILE src/vibration_poc/model/fno.py`
- Lifting → 4 FNO blocks → Projection
- Constructor: `input_channels, output_channels, hidden_dim, num_layers, modes`
- Same forward interface pattern as MeshGraphNet (takes dict, returns tensor)
- Acceptance: correct shapes, ~500K-800K params, can overfit single sample

**T46: Model factory** `MODIFY src/vibration_poc/model/__init__.py`
- `create_model(model_type, **kwargs)` → MeshGraphNet or FNO3d
- Acceptance: both model types instantiate correctly

### Phase 3C: Training

**T47: Grid physics losses** `MODIFY src/vibration_poc/physics.py`
- `compute_masked_mse(pred, target, mask)` — occupancy-masked MSE
- `compute_grid_smoothness_loss(pred, mask)` — finite-difference gradient penalty
- `compute_grid_bc_penalty(pred, bc_mask)` — boundary enforcement on grid
- Acceptance: losses compute, gradients flow, BC nodes get zero displacement

**T48: FNO trainer** `NEW src/vibration_poc/training/trainer_fno.py`
- `train_fno(config, dataset_config)` — full training loop
- CosineAnnealingLR scheduler, masked MSE + physics losses
- Same checkpoint pattern as current trainer
- Acceptance: loss decreases, checkpoints save/load

### Phase 3D: Inference + Integration

**T49: FNO inference** `NEW src/vibration_poc/inference/predict_fno.py`
- `predict_fno(model, graph, norm_stats, grid_config)` — single-step prediction
- `rollout_fno(...)` — autoregressive rollout (grid-based, results interpolated back to mesh)
- **Same output format** as current `rollout()` — list of dicts with world_pos, predicted_displacement, mesh_pos
- Acceptance: output compatible with existing visualization and FFT analysis

**T50: Engine update** `MODIFY src/vibration_poc/engine.py`
- `load_fno_model()` function
- `evaluate_design()` accepts either model type
- DesignMetrics format unchanged
- Acceptance: evaluation works with both models, same API

**T51: App update** `MODIFY src/vibration_poc/app.py`
- Model type selector in sidebar (MeshGraphNet / FNO)
- Load appropriate model, run appropriate inference
- Acceptance: UI works with both model types, visualization unchanged

### Phase 3E: Validation

**T52: Interpolation quality tests** `NEW tests/test_grid_interpolation.py`
**T53: FNO model tests** `NEW tests/test_fno.py`
**T54: Comparison notebook** `NEW notebooks/fno_vs_gnn_comparison.ipynb`
- Train both on same data, compare: single-step MSE, rollout accuracy at steps 1/10/25/50, frequency spectrum accuracy, runtime

### Task Dependencies

```
T41 (config) ─────→ T40 (interpolation) ─→ T42 (preprocess) ─→ T43 (dataloader)
                                                                        ↓
T44 (SpectralConv) → T45 (FNO3d) → T46 (factory)                      ↓
                                                                        ↓
T47 (grid physics) ─────────────────────────────→ T48 (trainer) ←──────┘
                                                        ↓
T40 + T45 ──→ T49 (inference) → T50 (engine) → T51 (app)
                                                    ↓
T52, T53, T54 (tests + comparison) ←───────────────┘
```

Phases 3A (data) and 3B (model) can run in parallel. 3C (training) needs both. 3D (inference) needs 3A+3B. 3E is last.

---

## Part 7: What Could Go Wrong

### High Risk

**Interpolation error too large** — mesh→grid→mesh round trip loses information. If >5%, predictions are garbage regardless of model quality. **Mitigation**: measure in T52 before training. If bad, increase grid resolution or move to Geo-FNO.

**3D FFT memory** — 32×64×16 grid with 64 channels, batch 8: ~67MB per tensor. With FFT intermediates + gradients + optimizer, could hit 4-8GB per layer. **Mitigation**: start batch_size=4, use gradient checkpointing, try 16×32×8 first.

**FNO still learns average** — if the issue is data diversity not architecture, FNO won't help. **Mitigation**: FNO's spectral bias should help, but if single-step FNO still shows this, add temporal conditioning (timestep as input channel).

### Medium Risk

**Occupancy mask edge artifacts** — grid points at the plate boundary may have partial occupancy. **Mitigation**: use smooth (distance-based) masks instead of hard binary.

**No temporal encoding** — current single-step has no notion of "where in the vibration cycle." Adding velocity (pos[t] - pos[t-1]) as input could help.

---

## Part 8: Expected Outcomes

| Metric | MeshGraphNet (current) | FNO (expected) |
|--------|----------------------|----------------|
| Single-step MSE | ~1e-6 | ~1e-7 (spectral bias helps) |
| Rollout accuracy (50 steps) | 2-4× off | <1.5× off |
| Frequency spectrum | Flat (no real modes) | Should show actual vibration modes |
| Inference time (50 steps) | ~5s CPU | ~1s CPU (potentially single-shot) |
| Training time (20 epochs) | ~1 hr Colab T4 | ~30 min (batching + FFT efficiency) |

---

## Part 9: References

### Essential Reading

1. **FNO paper** — Li et al., "Fourier Neural Operator for Parametric PDEs", ICLR 2021
   - [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)
   - Read: Sections 1-3 (theory), Section 4 (architecture). The key figure is Figure 1 showing the spectral convolution layer.

2. **Geo-FNO paper** — Li et al., "Fourier Neural Operator with Learned Deformations for PDEs on General Geometries", JMLR 2023
   - [arXiv:2207.05209](https://arxiv.org/abs/2207.05209)
   - Read: Section 3 (deformation layer). This is our upgrade path if grid interpolation isn't accurate enough.

3. **MeshGraphNet paper** — Pfaff et al., "Learning Mesh-Based Simulation with Graph Networks", ICLR 2021
   - [arXiv:2010.03409](https://arxiv.org/abs/2010.03409)
   - What we currently use. Good for understanding what we're replacing and why.

### Implementation References

4. **neuraloperator library** (by the FNO authors)
   - [github.com/neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator)
   - MIT licensed. Reference implementation of FNO, TFNO, Geo-FNO. Could use directly or as reference.

5. **NVIDIA PhysicsNeMo / Modulus**
   - [docs.nvidia.com/physicsnemo](https://docs.nvidia.com/physicsnemo/)
   - Has production-quality FNO for regular grids (`physicsnemo.models.fno`). Already in our pyproject.toml as optional dep. Does NOT handle irregular meshes — we still need our interpolation layer.

6. **FNO tutorial notebooks** (by Zongyi Li)
   - [github.com/neuraloperator/neuraloperator/tree/main/examples](https://github.com/neuraloperator/neuraloperator/tree/main/examples)
   - Worked examples on Navier-Stokes (2D), Darcy flow. Good for understanding the training loop.

### Background / Deeper Understanding

7. **Neural Operator review** — Kovachki et al., "Neural Operator: Learning Maps Between Function Spaces", JMLR 2023
   - [arXiv:2108.08481](https://arxiv.org/abs/2108.08481)
   - Comprehensive review of neural operators (FNO, DeepONet, etc.). Good for understanding the theoretical foundation.

8. **Physics-Informed Neural Networks (PINNs)** — Raissi et al., "Physics-informed neural networks", JCP 2019
   - [doi.org/10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)
   - Background on PDE-informed training losses. Relevant if we add PDE residual loss.

9. **DeepMind deforming plate dataset**
   - [github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)
   - Our dataset source. Documentation on what the fields mean.

---

## Verification

```bash
# Run all tests
uv run pytest tests/ -q --tb=short

# Verify interpolation quality
uv run pytest tests/test_grid_interpolation.py -v

# Verify FNO model
uv run pytest tests/test_fno.py -v

# Type check
uv run mypy src/ --strict

# Train FNO (Colab)
# Set model_type="fno" in notebook, run training cells

# Compare models
# Run notebooks/fno_vs_gnn_comparison.ipynb

# Demo app with FNO
make demo
# Select "FNO" model type in sidebar, upload mesh, run inference
```
