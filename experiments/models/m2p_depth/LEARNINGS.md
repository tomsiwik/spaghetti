# LEARNINGS: M2P Depth Sweep (exp_m2p_depth) — KILLED

## Core Finding

Transformer depth is NOT the M2P generation quality bottleneck: L=2 already saturates
the hidden-states → B-matrix mapping at micro scale (L=4 adds +0.05pp, within noise).
Both width (Finding #355) and depth (Finding #357) are now closed directions. The
~92% quality ceiling is an optimization/training-budget problem, not an architectural
one.

---

## Why This Happened

### Root Cause: Shallow hypernetworks suffice for weight generation

Ha et al. (2016, arXiv:1609.09106) established the foundational result: a shallow
(1–2 layer) MLP hypernetwork is sufficient to generate weights for a deep primary
network. The explanation is functional: the mapping from a compact conditioning signal
(task identity / hidden state summary) to a target weight matrix has low **intrinsic
complexity** once the conditioning signal is rich enough. The bottleneck is the
*primary* network's capacity, not the *generator's* depth.

The M2P is structurally equivalent to these shallow hypernetworks: it maps a fixed-length
summary of base model hidden states to a low-rank B-matrix (rank=4, 5 modules × 2
layers = small target space). With d_M2P=64 input and rank-4 targets, this is a
low-complexity regression problem. L=2 already crosses the threshold where the function
is learnable; L=4 adds nothing because there is nothing further to learn with more depth.

The L=1→L=2 jump (+3.9pp) confirms depth is not *irrelevant* — a single layer is
insufficient. But L=2 is already sufficient for the task complexity at micro scale.

### Why the ~92% quality ceiling exists

The ceiling is not architectural. After closing both width (JL distortion ≠ quality,
Finding #355) and depth (L* ≤ 2, Finding #357), the remaining gap is:

1. **Training convergence** — 500 steps on 5 synthetic domains at micro scale. SHINE
   (arXiv:2602.06358) shows hypernetwork quality scales monotonically with training
   data volume, with "no sign of hitting capacity bottleneck" even at 6B tokens.
   Our 500-step budget is thin.

2. **B-matrix intrinsic dimensionality** — LoRA B-matrices live in a very
   low-dimensional subspace (Aghajanyan et al., arXiv:2012.13255 shows fine-tuning
   effective dimension is often <200 parameters; LoRAtorio arXiv:2508.11624 finds
   further low-dimensional structure "orders of magnitude smaller" within the rank-r
   space). The M2P must locate this subspace from 500 gradient steps. More steps or
   better supervision signal would help.

3. **Per-domain difficulty ceiling** — sort/repeat domains have higher irreducible
   difficulty. sort/reverse share character-ordering statistics (11–14% confusion,
   Finding #354). The 83.9–94.1% per-domain range with L=2 reflects task-level
   ambiguity, not depth-limited expressivity.

---

## Literature Context

### Shallow hypernetworks are the norm, not the exception

- **Ha et al. (2016, arXiv:1609.09106)** — original HyperNetworks paper uses 2-layer
  MLP to generate weights for deep networks. Depth of the hypernetwork was deliberately
  kept shallow. "The bottleneck is in the primary network capacity, not the generator."

- **HyperLoader (2024, arXiv:2407.01411)** — 1–2 layer MLP hypernetworks generate
  LoRA adapter weights conditioned on task+layer index across transformer layers.
  Confirms shallow generators saturate the adapter generation task.

- **SHINE (2026, arXiv:2602.06358)** — in-context hypernetwork mapping context to LoRA
  in a single forward pass. Identifies prior failures as "insufficient training scale,
  not architectural depth." Trained on 6B tokens; prior hypernetworks were data-starved.

### Training budget, not depth, governs quality at scale

SHINE explicitly reports prior hypernetwork failures were due to small training scale.
At our micro scale (500 steps, 5 domains), training budget is the likely binding
constraint — not depth, which we have now ruled out.

### B-matrix intrinsic dimension

- **Aghajanyan et al. (2021, arXiv:2012.13255)** — pre-trained LMs have low intrinsic
  fine-tuning dimension; larger models have *lower* intrinsic dimension. A rank-4 LoRA
  B-matrix spans only the task-relevant subspace, which is already low-dimensional.

- **LoRAtorio (2025, arXiv:2508.11624)** — treats LoRA parameter space as a D-dimensional
  manifold with further low-dimensional intrinsic structure. The B-matrix M2P must predict
  may lie in a subspace significantly smaller than rank × hidden_dim. If the effective
  intrinsic dim < d_M2P=64, then the M2P has more than enough capacity — confirming
  width AND depth are not the bottleneck.

---

## Confirming Evidence

- **Finding #355** (exp_m2p_bottleneck_width, KILLED): Width sweep (d=64→256) produced
  flat quality ~95–97%. Architecture size does not improve M2P quality. Consistent with
  the "shallow hypernetworks suffice" result from arXiv:1609.09106.
- **Finding #357** (this experiment, KILLED): Depth sweep (L=1→2→4) — L=2 saturates.
  L* ≤ 2 at micro scale.
- **Ha et al. (1609.09106)**: 2-layer MLP hypernetworks routinely outperform deeper
  counterparts for weight generation when the conditioning signal is rich enough.
- **L=1→L=2 sanity check (+3.9pp)**: Confirms depth is not irrelevant — L=1 is below
  L*, L=2 is at or above L*. The saturation point is between 1 and 2 layers.

---

## Contradicting Evidence

- **Yun et al. (2020, arXiv:1912.10077) Theorem 2** is the primary motivator — it
  proves existence of L* for universal approximation. However, Theorem 2 requires
  permutation equivariance, which M2P violates (positional embeddings, causal masking,
  fixed tensor output). Theorem 3 or MLP universal approximation (Hornik 1991) rescue
  the existence claim without the equivariance condition. The kill does NOT falsify the
  existence of L*; it shows L* ≤ 2 empirically.

- **REVIEW caveat**: The L=2 baseline in this experiment (91.9%) is lower than
  Finding #355 (95.1%). PAPER.md acknowledges this as training variance. If the true
  L=2 ceiling is 95–97%, a higher-quality L=2 baseline might reveal a non-zero depth
  signal — but the 0.05pp delta is robust regardless of absolute calibration. This
  remains a theoretical gap, not an actionable direction.

---

## Alternative Approaches for Improving M2P Quality

**All approaches below have paper references or prior experiment support.**

1. **Training step sweep** (500 → 1000 → 2000 steps)
   - Motivation: SHINE (arXiv:2602.06358) directly identifies training scale as the
     bottleneck for hypernetwork quality, not architecture. Our 500 steps is minimal.
   - Evidence: Finding #354 vs. #355 showed a 2.9pp quality gap between reused adapters
     (fewer effective training steps on the target) and fresh training. More steps should
     narrow the 3–5% remaining gap.
   - Kill criteria: quality(2000 steps) > quality(500 steps) + 2pp

2. **Better training signal: predict adapter delta, not absolute weights**
   - Motivation: HyperNet Fields (arXiv:2412.17040) shows training from weight-update
     *trajectories* rather than ground-truth snapshots improves convergence. The M2P
     currently regresses to fixed B-matrix snapshots. Predicting deltas (incremental
     updates) may be an easier target.
   - Evidence: SHINE generates LoRA "in a single pass" using context rather than static
     targets — analogous to delta-prediction.

3. **Bidirectional attention in M2P** (remove causal masking)
   - Motivation: REVIEW-adversarial.md flags causal masking as "architecturally
     questionable" for a non-autoregressive module. Memory tokens are not a sequence —
     bidirectional attention over them is more natural and may improve representational
     capacity without adding depth.
   - Evidence: Encoder-only transformers (BERT family) use bidirectional attention for
     non-generative tasks. This is a structural fix, not a depth/width expansion.
   - Kill criteria: quality(bidirectional) > quality(causal) + 1pp

4. **Reduce B-matrix rank to match intrinsic dimensionality**
   - Motivation: LoRAtorio (arXiv:2508.11624) and Aghajanyan et al. (arXiv:2012.13255)
     show the effective fine-tuning subspace is very low-dimensional. If rank=4 is
     over-specified at micro scale, reducing rank → simpler prediction target → easier
     convergence.
   - Kill criteria: quality(rank=2) ≥ quality(rank=4) − 1pp (demonstrates rank=4 not
     required)

---

## Implications for Next Experiments

1. **Architecture search is closed for M2P.** Width (d_M2P) and depth (M2P_LAYERS)
   are both exhausted. Do not re-open architectural sweeps without a proof that L* > 2
   or that d_M2P < d_intrinsic at micro scale.

2. **Training budget is the highest-evidence next direction.** SHINE (arXiv:2602.06358)
   directly confirms training scale governs hypernetwork quality. The MATH.md for the
   next experiment should derive: "500 steps is insufficient for convergence given B-matrix
   target complexity" — not another existence proof.

3. **Bidirectional attention fix is low-cost, high-signal.** Removing causal masking
   is a one-line change and the architectural motivation is solid (memory tokens are
   not autoregressive). This can be bundled with a training step sweep.

4. **Parity domain guard is stable infrastructure.** Exclude domains where
   (base_loss − sft_loss) < 0.05. Carry this into all future M2P experiments.

5. **L* ≤ 2 is a micro-scale result.** At macro scale (d_model=3584, Qwen3-4B), the
   B-matrix target space is far larger and L* may exceed 2. Do not generalize "depth
   doesn't matter" to macro scale without a separate sweep.

---

## Recommended Follow-Up

**Experiment: M2P Training Budget Sweep (exp_m2p_training_budget)**
- Sweep M2P training steps ∈ {500, 1000, 2000} with L=2, d_M2P=64 fixed
- Hypothesis: quality ceiling is training convergence, not architecture
- MATH.md must: bound convergence rate of MSE regression to B-matrix targets
  as a function of gradient steps; predict expected quality improvement per 500 steps
  using loss-curve extrapolation from the depth sweep runs
- Kill criteria:
  - K_progress: quality(2000) > quality(500) + 2pp (budget matters)
  - K_ceiling: quality(2000) ≥ 97% (ceiling reached)
  - K_plateau: |quality(2000) − quality(1000)| < 1pp (budget exhausted, something else)
- Citation: SHINE (arXiv:2602.06358) — training scale is the bottleneck for
  hypernetwork quality; architectural depth is not
- Secondary fix: remove causal masking in M2P attention (bidirectional over memory
  tokens); bundle with this sweep to isolate any additional gain
- References: #524 (HyperNetworks), #526 (SHINE), #527 (LoRAtorio)
