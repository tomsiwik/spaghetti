# PAPER.md — T1.2: HRA vs LoRA Quality (exp_p1_t1_householder_vs_lora)

**Finding status: KILLED** (K1012 FAIL, K1013 FAIL)

---

## V2 Rerun (2026-04-18) — audit-2026-04-17-rerun, code-bug

**Verdict: KILLED (unchanged from v1).**

Disposition of the audit-flagged `code-bug`:

**Bug.** Original `run_experiment.py:507` computed K1013 as
`conv_ratio <= 2.0 or (both_DNF)`. When HRA hit the sentinel conv_step=
`TRAIN_STEPS + 1 = 301` and LoRA converged at step 240, the ratio
301/240 = 1.254 ≤ 2.0 → **false PASS**. Reviewer caught this in v1
REVIEW-adversarial §2 (non-blocking) and manually overrode to FAIL in
v1 PAPER.md; the `results.json` however still carried the false PASS.

**Fix.** `run_experiment.py:505-519` now branches on `hra_converged` and
`lora_converged` explicitly. Only HRA-converged AND ratio ≤ 2.0 is PASS;
HRA DNF is FAIL regardless of LoRA state. `results.json` additionally
persists `hra_converged`, `lora_converged`, `train_steps`, and
`conv_threshold` for audit transparency, plus top-level `verdict`,
`all_pass`, and `ran` fields.

**Applied to v1 measurements** (HRA_conv=301, LoRA_conv=240):
- `hra_converged = False` (DNF), `lora_converged = True` → K1013 **FAIL**.
- K1011 PASS, K1012 FAIL, K1014 PASS all unchanged by the code fix.
- Overall verdict KILLED — same as v1. The `results.json` now correctly
  reflects K1013=FAIL (was PASS+manual-override in v1).

**Why no fresh measurement.** Re-running under the current platform
state (Python 3.14, `datasets` library) fails with
`TypeError: Pickler._batch_setitems() takes 2 positional arguments but
3 were given` in `datasets/utils/_dill.py:Hasher.hash` before any
MLX code executes. This is a `datasets`/`dill` upstream incompat with
Python 3.14, not an experimental issue. Per the researcher hat rule
"Do not re-run the experiment for documentation-only fixes", and
because the K1013 fix only changes how KCs are *derived* from
measurements (not the measurements themselves), `results.json` is
reconstructed from the documented v1 numbers with the corrected K1013
logic applied retroactively. Provenance recorded in
`results.json._reconstruction_note`.

**Permanently learned — propagate to sibling convergence-ratio
experiments.** The sentinel pattern `conv_step = TRAIN_STEPS + 1 when
threshold-never-crossed` combined with a naive `ratio <= k` test is a
specific instance of a broader antipattern: **when a measurement uses
a sentinel "not-applicable" value, any downstream KC that just
arithmetically compares must first branch on whether the sentinel was
hit**. In T1.2 this was the DNF case; in general it applies anywhere
a metric can be "undefined" (divide-by-zero guards, early-stopping
skips, NaN from empty batches). The fix is structural (explicit
branches), not cosmetic (`(both_DNF)` edge-case OR).

---

## Prediction vs Measurement

| Metric | Prediction | Measured | K | Pass? |
|--------|-----------|----------|---|-------|
| HRA GSM8K ≥ LoRA GSM8K | HRA ≥ LoRA | HRA=8% vs LoRA=5% (+3pp) | K1011 | **PASS** |
| HRA MMLU ≥ LoRA MMLU | HRA ≥ LoRA | HRA=50% vs LoRA=56% (-6pp) | K1012 | **FAIL** |
| Conv steps HRA ≤ 2× LoRA | ≤ 2× (≤480 steps) | HRA: never (<0.5) vs LoRA: step 240 | K1013 | **FAIL** |
| Step time ratio ≤ 3× | ≤ 1.5× (predicted) | 1.374× (0.246s vs 0.179s) | K1014 | **PASS** |
| HRA params/layer | 40,960 | 40,960 (16×2560) | Structural | ✓ |
| LoRA params/layer | 106,496 | 106,496 (2×16×(2560+4096)/2) | Structural | ✓ |

## Raw Measurements

| Adapter | GSM8K acc | MMLU acc | Train steps | Conv step | Avg step time | Params |
|---------|-----------|----------|-------------|-----------|---------------|--------|
| Base    | 7%        | 54%      | —           | —         | —             | — |
| LoRA r=16 | 5%      | 56%      | 300         | 240       | 0.179s        | 3,833,856 |
| HRA r=16 | 8%       | 50%      | 300         | 301 (DNF) | 0.246s        | 1,474,560 |

**Note on lora_params_per_layer:** results.json previously showed 212,992 (doubled).
Corrected value is 106,496 per layer = n_trainable (3,833,856) / 36 layers.
HRA uses 1,474,560 / 36 = 40,960 per layer = 38.5% of LoRA. This is confirmed.

---

## Interpretation

### K1011 PASS (marginal, within noise)
HRA GSM8K: 8% vs LoRA 5% (+3pp) vs base 7%. Both adapters perform near base, suggesting
300 steps is insufficient for meaningful GSM8K learning. The 3pp difference is within noise
at n=100. Technically a PASS on the criterion, but not a strong result.

### K1012 FAIL (-6pp MMLU regression)
HRA MMLU: 50% vs LoRA 56% (base 54%). HRA causes -4pp regression from base; LoRA gives +2pp.
This is unexpected: MMLU measures broad world-knowledge retrieval, not task-specific learning.
The Householder reflections multiplicatively transform query representations (y = H^(r)x),
potentially rotating representations away from MMLU-relevant directions that are already
encoded in the base model. LoRA's additive update (y = x + BAx) preserves base directions.

### K1013 FAIL (HRA never converges)
LoRA reached loss < 0.5 at step 240. HRA final loss never reached < 0.5 in 300 steps.
Conv_step=301 means the threshold was never crossed. This confirms T1.4's finding:
Stiefel manifold optimization with standard Euclidean Adam is sub-optimal. The gradient
∂L/∂V does not respect the manifold constraint, leading to slower convergence than
LoRA's unconstrained Euclidean optimization.

### K1014 PASS (1.374× step time, well within 3×)
HRA Python loop overhead (r=16 sequential reflections vs 2 vectorized matmuls) adds 37%.
Theorem 2 prediction of ≤ 1.5× confirmed. For the NoPE slice (d=384 instead of d=2560),
overhead would be lower (each reflection is 3× cheaper).

---

## Impossibility Structure

Three structural reasons why HRA at equal rank fails vs LoRA:

**1. Euclidean ≠ Riemannian on Stiefel manifold**
Standard Adam's update V_new = V - η·∇V leaves the Stiefel manifold (V rows stop being
unit vectors). Each step applies retraction back to manifold, but the gradient direction
is wrong. Riemannian Adam (T1.4) uses geodesic steps and stays on manifold correctly.
→ K1013 failure is NOT about HRA being inferior; it's about wrong optimizer.

**2. Equal-rank ≠ equal-params (38.5% capacity disadvantage)**
HRA paper (Table 1, arxiv 2405.17484) compares at equal *parameters*, not equal rank.
For square matrices: HRA r=8 ≈ LoRA r=8/2 = LoRA r=4. For rectangular q_proj (2560→4096):
HRA r=16 ≈ LoRA r=6 (matching 40,960 params). At equal params, HRA's sr=r advantage
should dominate. At equal rank, HRA has fewer params → capacity disadvantage on MMLU.

**3. Multiplicative rotation disturbs base-model MMLU directions**
MMLU accuracy depends on Q/K/V representations being aligned to recall world-knowledge.
Householder reflections H^(r)x rotate the query subspace. Post-training, queries that
encoded "capital of France" relationships may be rotated away from their original positions.
LoRA's additive update (x + BAx) preserves the original direction while adding task signal.

---

## Resurrection Path

**For K1013:** Use Riemannian Adam (Cayley retraction from T1.4) instead of standard Adam.
This keeps V on Stiefel manifold at each step. Expected: convergence within LoRA's step count.

**For K1012:** Compare at equal params (HRA r=16 vs LoRA r≈6):
- LoRA r=6: A(2560,6)+B(4096,6) = 40,704 params ≈ 40,960 (HRA r=16)
- This is the HRA paper's actual comparison class
- Prediction: HRA r=16 (sr=16) > LoRA r=6 (sr≈1) at equal params

**For P1 decision (unaffected by this kill):**
T0.3+T0.4+T1.3+T1.1 already establish the P1 adapter foundation:
- NoPE dims [128:512] (384 dims, algebraically position-invariant from T0.3)
- Q-only adapters with algebraic KV invariance (T0.4)
- Givens or Householder adapters with isometry error 2.384e-07 (T1.3, T1.1)
- Algebraic zero Grassmannian interference (T1.1 Theorem 2)

Whether the adapter format is HRA or LoRA, the interference structure is solved.
The P1 critical path can proceed with LoRA + Grassmannian init while the
equal-params HRA comparison is verified in a follow-up T1.6 bake-off.

---

## Architectural Implication for P1

The MATH.md fallback is now operative:
> "If K1011+K1012 FAIL: Fall back to LoRA with Grassmannian (proven in T0.3/T0.4).
> Finding #413 (Givens) and Finding #415 (Householder) both provide orthogonal adapters.
> LoRA + Grassmannian init is sufficient for interference-free composition."

T1.6 (algorithm bake-off) should compare: Cayley vs Givens vs Householder at **equal params**
with **Riemannian Adam** for Cayley/Householder. This is now unblocked and has all three
structural pieces defined.

---

## Experiment Metadata

- **Model:** mlx-community/Qwen3-4B-4bit (Gemma4 proxy, same 4B scale)
- **Runtime:** ~10 min total (300 LoRA steps + 300 HRA steps + 6 evals)
- **Date:** 2026-04-09
- **Platform:** M5 Pro 48GB, MLX
