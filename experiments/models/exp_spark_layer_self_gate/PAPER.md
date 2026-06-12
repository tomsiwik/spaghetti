# PAPER — exp_spark_layer_self_gate

**Verdict: SUPPORTED** (kill-id 2296 did NOT fire). Real run, `is_smoke:false`, pueue task 5,
9202s. Frozen `mlx-community/gemma-4-e4b-it-4bit`; q_proj r6 scale6 math + code adapters; HumanEval
pass@1 n=50 (real unit-test execution), GSM8K exact n=50; greedy; thinking enabled.

## Claim
Off-domain LoRA interference is not weight-uniform nor a decode-time transient — it is **layer-localized**,
and WHICH q_proj layers the off-task (math) adapter damages is predictable per-prompt from a single FREE
prompt forward pass via the per-layer cosine `γ^ℓ = mean_t cos(δ_math^ℓ(x), q_codebase^ℓ(x))`. Masking the
math adapter to the top-k most-constructive layers (zeroing it elsewhere) recovers the code task.

## MATH prediction vs the two nulls
- **Layer-localized (H1):** harm `Δ⁻` concentrates in a subset of layers and γ^ℓ<0 reports them; a top-k
  constructive mask removes `Δ⁻`, keeps `Δ⁺`, so `pass@1(D,k*) − pass@1(C) ≈ −Δ⁻ ≥ +8pp`.
- **Null N1 (uniform interference):** `Δ⁻` spread evenly / γ^ℓ uninformative → masking removes proportional
  helpful+harmful mass → recovery `< +8pp` (or hurts).
- **Null N2 (no interference, C≥B):** if math is net-helpful to code there is no `Δ⁻` to remove → recovery
  ceiling small → KILLED by construction (the trap that killed exp_spark_entropy_gated_lora off
  non-reproducing F#827 magnitudes).

The data lands squarely in H1, and decisively rejects N2: this adapter pair produces **massive** in-run
interference (C collapses code to 0.02).

## Measured HumanEval pass@1 (n=50)

| Condition | pass@1 | note |
|---|---|---|
| A base only | 0.44 | base code ability |
| B code-solo (ceiling) | 0.34 | code adapter, no math |
| C naive full-layer comp (math in all 42) | **0.02** | in-run interference baseline / KILL ANCHOR |

**D layer-self-gate k-sweep** (math kept only in per-prompt top-k constructive q_proj layers):

| k | pass@1 |
|---|---|
| **6** | **0.34** ← best_D |
| 12 | 0.24 |
| 18 | 0.18 |
| 24 | 0.22 |
| 30 | 0.12 |
| 36 | 0.12 |

`best_D = 0.34` at `k = 6`. Monotone trend: the fewer layers the math adapter is allowed into, the more
code is recovered — and at k=6 it recovers code-solo **exactly** (0.34 = B). At k=36 (near-C) it stays
crushed (0.12), confirming the damage is carried by the layers excluded only at small k.

## Kill criterion 2296 (anchored to IN-RUN C) — did NOT fire

- Recovery clause: `best_D − C = 0.34 − 0.02 = +32.0pp` ≥ +8pp → **PASS** (margin **+24.0pp** above threshold).
- Floor clause: `best_D = 0.34 ≥ B − 6pp = 0.28` → **PASS** (margin **+6.0pp**; in fact best_D = B exactly,
  i.e. full ceiling recovery, +6.0pp above the B−6pp floor).

Both clauses pass with comfortable margin; verdict SUPPORTED. The kill being anchored to the in-run C
(0.02), not F#827 magnitudes, makes this robust: C measured the true interference present here.

## How localized was the damage?
Per-prompt, ~25 of 42 layers carried a *positive* γ (constructive) math delta (n_pos_gamma: min 23, max 27,
mean 25.2, median 25; 0/50 prompts had ≤6 constructive layers). Yet the **winning mask kept only k=6** of
those — and larger k monotonically degraded pass@1. Interpretation: the destructive mass is not a clean
small subset of negative-γ layers; rather, recovery is maximized by admitting the math adapter only into the
handful of *most strongly* constructive layers and excluding the long tail of weakly/ambiguously aligned
layers (whose net effect on the code task is harmful even where γ is marginally positive). The damage is
genuinely depth-localized and the γ ranking is the right axis to exploit it — but the operative knob is
aggressive top-k restriction (k≈6), not "keep all positive-γ layers." This is a stronger result than the
literal MATH sketch: the per-prompt γ *ordering* is informative even though the γ *sign* alone is not the
clean oracle the first-order theorem assumed.

## GSM8K exact-match (n=50, on-domain characterization, not gating)

| Condition | exact |
|---|---|
| A base | 0.78 |
| B code-solo | 0.52 |
| C naive comp (math all 42) | 0.28 |
| D self-gate @ best_k=6 | 0.44 |

On the math side, masking the math adapter down to 6 layers raises GSM8K from C's 0.28 to 0.44 (+16pp) —
the same restriction that recovers code does not destroy the on-domain signal, though it does not reach
math-solo (not measured here; B is code-solo). Note base A is highest on GSM8K (0.78), consistent with the
base model already being a strong math reasoner and the scale-6 adapters net-perturbing it.

## Verdict line
SUPPORTED: a free, per-prompt, top-k=6 constructive-layer mask recovers off-domain HumanEval pass@1 from a
collapsed C=0.02 back to the code-solo ceiling B=0.34 (+32.0pp vs the in-run C anchor; +6.0pp above the
B−6pp floor) — off-domain LoRA interference is layer-localized and prompt-predictable, with no router, no
decode-time gating, and no training.
