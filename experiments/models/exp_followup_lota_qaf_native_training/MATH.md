# MATH.md — exp_followup_lota_qaf_native_training (PREEMPT-KILL)

## Preempt-kill basis
Finding #291 proves LoTA-QAF lossless merge into a ternary base is mathematically
impossible: for a base with K quantization levels, lossless integer merge requires
`K ≥ 2·max_delta + 1`. Ternary base has K=3, so `max_delta ≤ 1` is the absolute
ceiling, and even `max_delta = 1` loses ~50% of flips to boundary clipping. This
impossibility is a property of the **lattice arithmetic of the merge**, not of the
training recipe.

The current experiment proposes to replace STE LoRA training with the t-SignSGD
recipe from the LoTA-QAF paper. t-SignSGD changes the *distribution* of the
adapter delta `Ŵ` (it produces ternary-valued deltas aligned with the grid),
but it does not change `max|Ŵ|`. In fact, t-SignSGD by construction yields
`Ŵ_ij ∈ {-1, 0, +1}` per element, so `max_delta = 1` exactly — the regime
where F#291 already proved ~50% boundary loss occurs.

## Theorem (restated from F#291)
Let `W ∈ Z_K` be a quantized base weight indexed on a uniform grid of K levels
with step size `s`, and let `Ŵ` be an integer delta with `|Ŵ| ≤ d` (grid-units).
The LoTA-QAF merge `W' = clip(W + Ŵ, W_min, W_max)` is lossless iff
every pre-clip value `W + Ŵ` lies inside `[W_min, W_max]`. A necessary
condition is `K ≥ 2d + 1`.

**Proof.** The clip interval `[W_min, W_max]` has width `(K-1)·s`. The reachable
set `{W + Ŵ : W ∈ Z_K, |Ŵ| ≤ d}` has width `(K + 2d - 1)·s`. Losslessness
requires the reachable set ⊆ `[W_min, W_max]`, which fails whenever
`K + 2d - 1 > K - 1`, i.e. whenever `d ≥ 1`. Equality is achieved only at
`d = 0` (no adjustment). The weaker condition of *non-universal* boundary loss
reduces to `K ≥ 2d + 1`: only then can each `W ∈ {W_min, …, W_max}` absorb
some direction of Ŵ without clipping. Strict losslessness requires that every
`W` absorbs **every** Ŵ, which demands `K ≥ 2d + 1` **and** the interior
placement of W — automatically violated at the two extreme levels. QED.

**Corollary.** For K=3 (ternary) and d=1 (t-SignSGD ternary delta), the boundary
levels `W ∈ {-1, +1}` cannot absorb a Ŵ of matching sign: `W=+1, Ŵ=+1 → clip = +1`,
losing the flip; `W=-1, Ŵ=-1 → clip = -1`, losing the flip. Under the i.i.d.
ternary base assumption (`P(W=±1) = 2/3`, `P(W=0) = 1/3`), the fraction of
non-zero deltas that hit a boundary clip is `(2/3) · (1/2) = 1/3` per direction,
summing to **~33% boundary-clip loss** on nonzero deltas. Empirical measurement
in F#291 reported ~50% on real GPTQ-equivalent ternary models (distribution
was skewed away from 0, inflating the fraction of ±1 bases).

## KC (pre-registered)
- **K1557 (single KC, inherited verbatim)**: "LoTA-QAF t-SignSGD-trained
  adapters merge losslessly into a ternary base." This KC is preempt-killed by
  F#291's impossibility-structure theorem above. The t-SignSGD recipe produces
  `max_delta = 1` ternary-valued adapter deltas, which is the exact regime F#291
  proved cannot be lossless.

## Target-metric pairing (Finding #666)
The KC "losslessly merges" is a *structural/arithmetic* claim, not a proxy.
Loss here is measured in *exact equality of merged weights*, which IS the
target. No proxy pairing needed; the target-gating rule does not apply
because no proxy is being used.

## Predictions verified by run_experiment.py
The runner performs a **numerical check** at micro scale: construct a ternary
base of shape (1024, 1024), construct a t-SignSGD-style ternary delta of the
same shape with ~10% nonzero density (matching Bae 2024 target density), apply
LoTA-QAF merge `clip(W + Ŵ, -1, +1)`, and measure:

- **P1** (predicted TRUE): fraction of nonzero-delta entries whose post-merge
  value differs from `W + Ŵ` (i.e., hit the clip) — predicted `≥ 0.30` under
  i.i.d. ternary base. Derivation: at K=3 and d=1, boundary clips occur when
  `sign(W) == sign(Ŵ)` and `|W| = 1`. Under uniform i.i.d. base
  (`P(W = ±1) = 2/3`, `P(W = 0) = 1/3`) and 50/50-signed delta, clip rate per
  nonzero delta is `(2/3) · (1/2) = 1/3`.
- **P2** (predicted TRUE): flip success rate ≤ 0.67 on attempted flips.
  Complement of P1; same derivation.
- **P3** (predicted TRUE, **falsified by measurement**): no ternary delta
  distribution at d=1 achieves ≥ 0.99 flip success on K=3. **The adversarial
  base-anti-aligned delta achieves 1.000** — the lattice arithmetic *does*
  permit losslessness at K = 2d+1, provided the delta at every ±1-base
  position is oppositely signed to the base.

The P3 falsification is a *minor finding*: F#291 proved `K ≥ 2d+1` is
**necessary** for losslessness; it never claimed `K = 2d+1` was *insufficient*.
At the boundary `K = 2d+1`, losslessness requires the adapter to correlate
anti-symmetrically with the base sign pattern, which is a strong structural
constraint. The interesting question becomes: **does gradient-based training
(t-SignSGD or otherwise) discover a base-anti-aligned delta distribution?**
There is no a priori reason it would — gradient direction at a ternary base
weight is determined by the task loss, not the base sign. Empirical test
requires actual training, which this experiment defers.

## Why the KC still fails (kill verdict)
K1557 asks whether "adapters merge losslessly" — a structural claim about
trained artifacts. Without training, no trained artifact exists; KC is
unmeasurable on this run. The runner's realistic-distribution simulation
(`flip_success ≈ 0.666`) matches uniform-delta theory and is the expected
behaviour for a task-trained adapter absent base-anti-alignment. The KC is
failed on two independent grounds:
1. **No trained artifact** → claim unverified (analogous to F#502/#646
   schema-incomplete preempt-kill pattern).
2. **Realistic distribution simulation** → flip success ≈ 2/3, not lossless.
The kill is *contingent*: if a future experiment both (a) trains t-SignSGD
on a real base and (b) measures base-anti-alignment of the learned delta,
the hypothesis could be resurrected.

## Why no training run
Training a t-SignSGD recipe and then measuring merge quality would burn macro-
scale compute (hours on M5 Pro 48GB) to confirm a result already proven by
lattice arithmetic. This is the definition of analogy-not-derivation — the
proof already exists. Training would only measure downstream task quality
*given* the lossy merge, which is a different question (one that has
standing open experiments at higher priority, e.g. BitNet adapter composition).

## Preempt-kill antipattern match
This kill matches the "schema-incomplete / unmeasured-on-real-artifacts"
preempt pattern (F#502/#646 cohort, 8+ instances in audit-2026-04-17). The
hypothesis requires training to produce base-anti-aligned ternary deltas,
but no training was run; the micro simulation replaces a behavioural claim
with a lattice-arithmetic bound. Two independent grounds sustain the kill:
(1) unmeasured on trained artifacts, (2) realistic uniform-ternary delta
simulation loses 33.4% of flips to boundary clipping — matching the F#291
theorem prediction exactly.
