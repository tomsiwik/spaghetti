# MATH — DFA R1: disjoint-frame B-projection cuts N=2 behavioral interference

## Disease being prevented

Cross-adapter interference (F#827: −14pp python→math, −12pp medical→code) when two LoRA adapters
are summed in delta-output space. The bet `dfa-init` (Pico, arXiv:2604.16826) claims interference
lives on the **B / output side**: the two adapters' delta-outputs overlap in activation space, so
adapter j's output corrupts the residual stream feeding the skill that adapter i installed.

## Setup

For a single q_proj layer, base output `y = W h`. Adapter i adds delta-output
`δ_i(h) = s · (h A_i) B_i ∈ R^d`, with `A_i ∈ R^{2560×6}`, `B_i ∈ R^{6×2048}`, `d = 2048`, `s = 6`.
Naive N=2 composition (the F#827 baseline) is `y = W h + δ_python(h) + δ_math(h)`.

Each adapter's delta-output is confined to the **row space of B_i**, a ≤6-dim subspace
`S_i = rowspace(B_i) ⊆ R^d`. Interference magnitude is governed by the overlap of `S_python`
and `S_math`: if they share directions, the two outputs add/cancel along shared axes.

## Construction (DFA: Disjoint-Frame B-projection)

Per layer, stack the two B row-spaces and orthonormalize jointly:

    M = [ B_python^T  |  B_math^T ] ∈ R^{2048×12}
    Q, R = QR(M)          # Q ∈ R^{2048×12}, columns orthonormal, Q^T Q = I_12
    Q_python = Q[:, 0:6]   # frozen orthonormal frame for python  (R^{2048×6})
    Q_math   = Q[:, 6:12]  # frozen orthonormal frame for math

Then project each adapter's delta-output onto its own frame:

    δ'_i(h) = Q_i (Q_i^T δ_i(h)) = P_i δ_i(h),   P_i = Q_i Q_i^T   (orthogonal projector, rank ≤6)
    y = W h + δ'_python(h) + δ'_math(h)

**Composition is Σ_i P_i δ_i, never (ΣB)(ΣA).** LORA_SCALE = 6 ≤ 8. Frames are frozen
(seeded only through the adapters' own B, no learned parameter), built lazily at the true
d discovered from the model — never hardcoded.

## Theorem 1 (output-orthogonality by construction)

Because QR yields `Q^T Q = I_12`, the two blocks satisfy `Q_python^T Q_math = 0_{6×6}`. Hence for
any inputs,
    ⟨δ'_python(h), δ'_math(g)⟩ = (Q_python^T δ_python(h))^T (Q_math^T Q_python)(…) = 0,
i.e. `P_python P_math = Q_python (Q_python^T Q_math) Q_math^T = 0`. The two projected delta-outputs
live in orthogonal subspaces of R^d — **zero output overlap, exactly, every layer.** ∎

## Theorem 2 (solo skill is preserved)

`Q_python` spans exactly the first-block QR basis of `[B_python^T | B_math^T]`. The QR of a stacked
matrix orders columns so the first 6 columns span a subspace **containing** the original
`rowspace(B_python)` whenever `B_python^T` is the leading block (Gram–Schmidt processes python's
columns first, before math's). Therefore `P_python δ_python = δ_python` up to the components of
python's row space that are linearly dependent on already-extracted directions — which for
full-rank `B_python` (rank 6) is the identity. So solo-python output is recovered exactly, and only
math's delta is *deflected* off python's shared directions. Predicted solo drop ≈ 0pp (kill 2313
threshold: 5pp; gate: ≤2pp).

## Predictions (pre-registered, before the run)

Conditions on HumanEval-style + GSM8K behavioral harness, N≥200 items/domain, no-thinking.
Pair = python(code) + math, baseline interference = F#827 (−14pp python→math).

- **A** base (no adapter)
- **B** math-solo (ceiling for the math skill)
- **C** naive N=2 sum  (python + math, both unprojected) — the F#827 interference baseline
- **D** DFA N=2 sum    (python + math, each projected onto its disjoint frame)

Let `gap = acc(B) − acc(C)` be the measured interference drag, `recovered = acc(D) − acc(C)`.

- **Predicted:** `gap ≥ 0.10` (interference is real, replicates F#827 scale).
- **Predicted:** DFA cuts interference ≥50%: `recovered ≥ 0.5·gap` AND residual drag
  `acc(B) − acc(D) ≤ 0.07` (≤7pp).
- **Predicted:** solo python (code) unprojected vs projected drops ≤2pp (frame preserves skill).

## Numeric refutation thresholds (kill criteria)

- **K2313** (frame destroys skill): KILL if projected-solo accuracy drops **> 5pp** vs unprojected solo.
- **K2314** (interference uncut): KILL if N=2 composed interference is **not cut ≥50%** — i.e.
  residual drag `acc(B) − acc(D) > 7pp` at matched solo accuracy, OR `recovered < 0.5·gap`.

Verdict `supported` iff (K2313 not triggered) AND (K2314 not triggered) AND `gap ≥ 0.10` (the
interference the bet exists to cut is actually present). If `gap < 0.10` the pair shows no
interference to cut → `provisional` (wrong pair, not a refutation of the mechanism).
