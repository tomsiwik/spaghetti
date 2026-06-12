# bet: dfa-init — composability by construction (objective + init)

**Thesis.** Cross-adapter interference lives on the **B/output side**, not the A/input side pierre
defended (Pico, "Crowded in B-Space", arXiv:2604.16826). Force each adapter's B columns into a
**frozen disjoint-block orthonormal frame** (DFA) — cross-adapter output overlap is then zero by
construction, no merge-time cleverness needed. Escalate to a shared-frozen-predictor latent-alignment
objective (LLM-JEPA, arXiv:2509.14252) so adapters *train into* one compatible subspace.

**Why this can be a breakthrough:** every prior pierre experiment rearranged frozen vectors
(provable ceiling +2–4pp, F#827/837/844). This bet adds degrees of freedom at TRAIN time — the
one layer the old loop never touched.

**Pierre branch:** `bet/dfa-init` · **Baseline to beat:** F#827 interference (−14pp python→math, −12pp medical→code).

## Ladder
- **R1 — DFA N=2 composition ablation** (days; ~30-line B-init change + existing merge).
  Project the EXISTING math/python/medical adapters onto disjoint B-blocks (QR per block), compose
  N=2, run the behavioral harness (GSM8K/HumanEval/MedQA slices, ≥200 items, no-thinking harness —
  adapter benefit sign-flips with thinking mode, so the harness must match training).
  **Gate:** composed-vs-solo interference cut ≥50% (≤7pp drag) at matched solo accuracy (−≤2pp).
  **Kill:** projection costs >5pp solo accuracy (the frame destroys the skill) or interference uncut.
- **R2 — train WITH the frame** (≈1 week). Retrain the 3 adapters from the DFA init (B frozen-frame,
  A free; one Stiefel retraction per step if B trainable). Same gate, plus N=3 composition.
- **R3 — JEPA shared-predictor objective.** Add the latent-alignment term (shared frozen predictor)
  to adapter training; measure composition additivity directly (logit-space deviation from Σ).
  **Gate:** N=3 composed ≥ best-solo-per-domain −3pp (the "free composition" milestone).
- **R4 — SHIP:** fold the winning init+objective into pierre's training path on `bet/dfa-init`;
  league-score the branch.

## Honest risk
Param-space disjointness may not buy behavioral non-interference (the gap that killed
A-orthogonality). R1 is designed to expose exactly that for <$1 of compute — if it does, R3's
function-space objective becomes the bet and R2 is skipped.
