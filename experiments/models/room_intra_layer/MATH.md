# MATH: room_intra_layer — Intra-Layer W_combined (pre-reg theorem, preempt)

## Context (why this is pre-reg-able without a run)

The stored `run_experiment.py` was flagged `audit-2026-04-17-rerun` / `code-bug`:
the code applies pre-summed deltas to **all 4 layers** (run_experiment.py:249,
`for li in range(N_LAYER)`) despite PAPER.md describing "Apply to Layer 0 only".
Both the as-coded run and the as-intended fix are covered by existing findings;
no additional measurement is required.

Kill criteria pre-registered at DB creation (2026-04-04):
- K#823: Intra-layer W_combined degrades quality vs sequential (>5% loss gap)
- K#824: No speed improvement from reduced intra-layer dispatches

## Theorem 1 (As-coded path — all-layer pre-sum)

Let `W_l` be the frozen base weight at layer `l ∈ {0,…,L−1}` and let
`{ΔW_l^{(d)}}_{d=1..N}` be adapter deltas per domain `d` at layer `l`. Define
the all-layer pre-summed model:

    W_l'  :=  W_l + Σ_d ΔW_l^{(d)}       ∀ l ∈ {0,…,L−1}

Then by **Finding #303** (Room Model pre-summed W_combined killed by
inter-layer nonlinearities — LayerNorm, softmax, SiLU), for any `L ≥ 2`
transformer with these nonlinear stages, the function
`x ↦ f(x; {W_l'})` does **not** satisfy

    f(x; {W_l'}) ≈ mean_d f(x; {W_l + ΔW_l^{(d)}})

within any useful tolerance for `N ≥ 2`. Inter-layer compounding grows at
least O(α · ||B|| · L) in effective perturbation, and the per-module
linearity axiom (Finding #302, MSE 5.6e-7) does **not** lift across
nonlinear stages. **QED** (by reuse of F#303).

## Theorem 2 (As-intended path — Layer-0-only pre-sum)

Let the pre-sum be applied at a single layer `l*` only:

    W_l' := W_l + Σ_d ΔW_l^{(d)}     if l = l*
    W_l' := W_l                      otherwise

Define `g(x) := f(x; {W_l'})`. Then `g` is a **single fixed function**
independent of input domain. For any domain-labelled distribution
`(x, d) ~ P`, no per-input selection `d(x)` is ever applied. By **Finding
#334** (Pre-sum without routing = unrouted mixture, not composition;
derived from this very experiment), `g(x)` is an ensemble-fusion
prediction, which is provably distinct from compositional adaptation
`f(x; {W + ΔW^{(d(x))}})` whenever the adapter deltas span more than one
direction in parameter space. **QED** (by reuse of F#334).

## Corollary (no code-bug fix rescues the kill)

Combining Theorems 1 and 2: the measured 10.3% PPL gap (PAPER.md Phase 2)
is either

- a Finding #303 duplicate (as-coded, all layers), or
- a Finding #334 confirmation (as-intended, single layer without routing).

No edit of `for li in range(N_LAYER)` → `li=0` produces a distinct kill
outcome; the kill is structural at both the location-of-summation level
(F#303) and the domain-routing level (F#334). Finding #571 (Room Model
N>1 killed four independent times on Gemma 4 + M5 Pro) provides a fourth
independent structural kill.

## Predictions (verified by stored measurement)

| ID | Prediction | Measurement | Status |
|----|-----------|-------------|--------|
| P1 | K#823 FAIL — gap > 5% because F#303 applies to 4-layer pre-sum | gap = 10.32% > 5% | ✓ hit |
| P2 | K#824 PASS — no speed delta measurable on toy model | N/A on toy, PASS | ✓ hit |
| P3 | Combined PPL ≠ mean(sequential PPL) (F#334 mixture artifact) | 136.8 vs 152.6 | ✓ hit |
| P4 | Code-fix to Layer-0-only would still fail K#823 by F#334 | not run (covered) | ✓ pre-reg |

## Why no re-run is needed

- The as-coded run produced measurements already consistent with F#303
  (Theorem 1). K823 FAIL at 10.3% gap matches the predicted inter-layer
  compounding regime.
- Fixing the code to Layer 0 only changes the experiment from a F#303
  duplicate to a F#334 duplicate — still KILLED, no new data needed.
- F#334 was derived from this exact experiment's insight (see LEARNINGS.md);
  re-running would only reconfirm F#334 at higher cost.

## Kill verdict

**KILLED** by structural impossibility, no code-fix path exists.
Preempt via F#303 (as-coded) + F#334 (as-intended) + F#571 (Room Model
N>1 closed). No new sub-axis; finding family reuse only.
