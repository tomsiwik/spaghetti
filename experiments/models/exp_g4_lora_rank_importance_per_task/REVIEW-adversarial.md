# REVIEW-adversarial.md — exp_g4_lora_rank_importance_per_task

## Verdict: KILL (preempt-structural, F#666-pure standalone — ~29th drain-window)

Review written after-the-fact: the researcher emitted `experiment.done` → analyst wrote LEARNINGS.md without the reviewer step being consumed. DB status is already `killed`, F#759 registered, all other required artifacts present. This pass closes the artifact gap and verifies the preempt-structural kill stands.

## Adversarial checklist (reviewer.md §3)

**Consistency (highest priority):**
- (a) `results.json["verdict"]` = `"KILLED"`; DB status = `killed`. **PASS** — consistent.
- (b) `results.json["all_pass"]` = `false`; both KCs `untested`; status = `killed`. **PASS** — no silent upgrade to supported.
- (c) PAPER.md verdict line: `"KILLED (preempt, F#666-pure standalone — ~29th drain-window)"`. **PASS** — no `PROVISIONAL` / `PARTIALLY SUPPORTED` / `INCONCLUSIVE` mismatch.
- (d) `is_smoke` = `false`; KILL is preempt-structural, not smoke. **PASS**.

**KC integrity:**
- (e) No post-run KC mutation — preempt-stub ran `<1s`, no data could drive KC relaxation. **PASS** (N/A).
- (f) Tautology sniff: KC pair K1941 (uniform argmax) XOR K1942 (variance > 4×) is explicitly flagged in MATH §1 step 5 and PAPER.md 4-cell truth table as *internally XOR-inconsistent* — this is the **reason for the preempt-kill**, not a covert tautology in the scaffold. No `e=0→0`, `x==x`, same-expression-twice, or kappa-with-shared-rater patterns. **PASS**.
- (g) K-IDs match MATH.md §1 (K1941, K1942) and results.json. **PASS**.

**Code ↔ math:**
- (h)–(l) N/A — `run_experiment.py` is a preempt-stub (`sys.exit(1)` after explanatory stderr print). No composition math, no `LORA_SCALE`, no `shutil.copy`, no hardcoded `{"pass": True, ...}`, no routing on single sample. **PASS** by vacuous quantification.
- (m) Base-model substitution: MATH §0 pins `mlx-community/gemma-4-e4b-it-4bit` per F#627 and states explicitly "Not loaded." **PASS**.
- (m2) Skill invocation evidence: MATH §0 states `/mlx-dev` + `/fast-mlx` "Not invoked — no MLX code written; honest disclosure per reviewer checklist item (m2)." Per F#687/F#700/F#703/F#757 precedent, preempt-structural kill carves out (m2) — the skills list is documented, and no MLX training-loop code exists to invalidate. **PASS** (carve-out applies).

**Eval integrity:**
- (n) N/A — no eval, no `avg_thinking_chars` claim.
- (o) N/A — no headline `n`.
- (p) N/A — no synthetic padding.
- (q) N/A — no baseline cited as measured.
- (t) **Target-gated kill (F#666)**: reviewer.md §5 preempt-structural carve-out (F#666-pure standalone clause) **explicitly exempts** this from (t) — F#666 is the *reason* for the preempt, not a blocker on it; no KC was measured (proxy or target), so proxy-FAIL-with-target-PASS ambiguity cannot arise. MATH §1 step 2 + results.json `kc_set_gating` document the F#666 violation as the kill ground. **PASS** (carve-out applies).
- (u) **Scope-changing fix antipattern**: no SFT↔LoRA swap, no `max_length` reduction, no `trackio` disable, no base-model downgrade, no KC-complexity drop. Graceful-failure preempt-stub is the canonical preempt-structural artifact per reviewer.md §5 F#666-pure standalone clause. **PASS** (not a scope change).

**Deliverables:**
- (r) PAPER.md contains prediction-vs-measurement table with both KCs as "not measured". **PASS**.
- (s) Math/preempt theorem: MATH §1 derives 5-step proof (proxy kind → forbidden-solo → standalone-no-fallback-target → 4-cell decision table → unanchored threshold). Both KCs correctly classified as proxies (dimensionless argmax-distribution statistics). XOR-inconsistency correctly identified. No unsupported claims. **PASS**.

## Taxonomic verification

- 13th g4-ablation super-family sub-type (rank-importance-per-task; NEW). g4-ablation non-saturating — confirmed distinct from prior 12 sub-types (per-layer-cos, PPL-drift, canary-FNR, routing-collision, hash-ring-PPL, routing-family, gumbel-routing, perturbation-stability, SVD-rank-delta, SVD-denoise-PPL, init-method-comparison F#723, seed-determinism F#757).
- 1st structural-hyperparameter argmax-divergence proxy-bucket form (NEW). Distinct from cos-sim (4 forms), routing-acc (4 forms), infra-bench (5 forms), training-axis-efficiency (1 form). Key insight captured in LEARNINGS: argmax operation is dimensionless and inherits forbidden-solo even if M is target-anchored.
- ~29th F#666-pure standalone preempt-KILL in drain window. Running tally in MATH §2 matches prior finding chain.

## Assumptions

- The researcher's claim that KC pair is internally XOR-inconsistent (K1941 PASS ∧ K1942 PASS cannot both hold) is algebraically correct: variance > 4× implies non-uniform argmax, contradicting uniform-argmax PASS. Confirmed.
- The 4× threshold critique (unanchored against geometric rank grids and F#742 noise floor) is consistent with F#742's C_20=0.335 weak structural concentration — adopted without further verification given the direct-anchor citation.
- The decision to NOT file an `_impl` companion matches F#687/F#698/F#700/F#701/F#703/F#757 precedent for F#666-pure preempt-structural kills — unblock is pre-reg-external (v2 with target pair), not follow-up `_impl`.
- Review is being written after analyst already emitted `learning.complete`; LEARNINGS.md is verified consistent with this review. No re-analyst pass required.

## Non-blocking observations

1. Pre-reg lacks operational `M(r, task)` spec, rank grid, and task set — LEARNINGS and PAPER Path-A correctly flag these as v2 requirements.
2. `run_experiment.py` raises `SystemExit(1)` rather than writing `results.json` directly. `results.json` is independently well-formed so the preempt-KILL verdict stands (non-blocking, same note appears in F#758 review).

## Routing

Emit `review.killed` — DB already `killed`, F#759 registered, LEARNINGS.md already drafted by analyst. This review closes the artifact set.
