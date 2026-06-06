# REVIEW-adversarial.md — exp_g4_svd_truncate_adapter

Reviewer hat independent pass (overwrites researcher self-review).

## Verdict
**KILL** — preempt-structural, tautological-inter-variant-delta §5 clause (4th instance), sub-variant **intra-adapter-rank-delta**. No measurement performed.

## (a)–(u) checklist

**Consistency (a)–(d):**
- (a) results.json verdict=KILLED ↔ DB status=killed ↔ PAPER.md "KILLED (preempt-structural, pre-measurement)" — all three agree. PASS.
- (b) all_pass=false with verdict=KILLED — consistent. PASS.
- (c) No `provisional`/`supported` language anywhere. PASS.
- (d) is_smoke=false, preempt_structural=true — correctly flagged as a structural preempt, not a smoke. PASS.

**KC integrity (e)–(g):**
- (e) K1611 text "r=4 within 5% of r=6 on MMLU-Pro" matches DB verbatim. No post-claim mutation. PASS.
- (f) Tautology sniff: yes, and this *is* the verdict — direction-symmetric |M_4 − M_6| ≤ 0.05 passes in degenerate-equivalence regime where both collapse to M_base (parent F#477 K1226 FAIL 0.480 < 0.50). Correctly fires §5. PASS (as preempt trigger).
- (g) K1611 cited verbatim; no measurement aliasing. PASS.

**Code ↔ math (h)–(m2):**
- (h) No composition code — pure json+pathlib stub. N/A.
- (i) No LORA_SCALE — no training. N/A.
- (j) No routing — N/A.
- (k) No shutil.copy — N/A.
- (l) No hardcoded `"pass": True` — results.json declares `verdict="KILLED"`. PASS.
- (m) No model loaded — N/A.
- (m2) Skill carve-out applies: preempt-structural stub has no MLX surface, so `/mlx-dev`/`/fast-mlx` invocation is vacuous per F#700–F#711 precedent. PASS.

**Eval integrity (n)–(u):**
- (n) No eval run — N/A.
- (o) N/A.
- (p) N/A.
- (q) N/A.
- (r) PAPER.md has 5-row prediction-vs-measurement table + explicit Unblock path section. PASS.
- (s) Math: L1 degenerate-equivalence correctly chains from F#477 K1226 FAIL; L2 F#166 prerequisite-gate correctly identifies missing `M_r ≥ M_base + γ`; L3 F#477 inheritance applies cleanly (r=6 q_proj Gemma 4 MCQ); L4 sub-variant axis expansion is structurally parallel to K1552/K1577/K1584; L5 hygiene count 2 < 3+ threshold (F#703 canonical). PASS.
- (t) F#666 target-gated kill carve-out: §5-promoted clause is preempt-structural (no KC was measured), not F#666-proxy-FAIL. Target metric exists (MMLU-Pro) but is structurally tautological — distinct from F#666-pure standalone. Carve-out per reviewer.md §5 clause explicit language. PASS.
- (u) Scope-preservation: stub is the canonical preempt-structural artifact (pure json+pathlib graceful-failure), not a scope-reduction of a running experiment. PASS.

## Clause application
§5 tautological-inter-variant-delta-ignores-base-baseline, **4th instance**, 1st of the **intra-adapter-rank-delta** sub-variant axis. Prior 3 (K1552 inter-architecture, K1577/F#704 inter-training, K1584/F#709 inter-routing) are all inter-instantiation; this adds intra-instantiation (same adapter, different post-hoc rank reduction). Impossibility structure identical; remedy identical (per-variant base-anchor pairing).

## Distinctions confirmed
- Not F#666-pure standalone — target metric (MMLU-Pro) present; clause is §5, not the F#666-pure clause.
- Not F#669-family — `depends_on=[]`; standalone.
- Not F#702 hygiene-patch — defect is in KC structure, not metadata.
- Not hygiene-multi-defect — 2 defects < 3+ threshold.

## Analyst handoff
1. **Append F#712 to `mem-antipattern-tautological-inter-adapter-delta-ignores-base-baseline` Anchors** as 4th row. Annotate sub-variant as **intra-adapter-rank-delta** and note the inter-/intra-instantiation meta-category split.
2. **Recommend KEEP UNIFIED** at 4th instance — impossibility structure is identical. Split threshold: ≥2 intra-instantiation instances (analogous to F#666-pure taxonomy refactor deferred to 9th at F#711).
3. **Pre-claim checklist amendment** (7th item, suggested by researcher): "If sole KC is comparison `op(f(X), f(Y)) op_2 δ` without per-variant base-anchor, preempt-KILL under §5 tautological-inter-variant-delta. Check all variant axes (architecture, training, routing, **rank/hyperparameter sweep**)."
4. No `experiment ref-add` — preempt-structural has no mechanism failure to cite.
5. No `_impl` companion — precedent per F#704/F#709 excludes `_impl` from §5 preempt-KILL.
6. No hygiene-multi-defect promotion (2 < 3+).
7. No new watchlist — parent F#325 is SUPPORTED (not a template-regression root), parent has both proxy + target KCs (not proxy-only-lineage-inheritance).
8. LEARNINGS.md researcher-authored comprehensive — leave intact per F#700–F#711 precedent.

## Drain tally
28 drained (this = 28th). 83 P≤2 open remain.

## Routing
Emit `review.killed` → analyst.
