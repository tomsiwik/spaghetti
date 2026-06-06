# REVIEW-adversarial.md — exp_hedgehog_pair_composition_polite_refactor

**Verdict:** KILL (preempt-structural, F#669-family clause + F#666-pure compound).
**Date:** 2026-04-25 · drain-window iter ~56 · reviewer hat.
**doom_loop.py:** exit=0.

## Adversarial checklist (18 items)

**Consistency:**
- (a) `results.json["verdict"]="KILLED"` ↔ proposed status `killed` → **MATCH**.
- (b) `all_pass=false` ↔ KILLED claim → **MATCH**.
- (c) PAPER.md verdict line "KILLED (preempt-structural; no measurement performed)" — no PROVISIONAL/PARTIALLY/INCONCLUSIVE/DEGENERATE language → **MATCH**.
- (d) `is_smoke=false` ↔ KILLED claim → **MATCH**.

**KC integrity:**
- (e) K#1846 + K#1847 byte-for-byte identical to 2026-04-23 DB (`experiment get` confirms unchanged) → **PASS**.
- (f) Tautology — N/A (no KC measurement performed; carve-out).
- (g) KC IDs in `run_experiment.py` (1846, 1847) ↔ DB → **MATCH**.

**Code ↔ math:**
- (h) `sum(lora_A` / `add_weighted_adapter("linear")` — N/A (no composition computed).
- (i) `LORA_SCALE` ≥ 12 — N/A (no LoRA initialized).
- (j) Single-sample-then-broadcast routing — N/A.
- (k) `shutil.copy` of sibling adapter — N/A (no adapter swap).
- (l) Hardcoded `{"pass": True, ...}` — N/A (kill_criteria result = "fail (untested-preempt)"; carve-out per F#669-family).
- (m) Target model substitution — N/A (no model loaded).
- (m2) **Skill attestation carve-out applies** — F#669-family precedent F#775/F#777/F#779. `run_experiment.py` imports only `json` + `pathlib`; no `mlx_lm.load`, no `nn.value_and_grad`, no `mx.eval`. MATH.md §0 explicitly attests carve-out. → **CARVE-OUT N/A**.

**Eval integrity:**
- (n) Base accuracy = 0% with avg_thinking_chars = 0 — N/A (no eval).
- (o) Headline n < 15 — N/A.
- (p) Synthetic padding — N/A.
- (q) Cited baseline drift — N/A.
- (t) **Target-gated kill (F#666) carve-out applies** — F#669-family preempt-KILL is a structural verdict where NO KC was measured (no proxy-FAIL with target-PASS scenario possible). Per reviewer hat clause: "(t) does NOT apply to preempt-KILL — F#666 gates kills on proxy-FAIL; preempt-KILL is a structural verdict where NO KC was measured." → **CARVE-OUT N/A**.
- (u) Scope-changing fix — N/A (graceful-failure stub is the canonical preempt-structural artifact, not a scope reduction).

**Deliverables:**
- (r) PAPER.md prediction-vs-measurement table present (§2, 7 rows) → **PASS**.
- (s) Math/argument adversarial pass — F#669 cascade derivation in MATH.md §2 is sound: K#1846 needs `acc_polite_iso`, `acc_refactor_iso`, `acc_polite_pair`, `acc_refactor_pair`, all of which require trained polite + refactor adapter weights; both PROVISIONAL parents have all 4 KCs each untested ([·] verified via `experiment get`); pair adapter is unconstructable. F#666 derivation in MATH.md §3 is sound: K#1846 (axis-accuracy delta) and K#1847 (per-layer cos-sim) are both proxy-only with no target-pair counterpart. Compound 2-parent + 2-KC-proxy structural impossibility. → **PASS**.

## Verdict-consistency pre-flight (6/6)

1. results.json `verdict="KILLED"` ↔ DB action `--status killed` → **MATCH**.
2. results.json `all_pass=false` ↔ KILLED → **MATCH**.
3. PAPER.md verdict line ≠ PROVISIONAL/PARTIAL/etc → **MATCH**.
4. `is_smoke=false` ↔ KILLED → **MATCH**.
5. KC git-diff (vs DB 2026-04-23): unchanged → **MATCH**.
6. Antipattern match: F#669-family + F#666-pure compound carve-outs apply per (m2)+(t) → **MATCH**.

## Decision

**KILL (preempt-structural, F#669-family clause + F#666-pure compound).** All 18 adversarial items PASS or have a documented carve-out (F#669-family for m2 and t). All 6 verdict-consistency pre-flight items PASS. F#669 reuse #17, 2-parent F#669 cardinality 1st observation. F#780 sub-axis 2nd-instance same-cluster (Hedgehog→Hedgehog), advancing 1/3 → 2/3 toward canonicalization (NOT cross-cluster — same-cluster only).

## Doom-loop self-check

- `doom_loop.py` exit=0.
- 4th consecutive researcher preempt-KILL, but each iter substantively distinct per `mem-pattern-triple-fire` (different mechanism / cluster / finding-pair):
  - iter ~47/~48: rank_ablation (post-F#770 Hedgehog F#669 1st)
  - iter ~49/~50: jepa_scale_sweep (post-F#770 JEPA F#669)
  - iter ~52/~53: cross_axis (pre-F#770 compound, 1-parent cardinality)
  - iter ~55/~56 (this): pair_composition (pre-F#770 compound, **2-parent cardinality NEW**)
- The pattern of preempt-KILL itself has canonicalized; HALT_ESCALATION addendum requested at analyst pass per researcher routing recommendation. Analyst should write addendum: macro-budget cap + cascade saturation blocker stack.

## Findings to file (post-`experiment complete`)

1. **F#NEW1** (killed) — F#669 17th reuse, 2-parent F#669 cardinality 1st observation, 3rd Hedgehog-cluster F#669, 2nd Hedgehog pre-F#770-repair compound.
2. **F#NEW2** (provisional) — F#780 sub-axis 2nd-instance same-cluster (Hedgehog→Hedgehog); 1/3 → 2/3 toward canonicalization. Distinct from F#776/F#778 schema-repair-reveals-F#669 path.

## Routing

`review.killed` → 🧠 Analyst.
- Analyst payload: ratify F#NEW1+F#NEW2 in LEARNINGS.md.
- Analyst secondary: write HALT_ESCALATION.md addendum (macro-budget cap + cascade saturation, distinct from 2026-04-19 Py3.14 stack). Recommendations:
  1. Promote `exp_hedgehog_behavior_adapter_politeness_impl` (P=1 macro 4-6h, F#683-cluster highest-leverage).
  2. Promote `exp_hedgehog_procedural_adapter_refactor_impl` (P=1 micro 4-6h).
  3. Defer non-Hedgehog macros (memento, class_composition).
  4. Separate orchestration for `rdt_loop_kv_cache_impl` (2h, parent scope-deferred).
- Analyst tertiary recommendation for next CLAIM-AND-AUDIT: triple_composition would be 5th consecutive preempt; if no non-cascade P≤2 arrived, recommend HALT for orchestrator unblock instead of routing yet another preempt.

## Assumptions

- F#669-family carve-out for (m2) and (t) is canonical per F#775/F#777/F#779 precedent (3-instance threshold met).
- F#NEW2 is correctly classified as `provisional` (compound F#666+F#669 pre-F#770-repair sub-axis is at 2/3 same-cluster instances; canonicalization at 3/3, ideally cross-cluster).
- Researcher's 2-parent F#669 cardinality novelty claim verified: prior 16 F#669 reuses had cardinality 1.
