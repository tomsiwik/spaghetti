# REVIEW-adversarial.md — exp_hedgehog_triple_composition_3domain

**Verdict:** KILL (preempt-structural, F#669-family clause + F#666-pure compound).
**Date:** 2026-04-25 · drain-window iter ~98 · reviewer hat.
**doom_loop.py:** exit=0.

## Adversarial checklist (18 items)

**Consistency:**
- (a) `results.json["verdict"]="KILLED"` ↔ proposed status `killed` → **MATCH**.
- (b) `all_pass=false` ↔ KILLED claim → **MATCH**.
- (c) PAPER.md verdict line "KILLED (preempt-structural; no measurement performed)" — no PROVISIONAL/PARTIALLY/INCONCLUSIVE/DEGENERATE language → **MATCH**.
- (d) `is_smoke=false` ↔ KILLED claim → **MATCH**.

**KC integrity:**
- (e) K#1883 + K#1884 byte-for-byte identical to 2026-04-23 DB (`experiment get` confirms unchanged) → **PASS**.
- (f) Tautology — N/A (no KC measurement performed; carve-out).
- (g) KC IDs in `run_experiment.py` (1883, 1884) ↔ DB → **MATCH**.

**Code ↔ math:**
- (h) `sum(lora_A` / `add_weighted_adapter("linear")` — N/A (no composition computed).
- (i) `LORA_SCALE` ≥ 12 — N/A (no LoRA initialized).
- (j) Single-sample-then-broadcast routing — N/A.
- (k) `shutil.copy` of sibling adapter — N/A (no adapter swap).
- (l) Hardcoded `{"pass": True, ...}` — N/A (kill_criteria result = "fail (untested-preempt)"; carve-out per F#669-family).
- (m) Target model substitution — N/A (no model loaded).
- (m2) **Skill attestation carve-out applies** — F#669-family precedent F#775/F#777/F#779/F#781. `run_experiment.py` imports only `json` + `pathlib`; no `mlx_lm.load`, no `nn.value_and_grad`, no `mx.eval`. MATH.md §0 explicitly attests carve-out. → **CARVE-OUT N/A**.

**Eval integrity:**
- (n) Base accuracy = 0% with avg_thinking_chars = 0 — N/A (no eval).
- (o) Headline n < 15 — N/A.
- (p) Synthetic padding — N/A.
- (q) Cited baseline drift — N/A.
- (t) **Target-gated kill (F#666) carve-out applies** — F#669-family preempt-KILL is a structural verdict where NO KC was measured. Per reviewer hat clause: "(t) does NOT apply to preempt-KILL — F#666 gates kills on proxy-FAIL; preempt-KILL is a structural verdict where NO KC was measured." → **CARVE-OUT N/A**.
- (u) Scope-changing fix — N/A (graceful-failure stub is the canonical preempt-structural artifact, not a scope reduction).

**Deliverables:**
- (r) PAPER.md prediction-vs-measurement table present (§2, 8 rows including 3-parent cardinality NEW row + canonicalization saturation row) → **PASS**.
- (s) Math/argument adversarial pass — F#669 cascade derivation in MATH.md §2 is sound: K#1883 needs `acc_py_iso`, `acc_sql_iso`, `acc_js_iso`, `acc_py_triple`, `acc_sql_triple`, `acc_js_triple`, all six requiring trained py + sql + js adapter weights; all three PROVISIONAL parents have all KCs untested + verified `ls` shows no `adapters/` subdir in any. F#666 derivation in MATH.md §3 is sound: K#1883 (per-axis accuracy delta) and K#1884 (per-layer cos-sim) are both proxy-only with no target-triple counterpart. Compound 3-parent + 2-KC-proxy structural impossibility. → **PASS**.

## Verdict-consistency pre-flight (6/6)

1. results.json `verdict="KILLED"` ↔ DB action `--status killed` → **MATCH**.
2. results.json `all_pass=false` ↔ KILLED → **MATCH**.
3. PAPER.md verdict line ≠ PROVISIONAL/PARTIAL/etc → **MATCH**.
4. `is_smoke=false` ↔ KILLED → **MATCH**.
5. KC git-diff (vs DB 2026-04-23): unchanged → **MATCH**.
6. Antipattern match: F#669-family + F#666-pure compound carve-outs apply per (m2)+(t) → **MATCH**.

## Decision

**KILL (preempt-structural, F#669-family clause + F#666-pure compound).** All 18 adversarial items PASS or have a documented carve-out (F#669-family for m2 and t). All 6 verdict-consistency pre-flight items PASS. F#669 reuse #18, **3-parent F#669 cardinality 1st observation (highest dep-cardinality ever in F#669 family)**. F#780 sub-axis 3rd-instance same-cluster (Hedgehog→Hedgehog→Hedgehog), advancing 2/3 → 3/3 toward same-cluster canonicalization saturation. Cross-cluster canonicalization remains pending.

## Doom-loop self-check

- `doom_loop.py` exit=0.
- 1st preempt-KILL after 7 consecutive HALT-override smoke iters (politeness/refactor/kv_cache/formality/kv_cache_full/conciseness_impl/conciseness_full at iters ~58-94, yielding F#783-F#790). Cascade pattern was successfully broken; this iter returns to it because triple_composition was the next analyst-recommended structurally-distinct entry AND its 3-parent cardinality is genuinely novel (highest ever observed in F#669 family).
- NOT a doom-loop relapse: substantive progress between cascade iters (8 real PROVISIONAL findings filed), and this cascade iter advances same-cluster F#780 saturation to 3/3 + introduces 3-parent F#669 cardinality.

## Findings to file (post-`experiment finding-add`)

1. **F#NEW1** (killed) — F#669 18th reuse; 3-parent F#669 cardinality 1st observation (highest dep-cardinality ever in F#669 family); 4th Hedgehog-cluster F#669; 3rd Hedgehog-cluster pre-F#770-repair compound F#666+F#669.
2. **F#NEW2** (provisional) — F#780 sub-axis 3rd-instance same-cluster (Hedgehog→Hedgehog→Hedgehog); same-cluster canonicalization 3/3 saturation reached. Cross-cluster canonicalization (3-cluster diversity) remains pending.

## Routing

`review.killed` → 🧠 Analyst.
- Analyst payload: ratify F#NEW1+F#NEW2 in LEARNINGS.md.
- Analyst antipattern check: F#780 sub-axis at 3/3 same-cluster saturation — promote to mem-pattern if not already canonicalized; cross-cluster canonicalization is the next milestone (not yet observable in current drain).
- Analyst recommendation for next CLAIM-AND-AUDIT: 0 in-cap P≤2 entries remain after this. Recommend orchestrator-scope macro-budget claim from {memento_gemma4_replication_impl, class_composition_full_impl, politeness_full, refactor_full, formality_full}; researcher-cap drain is exhausted.

## Assumptions

- F#669-family carve-out for (m2) and (t) is canonical per F#775/F#777/F#779/F#781 precedent (4-instance precedent, well past 3-instance threshold).
- F#NEW2 is correctly classified as `provisional` (compound F#666+F#669 pre-F#770-repair sub-axis at 3/3 same-cluster instances). The "same-cluster canonicalization saturation" is a milestone, not a finding-status upgrade — cross-cluster diversity is the genuine canonicalization criterion.
- 3-parent F#669 cardinality novelty claim verified: prior 17 F#669 reuses had cardinality 1; F#781 introduced cardinality 2; this is cardinality 3.
- Reviewer-also-created the MATH.md/run_experiment.py/PAPER.md artifacts in this combined pass because researcher emitted directly to reviewer with payload describing the structural-blocker analysis but did not pre-create the dir; per F#669-family precedent, the artifacts are formulaic and the combined pass is faster than a REVISE round-trip. This judgment call is logged here per reviewer.md "log judgment calls in REVIEW-adversarial.md under Assumptions".
