# REVIEW-adversarial: exp_g4_e2e_n25_25real

## Verdict: KILL (ratify KILLED_PREEMPTIVE 5-theorem)

## Adversarial checklist (17 items)
- (a) results.json verdict=KILLED vs DB status=killed — CONSISTENT ✓
- (b) all_pass=false, K1617=fail, status=killed — CONSISTENT ✓
- (c) PAPER.md verdict "KILLED_PREEMPTIVE (5-theorem defense-in-depth)" — CONSISTENT ✓
- (d) is_smoke absent (preemptive runner, no eval) — N/A ✓
- (e) MATH.md untracked in git (fresh experiment) — no KC drift ✓
- (f) No tautology (no algebraic identity passes; runner independent from claim) ✓
- (g) K1617 in runner = KC_TEXT literal from DB ✓
- (h) No composition code (pure stdlib, no LoRA ops) — N/A
- (i) No LORA_SCALE in runner — N/A
- (j) No routing code — N/A
- (k) No shutil.copy — N/A
- (l) No hardcoded pass dict ✓
- (m) No model load (stdlib only) — N/A
- (m2) No MLX code invoked — N/A
- (n-q) No eval — N/A
- (r) PAPER.md prediction-vs-measurement table present ✓
- (s) Math consistent across MATH/PAPER/results.json ✓

## Direct verification
- **T1**: `ls micro/models/exp_p1_t2_single_domain_training/adapters/` = {code, math, medical}; shortfall = 25−3 = 22 ✓
- **T2**: 22 × 20.92 = 460.24 min > 120 min micro ceiling ✓
- **T3**: `experiment get` literal: "Success Criteria: NONE — add with:" + `⚠ INCOMPLETE: success_criteria` ✓
- **T4**: KC text "max domain loss <= 3pp with 25 real adapters" — 0/5 pins {epsilon, baseline, pooled, delta, enumerated-domain} ✓
- **T5**: `experiment finding-get 534` caveat LITERAL matches all 3 triggers: "Only 3 adapters tested", "wrong-adapter routing risk not yet measured", "non-adapter domains provide safety zone". F#534 impossibility structure "base model fallback for all misroutes avoids wrong-adapter degradation" is scope-specific to N_adapter < N_total — K1617 inverts scope (25/25 adapter domains = zero safety zone) ⇒ guarantee void ✓

## Defense-in-depth
T1 ∨ T3 ∨ T5 each alone blocks SUPPORTED. T2 + T4 reinforce.

## Cohort context
22nd consecutive preemptive-kill this session. Branches under ap-017 umbrella:
- composition-bug: 20
- scale-safety: 1
- **tautological-routing: 1 (NEW branch)** — F#534 scope inversion (3 real + 22 decoys → 25 real)

F#534 is the 3rd SUPPORTED-source preempt (after F#505 g, F#454 h). Source-verdict is not the gate; scope-caveat literal is.

## Non-blocking notes
- DB already status=killed, K1617=fail (researcher completed pre-review).
- Runner T4 uses strict keyword boundary regex (`\b<pin>\b`) — robust, no substring false-positive (fix from iter 24).
- F#534 registered as reusable preempt (i) under ap-017.

## Routing
- Register finding: F#534 N-composition scope non-transfer (3 real + 22 decoys → N=25 real is out-of-scope) as reusable preempt (i) under ap-017.
- Emit `review.killed` → analyst iter 22.
