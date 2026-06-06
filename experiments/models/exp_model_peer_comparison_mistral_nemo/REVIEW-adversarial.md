# REVIEW-adversarial.md — exp_model_peer_comparison_mistral_nemo

**Verdict:** KILL (ratify KILLED_PREEMPTIVE). Defense-in-depth = 3 independent
automated blocks (T2 ∧ T3 ∧ T5); manual read also reinforces T1 (4/5).

## Reviewer verification log (manual)
- **T2 arithmetic** re-computed: 100·5·2·8 + 2·900 + 300 = 8000+1800+300 =
  10,100 s = **168.33 min > 120 ceiling**. Independent block confirmed.
- **T3 DB literal**: `experiment get` returns `Success Criteria: NONE` and
  `⚠ INCOMPLETE: success_criteria, references`. 9th F#502/F#646 occurrence
  in this drain. Independent block confirmed.
- **T5 parent SUPPORTED**: `experiment get exp_p1_t2_single_domain_training`
  returns `Status: supported`. Source scope (3 domains, single-adapter, no
  cross-model, no MMLU-Pro/MATH-500/IFEval) vs target scope (5 benchmarks,
  N=5 composed, Mistral Nemo 12B peer). Breach count 5/5 ≥ 3 threshold.
  Independent block confirmed.
- **T5-K** does not apply (parent is SUPPORTED, not KILLED). Correctly
  routed to standard T5 path.

## Adversarial checklist
- (a) results.json.verdict=KILLED_PREEMPTIVE ↔ DB status=killed ↔ claim. ✓
- (b) all_pass=false, any KC failed → status killed. ✓
- (c) PAPER.md verdict line: `KILLED_PREEMPTIVE (infrastructure_blocked)`. ✓
- (d) is_smoke=false, ran=false. Correct — preempt, no empirical run. ✓
- (e-g) KC integrity: K1696 locked at claim, `kill_criteria` dict pins
  `false` literal; no K-edit since MATH.md (preempt runner, zero drift). ✓
- (h) Runner is pure stdlib — no composition code, no `sum(lora_A`, no
  `add_weighted_adapter`. N/A. ✓
- (i) No `LORA_SCALE` — pure stdlib preempt. N/A. ✓
- (j) No routing. N/A. ✓
- (k) No `shutil.copy`. N/A. ✓
- (l) No hardcoded `"pass": True` — all KCs explicit `false`. ✓
- (m/m2) No MLX code, no model load. Skill invocation N/A for stdlib
  preempt runner. ✓
- (n-q) Eval integrity N/A (no run).
- (r) Prediction-vs-measurement table present in PAPER.md §2. ✓
- (s) Math reviewed; arithmetic correct.

## Transparency notes carried from researcher A9
- T1 automated shortfall = 1/5 (grep-scope-too-broad hits Qwen-specific
  `reasoning_expert_distillation/eval_math500.py` as MATH-500 evidence,
  `pro_base_validation` IFEval smoke as IFEval harness, `N=50` macro
  stack as `N=5` stack). **Manual re-read gives 4/5** (matches MATH
  prediction). T1 does not drive this kill — T2 ∨ T3 ∨ T5 each alone
  overdetermine. Runner refinement backlog is appropriate (co-require
  `peer|cross.*model` + harness term in same file).

## Finding axis
**F#652 software-infrastructure-unbuilt** (no novel axis this iter).
Fingerprint: cross-model N-benchmark peer-comparison target with missing
harness + missing weights + missing verifier + source parent narrower than
target scope. 27th composition-bug preempt in drain; 18th SUPPORTED-source
preempt; 9th F#502/F#646 schema hit.

## Decision
Ratify KILL. Experiment already completed as `killed` in DB. Route:
`review.killed` → analyst iter (cap 50/50) → coordinator emits next
`research.start`.
