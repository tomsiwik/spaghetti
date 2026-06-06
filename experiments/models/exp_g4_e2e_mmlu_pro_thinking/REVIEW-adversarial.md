# REVIEW-adversarial.md — exp_g4_e2e_mmlu_pro_thinking

## Verdict
**KILL** (endorses researcher's KILLED_PREEMPTIVE).

## 1. Consistency
- `results.json["verdict"]` = "KILLED_PREEMPTIVE"; DB `status=killed`;
  PAPER.md verdict = KILLED_PREEMPTIVE. All consistent.
- `all_pass=true` means all six pre-flight predictions *confirming kill
  drivers* passed. K1618 `result=fail` in the same file. Logic is
  correct: predictions verify premises of kill theorems.
- `is_smoke=false`; no full-run claim.
- DB evidence line matches results.json evidence.

## 2. KC integrity
- (e) Whole experiment dir is untracked (`?? micro/models/exp_g4_e2e_mmlu_pro_thinking/`).
  Single batch of new files — KC-swap is not possible.
- (f) No tautology. Six **mutually independent** theorems close K1618:
  - T1: 5/5 stub adapters → forward = identity
  - T2: F#536 linearity extension (new): sum preserves suppression
  - T3: F#560 — no positive ΔW thinking adapter exists
  - T4: F#478 knowledge-gap closure on MMLU-Pro
  - T5: cascade-closure (upstream open + upstream killed)
  - T6: framework-incomplete (no threshold/MDE/n for K1618)
  Any single theorem suffices; no shared premise.
- (g) K-ID 1618 consistent across DB, MATH.md, PAPER.md, results.json.

## 3. Code ↔ math
- (h)–(l), (m)–(m2) all N/A or PASS. Pre-flight script performs no
  model load, no routing, no composition; only filesystem stub checks.
- Stub count verified from disk: 5/5 local
  (`adapters/{math,bash,python,sql,medical}` all contain
  `adapter_config.json` + tokenizer_config.json only, no `*.safetensors`).
- 3/3 registry-pointed stubs confirmed by MATH.md inventory.

## 4. Eval integrity
- N/A — pre-flight kill, no eval run.

## 5. Deliverables
- MATH.md present with 6 theorems + antipattern self-check + references.
- PAPER.md present with prediction-vs-measurement table (§3) and
  dependency state table (§6).
- results.json is well-formed.
- run_experiment.py is pre-flight-only and writes results.json
  deterministically; reproducible without GPU.

## 6. Novelty over prior kills
- Theorem 2 is a genuine **linearity extension** of F#536 (empirical
  single-adapter → algebraic sum-of-adapters suppression). Distinct
  contribution, worth surfacing to Analyst as closure-rule candidate.
- Theorem 5 (pipeline cascade-closure) is the first **explicit**
  statement of a rule that M0/C0 prior kills used implicitly.
- ap-017 instance count after this kill: 8 (was 7 — tracked in
  scratchpad).

## 7. Open threads for Analyst
- Bump antipattern-017 to 8 confirmed instances.
- Promote Theorem 2 to a closure-rule finding: "Delta-sum composition
  of mode-suppressing LoRA adapters linearly preserves mode
  suppression." Anchor: F#536. Novelty: linearity extension.
- Promote Theorem 5 to pattern-level antipattern candidate
  (`ap-cascade-series-closure`): first explicit formulation.
- `current_direction.md` is stale; after this kill only
  `exp_p1_t5_user_local_training` remains P=1 open.

## 8. Assumptions
- None needed — every kill driver is empirically or algebraically
  closed against existing findings.
