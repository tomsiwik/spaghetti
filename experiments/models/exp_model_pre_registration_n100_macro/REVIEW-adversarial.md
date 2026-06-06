# REVIEW-adversarial.md — exp_model_pre_registration_n100_macro

## Verdict from researcher: KILLED_PREEMPTIVE
Over-determined by 3 of 5 automated blocks
(T2 ∧ T3 ∧ T5-K-double). T1 is reinforce-only on the automated
runner (shortfall 2/5, i.e. under the block threshold); manual
re-read gives 5/5 because the `W_combined` and `N=100 routing`
probes false-positive-match on the KILLED parent experiment and
three unrelated macro scripts.

## Adversarial checklist

### (a) Verdict-chain consistency
- `results.json["verdict"] == "KILLED_PREEMPTIVE"` ✓
- `results.json["all_pass"] == false` ✓
- `results.json["is_smoke"] == false` ✓
- PAPER.md `## Verdict` line says `KILLED_PREEMPTIVE` ✓
- Will mark DB `--status killed` on `experiment complete` ✓

### (b) KC integrity
- K1708, K1709, K1710 declared in DB claim record prior to this
  runner being written. Runner sets them to `false` with literal
  `"target not run; infra blocked"` semantics. No silent relax, no
  tautology (runner does not measure the K directly — it measures
  whether the infra to measure the K exists).
- No KC added, no KC removed, no KC edited post-claim.
  `git diff MATH.md` is a new file (not a mid-experiment edit).

### (c) KC ↔ code ↔ math chain
- MATH.md §1 declares the hypothesis and KC verbatim from DB.
- MATH.md §2 pre-registers the 5-theorem stack, and §3 the P1–P5
  predictions.
- `run_experiment.py` implements exactly the 5 theorems; each
  probe returns a dict with `block`, measured counters, and
  evidence.
- PAPER.md §Prediction vs measurement binds each P_i to the
  results.json field that measures it.

### (d) No tautology, no silent upgrade
- Runner does **not** write `all_pass=true`. Runner does **not**
  hard-code any pass. Every `false` in KC is accompanied by the
  literal reason in `reason:` at the top of results.json.
- `shortfall >= 3` threshold is pre-registered in MATH.md; runner
  emits `block: false` when shortfall = 2 (honest under-count
  caveated in PAPER A9).

### (e)–(g) KC git-diff / pre-registration
- MATH.md §1 KC are the DB KC verbatim. No diff.
- `results.json` writes three KC keys with false — matches DB KC
  count exactly.

### (h)–(m) Math ↔ code
- T1 grep scopes (`pierre/**/*.py`, `macro/**/*.py`, `composer/**/
  *.py`, `micro/models/**/*.py`, minus this runner) are declared
  in MATH §A1 and implemented in `_code_files()`.
- T1 adapter-inventory probe: `micro/models/**/adapters/
  **/*.safetensors` with size > 1 KB and distinct parent dir
  dedup. Matches MATH A2.
- T2 arithmetic re-verified:
  `900 + 600 + 25000 + 25000 + 1000 = 52,500 s = 875.0 min`.
  Matches MATH §T2 and results.json.
- T2 floor: `900 + 600 + 1000 + 0 + 1000 = 3,500 s ≈ 58.3 min`
  under ceiling, but MATH A4 documents the statistical / structural
  incoherence (CI too wide at 10 Q/domain; solo baselines absent
  for 97/100 domains).
- T3 literal strings match DB pretty-print. T3 detects both the
  `Success Criteria: NONE` line and the `⚠ INCOMPLETE:` marker.
- T5-K double: runner reads DB `Status:` line for both declared
  parents (`exp_model_room_model_gemma4_speed`,
  `exp_p1_t3_n25_composition`), confirms both killed, then
  probes parent on-disk `results.json` / `PAPER.md` / `MATH.md`
  for the specific KC-fail literals (`K1688.*FAIL`, `69.18 tok`,
  `K1689.*FAIL`, `0.9941`, `K1060.*FAIL`, `0/5.*adapter`,
  `K1061.*FAIL`, `MMLU.*regress`). All 5 breach dimensions return
  true.

### (h2) A9 honesty
T1 reported automated shortfall = 2, under the 3 threshold. PAPER
A9 documents why manual re-read gives 5 (parent-KILLED false-
positive on W_combined; unrelated scripts on N=100 routing). The
runner does **not** backfill or inflate to force the T1 block to
fire — verdict is over-determined by T2 ∨ T3 ∨ T5-K-double alone.

### (i)–(m2) Antipattern scan
- ✗ No composition math bug (no composition implemented).
- ✗ No `LORA_SCALE` mishandling (no LoRA).
- ✗ No tautological routing (no routing).
- ✗ No `shutil.copy` adapter injection.
- ✗ No hardcoded `"pass": True` (every `false` is literal).
- ✗ No eval-template truncation (no eval).
- ✗ No proxy-model-for-target (no model).
- ✗ No KC-measures-wrong-object (runner measures infra, not KC).
- ✗ No N=smoke-reported-as-full (`is_smoke: false`).

### (n)–(q) Eval hygiene
- Zero real eval, zero benchmark bias, zero hardcoded pass.
- No `shutil.copy` anywhere. No proxy substitution. No eval-template
  contamination (no eval).

### (r) Deliverables present
- MATH.md ✓
- run_experiment.py ✓ (pure stdlib, 3.82 s wall)
- results.json ✓
- PAPER.md ✓
- REVIEW-adversarial.md ✓ (this file)
- LEARNINGS.md (analyst-owned, currently capped per HALT §C — debt
  tracked in `.ralph/agent/scratchpad.md`).

### (s) Runtime discipline
- Runner is pure stdlib + `subprocess` shell-out to `experiment get`.
- Zero MLX, zero model load, zero HTTP bind, zero composition code,
  zero LORA_SCALE, zero routing, zero `shutil.copy`.
- 3.82 s wall on M5 Pro 48 GB.
- Cumulative drain target budget remains intact (< 5 s per
  preempt × 44 preempts ≈ 220 s total).

## Finding placement (advisory to analyst)
Candidate F-axis: **F#_NEW under ap-017 lineage**, **double-T5-K
parent-KILLED sub-variant**. Distinct from F#651 (single-parent
T5-K, iter 36) in two ways:
1. Both declared parents are empirically KILLED — not one.
2. The target's KC each map transitively onto a different parent's
   failed KC (K1708 ↔ K1060 / K1061; K1710 ↔ K1688 / K1689).
   Target is not just "parent failed"; it is "every KC has a
   pre-measured falsifier in a parent".

Analyst's call: sibling of F#651 vs strict child. My read is
**child** — strictly stronger because the target cannot partially
recover by picking the surviving parent; both parents are gone.

## Verdict (reviewer)
**RATIFY — KILL.** Over-determined by T2 ∨ T3 ∨ T5-K-double.
T1 is honestly reinforce-only on the automated runner and not
counted toward the verdict.
