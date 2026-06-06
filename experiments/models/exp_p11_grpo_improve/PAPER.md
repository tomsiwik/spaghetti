# PAPER — P11.G0: GRPO Refinement from F0 Initialization

**Status: KILLED (preemptive, dependency-chain)** — both upstream reasoning-SFT adapters (F0 s1K, F1 LIMO) are killed and no usable warm-start adapter exists. Theorem 1's premise (p_SFT > p_base, measured on F0) cannot be evaluated; the experiment is not executable.

## Summary
G0's MATH.md assumes a valid F0 adapter exists with accuracy p_SFT on MMLU-Pro, and that GRPO refinement on top of F0 will compound SFT+RL gains. Both upstream dependencies were killed before producing a usable adapter:

- **F0** (`exp_p11_s1k_reasoning_train_eval`, killed 2026-04-17): `mlx_lm.lora` subprocess crashed at step 3 after ~31 min on the full run. `adapters/math-s1k-reasoning-v0/` contains only `adapter_config.json` — no `.safetensors`. Root-cause hypothesis: OOM at MAX_SEQ_LEN=8192 on long s1K traces, compounded by `capture_output=False` discarding stderr and `save-every=200` meaning no checkpoint landed before the crash.
- **F1** (`exp_p11_limo_reasoning_train_eval`, killed 2026-04-14): preemptively killed upstream due to catastrophic-forgetting impossibility structure.

Running `run_experiment.py` now would crash at phase1: `load(MODEL_ID, adapter_path=F0_ADAPTER)` calls mlx_lm which expects `adapters.safetensors` in the adapter dir. The existence check (`F0_ADAPTER.exists()`) passes (the dir exists) but the actual weight load fails.

## Prediction vs Measurement

| KC | Prediction (Theorem) | Measured | Verdict |
|----|----------------------|----------|---------|
| K1514: G0 MMLU-Pro+thinking ≥ 70% | Compound gains (Thm 1+2) predict possible 67–72% | Not measured — no adapter | FAIL (unmeasurable) |
| K1515: G0 GSM8K ≥ F0 GSM8K | Thm 2 (D_train=MMLU-Pro → no MMLU-Pro regression, does not protect GSM8K) | Not measured | FAIL (unmeasurable) |
| K1516: G0 ≥ F0 + 3pp (either bench) | Thm 1 (better init → lower var → faster convergence) | Not measured | FAIL (unmeasurable) |

All three KCs fail as **unmeasurable** — the kill is structural, not empirical.

## results.json (verbatim)
```
{
  "verdict": "KILLED",
  "killed": true,
  "killed_reason": "upstream_dependencies_killed",
  "f0_adapter_exists": true,
  "f0_adapter_usable": false,
  "phase1_executed": false,
  "phase2_executed": false,
  "phase3_executed": false,
  "kill_criteria": {
    "K1514_g0_mmlu_ge_70pct": false,
    "K1515_g0_gsm8k_ge_f0": false,
    "K1516_g0_ge_f0_plus_3pp": false
  }
}
```

## Assumptions (per researcher-hat Autonomy rule)
- I did not re-attempt F0. Rationale: (a) F0 is registered as `killed` in the DB with a specific training-subprocess failure; rerunning G0's scaffolding would not fix F0's OOM. (b) A fresh F0-v2 is out of scope for a claim on G0 — it would require a new experiment (e.g. `exp_p11_s1k_reasoning_train_eval_v2`) with smaller MAX_SEQ_LEN, `capture_output=True`, and `save-every=50`.
- I did not attempt to substitute a different adapter (e.g. `math-gsm8k-knowledge-v0`, 36.1% MMLU-Pro) because doing so would violate Theorem 1's premise (p_SFT > p_base = 62.1%) — the substitute is *worse* than base on MMLU-Pro, invalidating the entire theoretical motivation.
- The SMOKE PAPER.md artifact (prior to this rewrite) showed phase1 yield=64.3% using a smoke F0 adapter (20 steps). That adapter is gone — the full run overwrote the config and never produced safetensors. The smoke result is not recoverable as evidence.
- KC IDs (1514, 1515, 1516) are unchanged from MATH.md registration; no post-hoc edits.

## Verdict Pre-flight Check (§PLAN.md §5)
1. `results.json["verdict"] = "KILLED"` ✓
2. `all_pass` — not applicable to kill
3. PAPER.md verdict line = **KILLED** ✓
4. `is_smoke = false` ✓
5. No KC edits post-MATH.md (git-clean on MATH.md KC table) ✓
6. Antipattern audit: no composition bugs, no tautological routing, no unsafe adapter scale, no `shutil.copy` as new adapter, no hardcoded `"pass": True`, no eval-template truncation, no proxy substitution, no N=smoke-reported-as-full. The kill is structural (missing upstream artifact), not an antipattern ✓

**Verdict: KILLED — dependency-chain preemptive**. No falsification of Theorem 1, 2, or 3 — the experiment was not executed.

## Next Experiment
Unblocking G0 requires producing a valid reasoning SFT adapter. Either:
1. **F0-v2** (`exp_p11_s1k_reasoning_train_eval_v2`): re-run s1K training with (a) `--max-seq-length 4096` (filter traces >4k tokens or truncate), (b) `capture_output` redirected to a log file for postmortem, (c) `save-every 50` so a partial checkpoint survives late crashes.
2. **F1-v2**: revisit LIMO with a structural fix for the catastrophic-forgetting impossibility (the 2026-04-14 kill reason).

Only after one of these produces a registered adapter can G0 be resurrected. This is a hypothesis-generation decision and out of scope for a Researcher-hat claim on G0; the next iteration (or a follow-up with fewer open P0-P2 tasks) should generate the unblocking experiment.

## References
- arXiv:2602.04118 (GRPO sample efficiency — premise requires trained SFT init)
- arXiv:2402.03300 (GRPO; Shao et al. 2024)
- arXiv:2501.12948 (DeepSeek-R1 RS-SFT as GRPO warmup — same dependency)
- arXiv:1612.00796 (EWC; see REVIEW-adversarial.md NB1 — EWC citation acknowledged as misapplied, fix deferred to a G0-v2 under a future claim)
