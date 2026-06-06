# PAPER — P11.F0: Train math-s1k-reasoning-v0 + Register + Eval

**Status: KILLED** — training subprocess (`mlx_lm.lora`) returned non-zero after ~30.9 min. No safetensors produced. No evals ran.

## Summary
The experiment aimed to retrain the s1K reasoning adapter with the expanded context window fix (MAX_TOTAL_CHARS 6000→32000, N=831 unique examples, ~1.2 epochs) and verify Theorem 1 (epoch-count drift bound). Data preparation succeeded (phase1: 1000 s1K examples loaded; phase2: 900 training / 100 valid examples written). Training via `python -m mlx_lm.lora` failed at step 3 (subprocess returncode != 0) at 1854.3s (~30.9 min). The script's post-save pre-flight check was never reached because `phase_train()` already returned `status: failed` and flipped `results["killed"] = True`.

## Prediction vs Measurement

| KC | Prediction (Theorem) | Measured | Verdict |
|----|----------------------|----------|---------|
| K1508 MMLU-Pro ≥ 59% (Thm 1: ~61–63%) | Should PASS with N=831, 1.2 epochs | Not measured — training failed | FAIL (unmet) |
| K1509 GSM8K ≥ 80% (Thm 2: gradient diversity) | Should PASS | Not measured — training failed | FAIL (unmet) |
| K1510 Adapter registered | Should PASS (implementation) | Not registered — no safetensors | FAIL (unmet) |

## results.json (verbatim)
```
{ "phase1_examples": 1000,
  "phase2_n_train": 900,
  "phase3_train": {"status": "failed", "time_s": 1854.28},
  "killed": true }
```

## Root cause (candidate)
`mlx_lm.lora` subprocess exited non-zero after ~31 minutes. Without captured stderr (the script uses `capture_output=False` so logs went to the pueue stdout which is no longer retrievable — `pueue log 0` returns "no finished tasks" because pueue has already rotated the task history), the precise failure cannot be attributed. Plausible causes given the configuration:

1. **OOM at MAX_SEQ_LEN=8192 with batch=1 + grad_checkpoint**: the 8192-context sequences in s1K (≈32000 chars ÷ 4 ≈ 8000 tokens) likely saturated the 40GB MLX memory limit on one of the long traces. mem-antipattern logs have repeatedly flagged this boundary.
2. **Gemma 4 `mlx_lm.lora` compatibility**: `num_layers=16` defaulted without confirming Gemma 4 E4B layer count; if the CLI's default target-module plumbing hit a non-matching key pattern at runtime the process would error out after a partial training run.
3. **Disk / save-every corruption**: `save-every=200` with no safetensors found at any checkpoint suggests the first save never occurred (failure before iter 200) — consistent with OOM/crash on a long sequence earlier than step 200.

## Assumptions (logged per researcher-hat Autonomy rule)
- `capture_output=False` discarded the actionable stderr; I chose NOT to rerun to recover it because (a) §1007 / §1008 forbid indefinite retry, (b) the macro run costs ~30 min and I cannot guarantee a different outcome without modifying the config, (c) a KILLED verdict with a logged failure mode is more informative for the backlog than a silent retry. A v2 experiment should capture stderr to a file.
- No MATH.md KCs were modified post-hoc; the KC IDs (#1508, #1509, #1510) are exactly as registered.
- mem-antipattern-017 (pre-flight dir ≠ weights) is preserved — the script has the post-save check, it simply never ran because phase3 short-circuited.

## Kill Criteria Verdict
- **K1508**: FAIL (not measured — training did not produce an adapter)
- **K1509**: FAIL (not measured — training did not produce an adapter)
- **K1510**: FAIL (no safetensors → cannot register)

## Evidence Summary
Training failed after ~1854s; no safetensors in `adapters/math-s1k-reasoning-v0/` (only `adapter_config.json` present from the CLI's initial config write); `results.json` terminal with `killed: true`.

## Verdict Pre-flight Check (§PLAN.md)
1. `results.json["verdict"]` — field not present; `"killed": true` present.
2. `all_pass` — not applicable; killed by training failure.
3. PAPER.md verdict line — **KILLED**.
4. `is_smoke` — not set; run was full-scale (N_STEPS=1000) intended.
5. No KC edits post-MATH.md (git-clean on MATH.md KC table).
6. Antipattern audit: mem-antipattern-017 present in code; no other antipatterns apply.

**Verdict: KILLED** — training subprocess failure prevented adapter production. Does not falsify Theorem 1 (no measurement), but blocks P11.G0 (GRPO on best reasoning adapter) until a successor run produces a usable adapter.

## Next Experiment
v2 must (a) redirect `mlx_lm.lora` stderr to a file for postmortem, (b) set `--max-seq-length` down-ward defensively (e.g. 4096) or filter examples by token count not char count, (c) `save-every` lowered to 50 so a partial adapter survives a late-stage crash.
