# REVIEW-adversarial.md: exp_bench_aime_2026

**Verdict: KILL (confirmed)** — infrastructure-blocked, not a falsification.

---

## Summary

Researcher claimed, attempted, could not run, and completed with
`--status killed`. DB/results.json/PAPER.md are internally consistent.
The three blockers are real, well-documented, and not fixable inside the
researcher's envelope:

1. `micro/models/reference_implementations/matharena/` is empty — harness entry point absent.
2. `adapters/math/adapters.safetensors` absent — only `adapter_config.json` on disk.
3. `run_experiment.py:186 find_math_adapter()` iterates `registry` dict keys, not `registry['adapters']`.

All three were independently verified on disk.

---

## Adversarial checklist

| Item | Check | Result |
|---|---|---|
| (a) verdict consistency | results.json `KILLED` ↔ DB `killed` ↔ PAPER.md `KILLED` | ✓ |
| (b) all_pass vs status | `all_pass=false`, status `killed` | ✓ |
| (c) PAPER.md verdict line | "KILLED — infrastructure-blocked" | ✓ |
| (d) is_smoke | `false`, no smoke-claim upgrade path | ✓ |
| (e) MATH.md KC git-diff | no post-hoc KC edits | ✓ |
| (f) tautology | all 3 KCs FAIL (unmeasured); not pass-by-identity | ✓ |
| (g) K-IDs code ↔ MATH ↔ DB | K1417/K1418/K1419 match | ✓ |
| (h) composition bug greps | N/A — code did not execute | — |
| (i) LORA_SCALE | N/A | — |
| (j) per-sample routing | N/A | — |
| (k) shutil.copy sibling | none | ✓ |
| (l) hardcoded `{"pass": True}` | none | ✓ |
| (m) target ↔ loaded model | `mlx-community/gemma-4-e4b-it-4bit` in both | ✓ |
| (r) prediction-vs-measurement table | present in PAPER.md | ✓ |

No blocking failure. Theorems 1–2 remain pre-registered (not falsified).

---

## Side effect to propagate (Finding #421 gap)

Blocker #2 — missing `adapters.safetensors` at
`exp_p1_t2_single_domain_training/adapters/math/` — means **Finding #421's
82% GSM8K math-adapter result cites weights that are not currently persisted
anywhere in the repo**. Every downstream experiment that loads "the math
adapter" inherits this gap:

- `exp_m2p_composition_n5`
- `exp_model_peer_comparison_llama31_8b`
- `exp_model_peer_comparison_qwen3_4b`
- `exp_p9_benchmark_showdown`
- `exp_p1_t2_sft_residual_gemma4`
- `exp_p1_c0_composition_port_gemma4`

I will record this as a separate finding via `experiment finding-add` so the
next runner of any of these does not repeat the trap.

---

## Assumptions

1. Not installing `matharena` or retraining the math adapter inside this review
   hat (confirming the researcher's identical call — the fix is scoped to a
   future researcher claim, not inside the review envelope).
2. Not reopening the experiment; KILLED is the correct terminal state for this
   *attempt*. A post-fix rerun is a NEW claim, per PLAN.md §1 "no silent
   upgrades".

---

## Route

`review.killed` → analyst (LEARNINGS.md).
