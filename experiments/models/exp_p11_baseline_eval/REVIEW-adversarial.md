# Adversarial Review: exp_p11_baseline_eval

**Verdict: KILL**

## One-line

2/3 pre-registered KCs falsified: 0/5 P1 adapter safetensors exist on disk (K1505); base MMLU-Pro+thinking 40.7% is 21.4pp below the 62.1% Finding #530 prediction (K1506).

## Results summary

| Metric | Predicted | Measured | Delta |
|---|---|---|---|
| Base MMLU-Pro (thinking=OFF) | ~40% | **45.0%** (126/280) | +5pp ✓ |
| Base MMLU-Pro (thinking=ON) | ~62.1% (Finding #530) | **40.7%** (114/280) | **−21.4pp ✗** |
| Base GSM8K (thinking=ON) | ~77% | HTTP 422 (ext. API) | — |
| 5× P1 adapters | evaluated | 0/5 (safetensors missing) | — |
| avg_thinking_chars (ON) | >0 (not truncated) | 2931 | ✓ (rule n N/A) |

## Kill-criteria verdict (from results.json)

- **K1505** (all 5 adapters evaluated): **FAIL** — value=0.
  Every adapter load errored with `[load_safetensors] Failed to open file …/adapters.safetensors`. Verified on disk: every P1 adapter dir contains only `adapter_config.json`; the weight files were never committed or were deleted. Only `exp_p11_cloq_calibrated_init/adapters/…` has real safetensors anywhere in the tree.
- **K1506** (base MMLU-Pro+thinking ≈62.1% ±5pp): **FAIL** — value=40.7%.
  Thinking ON actually *hurt* base accuracy (−4.3pp vs OFF). avg_thinking_chars=2931 confirms thinking is generated, not truncated — so this is a genuine prediction miss, not antipattern (n).
- K1507 (registry.json updated): PASS, but meaningless given K1505=fail (no adapter rows updated).

## Adversarial checklist

| Rule | Status | Note |
|---|---|---|
| (a) results.json verdict ↔ DB status | OK | no `verdict` key; KC failures routed to KILL. |
| (b) all_pass vs claim | OK | K1505/K1506 fail → not claiming supported. |
| (c) PAPER.md verdict line | N/A | PAPER.md was never written (KILL now; Analyst handles). |
| (d) is_smoke full-run mislabel | OK | `smoke_test: false`, `eval_per_cat: 20`. |
| (e) KC edited after first run | OK | MATH.md has one commit only (de38e37). |
| (f) tautology KC | OK | measurements are real. |
| (g) K-code ↔ MATH semantics | OK | K1505 counts non-error results; K1506 uses `abs(x − 62.1) ≤ 5.0`. |
| (h)–(l) composition/scale/routing | N/A | eval-only experiment. |
| (m) target model ≠ loaded | OK | `mlx-community/gemma-4-e4b-it-4bit` in both. |
| (m2) `/mlx-dev` skill evidence | Weak | not cited, but code uses only stock `mlx_lm.load/generate` + `mx.clear_cache()`. Non-blocking for eval harness. |
| (n) thinking truncated → false gain | OK | avg_thinking_chars=2931, not 0. |
| (o) headline n<15 | OK | n=280 per condition. |
| (p) synthetic padding | OK. |
| (q) cited baseline drift | **FLAG** | Finding #530 cited at 62.1%; measured 40.7%. Either #530 used a different prompt/format/quant, or it must be re-examined. |
| (r) PAPER.md prediction-vs-measurement table | Absent | KILL makes it a follow-up Analyst task, not a REVISE blocker. |

## Root causes

1. **K1505 — pre-run check missed the file, not the directory.** The design-phase review verified `adapter_config.json` directories exist; it never checked for `adapters.safetensors`. The P1 training experiments (`exp_p1_t2_single_domain_training`, `exp_p1_t2_multi_domain_5`) apparently never persisted weight files to disk (or they were deleted). This is a *data integrity* failure, not a code bug in this experiment.
2. **K1506 — prompt format or quant effect.** Finding #530 was measured with a potentially different prompt template and/or model precision. In this eval (4-bit Gemma 4 e4b-it, plain "Answer:" chat prompt, 2048-token thinking budget), thinking actively degrades MCQ accuracy. Candidate causes to document (for the analyst / follow-up experiment): (i) thinking uses the prompt-supplied "Answer:" as an invitation to over-generate, not to extract a letter; (ii) parse_mcq regex picks the last A–J token from a 3000-char reasoning trace, biasing toward parsing errors; (iii) 4-bit quantization of a non-thinking-trained base model may not benefit from the thinking channel at all.

## Why KILL (not REVISE)

- Missing safetensors cannot be fixed by editing `run_experiment.py`. Fixing requires re-running the P1 training experiments — a separate, larger piece of work that belongs to its own experiment.
- The K1506 miss is a substantive measurement that contradicts a prior finding. Revising the eval code won't move the number 21pp.
- Pre-registered KCs are falsified → per PLAN.md §1 kill-criteria discipline, this is `killed`, not "criterion reformulated".

## Assumptions / judgment calls

- Treating Finding #530's 62.1% as the canonical target even though the source eval protocol is not re-verified here. The analyst should compare the two protocols in LEARNINGS.md.
- The GSM8K HTTP 422 is an external-service failure; not counted as a KC miss (K1505 focuses on adapters; base GSM8K was a secondary prediction). It is however worth noting that the datasets-server URL scheme is brittle; follow-up experiments should cache GSM8K locally.

## Follow-up work (for Analyst / next experiments)

1. Re-run P1 single-domain + multi-domain training experiments with weights-save enabled, then re-run P11.E0.
2. Investigate the 21pp MMLU-Pro+thinking shortfall: prompt-format ablation vs Finding #530's original setup; compare `max_tokens`/thinking-budget sensitivity.
3. Local cache GSM8K test set to remove datasets-server dependency from baseline eval.
