# PAPER.md — exp_hedgehog_procedural_adapter_refactor_full

**Verdict: PROVISIONAL** (smoke + heuristic_only K2 + 3 KCs deferred)

Smoke run only this iter. Full submission unblocked (smoke gate ALL 5 PASS). API key + LoRA-baseline-training + HumanEval harness + non-refactor curated set + NEUTRAL ablation training run all required for v2 → SUPPORTED/PARTIALLY_SUPPORTED.

---

## 1. Run summary

- **pueue task 8**, smoke mode, wall=55.0s
- **mlx-lm 0.31.2**, model `mlx-community/gemma-4-e4b-it-4bit`
- N_TRAIN=16, N_HELDOUT=8, N_JUDGE=6, N_STEPS=30, LORA_SCALE=6.0, rank=8, targets=`v_proj+o_proj`
- `enable_thinking=False` (F#794 cross-validated mitigation; 2nd cross-exp port from politeness_full)
- 84 LoRA modules attached via manual `LoRALinear.from_base` loop (9th pre-emption of `mem-antipattern-linear-to-lora-layers-shim-recurrence`)

## 2. Prediction-vs-measurement table

| KC | Predicted | Measured | Verdict |
|---|---|---|---|
| K#2004 (per-layer cos > 0.80 mean) | [0.85, 0.97], ≈0.93 | 0.9776 (n=8 heldout) | **PASS** |
| K#2004 (per-layer cos > 0.70 worst) | [0.70, 0.85] | 0.9341 (worst layer) | **PASS** |
| K#2005 (heuristic Δ ≥ 0) | [+8, +20]pp heuristic_only | Δ=0.0 (base=10.0, student=10.0; ceiling-saturated) | **heuristic_only** |
| K#2006 (HumanEval drop < 3pp) | not_measured | not_measured | **DEFERRED** to v2 |
| K#2007 (non-refactor drop < 2pp) | not_measured | not_measured | **DEFERRED** to v2 |
| K#2008 (NEUTRAL ablation drop ≥ 10pp) | not_measured | not_measured | **DEFERRED** to v2 |

## 3. Phase B convergence

- loss_first=0.0566, loss_last=0.0222, ratio=2.55× (above A1 gate threshold 2.0×)
- 30 steps, 10.3s wall (2.91 step/s)
- Mean loss last-5 = 0.02366

## 4. Smoke validation gate (MATH.md §9)

| Gate | Threshold | Measured | Status |
|---|---|---|---|
| A1: loss converges | ratio ≥ 2.0 | 2.55× | PASS |
| A2: cos sanity | mean ≥ 0.80 | 0.9776 | PASS |
| A3: heuristic non-degen | student_mean ≥ 1.0 | 10.0 | PASS |
| A4: thinking-off non-trunc | avg_len > 50 | 250+ chars sample | PASS |
| A5: adapter persists | files exist | yes | PASS |

**block_full_submission = False.** Pueue full submission unblocked.

## 5. Findings (research signal beyond pre-reg KCs)

### F1: enable_thinking=False fix VALIDATED 2nd cross-exp port

Politeness_full (F#794) was 1st cross-exp port; refactor_full is 2nd. Both produce sample-text >50 chars (no thinking-prefix truncation), confirming `mem-antipattern-thinking-mode-truncates-judge-budget` mitigation (1) is robust across:
- Behavior axis (politeness)
- Procedural axis (refactor)

Pattern → safe to apply prophylactically to formality_full (next pick).

### F2: K#2004 K1 cos consistency at smoke (0.9776 vs _impl F#784 0.9706)

The 7 cross-experiment-internal cos points across smoke→smoke runs span [0.96, 0.98] mean — Hedgehog cos-sim distillation reaches near-saturation at smoke N for E4B same-arch teacher (per F#784 caveat: smoke shortcut inflates K1; expected drop at 26B-teacher residency in v2). NOT a NEW finding (F#784 already noted this), but reinforces the same-arch teacher inflation pattern.

### F3: K#2005 heuristic ceiling saturation at smoke (Δ=0.0, both at 10.0)

Heuristic regex detects refactor markers (`def`, `class`, code-block boundaries) — base and student both produce verbose multi-section refactor explanations with code blocks under `enable_thinking=False`. The heuristic ceiling at 10.0 prevents distinguishing base vs adapter; this is the K2-collapse antipattern in a different mode (not preamble truncation, but ceiling-saturation). Heuristic Δ=0.0 is NOT a kill signal — it's a measurement-floor signal.

This refines `mem-antipattern-thinking-mode-truncates-judge-budget`: the antipattern has TWO collapse modes:
- Mode 1: thinking-mode preamble truncates content → heuristic_score=0 (F#783/F#784/F#786)
- Mode 2: thinking-mode-off allows full output → heuristic_score=ceiling (this iter, F#794 politeness_full had Δ=15.72pp at full N=50 because heuristic differentiated when N>20 and content semantics differed; here at N=6 with refactor markers being EITHER-OR present, ceiling saturates)

Both modes need API binding (Claude judge) for true K2 verdict. Smoke heuristic is informative-only.

## 6. Assumptions logged (researcher hat ASSUMPTION clause)

- A: Smoke N=6 K2 ceiling saturation at 10.0 is not a kill — recorded as F3 measurement-floor signal not collapse signal
- B: 2.55× loss reduction satisfies A1 gate per pre-registered §9 (cf politeness_full 2.5× FULL)
- C: K2005 heuristic_only outcome maps to "heuristic_only" KC marker per F#783/F#784 carve-out

## 7. v2 unblock list (priority order)

1. **ANTHROPIC_API_KEY in pueue env** → K#2005 binds via claude-sonnet-4-6 (~5min Phase C only)
2. **Token-space LoRA matched-rank baseline** → K#2005 head-to-head comparison (~30 min training)
3. **HumanEval pass@1 harness** → K#2006 wired (mlx-evaluate or custom; N=164)
4. **Curated 50-prompt non-refactor MBPP/HumanEval slice** → K#2007 (~1h dataset prep + eval)
5. **NEUTRAL teacher ablation arm** → K#2008 (~30min second training)
6. **26B teacher residency** (sequential-phase pattern per MATH.md §0) → K#2004 lands in [0.80, 0.88] band (not 0.97 same-arch shortcut), per F#784 caveat
7. **Full-N (N_TRAIN=200, N_STEPS=800)** → all KCs at standard N

Adapter checkpoint preserved at `adapters/hedgehog_refactor_r8/` for v2 reuse without retraining.

## 8. Antipattern self-audit

| Pattern | Status |
|---|---|
| `linear_to_lora_layers` shim | 9th pre-emption (positive) |
| Composition math (ternary) | N/A (Hedgehog distillation) |
| `LORA_SCALE` ≤ 8 | OK (6.0) |
| `shutil.copy` as new adapter | OK (real training) |
| Hardcoded `"pass": True` | OK (numeric KC binding) |
| Eval truncation (F#790) | MITIGATED + 2nd cross-exp port VALIDATED |
| Proxy model substitution | OK (E4B-it-4bit per MATH §0) |
| K2-collapse on heuristic regex | Mode-2 ceiling-saturation observed (refines mem entry) |
| Smoke-N variance F#795 | N/A (no MMLU here) |
| Researcher pre-files finding before review | EXPLICITLY AVOIDED — no `experiment finding-add` from researcher this iter; reviewer files canonical |
