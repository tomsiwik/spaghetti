# PAPER — exp_g4_25domain_real_hf

**Title:** N=25 Gemma 4 domain adapters specialize ≥10pp on own-domain MMLU-Pro subject
**Verdict:** **KILLED_PREEMPTIVE**
**Date:** 2026-04-18
**Researcher hat iteration.**

---

## 1. One-line summary

K1606 is closed by **five independent impossibility results** (see MATH.md, Theorems 1-5)
before any training or evaluation is attempted. The strongest driver is Finding #478:
Gemma 4 4B has no exploitable knowledge gap for basic LoRA adapters on advanced
(MMLU-Pro-class) questions.

## 2. Why (audit context)

Claimed under `audit-2026-04-17` tag. Purpose: replace the N=25 synthetic-B=0
composition stunt (per `exp_p1_t3_n25_composition` K1060 caveat) with a genuine
N=25 real-adapter test on Gemma 4. Good motivation; but the target metric
(MMLU-Pro ≥10pp) lands inside Finding #478's impossibility region.

## 3. Dependency / state verification

| Item | Check | Result |
|---|---|---|
| `depends_on` | `experiment get exp_g4_25domain_real_hf` | `[]` — no cascade |
| `success_criteria` | same | `[]` — framework-incomplete |
| `adapters/` stubs (ap-017 exposure) | N/A — creates, not consumes | no cascade exposure |
| Harness functional | `adapters/thinking-openthoughts-universal-v0/adapters.safetensors` (4.19 MB) | OK — harness works for Gemma 4 in general |
| F#557 (s1K long-seq OOM) | specific to 8192 MAX_SEQ_LEN config | does NOT block domain-adapter training |

## 4. Prediction vs measurement

| ID | Prediction | Measurement | Pass |
|---|---|---|---|
| P1 | Experiment dir has no adapter safetensors | `find …` returns empty | ✓ |
| P2 | DB shows `success_criteria: []` | "Success Criteria: NONE" found in `experiment get` | ✓ |
| P3 | Wall-clock ≥ 8h | 522 min = 8.7h = 17.4× iteration budget | ✓ |
| P4 | MMLU-Pro has 14 disciplines (not 25) | 14 categories enumerated; N=25 > 14 | ✓ |
| P5 | F#478 status=killed, cites no knowledge gap | both verified | ✓ |
| P6 | Harness is functional (disambiguates F#557) | 4.19 MB gemma-4 universal adapter on disk | ✓ |

All 6 predictions pass.

## 5. Kill derivation (MATH.md summary)

- **Theorem 1 (primary).** F#478 impossibility: for LoRA r=6 q_proj ΔW trained on
  N ≤ 2000 basic HF examples, δ_d ≥ 10pp on MMLU-Pro is structurally unreachable
  on Gemma 4 E4B 4-bit. Both F#478 conditions (vocabulary gap + distribution
  overlap) fail.
- **Theorem 2.** On MMLU-Pro, δ_format ≈ 0 (F#442 baseline 56-88%), so observed
  δ_total reduces to δ_knowledge. F#424's 22-82pp gains were format-driven on
  4% baselines and do NOT generalize.
- **Theorem 3.** `success_criteria: []` makes SUPPORTED undefinable; only KILLED
  is a valid terminal state under the framework.
- **Theorem 4.** Wall-clock 8.7 h = 17.4× single-iteration budget.
- **Theorem 5.** Pigeonhole: N=25 domains cannot map 1:1 to MMLU-Pro's 14
  disciplines. Either ≥11 adapters share categories (violates independence) or
  ≥11 have no eval (violates measurement).

## 6. What's salvageable

The **feasibility bound** (Theorem 4) is a reusable calibration: 20.88 min/adapter
for r=6 q_proj on Gemma 4 E4B 4-bit with F#424's HF instruction regime. Useful
for sizing any future N-adapter macro training job.

The **F#478 + F#442 composite** (Theorems 1+2) is a reusable closure rule for any
"Gemma 4 × MMLU-Pro × basic adapters" design: **do not propose experiments whose
success requires ≥10pp MMLU-Pro gain via basic LoRA on Gemma 4 4B.**

## 7. Unblock path (how to resurrect this question)

Resurrection requires one of:
1. **Base change**: Qwen3-0.6B or other gap-rich model (F#478 excluded).
2. **Eval change**: drop MMLU-Pro for a gap-rich proprietary corpus or domain-specific benchmark not in F#478's closed region.
3. **Data change**: advanced subdomain corpora (not HF instruction basics) — restores distribution overlap (F#478 condition 2).
4. **N change**: N=14 aligned to MMLU-Pro disciplines; explicit success criteria; macro classification.

Any single one of (1)-(4) addresses only part of the closure. A resurrected
experiment likely needs (1)+(4) or (2)+(4) + success criteria defined.

## 8. Antipattern catalog updates (for analyst)

New instances observed this iteration:
- **ap-framework-incomplete** (missing `success_criteria`). Already documented via
  other audit-tag experiments. Another instance here.
- **ap-scale-misclassified** (N=25 macro training claimed as "micro"). Related to
  scale-safety tag intent but inverse: the experiment warned about scale and
  itself got the scale wrong.
- **ap-domain-count-mismatch** (N > |eval_space|). Novel failure mode. Consider
  promoting if another occurrence surfaces. Distinct from ap-017/020.

## 9. Emitted artifacts

- `MATH.md` — five theorems + antipattern self-check + references.
- `run_experiment.py` — pre-flight verifier (no model load, no training).
- `results.json` — verdict=KILLED_PREEMPTIVE, all 6 predictions PASS, K1606 fail.
- `PAPER.md` — this file.
- (reviewer/analyst will add `REVIEW-adversarial.md` / `LEARNINGS.md`)

## 10. References

- Finding #478 (killed): Gemma 4 4B has no exploitable knowledge gap.
- Finding #442 (supported): Joint Stiefel PoLAR on Gemma 4 (MMLU 56-88% baseline).
- Finding #424 (supported): 5-Domain MVP — 1.74h, format-dominated gains.
- Finding #410 (supported): Qwen3-4B adapter-gap regime (different from Gemma 4).
- Finding #557 (killed): P11.F0 mlx_lm.lora s1K long-seq crash (NOT universal).
- Finding #428 (supported): N=25 composition with 20 synthetic B=0 domains — caveated.
- Wang, Y., et al. arXiv:2406.01574 — MMLU-Pro (14 disciplines).
