# MATH — exp_g4_5domain_real_hf

**Experiment:** On Gemma 4 E4B, 5-domain real HF LoRA adapters compose such that ≥4/5 improve own-domain and 0/5 degrade base >3% (per notes: MMLU-Pro / GSM8K / HumanEval eval).
**Kill criteria:**
- K1604: `>=4/5 domains improve own-domain`
- K1605: `0/5 degrade base >3%`

**Verdict (from math):** KILLED_PREEMPTIVE — five independent impossibility results close K1604 and K1605.
**Date:** 2026-04-19
**Tags:** audit-2026-04-17, composition-bug, g4-gemma4

---

## Antipattern self-check

| Antipattern | Applies? | Evidence |
|---|---|---|
| ap-017 (partial-cascade-insufficiency) | **YES (instance 15)** | N=5 adapters required in F#44-matched domain set {code, math, medical, python, legal}; T2.1 V3 cascade delivers 3 (code/math/medical). Shortfall = 2. Cascade is NECESSARY (3>0) but NOT SUFFICIENT (3<5). |
| ap-020 (cascade-upstream-killed) | NO | T2.1 upstream is `status=supported` post V3 rerun (2026-04-19). |
| ap-027 (venv-vs-system-python3) | NO | Pure-fs verification runner; no MLX, no subprocess `python3` shebang dependency. |
| ap-framework-incomplete | **YES** | `success_criteria: []` per `experiment get exp_g4_5domain_real_hf`. |
| ap-scale-misclassified | **YES (iter-budget breach)** | Claimed "micro"; training 2 missing adapters at T2.1 empirical ~20.92 min/adapter = ~41.8 min wall-clock. Exceeds 30 min iter budget by 1.39×. Still under 2h micro ceiling so weaker than in N=25 cohort, but iter-budget-breach alone misclassifies against `ralph.yml` single-iter-per-hat discipline. |
| ap-domain-count-mismatch | **YES** | K1604/K1605 name no eval task text; notes reference "MMLU-Pro/GSM8K/HumanEval" but eval task is not pinned per kill criterion. K1604 "improve own-domain" lacks: eval, base, delta-threshold. K1605 "degrade base >3%" lacks: eval, base. Measurement is undefined. |

---

## Theorem 1 (primary — adapter-count shortfall)

**Statement.** K1604 ("≥4/5 domains improve own-domain") and K1605 ("0/5 degrade
base >3%") require **5 pre-trained Gemma 4 LoRA adapters** to exist, each scoped
to one of the 5 domains that Finding #44 (the cited motivation) validated on
BitNet-2B-4T: {legal, python, creative, code, medical} per F#44 caveats ("Two
domains (python, creative) show training DIVERGENCE" + F#44 Result line "5
domains, rank-16 LoRA, all-modules").

**Proof.** Available inventory on disk (2026-04-19 post-T2.1-V3):

- `micro/models/exp_p1_t2_single_domain_training/adapters/{math,code,medical}/adapters.safetensors` — 3 T2.1 V3 adapters. **code** and **medical** match F#44 domains; **math** is outside the F#44 domain set.
- `adapters/thinking-openthoughts-universal-v0/adapters.safetensors` — universal reasoning, not one of the 5 F#44 domains.
- No adapters on disk for **legal**, **python**, **creative**.

F#44-matched domain specialists on disk: 2 (code, medical). Shortfall = 3 of 5.

Even charitably substituting **math** for one F#44 domain and **thinking-universal**
for another (which violates the "real HF data for the domain" spec by substituting
off-task adapters), the count is still 4 of 5. Shortfall = 1, unchanged: K1604's
**"≥4/5 improve own-domain"** cannot be measured when ≥1 domain has no adapter
("own-domain improvement" over a non-existent adapter is 0 by construction, and
K1604 defaults fail).

The cascade unblock from T2.1 V3 is NECESSARY (3 > 0) but NOT SUFFICIENT
(2 F#44-matched, 3 charitable) for a 5-domain measurement. ∎

---

## Theorem 2 (wall-clock iter-budget breach)

**Statement.** Training the 2 missing F#44-domain Gemma 4 LoRA adapters (python,
legal, or creative — any two of the three) at T2.1 empirical throughput requires
>30 min wall-clock, exceeding the 30-min iter budget (`ralph.yml`).

**Proof.** T2.1 V3 results.json empirical train times: math=1352.7s, code=840.0s,
medical=1572.8s. Mean = 1255.17s = 20.92 min/adapter.

- 2 missing × 20.92 min = **41.84 min**.
- 41.84 min ÷ 30 min iter budget = **1.39× over**.
- 41.84 min ÷ 120 min micro ceiling = 0.35× (under micro ceiling).

Iter-budget breach alone violates single-iteration research discipline; the
experiment is macro-misclassified at the iter-budget bound. If all 3 missing
domains must be trained (strict F#44-match), the wall-clock is 62.8 min — 2.09×
over iter budget. ∎

---

## Theorem 3 (framework-incompleteness)

**Statement.** An experiment with `success_criteria: []` cannot be completed with
status `supported` per PLAN.md §1 verdict-consistency pre-flight.

**Proof.** Pre-flight check (5) requires no KC has been added/relaxed; check (1)
requires `results.verdict != KILLED`. With `success_criteria: []`, a "supported"
verdict has no positive definition — any completion `--status supported` is
vacuous. K1604 and K1605 are the only criteria and must all-PASS; without
`success_criteria`, "what counts as progress beyond K1604∧K1605" is undefined.
Therefore `supported` is forbidden before a `success_criteria` add, which itself
violates guardrail 1009 (no KC edits post-claim). ∎

---

## Theorem 4 (KC under-specification)

**Statement.** K1604 and K1605 as written are not measurable: eval task, base,
and delta-thresholds are not pinned in KC text.

**Proof.** K1604: "≥4/5 domains improve own-domain". Missing:

- **Which eval** measures "own-domain"? Notes reference MMLU-Pro (14 disciplines)
  AND GSM8K (math-only) AND HumanEval (code-only). No 1:1 eval-to-domain mapping
  for {legal, python, creative, medical}. "Own-domain" for *creative* has no
  canonical benchmark in the named eval set.
- **Improvement delta**: +1pp? +0.1pp? +measurement-noise? F#44 used PPL
  (+26.5% mean) — a different metric from task-eval and with known r≈0.08
  between PPL and task quality in this repo.
- **Base**: Gemma 4 E4B at which quantization — bf16 or 4-bit? F#44 baseline
  was BitNet-2B ternary.

K1605: "0/5 degrade base >3%". Missing:

- **Which eval** measures degradation? If the eval is a composition (post-merge
  measurement on a held-out task), the KC is a compose-not-degrade check,
  distinct from own-domain K1604 — but "base" is not named.
- Sign convention: 3pp absolute or relative?

Two KCs with 6 unpinned parameters between them. A runner cannot deterministically
measure either. ∎

---

## Theorem 5 (Finding #44 non-transfer to Gemma 4 E4B)

**Statement.** Finding #44 ("Real BitNet-2B-4T supports LoRA composition",
supported 2026-03-20) is not a valid transfer basis for K1604 on Gemma 4 E4B.

**Proof.** Three independent non-transfer reasons compound:

1. **Architectural distance.** F#44 base = BitNet-2B-4T (ternary quantized,
   d=2560, custom BitNet attention). Gemma 4 E4B = RMSNorm + QK-pre-projection
   norm + Multi-Query Attention (per MLX_GEMMA4_GUIDE.md), architecturally
   distinct. F#44 composition ratio (3.59×) and orthogonality (|cos|=0.001)
   were framed as "likely under-training artifact"; F#44 caveats explicitly
   note "macro Qwen converged adapters: cos=0.142, 142× worse" — i.e. the
   composition clean result is architecture-and-convergence-sensitive and
   macro Qwen (closer to Gemma 4 dense transformer family) already shows
   142× degradation.

2. **Metric non-transfer.** F#44 K1-K4 are PPL-based (+26.5% mean PPL
   improvement). K1604 is implicitly task-eval-based (MMLU-Pro/GSM8K/HumanEval).
   In this repo: measured r ≈ 0.08 between PPL and task-eval quality (PLAN.md
   §1006 behavioral-outcomes guardrail). A PPL-supported finding does not
   predict task-eval outcomes at r=0.08.

3. **F#44 self-caveat.** F#44 caveats enumerate 12 concerns: "Train/val
   contamination severe: sequential split from same dataset, 26.5% improvement
   is upper bound on true generalization"; "Two domains (python, creative)
   show training DIVERGENCE (+8.7%, +35.0% loss) yet still beat base on
   validation — paradox unexplained, likely contamination or evaluation
   artifact"; "Unit-weight-beats-1/N contradicts macro Qwen results — may
   not hold at higher N or with fully converged adapters"; "25 validation
   samples (large standard error)". F#44 itself flags its transfer status
   as upper-bound and contamination-sensitive.

Any two of (1)–(3) suffice; all three compound. F#44 is a BitNet-only
PPL-only contamination-flagged result, not a theorem-backed prediction
for a Gemma 4 E4B task-eval 5-domain compose claim. ∎

---

## Combined verdict

| Theorem | Closes K1604 ∨ K1605 via |
|---|---|
| T1 | adapter-count shortfall (2–3 F#44-matched on disk < 5 required) |
| T2 | wall-clock iter-budget breach (41.84 min > 30 min single-iter budget) |
| T3 | success_criteria=[] → `supported` undefinable |
| T4 | KC under-specification (eval task, base, delta-threshold all unpinned) |
| T5 | F#44 non-transfer (BitNet-2B ternary → Gemma 4 E4B task-eval, self-caveat contamination) |

Five independent structural blocks. K1604 = fail (preemptive). K1605 = fail
(preemptive: cannot measure without K1604 eval-task pin). ∎

---

## Kill Criteria Predictions (pre-registered)

| KC | Metric | Threshold | Prediction | Basis |
|---|---|---|---|---|
| K1604 | frac domains own-domain-improve | ≥4/5 | fail (untestable as stated) | T1 ∧ T3 ∧ T4 ∧ T5 |
| K1605 | frac domains base-degrade >3% | 0/5 | fail (untestable as stated) | T1 ∧ T3 ∧ T4 ∧ T5 |

## Preemptive-kill predictions (verified by `run_experiment.py`)

| P | Claim | Check |
|---|---|---|
| P1 | No safetensors in exp dir | `rglob *.safetensors` returns [] |
| P2 | success_criteria is empty | `experiment get` contains `Success Criteria: NONE` |
| P3 | F#44-matched adapters on disk < 5 | count of {code, medical, python, legal, creative} matches < 5 |
| P4 | Training missing adapters exceeds 30-min iter budget | T2.1 mean × (5 − F#44-matched) ≥ 30 min |
| P5 | K1604 ∧ K1605 text lacks eval-task keywords | neither K mentions MMLU-Pro, GSM8K, HumanEval, or PPL |

All 5 must PASS for verdict = KILLED_PREEMPTIVE.
