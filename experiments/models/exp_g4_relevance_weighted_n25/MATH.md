# MATH — exp_g4_relevance_weighted_n25

**Experiment:** Relevance-weighted compose on Gemma 4 N=25 beats equal-weight by >=5pp MMLU-Pro.
**Kill criterion K1602:** `diff >= 5pp` (relevance-weighted - equal-weight on MMLU-Pro @ N=25 on Gemma 4 E4B).
**Verdict (from math):** KILLED_PREEMPTIVE — five independent impossibility results close K1602.
**Date:** 2026-04-19
**Tags:** audit-2026-04-17, composition-bug, g4-gemma4

---

## Antipattern self-check

| Antipattern | Applies? | Evidence |
|---|---|---|
| ap-017 (stub adapters) | **YES (partial-cascade-insufficiency)** | N=25 adapters required; only 3 exist (`exp_p1_t2_single_domain_training/adapters/{math,code,medical}/adapters.safetensors`, 4.9 MB each, V3 SUPPORTED). 22/25 = 88% missing. Per ap-017 scope addendum (2026-04-19 iter 13): cascade unblock is NECESSARY but NOT SUFFICIENT when KC denominator > cascade-count. |
| ap-020 (cascade-upstream-killed) | NO | T2.1 upstream is `status=supported` post V3 rerun (2026-04-19). |
| ap-027 (venv-vs-system-python3) | NO | Pure-fs verification runner; no MLX, no subprocess `python3` shebang dependency. |
| ap-framework-incomplete | **YES** | `success_criteria: []` per `experiment get exp_g4_relevance_weighted_n25`. |
| ap-scale-misclassified | **YES** | Claimed "micro"; training 22 missing adapters at T2.1 empirical ~20.9 min/adapter = ~7.66h wall-clock (macro by any definition — iter budget 30 min, micro ceiling 2 h). |
| ap-domain-count-mismatch | **YES** | MMLU-Pro has 14 disciplines (Wang et al. 2024); N=25 > 14 violates pigeonhole for "own-domain" weighting. Relevance-weighting over 25 adapters on 14 categories inherits the same mismatch as exp_g4_l2_norm_compose_n25. |

---

## Theorem 1 (primary — adapter-count shortfall)

**Statement.** K1602 ("diff >= 5pp relevance-weighted vs equal-weight compose at N=25")
requires **25 pre-trained Gemma 4 LoRA adapters** to exist at experiment-run time.
The "compose at N=25" phrase fixes the adapter count at 25; a relevance-weighted
composition over <25 specialists does not measure K1602.

**Proof.** "Compose at N=25" is a scalar equation `w ∘ {A_1, ..., A_25}` requiring
exactly 25 distinct adapter pairs. Available inventory on disk (2026-04-19 post-T2.1-V3):

- `micro/models/exp_p1_t2_single_domain_training/adapters/{math,code,medical}/adapters.safetensors` — 3 adapters.
- `adapters/thinking-openthoughts-universal-v0/adapters.safetensors` — 1 adapter (generic thinking; not a domain specialist per K1602 "domain-relevant weighting" interpretation).

3 domain adapters + 1 universal = 4 adapters maximum. 4 < 25. The required 21+ additional
adapters must be trained (see Theorem 2) to even define the LHS of K1602. The cascade
unblock from T2.1 V3 is NECESSARY (4 > 0) but NOT SUFFICIENT (4 < 25). ∎

---

## Theorem 2 (wall-clock macro-scale bound)

**Statement.** Training the 21 missing Gemma 4 LoRA adapters at T2.1 empirical
throughput requires >2h wall-clock, exceeding both the 30-min iter budget and the
2h micro-scale ceiling.

**Proof.** T2.1 V3 results.json empirical train times: math=1352.7s, code=840.0s,
medical=1572.8s. Mean = 1255.2s = 20.92 min/adapter. 21 missing adapters × 20.92 min
= 439.3 min = 7.32h.

- 7.32 h ÷ 30 min iter budget = 14.64× over.
- 7.32 h ÷ 2 h micro ceiling = 3.66× over.

Either bound alone misclassifies the experiment as macro-scale. Per PLAN.md Part 2,
macro-scale training requires operator approval. ∎

---

## Theorem 3 (framework-incompleteness)

**Statement.** An experiment with `success_criteria: []` cannot be completed with
status `supported` per PLAN.md §1 verdict-consistency pre-flight.

**Proof.** Pre-flight check (5) requires no KC has been added/relaxed; check (1)
requires `results.verdict != KILLED`. With `success_criteria: []`, a "supported"
verdict has no positive definition — any completion `--status supported` is
vacuous. Therefore `supported` is forbidden before a KC-add, which itself violates
guardrail 1009 (no KC edits post-claim). K1602 exists but is the *only* criterion
and must all-PASS; the missing `success_criteria: []` leaves "what counts as
progress" undefined beyond K1602's negative framing. ∎

---

## Theorem 4 (MMLU-Pro discipline-count pigeonhole)

**Statement.** MMLU-Pro has 14 disciplines (Wang et al. 2024, arxiv:2406.01574
Table 2). Any compose-at-N=25 where N exceeds the category count forces ≥12 adapters
to share disciplines with other adapters, violating the "one-adapter-per-category"
reading of relevance-weighted composition.

**Proof.** |disciplines| = 14 = {biology, business, chemistry, computer_science,
economics, engineering, health, history, law, math, philosophy, physics, psychology,
other}. N=25 > 14 by pigeonhole forces min 25 − 14 = 11 adapters to collide with a
sibling. K1602's "relevance-weighted" interpretation over disjoint domains is
ill-defined under collision. Either K1602 is reinterpreted (changes KC — forbidden)
or the design violates pigeonhole. ∎

---

## Theorem 5 (Finding #137 non-transfer to Gemma 4)

**Statement.** Finding #137 (`Relevance-weighted composition resolves dilution`,
supported 2026-03-28) is not a valid transfer basis for K1602's Gemma 4 E4B claim.

**Proof.** Finding #137 established PPL-probe weighting at +9.34pp over equal-weight,
probe-oracle r=0.990, on a different base model (BitNet-2B per the micro cohort of
that date) with N=10 composition types. Gemma 4 E4B uses RMSNorm + QK-pre-projection
norm + Multi-Query Attention (per MLX_GEMMA4_GUIDE.md), architecturally distinct
from BitNet-2B's ternary quantized base. The relevance-weighting mechanism in F#137
(PPL-probe per-adapter) measures *ppl on a held-out probe* — a quantity whose
calibration is quantization-sensitive and has never been validated on 4-bit Gemma 4.
Transfer is unestablished; K1602's +5pp target is not a theorem-backed prediction
on the new base model. ∎

---

## Combined verdict

| Theorem | Closes K1602 via |
|---|---|
| T1 | adapter-count shortfall (3/25 or 4/25 available) |
| T2 | wall-clock macro-scale breach (7.32h >> 2h micro ceiling) |
| T3 | success_criteria=[] → supported undefinable |
| T4 | MMLU-Pro 14-discipline pigeonhole violation |
| T5 | Finding #137 non-transfer (BitNet-2B → Gemma 4 E4B) |

Five independent structural blocks. K1602 = fail (preemptive). ∎

---

## Kill Criteria Predictions (pre-registered)

| KC | Metric | Threshold | Prediction | Basis |
|---|---|---|---|---|
| K1602 | diff (rel-wtd - equal-wtd) MMLU-Pro | >=5pp | fail (untestable as stated) | T1 ∧ T2 ∧ T3 ∧ T4 ∧ T5 |

## Preemptive-kill predictions (verified by `run_experiment.py`)

| P | Claim | Check |
|---|---|---|
| P1 | No safetensors in exp dir | `rglob *.safetensors` returns [] |
| P2 | success_criteria is empty | `experiment get` contains `Success Criteria: NONE` |
| P3 | Available adapters < 25 | disk count < 25 |
| P4 | Training missing adapters exceeds 2h macro threshold | from T2.1 mean × (25 - available) ≥ 120 min |
| P5 | MMLU-Pro has 14 disciplines < 25 | `len(MMLU_PRO_CATS) == 14 and 25 > 14` |

All 5 must PASS for verdict = KILLED_PREEMPTIVE.
