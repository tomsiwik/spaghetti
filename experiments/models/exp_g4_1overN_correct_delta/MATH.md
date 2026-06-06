# MATH — exp_g4_1overN_correct_delta

**Experiment:** On Gemma 4 E4B @ N=25 MMLU-Pro, 1/N scaling beats equal(scale=1) and additive by ≥3pp (clean delta-sum).
**Kill criterion K1603:** `1/N beats others by 3pp` (delta between 1/N and best of {equal(scale=1), additive} on MMLU-Pro @ N=25 on Gemma 4 E4B).
**Verdict (from math):** KILLED_PREEMPTIVE — five independent impossibility results close K1603.
**Date:** 2026-04-19
**Tags:** audit-2026-04-17, composition-bug, g4-gemma4

---

## Antipattern self-check

| Antipattern | Applies? | Evidence |
|---|---|---|
| ap-017 (stub adapters) | **YES (partial-cascade-insufficiency, instance 13)** | N=25 adapters required; only 3 exist (`exp_p1_t2_single_domain_training/adapters/{math,code,medical}/adapters.safetensors`, 4.9 MB each, V3 SUPPORTED). 22/25 = 88% missing. Per ap-017 scope addendum (2026-04-19 iter 13): cascade unblock is NECESSARY but NOT SUFFICIENT when KC denominator > cascade-count. |
| ap-020 (cascade-upstream-killed) | NO | T2.1 upstream is `status=supported` post V3 rerun (2026-04-19). |
| ap-027 (venv-vs-system-python3) | NO | Pure-fs verification runner; no MLX, no subprocess `python3` shebang dependency. |
| ap-framework-incomplete | **YES** | `success_criteria: []` per `experiment get exp_g4_1overN_correct_delta`. |
| ap-scale-misclassified | **YES** | Claimed "micro"; training 22 missing adapters at T2.1 empirical ~20.9 min/adapter = ~7.66h wall-clock (macro by any definition — iter budget 30 min, micro ceiling 2 h). |
| ap-domain-count-mismatch | **YES** | MMLU-Pro has 14 disciplines (Wang et al. 2024); N=25 > 14 violates pigeonhole. 1/N scaling assumes adapter domains are disjoint so that per-domain contribution is `(1/N) · Δ_i`; under pigeonhole collisions (≥11 colliding specialists), 1/N understates per-category contribution relative to additive, biasing K1603 in an uncontrolled direction. |

---

## Theorem 1 (primary — adapter-count shortfall)

**Statement.** K1603 ("1/N beats others by 3pp at N=25 MMLU-Pro compose") requires
**25 pre-trained Gemma 4 LoRA adapters** to exist at experiment-run time.
"Compose at N=25" fixes the adapter count; a 1/N-vs-equal(1)-vs-additive comparison
over <25 specialists does not measure K1603.

**Proof.** The three compose schedules under test are functions of the same adapter
set: `W_1/N = W_0 + (1/N) Σ_{i=1}^{N} Δ_i`, `W_eq = W_0 + Σ_{i=1}^{N} Δ_i` (with
scale=1, which is equivalent to additive when N>1 — the distinction in this KC is
between uniform-weight-1/N and uniform-weight-1 merging before vs. after delta-sum
correction). All three sums require N distinct `Δ_i = B_i A_i` pairs. Available
inventory on disk (2026-04-19 post-T2.1-V3):

- `micro/models/exp_p1_t2_single_domain_training/adapters/{math,code,medical}/adapters.safetensors` — 3 adapters.
- `adapters/{math,code,medical,sql,python,bash,math-s1k-reasoning-v0,math-star-r1-v0,thinking-openthoughts-universal-v0}/adapters.safetensors` — 9 additional adapter files, but 3 of them (math/code/medical) duplicate the T2.1 set, and 3 (math-s1k-reasoning-v0, math-star-r1-v0, thinking-openthoughts-universal-v0) are domain-independent thinking/reasoning adapters not pre-trained on the 25 MMLU-Pro-indexed categories that a 1/N K1603 claim would index against.
- Counting non-duplicate specialist candidates charitably: {math, code, medical, sql, python, bash} = 6 domain specialists. 6 < 25.

The required 19+ additional adapters must be trained (see Theorem 2) to even define
the RHS of K1603. The cascade unblock from T2.1 V3 is NECESSARY (3 > 0) but NOT
SUFFICIENT (6 < 25). ∎

---

## Theorem 2 (wall-clock macro-scale bound)

**Statement.** Training the 19 missing Gemma 4 LoRA adapters at T2.1 empirical
throughput requires >2h wall-clock, exceeding both the 30-min iter budget and the
2h micro-scale ceiling.

**Proof.** T2.1 V3 results.json empirical train times: math=1352.7s, code=840.0s,
medical=1572.8s. Mean = 1255.2s = 20.92 min/adapter. 19 missing adapters × 20.92 min
= 397.5 min = 6.62h.

- 6.62 h ÷ 30 min iter budget = 13.25× over.
- 6.62 h ÷ 2 h micro ceiling = 3.31× over.

Either bound alone misclassifies the experiment as macro-scale. Per PLAN.md Part 2,
macro-scale training requires operator approval. ∎

---

## Theorem 3 (framework-incompleteness)

**Statement.** An experiment with `success_criteria: []` cannot be completed with
status `supported` per PLAN.md §1 verdict-consistency pre-flight.

**Proof.** Pre-flight check (5) requires no KC has been added/relaxed; check (1)
requires `results.verdict != KILLED`. With `success_criteria: []`, a "supported"
verdict has no positive definition — any completion `--status supported` is
vacuous. K1603 exists but is the *only* criterion and must all-PASS; the missing
`success_criteria: []` leaves "what counts as progress" undefined beyond K1603's
negative framing. Therefore `supported` is forbidden before a KC-add, which itself
violates guardrail 1009 (no KC edits post-claim). ∎

---

## Theorem 4 (MMLU-Pro discipline-count pigeonhole)

**Statement.** MMLU-Pro has 14 disciplines (Wang et al. 2024, arxiv:2406.01574
Table 2). Any compose-at-N=25 where N exceeds the category count forces ≥11
adapters to share disciplines with other adapters. For 1/N scaling specifically,
colliding specialists add sub-linearly (each category's delta-sum is dominated
by the pigeonhole bucket-size, not by 1/N), biasing K1603 in an uncontrolled
direction regardless of compose schedule.

**Proof.** |disciplines| = 14 = {biology, business, chemistry, computer_science,
economics, engineering, health, history, law, math, philosophy, physics,
psychology, other}. N=25 > 14 by pigeonhole forces min 25 − 14 = 11 adapters to
collide with a sibling. Under 1/N: per-category contribution = (1/25) · Σ_{i∈bucket}
Δ_i, scaling with bucket-size not with 1. Under additive(scale=1): same sum but
with factor 1. The ratio (1/N-vs-additive) at a category depends on bucket-size,
not on a schedule invariant. K1603's 3pp gap is therefore a measurement of the
pigeonhole distribution, not of the "1/N resolves composition catastrophe"
hypothesis. Either K1603 is reinterpreted (changes KC — forbidden) or the
measurement is confounded. ∎

---

## Theorem 5 (Finding #13 / #14 non-transfer to Gemma 4)

**Statement.** Findings #13 ("Pre-merge composition preserves gains over base",
supported 2026-03-15) and #14 ("1/N scaling resolves composition catastrophe",
supported 2026-03-28) are not valid transfer bases for K1603's Gemma 4 E4B
N=25 claim.

**Proof.** F#13 was validated at macro scale with N=5 (not N=25) on BitNet-2B
(architectural precondition). F#14 was validated on BitNet-2B with N=5 adapters
(sql, python, bash, math, medical), reporting PPL trillions→2.36 for 1/N vs.
naive-sum. Gemma 4 E4B uses RMSNorm + QK-pre-projection norm + Multi-Query
Attention (per MLX_GEMMA4_GUIDE.md), architecturally distinct from BitNet-2B's
ternary quantized base. Additionally, F#13 caveats explicitly: "UPDATE
2026-03-26: exp_cross_adapter_knowledge_transfer KILLED this interpretation;
0/20 pairwise transfers >2%, the benefit is 1/N regularization not knowledge
sharing." The "1/N as regularization" interpretation is architecture-sensitive
(4-bit Gemma 4 regularization dynamics never measured), and the N-extrapolation
from 5 → 25 at the PPL trillions→2.36 scale is unvalidated. Transfer is
unestablished; K1603's +3pp target is not a theorem-backed prediction on the
new base model at the new N. ∎

---

## Combined verdict

| Theorem | Closes K1603 via |
|---|---|
| T1 | adapter-count shortfall (3/25 T2.1 + 3-6 non-duplicate candidate < 25 required) |
| T2 | wall-clock macro-scale breach (6.62h >> 2h micro ceiling) |
| T3 | success_criteria=[] → supported undefinable |
| T4 | MMLU-Pro 14-discipline pigeonhole confounds schedule comparison |
| T5 | Findings #13/#14 non-transfer (BitNet-2B N=5 → Gemma 4 E4B N=25) |

Five independent structural blocks. K1603 = fail (preemptive). ∎

---

## Kill Criteria Predictions (pre-registered)

| KC | Metric | Threshold | Prediction | Basis |
|---|---|---|---|---|
| K1603 | diff (1/N - best(equal(1), additive)) MMLU-Pro | >=3pp | fail (untestable as stated) | T1 ∧ T2 ∧ T3 ∧ T4 ∧ T5 |

## Preemptive-kill predictions (verified by `run_experiment.py`)

| P | Claim | Check |
|---|---|---|
| P1 | No safetensors in exp dir | `rglob *.safetensors` returns [] |
| P2 | success_criteria is empty | `experiment get` contains `Success Criteria: NONE` |
| P3 | Available adapters < 25 | disk count < 25 |
| P4 | Training missing adapters exceeds 2h macro threshold | from T2.1 mean × (25 - available) ≥ 120 min |
| P5 | MMLU-Pro has 14 disciplines < 25 | `len(MMLU_PRO_CATS) == 14 and 25 > 14` |

All 5 must PASS for verdict = KILLED_PREEMPTIVE.
