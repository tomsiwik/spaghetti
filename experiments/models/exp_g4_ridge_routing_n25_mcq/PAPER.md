# PAPER.md — exp_g4_ridge_routing_n25_mcq

## Experiment
Ridge routing on Gemma 4 E2B hidden-state features over N=25 MMLU subjects with
disjoint train/test partitions and 10 hard-negative pairs.

**KC:** K1616 (primary, pre-registered) — test accuracy ≥ 90% per-sample.

**Scale:** **full** (`IS_SMOKE=0`): N_TRAIN=100, N_TEST=40 per domain.
Actual yield after per-subject capacity limits: 2260 train / 930 test samples
(some MMLU subjects have < 100+40=140 rows, script falls back to 70/30 split).
Matches MATH.md's declared full scale; still below F#502 (200/80) but within the
same measurement regime.

## Prediction vs Measurement
| # | Claim | Target | Measured | Pass | Source |
|---|-------|--------|----------|------|--------|
| P1 (K1616) | ridge test acc at N=25, disjoint + hard-neg | ≥ 0.90 | **0.8387** | **FAIL** | results.json `test_acc` |
| P2 | worst-domain acc (no feature collapse) | ≥ 0.60 | **0.2500** | **FAIL** | results.json `per_domain_acc[clinical_knowledge]` |
| P3 | ridge fit time | ≤ 60 s | 0.12 s | PASS | results.json `fit_time_s` |
| P4 | per-sample inference latency | ≤ 10 ms | 0.0004 ms | PASS | results.json `infer_ms_per_sample` |

K1616 FAILs by 6.1 pp at full scale. Best α=100 (inner-val 82.6% / test 83.9% —
no generalization gap; the 6.1pp shortfall is structural, not overfit).

## Verdict
**KILLED** — K1616 fails at full (non-smoke) scale with clean measurement.
`is_smoke=false` in results.json; guardrail #1009 satisfied. Per MATH.md's
"What falsifies the theorem" section, two of three falsification clauses trigger
(P1 acc < 86%-lower-bound is near-miss at 83.87%; P2 worst-domain < 30% is
clean trigger). Theorem 3's empirical prediction of [86%, 98%] does not hold
at Gemma 4 E2B + smoke-feature-extraction + N=25 + F#502 methodology.

## Key quantitative comparison
| Method / study | N=25 methodology | Test acc | Source |
|----------------|------------------|----------|--------|
| TF-IDF centroid-collapse (tautological) | F#458 | 98.8% | finding #458 |
| TF-IDF + disjoint + hard-neg | F#502 | 84.2% | finding #502 |
| **This work** — Gemma 4 hidden state + disjoint + hard-neg | full | **83.9%** | results.json |
| Gemma 4 hidden state @ smoke (N_TRAIN=25) | smoke | 76.5% | prior run (now archived to PAPER §Appendix) |

**Hidden-state features do NOT close the 84.2% TF-IDF gap.** Gemma 4 E2B
mean-pooled last-hidden-state is essentially equivalent to TF-IDF for ridge
routing at N=25 with hard negatives (83.9 vs 84.2 within 0.3pp — noise
floor). F#310's +1.7pp hidden-vs-TF-IDF advantage at N=5 does not extrapolate
to N=25.

## Relation to MATH.md theorems
**Theorem 1 (ridge existence/optimality):** holds unconditionally; closed-form fit
converged in 0.12s. Not falsified.

**Theorem 2 (ridge acc upper-bounded by `1 − sup s_ij − O(1/√n)`):** holds
structurally; the alias pair `medical` ⇔ `clinical_knowledge` has `s_ij = 1` by
construction (identical MMLU subject), pushing the per-pair upper bound to 0 and
contributing a ~4pp drag on overall accuracy. Not contradicted.

**Theorem 3 (Gemma 4 separates N=25):** **falsified**. Predicted interval
[86%, 98%] does not contain measurement 83.9%. Extrapolation via F#502's
N=25/N=5 scaling ratio and F#310's +1.7pp hidden-state gain was optimistic —
F#310 was measured at N=5 where s_ij slack is abundant; at N=25 with hard
negatives, hidden-state and lexical features carry equivalent discriminative
information.

## Dominant failure modes (ranked)
1. **Alias pair `medical` ⇔ `clinical_knowledge`** (identical MMLU data, two
   labels): 25.0% and 27.5% — consistent with 50% Bayes-optimal, degraded by
   α-regularized centroids not landing at the symmetric split. Contributes ~4pp
   accuracy drag (2/25 classes × 50% below perfect).
2. **`college_mathematics` vs `math`/`abstract_algebra`** (hard-negative trio):
   56.7% for college_mathematics. Ridge confuses math domains that share
   graduate-level vocabulary. Expected per MATH.md §"Theorem 2 s_ij bound".
3. **`code` / `math` / `jurisprudence`** in the 73-79% band: partial confusion
   with their hard-neg counterparts (ML, college_mathematics, legal). Each
   contributes ≤1pp to aggregate drag.

Aggregate: removing the alias pair lifts the accuracy ceiling from 83.9% to an
estimated ~88% (23 non-alias classes at their measured mean 0.88 + 2 alias at
0.265). Even with alias removed, **K1616 (90%) is not reachable under this
methodology.** The finding that Gemma 4 hidden states do not outperform TF-IDF
at N=25 is robust to the alias choice.

## Antipattern self-check
- `ap-017` stub-cascade: N/A — no LoRA adapter loaded.
- `ap-020` cascade-dependent: N/A — `depends_on: []`.
- `ap-tautological-routing`: CLOSED — disjoint train/test per subject
  (seed-partitioned index), hard-negative pairs, real Gemma 4 forward passes.
  F#458's tautology (synthetic centroids, no disjoint test) is absent.
- `ap-no-knowledge-gap`: N/A — routing experiment, not LoRA-training-on-hard-eval.
- `ap-smoke-reported-as-full`: CLOSED — `is_smoke=false` on the final run.
  Smoke run is archived in §Appendix (historical only).
- `ap-hardcoded-pass`: CLOSED — K1616 gated on measured accuracy.
- `ap-composition-bug`: N/A — no adapter composition.

## Patch summary (for reviewer)
Three minimal fixes to `run_experiment.py` vs the pre-existing file:
1. `load_mmlu_split`: replaced `datasets.load_dataset("cais/mmlu", subject,
   split="test")` (fails on Python 3.14 via dill `Pickler._batch_setitems`
   signature mismatch) with `hf_hub_download` + `pd.read_parquet` against
   `cais/mmlu/all/test-00000-of-00001.parquet`, filter by `subject`.
   Cached once per process.
2. `extract_features`: added `.astype(mx.float32)` before `mx.eval(pooled)`.
   mlx-lm Gemma 4 E2B 4-bit emits bf16 hidden states; numpy does not support
   bf16 via `__array__`. Fixes
   `RuntimeError: Item size 2 for PEP 3118 buffer format string B ...`.
3. Default `IS_SMOKE=0` (was `1`). Script now runs at MATH.md's declared
   full scale (100/40) by default.

**No KC edited. No MATH.md edited. Git diff is scoped to loader + dtype cast + default.**

## Assumptions (logged, per researcher.md)
- **A1:** MMLU "all" parquet contains all 25 subjects used here; confirmed by
  sanity-check (all 25 subjects loaded; no `failed_domains`).
- **A2:** The alias `medical` ⇔ `clinical_knowledge` is intentional per MATH.md §A1
  and `run_experiment.py` inline comment (matches F#502 methodology for
  comparability). Removing the alias would raise the ceiling but deviate from F#502.
- **A3:** F#310 was measured on different base (Qwen? or Gemma 3?) and a
  different N=5 domain set; extrapolating its +1.7pp to Gemma 4 E2B at N=25
  is what the experiment tested and falsified.
- **A4:** Some MMLU subjects have fewer than 140 rows, script falls back to
  70/30 split. Final per-domain yields 30-40 test samples; all 25 domains
  represented. Not a methodology break — disjointness still holds.

## Unblocks / open threads
- **Salvageable finding candidate:** "Gemma 4 E2B mean-pooled last-hidden-state
  features and TF-IDF are statistically tied for N=25 MMLU-subject ridge routing
  with hard negatives (83.9% vs 84.2%)." Distinct from F#310 (N=5), F#458
  (tautological), F#502 (TF-IDF baseline). Useful as a negative-result closure:
  future experiments proposing "hidden states beat lexical features" at N≥25
  need to justify the mechanism that this experiment did not find.
- **v2 design notes (not in scope here):** a variant that (a) drops the alias,
  (b) uses per-token (not mean-pool) features, or (c) trains the classifier on
  adapter-output features (not base hidden states) could close the 6.1pp gap.
  None of these are covered by K1616 as pre-registered.

## §Appendix — smoke run (historical)
Prior smoke run at IS_SMOKE=1 (N_TRAIN=25, N_TEST=15 → 625 train / 375 test):
- test acc = 0.7653
- worst-domain = 0.2667 (medical / clinical_knowledge)
- same α=100, same failure pattern
- is_smoke=true → would have been provisional per #1009

Full-scale run adds +7.3pp over smoke, consistent with 4× data improvement.
