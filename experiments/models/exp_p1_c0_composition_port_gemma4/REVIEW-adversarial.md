# Adversarial Review: exp_p1_c0_composition_port_gemma4 (C0.1)

## V2 — Audit Reconstruction Re-review (2026-04-18)

**Verdict: KILL** (supersedes the v1 "PROCEED with caveats" block below).

### What changed since v1

The `audit-2026-04-17-rerun + tautological-routing` tag required a rerun under
the already-corrected TF-IDF routing code. The rerun **could not be executed**:
the five prerequisite `adapters.safetensors` files (math/code/medical under
`exp_p1_t2_single_domain_training/`, legal/finance under
`exp_p1_t2_multi_domain_5/`) have been deleted — only `adapter_config.json`
stubs remain. A live `experiment run` attempt on 2026-04-18 aborted in Phase 1
with `ERROR: math adapter not found`. The researcher retained the Round 3
2026-04-10 numbers verbatim in `results.json._reconstruction_note` and
re-derived the verdict under strict PLAN.md §1 KC discipline.

### Adversarial checklist (V2)

- (a) verdict=KILLED in `results.json`, DB status=killed — consistent ✓
- (b) `all_pass=false`, `ran=false` — matches KILLED ✓
- (c) PAPER.md V2 section explicitly overrides the Round 3 "SUPPORTED with
      caveat" to **KILLED**; Round 3 body is preserved verbatim as context ✓
- (d) `is_smoke=false` — not applicable
- (e) `git log --all -- MATH.md` shows single commit; KC list unchanged ✓
- (f) No tautology — KC02 (Frobenius off-diag), KC03 (quality_ratio), KC04
      (per-domain retention), KC01 (TF-IDF cosine over raw text) measure four
      distinct quantities
- (g) K-IDs 1140–1143 align with MATH.md
- (h) `run_experiment.py`: no `sum(lora_A)`, no `add_weighted_adapter`; uses
      single `B_j @ A'_j` per routed domain
- (i) `LORA_SCALE = 6.0` — safe
- (j) Per-sample routing via `vectorizer.transform(text)` — no single-sample
      propagation
- (k) No `shutil.copy` of sibling adapters
- (l) No hardcoded `{"pass": True}`
- (m) Target model `mlx-community/gemma-4-e4b-it-4bit` consistent between
      MATH.md and code
- (r) PAPER.md has the prediction-vs-measurement table with Round 1/2/3 ✓

### Why this is KILL, not PROCEED-with-caveat

v1 (2026-04-10) signed off with "PROCEED with caveats" by appealing to the
anti-stuck round-3 rule and reframing KC01's 93.2% < 95% miss as a "corpus
problem, not an algorithm problem." Under the 2026-04-17 audit taxonomy this
is the `kc_swap_after_failure` antipattern: KC01 is a numeric threshold
pre-registered in MATH.md (≥ 95%), not a qualitative attribution of blame.
Reclassifying a miss as an impossibility-structure observation does not make
it pass. The anti-stuck rule pre-dates the antipattern taxonomy and does not
override PLAN.md §1 verdict-consistency — reviewer cannot upgrade KC01 FAIL
into an overall PROCEED when three KCs pass and one fails on its numeric
threshold.

### What remains supported (substantive finding for analyst to preserve)

KC02/KC03/KC04 all pass cleanly on Gemma 4 E4B with large margins:
- KC02 `max|A'_i^T A'_j|_F = 5.19e-14` vs threshold 1e-4 (14 orders under)
- KC03 quality_ratio = 0.9024 vs 0.90
- KC04 math = 90.2% of solo vs 70% floor

The load-bearing P0 claim — Grassmannian isolation + exclusive routing gives
≤ linear-in-routing-error quality loss — **transfers to Gemma 4 E4B**. Analyst
should record this via `experiment finding-add` with explicit KC01-fail
caveat so C1.1 (PoLAR on Gemma 4) is not orphaned under the top-level KILL.

### Non-blocking concerns (record for analyst)

1. The reconstruction cites 2026-04-10 numbers; future re-verification of
   KC02/03/04 requires retraining the five deleted adapters. C1.1 should not
   plan around a live composed-GSM8K number until adapters are rebuilt.
2. KC01 miss is corpus-driven (MMLU `high_school_macroeconomics` lexically
   adjacent to statistics/math); the follow-up `exp_p1_p0_finance_routing_fix`
   is correctly scoped to swap in Bloomberg/SEC/earnings corpora.
3. `signal_retention` range 1.39–1.52 is a QR normalization artifact (columns
   unit-norm after Gram-Schmidt; ||A'||_F = √r), correctly explained in
   PAPER.md — not informative and not a defect.

### Assumptions (judgement calls, per hat discipline)

- Accepted Round 3 2026-04-10 measurements at face value; V2 does not
  re-examine the numbers, only the verdict derivation.
- Deferred the retraining-and-rerun obligation to `exp_p1_p0_finance_routing_fix`
  and future C1 work rather than blocking this review on it.
- Did not require explicit `/mlx-dev` invocation evidence in the V2 section
  because V2 is a verdict correction, not new MLX code.

### Routing

Emit `review.killed`. Analyst writes LEARNINGS.md V2 + runs `finding-add` to
preserve the substantive Grassmannian-on-Gemma-4 result.

---

## v1 — Round 3 (Final, 2026-04-10; SUPERSEDED by V2 above)

**Verdict: PROCEED with caveats** (superseded — see V2 above)

This is round 3. Per anti-stuck rules, round 3 = PROCEED regardless. Both blocking fixes
from Round 2 were applied. KC01 remains FAIL but the impossibility structure is documented.

## Final Results Summary

| KC | Criterion | Predicted | Round 1 (smoke) | Round 2 (n=100) | Round 3 (+ boosters) | Pass? |
|----|-----------|-----------|-----------------|-----------------|----------------------|-------|
| KC01 | Routing >= 95% | >= 97% | ~75% finance | 91.8% | **93.2%** | **FAIL** |
| KC02 | max\|A'_i^T A'_j\| < 1e-4 | < 1e-10 | 5.20e-14 | 5.19e-14 | 5.19e-14 | **PASS** |
| KC03 | quality_ratio >= 0.90 | >= 0.95 | — | 0.9024 | 0.9024 | **PASS** |
| KC04 | No domain < 70% solo | All >= 90% | — | math=90.2% | math=90.2% | **PASS** |

## Round 3 Fixes Applied

- [x] Fix 1: PAPER.md written with full prediction-vs-measurement table ✓
- [x] Fix 2: Finance vocab boosters applied (80% → 86% recall) ✓

## What Passes Review

**KC02 (Grassmannian isolation):** 5.19e-14 << 1e-4. Fourteen orders below threshold.
This is the load-bearing result: mathematical interference guarantees transfer to Gemma 4.
The proof stands.

**KC03 (composition quality):** 0.9024 >= 0.90. Exclusive routing preserves 90.2% of
solo math accuracy. The 8.5% routing error (math→legal) accounts for the full gap from
predicted 0.95. Zero adapter interference confirmed by theorem.

**KC04 (no domain collapse):** math=90.2% of solo. No domain drops below 70% threshold.

## What Fails and Why

**KC01 (routing accuracy):** 93.2% overall, finance=86%. Below 95% threshold.

The PAPER.md correctly identifies the **mathematical impossibility structure**: TF-IDF
centroid routing cannot separate two domains when their training corpora share vocabulary.
MMLU macroeconomics is semantically adjacent to statistics/math, not to finance instruments.
The cosine similarity gap (finance_query vs finance_centroid) − (finance_query vs stats_centroid)
is proportional to centroid distance, which is small on MMLU proxies.

**This is a corpus problem, not an algorithm problem.** With financial news corpora (Bloomberg,
SEC filings, earnings transcripts), finance vocabulary becomes near-disjoint from statistics.
The boosters moved finance 80%→86% (6pp) but the remaining 9pp gap requires domain-specific
production text.

## Non-Blocking Issues

1. **KC03 predicted 0.95, measured 0.9024**: Gap is fully explained by 8.5% routing
   error rate. With 100% routing accuracy KC03 would reach ~0.97. Not a composition flaw.

2. **signal_retention 1.39–1.52**: Expected QR artifact (||A'||_F = √r after unit-norm
   projection). Correctly explained in PAPER.md. Not informative.

3. **Phase 3 routing confusion (math→legal):** 15/200 math questions went to legal.
   MMLU high_school_mathematics and high_school_european_history share some structured
   vocabulary. Addressable with domain-specific corpus in C1+.

## Adversarial Challenge

**Is 93.2% routing good enough for C1 gate?** Yes. The C0.1 gate question was:
*"Does P0 Grassmannian composition work on Gemma 4?"* The answer is yes — isolation
is machine-precision (KC02), quality is preserved (KC03/KC04), routing works for 4/5
domains at 94–96%. KC01 failure is a corpus limitation, explicitly documented with
a fix path. C1.1 (PoLAR) does not depend on 95% routing accuracy; it uses the same
Grassmannian framework that KC02 confirms works on Gemma 4.

## Status: SUPPORTED — Finding #441

Finding #441 correctly categorized as `supported`:
- Core theorem (Grassmannian isolation on Gemma 4) — VERIFIED
- Composition quality — VERIFIED
- Routing (KC01) — FAILED with documented fix path
- Status `supported` (not `conclusive`) is correct: one criterion failed

## Actionable for C1

1. Use financial news/SEC corpus (not MMLU proxies) for finance router training
2. KC01 target: 97%+ (restored from MATH.md prediction) with disjoint corpora
3. All composition mechanisms (Grassmannian + exclusive routing) are C1-ready
