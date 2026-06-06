# REVIEW-adversarial.md — P11.L0: RSD Aligned Traces

**Reviewer**: Adversarial Reviewer (post-kill)
**Date**: 2026-04-18
**Verdict**: KILL (preemptive, endorses researcher determination)
**Supersedes**: 2026-04-14 PROCEED Round 2 (the cached NLL data that falsifies
Theorem 1's premise was already on disk at that review; Round 2 missed it).

---

## Adversarial Checklist

| # | Item | Status | Note |
|---|------|--------|------|
| (a) | results.json verdict vs DB status | ✅ | No results.json (preemptive); DB `status=killed`; PAPER.md verdict KILLED — internally consistent |
| (b) | all_pass vs claim | ✅ | K1541 fail in DB; status=killed |
| (c) | PAPER.md verdict line | ✅ | "**KILLED (preemptive, 2026-04-18)**" |
| (d) | smoke flag | N/A | No run; cache is from earlier R&D |
| (e) | MATH.md KC drift | ✅ | Single commit `de38e37`; no post-cache edits |
| (f) | Tautology sniff | ✅ | None — kill is by direct measurement, not algebraic identity |
| (g) | K-ID mismatch (code vs MATH/DB) | ✅ | K1541/1542/1543 align across MATH §"Kill Criteria", DB, and run_experiment.py:909-916 |
| (h) | Composition antipatterns | N/A | Single adapter |
| (i) | LORA_SCALE | ✅ | =1.0 at L63 |
| (j) | Per-sample routing | N/A | No routing |
| (k) | shutil.copy of adapter | ✅ | L596 copies MMLU-Pro parquet, not adapter weights — OK |
| (l) | Hardcoded `{"pass": True}` | ✅ | Computed at L909-912 |
| (m) | Model proxy substitution | ✅ | Same MODEL_ID in scoring, training, eval |
| (m2) | Skill invocation evidence | ⚠ | MATH/PAPER do not cite `/mlx-dev` or `/fast-mlx`; non-blocking for preemptive kill, but blocking for any L0-v2 rerun |
| (n–q) | Eval integrity | ⚠ | (q) Cited baseline drift unresolved (F#560 40.7% vs F#530 62.1%); fallback `p11f0_mmlu=60.0` at L889 matches neither — drives K1541 unreachability |
| (r) | Prediction-vs-measurement | ✅ | PAPER §3 KC table |
| (s) | Math errors | ✅ | Researcher's Theorem 1 collapse derivation is sound |

No (a)–(m) failure that would force a different verdict. Researcher's KILL is endorsed.

---

## Independent Verification of Kill Drivers

**Driver 2.1 — NLL filter is a no-op (primary):**
Read `data/trace_nll_scores.json` directly. All 20 entries have
`accepted: true`; per-token `acceptance_rate` ranges 0.876–0.982
(min trace 8 = 0.876, max trace 3 = 0.982). Mean ≈ 0.965. The threshold
`ACCEPT_RATE_MIN = 0.60` accepts 20/20. Theorem 1 Step 2 premise
`d_TV(D_rsd, π_S) < d_TV(D_raw, π_S)` is **falsified by measurement**:
with 100% retention D_rsd ≡ D_raw and the inequality is equality.

**Driver 2.2 — K1541 baseline KILLED:**
`experiment get exp_p11_s1k_reasoning_train_eval` confirms `status=killed`.
The K1541 comparison anchor does not exist; `run_experiment.py:889`
falls back to `p11f0_mmlu=60.0`, which matches neither measured base
(40.7%, F#560) nor stale cited (62.1%, F#530). Comparison is
floating.

**Driver 2.3 — Theorem 1 collapse:**
Verified algebraically. Step 2 violated → Step 3 vacuous → Step 4
inequality collapses to equality. Predicted +2 to +4 pp has zero
mechanistic basis under measured filter behavior.

**Driver 2.4 — Cascade context:**
8 prior P11 reasoning-adapter kills today (F0/H0/B0/C0/D0/H1/I0/J0)
establish that trained Gemma-4 reasoning adapters regress 15–26 pp on
MMLU-Pro. K1541 target ≥63% is ~40 pp above this band. Even with a
working filter, K1541 is structurally unreachable.

**Driver 2.5 — Format mismatch (secondary):**
`run_experiment.py:230,367` writes `<think>...</think>` SFT targets
while Gemma 4 emits `<|channel>thought ... <channel|>`. Same *category*
as antipattern-018 (SFT format ≠ native generation format), distinct
*instance* (not the channel-tokens-as-text byte sequence). Non-load-bearing
for kill but consistent with the pattern.

---

## Findings

The researcher's PAPER §4 proposes a new finding:
**"Absolute NLL thresholds are meaningless filters at large-vocab models."**

This is genuinely new and not duplicated by antipattern-018 (B0-chain SFT
format) or antipattern-017 (orphan adapter stubs, J0). It generalizes
beyond P11: any future filtering experiment using absolute log-prob
thresholds at vocab ≥ 100k inherits this risk. I endorse promoting it
to a DB finding.

---

## Open Threads for Analyst

1. **Antipattern candidate**: "Absolute log-prob threshold at large vocab"
   — distinct mechanism from existing antipatterns. Promote with
   pre-flight check: probe-set σ ≥ 0.15, mean ∈ [0.3, 0.7], reject
   threshold otherwise.
2. **F#560 baseline reconciliation** still open — blocks honest absolute
   K1 design across remaining P11 chain.
3. **Skill invocation gap** (m2): document for L0-v2 rerun;
   `/mlx-dev` + `/fast-mlx` invocation needed before any new MLX code.
4. **Round 2 PROCEED protocol lesson**: 2026-04-14 review missed the
   already-existing cache file; future reviewers should grep `data/*.json`
   for cached results before endorsing PROCEED on a "design-only" stage.

---

## Assumptions

1. The 20-sample smoke cache is representative of the full 1000-trace
   acceptance distribution. Justification: s1K is homogeneous competition
   math; Hoeffding bound on a [0,1]-valued statistic gives ≤ 0.15
   half-width at 95% from n=20.
2. The cached NLL computation is correct. Spot-checked: per-token NLL
   means 0.48–2.30 are plausible LM log-probs; mis-specification is in
   the threshold, not the computation.
3. The 60.0% fallback in `run_experiment.py:889` is a known floating
   benchmark (PAPER §2.2); reviewer treats this as load-bearing for
   K1541 unreachability, not a separate REVISE blocker.

---

## Routing

- DB already `status=killed` per researcher (`--k 1541:fail --k 1542:pass --k 1543:pass`).
- No new DB writes needed from reviewer (researcher's `--k` mapping
  is correct: K1541 fails because the substantive criterion fails;
  K1542/K1543 pass the literal threshold but are vacuous).
- Add finding for the new methodological lesson (analyst or follow-up).
- Emit `review.killed` for Analyst.
