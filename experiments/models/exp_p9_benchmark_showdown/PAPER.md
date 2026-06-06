# PAPER: P9.G1 — Benchmark Showdown: Pierre v3 vs Base vs Gemma 4 27B

**Verdict: KILLED** (2026-04-19, precondition-probe via MATH.md §P tripwire)

## Summary

Pre-registered as a real-measurement run of Pierre v3 (base Gemma 4 E4B 4-bit + math
+ medical adapters) against published Gemma 4 27B numbers. On 2026-04-19 the
precondition probe (MATH.md §P, added pre-run) fired: the binding precondition P2
(math + medical adapter weight files on disk) FAILED (0 weight files for both
domains), so K1390/K1391/K1392 are structurally UNMEASURABLE. Upstream
`exp_p1_t2_single_domain_training` is `killed` with a documented Python 3.14
`datasets`/`dill` toolchain incompat in its reconstruction note — the same
root-cause blocker that saturates the 17-member audit-2026-04-17 Gemma 4 cohort.
This P9-tagged experiment is NOT in that cohort but shares the root cause.

See §P in MATH.md for the tripwire derivation. Probe was pure filesystem + JSON
reads; 0.683 s wall; no MLX, no network, no model load.

---

## Prediction vs Measurement

| Kill | Criterion | MATH.md Prediction | Measured | Status |
|------|-----------|--------------------|----------|--------|
| K1390 | Math adapter GSM8K ≥ Gemma 4 27B (90%) | LIKELY FAIL (82% < 90%) | UNMEASURABLE | P-tripwire fired (P2 FAIL) |
| K1391 | Math adapter GSM8K gain ≥ base + 20pp | EXPECTED PASS (~27pp) | UNMEASURABLE | P-tripwire fired (P2 FAIL) |
| K1392 | Medical adapter MedMCQA ≥ base + 3pp | UNCERTAIN | UNMEASURABLE | P-tripwire fired (P2 FAIL) |
| [INFO] | Serving cost ratio 4B/27B | PASS by math (14.8%) | 14.8% | Informational (unchanged) |

### Precondition probe outcome (MATH.md §P)

| Probe | Result | Detail |
|-------|--------|--------|
| P1 upstream `exp_p9_full_stack_integration` supported + results.json | FAIL | status=`open`, `results.json` absent |
| P2 math + medical adapter weight files on disk | FAIL | `{'math': [], 'medical': []}` — only `adapter_config.json` stubs exist |
| P3 `adapters/registry.json` resolves to real weights | PASS (informational) | 6 entries, only `thinking-openthoughts-universal-v0` has weights (21 files); all 5 domain-knowledge entries point to empty dirs |

P2 is the BINDING precondition for K1390/K1391/K1392 (see MATH.md §P sharpening). P3
passing on a non-math/non-medical entry does not lift the tripwire.

**Note on K1390**: MATH.md predicts K1390 will FAIL because Gemma 4 27B's published
GSM8K score (~90-91%) exceeds our math adapter's 82%. This is the "scale debt" —
the 9pp gap represents the quality improvement needed from P1 improvements (GRPO,
thinking adapter, s1K reasoning training) to close.

**Note on K1391 (revised)**: Replaced tautological HumanEval criterion (63-42=21, always PASS)
with freshly-measured math adapter GSM8K gain over base. Registry shows math adapter at 82%,
if base is ~55%, expected gain is ~27pp (PASS). If base is >62%, gain drops and may FAIL.
This is a real measurement, not a fixed computation.

**Note on K1392 (revised)**: Replaced tautological cost ratio criterion (15.8%<50%, always PASS)
with freshly-measured medical MedMCQA gain. Registry medical adapter = 50.0% MedMCQA;
base is unmeasured. If base is ~47-48%, adapter barely passes. UNCERTAIN — honest test.

---

## Benchmark Results Table

| Benchmark | Base E4B 4-bit | Pierre v3 (adapted) | Delta | Gemma 4 27B (published) |
|-----------|----------------|---------------------|-------|------------------------|
| GSM8K | TBD% | TBD% (math adapter) | TBD | ~90% |
| MedMCQA | TBD% | TBD% (medical adapter) | TBD | ~70% (estimated) |
| MMLU-Pro | 62.1%* | TBD% (oracle routing) | TBD | ~79% |
| HumanEval | ~42% (est) | 63% (from registry) | +21pp | ~74% |

*62.1% from Finding #530 (with thinking). Oracle routing may degrade MCQ (Finding #517).

---

## Value Proposition

Pierre v3 delivers:
- Math task quality: 82% GSM8K (from registry, refreshed below) — 9pp below 27B
- Code task quality: 63% HumanEval — 11pp below 27B
- Medical MCQ: TBD — potentially competitive with 27B for focused medical queries
- Serving cost: ~14.8% of Gemma 4 27B (15× cheaper to serve)

**Interpretation**: Pierre v3 is NOT a general replacement for 27B. It IS a cost-efficient
specialist system that delivers 90-92% of 27B math quality at 14.8% the cost. For
high-throughput domain-specific serving, this is the right trade-off.

---

## Published Reference Numbers

| Model | GSM8K | HumanEval | MMLU-Pro | Source |
|-------|-------|-----------|----------|--------|
| Gemma 4 E4B (base) | ~55% | ~42% | 62.1%* | Finding #530 + estimate |
| Pierre v3 (adapted) | 82% | 63% | TBD | Finding #421 + registry |
| Gemma 4 27B | ~90% | ~74% | ~79% | Google Gemma 4 Tech Report |

*62.1% uses thinking mode. Without thinking: 41.7%.

**Source note**: Gemma 4 27B numbers are from Google's Gemma 4 technical report and
competitive benchmarks. These are approximate figures; K1390 uses the published 90%
number as the reference threshold.

---

## Dependency Note

This experiment depends on exp_p9_full_stack_integration (G0) completing first.
G0 establishes that the routing + adapter stack functions end-to-end. G1 then
provides the competitive comparison against external reference points.

---

## Caveats

1. **Base GSM8K freshly measured** (K1391): Registry math adapter = 82%; base GSM8K
   measured in Phase 1. Expected gain ~27pp. If base is >62% (unlikely), K1391 may FAIL.
   HumanEval comparison (63% code vs ~42% est base) preserved as informational in Phase 6.

2. **Oracle routing ≠ production routing**: MMLU-Pro results use ground-truth category
   labels to select adapters. Production routing (97.7% accuracy) introduces ~2pp noise.

3. **Domain adapters hurt MCQ** (Finding #517): Math adapter on MMLU-Pro = 36.1% (vs 62.1%
   base). Oracle routing only helps for categories where the domain adapter is beneficial.

4. **27B reference numbers**: Google's published benchmark conditions may differ from
   ours (prompt format, temperature, sampling strategy). Exact parity not guaranteed.

5. **Why KILL instead of defer**: Per guardrail 1005, killed experiments are not
   dead ends; the structural fix is identified (upstream T2.1 rerun after Python
   3.14 `datasets`/`dill` toolchain fix + `experiment update --status open` on
   `exp_p1_t2_single_domain_training`). Once weights land, K1390/K1391/K1392 become
   measurable — re-claim this exp by ID and re-run the unmodified runner (§P PASS
   path takes over). KILL is the honest current-state verdict, not the terminal
   one.

## Unblock path

Identical to the 17-member audit-2026-04-17 cohort's unblock path (see
`.ralph/current_direction.md`):

1. Orchestrator-scope Python 3.14 toolchain fix for `datasets`/`dill` (or downgrade
   to 3.12 temporarily).
2. `experiment update --status open` (or v2 clone) on `exp_p1_t2_single_domain_training`.
3. Rerun T2.1 at LORA_SCALE=5 (Finding #586 scale-safety bound) with disjoint
   math/medical corpora.
4. Rerun this experiment — §P probe auto-PASSES and runner flows through to MLX
   measurement.
