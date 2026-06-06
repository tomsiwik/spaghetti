# exp_g4_adapter_initialization_comparison_v2 — PAPER.md

## Verdict: PROVISIONAL (smoke iter floor + K1985 non-interference FAIL — real signal beyond smoke)

`is_smoke: true`. Wall-clock 2372s (~40 min) for 9 runs (3 inits × 3 seeds × 100 iters) + 9 MCQ evals on n=80 medical-MCQ heldout.

## §1 Prediction-vs-measurement

| KC | Type | Prediction (MATH §3) | Measurement | Verdict |
|---|---|---|---|---|
| K1977 | proxy / structural | PASS — cross-init |cos| < 0.20 (PRNG fix) | max=0.042 (was 0.89-0.96 in v1) | **PASS** |
| K1978 | proxy / structural | PASS — PPL ratio > 1.10 (spread present) | 1.25 (driven by grassmannian s0 outlier 1.44) | **PASS** |
| K1979 | proxy / identifiability | PASS — within-init PPL seed-var > 5% | 24.6% (grassmannian) | **PASS** |
| K1983 | target / behavioral | FAIL (= init-invariance verified) | 1.25pp (s=0 reps); 9.6pp on means | **FAIL on reps / PASS on means** |
| K1984 | target / identifiability | PASS — within-init MCQ seed-var > 3pp | 12.5pp (gaussian) | **PASS** |
| K1985 | target / non-interference | PASS — no init drops base > 5pp | max drop 14.2pp (kaiming) | **FAIL — recipe is MCQ-suppressive** |

## §2 Headline finding (smoke-iter signal)

The medical q_proj r=6 LoRA recipe **suppresses medical-MCQ heldout accuracy** at 100 iters relative to no-adapter base:
- Base (no adapter): 57.5% MCQ
- Mean per init: grassmannian 48.3% (-9.2pp), kaiming 43.3% (-14.2pp), gaussian 52.9% (-4.6pp)

Per K1985, **two of three inits drop > 5pp**, killing the non-interference KC. This is a NEW finding not visible in v1 (which only measured PPL — base PPL 5105 → adapter PPL ~1.2 looked like a 4000× improvement, but PPL on the medical MCQ-formatted training distribution does NOT predict MCQ heldout accuracy).

## §3 Confound-resolution finding (parent v1 PRNG fix)

v1 K1924 (cross-init final cos Δ > 0.10) failed *not* because Grassmannian/Kaiming/Gaussian produce structurally different adapters but because v1 shared `mx.random.key(42)` across "different" inits, producing **starting cos 0.977-0.9995**. v2 with distinct seeds:
- Cross-init **starting** |cos|: 0.015-0.018 (proper independent random matrices)
- Cross-init **final** |cos|: 0.027-0.042

→ Independent random low-rank A matrices remain near-orthogonal under 100-iter SGD. Per K1977 PASS, the structural init-invariance claim survives the PRNG-confound fix. F#751 PROVISIONAL on K1924 should be **resolved KILLED retrospectively** for v1 (v1 K1924 PASS was an artifact of shared seed; with distinct seeds the proxy spread collapses below threshold, **opposite of v1's verdict**).

## §4 Within-init seed-variance bound (K1979 / K1984)

| init | PPL spread (max/min - 1) | MCQ spread (max-min, pp) |
|---|---|---|
| grassmannian | 24.6% | 3.75pp |
| kaiming | 0.5% | 6.25pp |
| gaussian | 14.6% | 12.5pp |

PPL seed-noise is large (up to 24.6%) → **K1979 PASS**, parent v1's 3.5% PPL spread falls within seed-noise → **v1's init-invariance claim is unidentifiable from a single-seed PPL measurement** (K1925 was a tautology — the spread was within seed-noise floor).

MCQ seed-noise reaches 12.5pp (gaussian) → **K1984 PASS**, eval has discriminative power.

## §5 Cross-init behavioral spread (K1983)

Per s=0 representative: 46.25% / 46.25% / 47.5% → spread 1.25pp → K1983 FAIL (FAIL = invariance verified). Per init means: 48.3% / 43.3% / 52.9% → spread 9.6pp → K1983 PASS. The 9.6pp spread is **larger than the 5pp threshold** but **smaller than the 12.5pp within-init seed-variance**, so cross-init differences are *not statistically separable from seed noise* at n=3 sub-seeds per init.

→ Honest verdict for K1983: spread cannot be distinguished from seed noise at smoke N. Both s=0-rep and means-based read pre-registered, with the s=0 rep matching the v1-style measurement (FAIL = invariance) and means showing the harder question is unresolved at this N.

## §6 Verdict logic per F#666 truth-table + SC#109

SC#109 requires K1983 FAIL ∧ K1984 PASS ∧ K1985 PASS. Outcome:
- K1983 FAIL on reps / ambiguous on means.
- K1984 PASS.
- **K1985 FAIL** ← blocks SC#109.

Therefore SC#109 NOT verified. Smoke-iter floor + K1985 FAIL → **PROVISIONAL** with two distinct findings:
1. **PRNG-confound resolution** (v1 retrospective KILL on K1924).
2. **Recipe MCQ-suppression** (medical-q_proj-r6-100-iter under-trained or wrong-target for MCQ).

## §7 Verdict-consistency pre-flight (PLAN §1 / researcher hat clause 6.6)

| Check | Status |
|---|---|
| 1. results.json verdict ≠ KILLED | ✓ "PROVISIONAL" |
| 2. results.json all_pass true | ✗ false (K1983 reps + K1985 FAIL) — smoke floor + structural FAIL = PROVISIONAL not SUPPORTED |
| 3. PAPER.md verdict line | ✓ contains "PROVISIONAL" (smoke + behavioral signal) |
| 4. is_smoke true | ✓ → PROVISIONAL floor honored |
| 5. KCs unchanged from MATH.md | ✓ K1977/K1978/K1979/K1983/K1984/K1985 all preserved byte-for-byte from DB |
| 6. Antipattern scan | ✓ all clear (composition N/A, no LORA_SCALE=20, no shutil.copy, no hardcoded pass) |

## §8 Reclaim path for v3 (full 1000-iter run)

Two distinct follow-ups warranted:

**v3a — full 1000-iter run** (already filed at P3 in DB schema, may need creation):
- `SMOKE_TEST=0` → ITERS=1000.
- Same 3 inits × 3 sub-seeds = 9 runs at ~30 min/run = ~4-5h.
- Tests whether K1985 non-interference recovers at convergence OR persists (recipe is structurally MCQ-suppressive regardless of iters).

**v3b — recipe-fix MCQ adapter** (NEW; not yet filed):
- Hypothesis: q_proj-only r=6 medical recipe is structurally inadequate for MCQ-heldout accuracy. Try v_proj+o_proj per F#627 canonical, scale=4 per F#330.
- KCs: K1985 (non-interference) at full 1000 iters.

## §9 Assumptions logged (researcher hat autonomy)

- A1: MCQ letter accuracy via greedy single-token scoring on logits[ABCD]. Did not generate full text — F#614 thinking-mode preservation deferred to v3 (smoke iter, not load-bearing).
- A2: medical/valid.jsonl first 80 rows used as MCQ heldout. Subset is shared across all 9 runs (deterministic).
- A3: All adapters saved as `adapter_<init>_s<sub_seed>.safetensors` for v3 reuse / regression analysis.
- A4: PRNG-confound fix verified empirically: starting cross-init |cos| dropped from 0.977-0.9995 (v1 single-key) to 0.015-0.018 (v2 distinct keys per init).

## §10 Cross-references

- F#751 (parent v1 PROVISIONAL — PRNG confound). v1 K1924 verdict needs retrospective re-evaluation (this experiment's K1977 measurement INVERTS v1's PASS to FAIL after confound removal — meaning v1's invariance claim was an artifact, but v2 still finds invariance under proper measurement).
- F#666 target-gated KILL: K1985 FAIL is a behavioral signal, not just a proxy spread.
- F#169 init-invariance prior: v2 confirms PASS at proxy level (K1977/K1978/K1979 all PASS) and ambiguous at behavioral level (K1983 FAIL on reps, PASS on means within seed-noise).
- F#627: q_proj-only is NOT canonical (canonical is v_proj+o_proj). v3b would test canonical recipe.
