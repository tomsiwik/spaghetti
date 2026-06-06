# PAPER.md — exp_hedgehog_behavior_adapter_formality

## Verdict
PROVISIONAL (design-lock; novel KC design)

## Problem
Hedgehog cos-sim distillation has been validated on 1 behavior axis (politeness,
F#683 PROVISIONAL) and 5 domain axes (F#684/696/697/717/718, axis-extension
super-saturation closed). The behavior-axis sub-cluster is critically under-
represented at 1 instance. Does the framework generalize cleanly to a 2nd,
distinct behavior axis — formality — with style/substance orthogonality?

## Pre-registered KCs
| KC | Quantity | KILL condition (KILL if TRUE) | Type |
|---|---|---|---|
| K1879 | Δ formality-judge(adapter, base) over 50 held-out neutral prompts | Δ < +10 pp | target — behavioral acquisition |
| K1880 | \|Δ accuracy(adapter, base)\| over 100-question MMLU subset (seed=42) | > 2 pp (two-sided) | target — style/substance orthogonality (NEW KC class) |

## Prediction → measurement table
| KC | Predicted | Predicted PASS/FAIL | Measured | Outcome |
|---|---|---|---|---|
| K1879 | Δ ∈ [+5, +18] pp; mean +12 pp | PASS (Δ ≥ +10 pp) | UNTESTED | UNTESTED |
| K1880 | \|Δ\| ∈ [0, 4] pp; mean 1.8 pp degradation | PASS borderline (\|Δ\| ≤ 2 pp) | UNTESTED | UNTESTED |

## What the experiment does
PROVISIONAL design-lock at the researcher hat. Locks: (a) the dual-target /
zero-proxy KC design (FIRST in the Hedgehog-framework super-family), (b)
formal-register teacher prompt π_Formal (single canonical), (c) MMLU-100
subset at seed=42 for K1880, (d) F#666 verdict matrix, (e) the Phase
0/A/B/C/D/E protocol. The runtime scaffold validates the structure
(`results.json` writes correctly, KCs map cleanly, blocker accounting works);
training/eval loops are explicit `NotImplementedError` stubs deferred to
`_impl`. Total runtime ~1.6 s.

## KC-design taxonomy in Hedgehog-framework super-family
| Design pattern | Instances |
|---|---|
| target+proxy paired (canonical) | F#683 (K1 cos-sim + K2/K3a/K3b/K4 targets), F#684, F#696, F#697, F#717, F#718, F#719 (KL), F#720 (MSE), F#721 (layer-sel), F#722 (temperature) |
| target+proxy intra-stability | F#723 (cos-sim variance + behavioral target) |
| **dual-target / zero-proxy** | **THIS (1st in super-family)** |
| pure-proxy (preempt-KILL pattern) | not present in super-family — only in cousin §5 / F#666-pure preempt-KILL bucket |

K1880 is a NEW KC class — *style/substance orthogonality* — absent from every
prior Hedgehog-framework PROVISIONAL. K1879 is target-axis-acquisition (cousin
of F#683 K2 politeness-judge). Together they form a zero-proxy paired-target
design.

## Hedgehog-framework PROVISIONAL pile (post-this filing)
9 designs / 0 measurements.

| # | Finding | Sub-type |
|---|---|---|
| 1 | F#683 | axis-extension (behavior — politeness, 1st) |
| 2 | F#684 | axis-extension (domain — refactoring) |
| 3 | F#696 | axis-extension (domain — JS) |
| 4 | F#697 | axis-extension (domain — Python) |
| 5 | F#717 | axis-extension (domain — Rust) |
| 6 | F#718 | axis-extension (domain — SQL) |
| 7 | F#719 | loss-variant-ablation (KL) |
| 8 | F#723 | data-augmentation-ablation (5× rephrase) |
| 9 | **THIS** | **axis-extension (behavior — formality, 2nd)** |

26B teacher cache remains the standalone-prereq-task candidate blocking 9+
dependents.

## Hard-defer transitive blockers
- This adapter's `_impl` does NOT depend on F#683 `_impl` (different axis →
  different neutral prompt set, different teacher capture).
- Phase A teacher capture pipeline pattern is reusable from F#683 (saves
  engineering, not blocked on F#683 results).
- 26B teacher residency is the shared blocker (single standalone-prereq task
  unblocks 9+ dependents).

## Sibling-position table
| Axis | Sub-cluster | Finding | Verdict | Notes |
|---|---|---|---|---|
| Politeness | behavior | F#683 | PROVISIONAL (design-lock) | mature rubric; reusable Phase A |
| **Formality** | **behavior** | **THIS / F#724** | **PROVISIONAL (design-lock)** | **2nd behavior-axis; 1st zero-proxy KC design** |
| Refactoring | domain | F#684 | PROVISIONAL | code-domain |
| JS | domain | F#696 | PROVISIONAL | language-domain |
| Python | domain | F#697 | PROVISIONAL | language-domain |
| Rust | domain | F#717 | PROVISIONAL | language-domain |
| SQL | domain | F#718 | PROVISIONAL | language-domain (closed axis-extension) |

## Assumptions A1-A12 (per MATH.md §8)
- A1: behavior-axis sub-cluster opening is justified (was 1 instance vs 5
  domain instances; formality ⊥ politeness on canonical register theory).
- A2: dual-target / zero-proxy is novel KC design in super-family.
- A3: π_Formal is single canonical formal-register prompt.
- A4: K1880 power asymmetric — n=100 MMLU detects |Δ| > 2 pp confidently
  but cannot rule out true drift in [0, 5] pp.
- A5: full pipeline ~ 8–10 h ⇒ PROVISIONAL with `_impl` follow-up.
- A6: LORA_SCALE = 6.0 ≤ 8 (F#328/F#330).
- A7: K1879/K1880 only — no retro-attached KCs.
- A8: F#702 hygiene-patch APPLICABLE (not F#666-pure).
- A9: behavior-axis sub-cluster promotion to standalone memory at 3rd
  instance (this is 2nd).
- A10: 9th Hedgehog-framework PROVISIONAL (8 → 9).
- A11: NOT a transitive dependent on F#683 `_impl`.
- A12: KC-design bifurcation rule extends to dual-target → PROVISIONAL.

## Unblock path
1. Land any Hedgehog `_impl` (F#683 `_impl` is closest to ready; would
   establish the Phase A/B/C training pipeline pattern reusable here).
2. Curate 250 neutral-formality prompts (Phase 0) — small task, parallel-
   izable to teacher-cache build.
3. 26B teacher cache build (standalone prereq blocking 9+ Hedgehog
   dependents).
4. Run Phase A/B/C/D in series via this scaffold.
