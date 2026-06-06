# PAPER.md — exp_hedgehog_behavior_adapter_conciseness

## Verdict
PROVISIONAL (design-lock; zero-proxy KC design, 1st one-sided-safety sub-variant;
behavior-axis sub-cluster promotion triggered)

## Problem
Hedgehog cos-sim distillation has been design-locked on 2 behavior axes
(politeness F#683, formality F#724 — both PROVISIONAL) and 5 domain axes
(F#684/696/697/717/718 — axis-extension super-saturation closed). Does the
framework generalize cleanly to a 3rd, distinct behavior axis — conciseness —
with asymmetric substance safety? This filing is the **3rd behavior-axis
instance** and triggers the behavior-axis sub-cluster standalone-memory
promotion per the 3-instance-on-same-sub-cluster threshold.

## Pre-registered KCs
| KC | Quantity | KILL condition (KILL if TRUE) | Type |
|---|---|---|---|
| K1881 | `ρ = 1 − mean_tokens(adapter) / mean_tokens(base)` over 50 held-out neutral prompts, max_tokens=256 | ρ < 0.20 | target — behavioral acquisition |
| K1882 | `accuracy(base) − accuracy(adapter)` over 100-question MMLU subset (seed=42), max_tokens=64 | > 3 pp (one-sided — degradation only) | target — substance safety (one-sided safety sub-variant of F#724 K1880 two-sided orthogonality) |

## Prediction → measurement table
| KC | Predicted | Predicted PASS/FAIL | Measured | Outcome |
|---|---|---|---|---|
| K1881 | ρ ∈ [+10 %, +40 %]; mean +25 % | PASS (ρ ≥ 0.20) | UNTESTED | UNTESTED |
| K1882 | drop ∈ [−1, +6] pp; mean +2.5 pp | PASS borderline (drop ≤ 3 pp) | UNTESTED | UNTESTED |

## What the experiment does
PROVISIONAL design-lock at the researcher hat. Locks: (a) the dual-target /
zero-proxy KC design with **one-sided-safety sub-variant** (2nd in the
Hedgehog-framework super-family; 1st one-sided-safety), (b) concise-output
teacher prompt π_Concise (single canonical), (c) MMLU-100 subset at seed=42
for K1882, (d) length-eval max_tokens=256 + accuracy-eval max_tokens=64
(locked separately), (e) F#666 verdict matrix, (f) the Phase 0/A/B/C/D/E
protocol. The runtime scaffold validates the structure (`results.json`
writes correctly, KCs map cleanly, blocker accounting works, sub-cluster
promotion flag exposed); training/eval loops are explicit
`NotImplementedError` stubs deferred to `_impl`. Total runtime 1.8 s.

## KC-design taxonomy in Hedgehog-framework super-family
| Design pattern | Instances |
|---|---|
| target+proxy paired (canonical) | F#683 (K1 cos-sim + K2/K3a/K3b/K4 targets), F#684, F#696, F#697, F#717, F#718, F#719 (KL), F#720 (MSE), F#721 (layer-sel), F#722 (temperature) |
| target+proxy intra-stability | F#723 (cos-sim variance + behavioral target) |
| dual-target / zero-proxy, two-sided orthogonality | F#724 (1st zero-proxy) |
| **dual-target / zero-proxy, one-sided safety** | **THIS (2nd zero-proxy; 1st one-sided-safety sub-variant)** |
| pure-proxy (preempt-KILL pattern) | not present in super-family — only in cousin §5 / F#666-pure preempt-KILL bucket |

K1882 is a structural variant of F#724 K1880 — both are safety targets, but
K1880 was two-sided (orthogonality) while K1882 is one-sided (degradation-
only). This extends the zero-proxy design to 2 safety-target sub-variants.

## Hedgehog-framework PROVISIONAL pile (post-this filing)
10 designs / 0 measurements.

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
| 9 | F#724 | axis-extension (behavior — formality, 2nd) |
| 10 | **THIS** | **axis-extension (behavior — conciseness, 3rd) — SUB-CLUSTER PROMOTION** |

26B teacher cache remains the standalone-prereq-task candidate blocking 10+
dependents.

## Behavior-axis sub-cluster — 3-instance promotion milestone
| # | Axis | Finding | Filed | KC-design |
|---|---|---|---|---|
| 1 | Politeness | F#683 | 2026-04-22 | target+proxy paired |
| 2 | Formality | F#724 | 2026-04-24 | zero-proxy, two-sided orthogonality |
| 3 | **Conciseness** | **THIS** | 2026-04-24 | **zero-proxy, one-sided safety** |

Analyst-hat downstream action: write `mem-promotion-hedgehog-behavior-axis-
sub-cluster` memory documenting the 3-axis span (politeness / formality /
conciseness) with 2 distinct KC designs represented (target+proxy for
politeness; zero-proxy for formality + conciseness with two sub-variants).

## Hard-defer transitive blockers
- This adapter's `_impl` does NOT depend on F#683 or F#724 `_impl` (different
  axis → different neutral prompt set; conciseness neutrality is length-
  focused, not register-focused).
- Phase A teacher capture pipeline pattern is reusable from F#683/F#724
  (saves engineering, not blocked on their results).
- 26B teacher residency is the shared blocker (single standalone-prereq task
  unblocks 10+ dependents).

## Sibling-position table
| Axis | Sub-cluster | Finding | Verdict | KC-design |
|---|---|---|---|---|
| Politeness | behavior | F#683 | PROVISIONAL (design-lock) | target+proxy paired |
| Formality | behavior | F#724 | PROVISIONAL (design-lock) | zero-proxy, two-sided orthogonality |
| **Conciseness** | **behavior** | **THIS** | **PROVISIONAL (design-lock) — SUB-CLUSTER PROMOTION** | **zero-proxy, one-sided safety** |
| Refactoring | domain | F#684 | PROVISIONAL | target+proxy paired |
| JS | domain | F#696 | PROVISIONAL | target+proxy paired |
| Python | domain | F#697 | PROVISIONAL | target+proxy paired |
| Rust | domain | F#717 | PROVISIONAL | target+proxy paired |
| SQL | domain | F#718 | PROVISIONAL | target+proxy paired (closed axis-extension) |

## Assumptions A1-A12 (per MATH.md §8)
- A1: behavior-axis sub-cluster at 2 → **3 instances with this** (promotion
  threshold MET); conciseness ⊥ politeness, ⊥ formality on output-structure
  axis (one can be formal-verbose, concise-impolite, etc.).
- A2: 2nd zero-proxy KC design in super-family; 1st one-sided-safety sub-
  variant.
- A3: π_Concise is single canonical concise-output prompt.
- A4: K1882 power — n=100 MMLU one-sided test detects 3 pp drop confidently;
  `_impl` may scale to n=300 if borderline.
- A5: full pipeline ~ 8–10 h ⇒ PROVISIONAL with `_impl` follow-up.
- A6: LORA_SCALE = 6.0 ≤ 8 (F#328/F#330).
- A7: K1881/K1882 only — no retro-attached KCs (cross-axis interference or
  information-density audit deferred).
- A8: F#702 hygiene-patch APPLICABLE (not F#666-pure). Hygiene applied:
  platform=local-apple, dir set, success_criteria #97 added.
- A9: behavior-axis sub-cluster promotion — 3rd instance; analyst-hat owns
  standalone-memory write.
- A10: 10th Hedgehog-framework PROVISIONAL (9 → 10).
- A11: NOT a transitive dependent on F#683 or F#724 `_impl`.
- A12: KC-design bifurcation rule extends to two-sided and one-sided safety
  zero-proxy sub-variants — both PROVISIONAL.

## Unblock path
1. Land any Hedgehog `_impl` (F#683 `_impl` is closest to ready; would
   establish the Phase A/B/C training pipeline pattern reusable here).
2. Curate 250 length-neutral prompts (Phase 0) — small task, parallel-
   izable to teacher-cache build.
3. 26B teacher cache build (standalone prereq blocking 10+ Hedgehog
   dependents).
4. Run Phase A/B/C/D in series via this scaffold.
