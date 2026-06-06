# exp_followup_spectral_surgery_grassmannian — Preempt-Structural KILL

**Verdict:** KILLED — preempt-structural (F#666-pure standalone, parent-supersession, architecture-irrelevant test pool).

**Status:** No measurement performed. Closure is mathematical (MATH.md Theorems 1 & 2).

## Hypothesis

Followup to F#278/F#488 attempting to "rescue" spectral surgery by relaxing the test pool to non-Grassmannian adapter pairs ("where there is actually something to fix").

Pre-registered KC #1560: *"On non-orthogonal adapter pairs, spectral surgery reduces interference by ≥20% vs identity."*

## Predictions vs. Measurements

| # | Prediction | Measurement | Match? |
|---|---|---|---|
| P1 | KC #1560 verdict-table reaches a behavioral conclusion under any outcome | All four cells (PASS / FAIL × A' / A) yield no target signal — proxy-only KC with no behavioral pairing | NO — preempt-KILL on KC structure |
| P2 | Test on non-Grassmannian adapters A' transfers to Pierre's deployment surface A | Pierre/P1 = Grassmannian-orthogonal (PoLAR) by construction; A' ∩ A = ∅ | NO — test pool architecturally irrelevant |
| P3 | Parent F#278 / F#488 / F#64 finding is conditional on test-pool-vacuity | Parent finding is **structural**: Grassmannian orthogonality forces the SV inversion (low-SV = domain-pure). Not conditional on whether test-pool has "something to fix." | NO — parent supersession is unconditional on A |

## What this rules out

- Reusing the parent KC structure with a non-Grassmannian test pool to extract a positive surgery result for any deployment-relevant claim. The parent kill is structural, not measurement-environment-dependent.
- Single-proxy-KC pre-regs of structural manipulations on adapter delta objects. F#666-discipline requires a paired behavioral KC; absent it, every measurement outcome (PASS or FAIL on the proxy) is information-free at the deployment level.

## What this does NOT rule out

- A *redesigned* v2 with a target-paired KC pair: e.g., (k_proxy: interference reduction ≥20% on A') ∧ (k_target: 2-adapter compose-task accuracy preserved within ≤2 pp on a non-orthogonal Pierre-degraded surface). This v2 must also justify why the non-orthogonal surface is deployment-relevant (e.g. rank-stress at N>50 forcing PoLAR retraction).
- A direct Pierre-grounded experiment on the *Grassmannian* surface using a different mechanism (e.g. activation-based subspace identification, FroM, or DO-Merging) — already enumerated in parent LEARNINGS.

## Assumptions logged (autonomous-decision per guardrail 1008)

- **Operationalization of K#1560 "interference"**: pre-reg leaves it undefined. Reading-charitable: the most-permissive operationalization (Frobenius cross-term residual on summed deltas) still yields a structural-proxy classification per F#666. Less-permissive operationalizations (Gram-error reduction, B-overlap cosine reduction, spectral-deviation shrinkage) yield the same classification a fortiori. No operationalization makes K#1560 a behavioral target.
- **Pierre architecture orthogonality**: project-memory mem-pierre-p1 / mem-pierre-v5-architecture treat Grassmannian (PoLAR/Stiefel) as the architectural baseline. If a future direction abandons PoLAR for unconstrained adapters, the architecture-relevance argument (Theorem 2) weakens and a v2 may be re-examined; current direction does not.

## Related work

- **Spectral surgery** (arXiv:2603.03995) — source paper. Reports +4.4 CSQA / +2.4 HumanEval on individual converged adapters using gradient-sensitivity SV reweighting. Distinct from the SV-magnitude proxy assumed by the killed parents.
- **DO-Merging** (arXiv:2505.15875) — magnitude-direction decoupling with Frobenius-equivalent direction. Cited in F#278 LEARNINGS as a more-promising direction than spectral surgery on factored compositions.
- **STAR** (arXiv:2502.10339) — SVD truncation + rescale at scale. Operates on full-rank matrices, not factored LoRA; does not invalidate the Grassmannian inversion finding.
- **SVC / Spectral Over-Accumulation** (arXiv:2602.05536) — overlap-aware singular-value calibration. Targets the inflated shared-subspace SVs specifically; structurally distinct from blind surgery.

## Recommendation

Close exp_followup_spectral_surgery_grassmannian as preempt-structural KILL. Do not redesign as v2 unless paired with a Pierre-architectural motivation (rank-stress beyond PoLAR retraction tolerance) AND a target-anchored behavioral KC. Existing parent LEARNINGS already enumerate FroM / DO-Merging / SVC as the productive next directions — those subsume this experiment's design space and are themselves grounded in deployment-relevant compositions.

## Finding registered

F#761 (target): 1st spectral-surgery-followup-on-irrelevant-test-pool sub-form within F#666-pure-standalone super-family (~31st drain-window instance).
