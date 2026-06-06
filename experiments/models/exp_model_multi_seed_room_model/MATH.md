# MATH.md — exp_model_multi_seed_room_model (KILLED_PREEMPTIVE)

## 1. Hypothesis (as declared by target)
Room Model N=5 composition on Gemma 4 E4B is **seed-robust**:
CV(MMLU-Pro score across 3 seeds) < 5 % with no seed producing a
catastrophic (>2σ below mean) outlier.

KC (pre-registered, locked by claim):
- K1711 — CV(MMLU-Pro score across 3 seeds) < 5 % for N=5 Room
  Model composition.
- K1712 — No seed produces catastrophic outlier (> 2 σ below mean).

Target declared parent: `exp_model_room_model_gemma4_speed`
(single-parent, current DB `Status: killed`).

## 2. Preempt theorem (defense-in-depth, 5-of-5 independent blocks)

**Theorem (preempt).** The empirical run is **impossible** or
**guaranteed-to-fail** iff at least **one** of the five blocks holds.
We show **four** hold independently (T1 ∧ T2 ∧ T3 ∧ T5-K) plus **one**
reinforces (T4). Any single block suffices. T5 is **T5-K variant**
(single-parent-KILLED, 2nd in drain after iter 36
`exp_model_loader_portability`).

### T1 — Artifact-absence block
Required artifacts (pre-reg, 3-seed Room Model composition at N=5):

1. **Room Model `W_combined` construction routine** passing parent
   KC K1688 (≥150 tok/s) and K1689 (logit cos > 0.999). Parent
   KILLED with 69 tok/s and cos 0.9941 — routine does not exist in
   a passing state.
2. **3 independently seeded training pipelines for N=5 Gemma 4 E4B
   LoRA adapters** — requires reproducible seed control across
   MLX array ops, sampler draws, and dataset shuffling for all five
   domains at all three seeds (15 adapter trainings total).
3. **MMLU-Pro eval harness** bound to the composed model for
   scoring parity across seeds (same prompt template, same decode
   policy, same N-Q sample).
4. **CV aggregator + 2 σ outlier test** over the 3-seed score
   tuple. Statistics code is trivial; the piece missing is the
   multi-seed harness that produces the tuple.
5. **Seed-controlled adapter-merge / compose invocation** — the
   target composes N=5 adapters per seed; requires that the
   merge routine is itself seed-deterministic (no nondet kernels).

Block fires if shortfall ≥ 3 of 5. Pre-analysis by code grep under
`pierre/`, `macro/`, `composer/`, `micro/models/` (excluding this
runner) plus on-disk adapter-count check for a 3-seed cohort:

- (1) Functional Room Model `W_combined` at N=5: **absent** — parent
  KILLED; see A1 A6. K1688 FAIL 69 tok/s, K1689 FAIL cos 0.9941.
- (2) Triple-seed N=5 adapter cohort on disk: **absent** — repo
  inventory shows solo-domain adapter checkpoints from
  `exp_p1_t2_single_domain_training` (single seed) and the
  T3/T3.5 composition experiments (single seed). No
  `seed_0/`, `seed_1/`, `seed_2/` cohort tree exists under
  `micro/models/**/adapters/`.
- (3) MMLU-Pro composed-model eval harness: **partial** — MMLU
  subsets exist in eval code but no composed-model binding at
  Room Model (`W_combined`) output, because (1) is absent.
- (4) CV + 2 σ outlier runner: **absent** as an integrated routine;
  fragments in analysis scripts.
- (5) Seed-controlled merge / compose invocation: **absent** —
  compose code in `pierre/` and `micro/` uses global RNG state
  or implicit defaults; no `seed=` parameter plumbed through the
  W_combined adder.

Shortfall ≥ 4/5. **T1 blocks** (over-determined).

### T2 — Cost-bound block
Conservative 3-seed N=5 Room Model composition + MMLU-Pro eval
on M5 Pro 48 GB, MLX, Gemma 4 E4B:

- Base cold-load × 3 seeds (serial; cannot reuse mx cache across
  seed swap safely): 3 × 15 min = 45 min wall.
- Room Model `W_combined` construction × 3 seeds: 3 × 10 min
  (parent failed K1710-equivalent at N=5; 10 min conservative
  budget per build) = 30 min.
- MMLU-Pro eval × 3 seeds, conservative 1 000 Q / seed @ 5 s/Q:
  3 × 5 000 s = 15 000 s = 250 min.
- 3 × N=5 adapter cold-load (if seed rotation reloads adapters):
  3 × 5 × 10 s = 150 s = 2.5 min.

Conservative total:
  `45·60 + 30·60 + 15000 + 150 = 2 700 + 1 800 + 15 000 + 150 =
  19 650 s ≈ 327.5 min`
vs **120 min ceiling**. Block fires by > 2.7×.

Floor (smoke: 3 seeds × 100 Q × 1 s = 300 s; no multi-seed
W_combined rebuild — assumes cached; no cold-load accounting):
`3·15·60 + 300 = 2 700 + 300 = 3 000 s ≈ 50 min` under ceiling,
but K1711 degenerates: at 100 Q the MMLU-Pro score half-width at
95 % CI is ≈ ± 10 pp; a 5 % CV threshold on scores in the 40–60 %
range (± 2 – 3 pp) is within noise. Floor is scientifically
incoherent.

**T2 blocks.**

### T3 — Schema-incomplete block
DB record (verbatim from `experiment get exp_model_multi_seed_room_model`):
  `Success Criteria: NONE`
  `⚠ INCOMPLETE: success_criteria, references, kill_results (all untested)`
  `references:` absent from pretty-print (zero entries).

Empty `success_criteria`, empty `references`, INCOMPLETE flag
simultaneously. F#502/F#646 antipattern: **13th occurrence** in
this drain (iter 44 was 12th). Stable, earned heuristic.
**T3 blocks.**

### T4 — Audit-pin reinforcer
Macro experiment with no prior runner, no DB diff in last 72 h,
no `.audit` directory. Pin-ratio measured post-run;
reinforce-only. **T4 reinforces (does not block alone).**

### T5 — Source-scope breach block (T5-K variant, single-parent KILLED)
Target declares one parent in `depends_on`:
- `exp_model_room_model_gemma4_speed` — current DB `Status:
  killed` (K1688 FAIL 69 tok/s vs 150; K1689 FAIL cos 0.9941 vs
  0.999; K1690 PASS).

The declared parent is KILLED. This is the **T5-K variant**
(parent-KILLED), single-parent form. 2nd single-parent T5-K in the
drain.

Transitive-kill breach dimensions (pre-reg ≥ 1 required for
T5-K; we count ≥ 3 for defense-in-depth):

  (A) **Room Model speed breach**. Parent measured 69 tok/s at
      N=5; target's multi-seed harness rebuilds `W_combined`
      three times over. Multi-seed cannot heal the per-seed
      speed deficit — it triples the cost.
  (B) **Room Model quality breach**. Parent measured cos 0.9941
      (routing ↔ W_combined) against the explicit-routing
      reference. If single-seed `W_combined` already drifts at
      cos 0.9941 vs the routing oracle, the "stable across seeds"
      claim (K1711) is asking for CV < 5 % on a method whose
      **single-seed** deviation from ground-truth is already
      outside its own tolerance band. Stability is measured
      relative to a moving target.
  (C) **F#571 memory-breach**. Memory `project_room_model.md`
      records Finding #571: Room Model SUPERSEDED for N>1 (killed
      4×). Target runs at N=5. The supersession is an empirical
      kill, not a stylistic preference.
  (D) **K1690 N=1 scope breach**. Parent's one PASS was K1690
      (bitwise-exact reversibility via `W_combined += / -=`
      delta), explicitly scoped in memory notes to **N=1
      hot-merge** reuse. Target N=5 composes 5 deltas; K1690's
      N=1 scope does not transfer.
  (E) **KC-target coupling breach**. K1711 measures
      CV(MMLU-Pro) across seeds **for N=5 Room Model
      composition**. The composed object is the parent-KILLED
      routine. A seed-robustness claim about a killed routine
      is ill-posed: the method's central claim already failed
      on seed-0; varying seed does not recover a failed method.

Count = **5/5 breaches**. All five are **transitive-kill**
breaches under the T5-K variant (parent-killed). **T5 (T5-K
single-parent) blocks** with wide margin.

**Theorem conclusion.** Verdict is **4-of-5 independent blocks** (T1 ∧
T2 ∧ T3 ∧ T5-K-single) plus **1 reinforcing** (T4). Any single block
suffices. Target is unrunnable on `local-apple` / MLX / 48 GB M5 Pro
within a 120 min budget without operator action (resurrect the KILLED
Room Model `W_combined` routine, train 15 seeded Gemma 4 E4B adapters,
bind them to an MMLU-Pro composed-model harness, register success
criteria and references, and — on top of all that — empirically flip
the parent-KILLED result).

## 3. Predictions (pre-registered)

| ID | Prediction | Measurement |
|----|------------|-------------|
| P1 | T1 shortfall ≥ 3 of 5 required artifacts | code grep under pierre/, macro/, composer/, micro/models/ + on-disk seed-cohort inventory under micro/models/**/adapters/ |
| P2 | T2 timing ≥ 120 min (conservative; floor scientifically incoherent) | arithmetic on 3-seed × N=5 × MMLU-Pro protocol |
| P3 | T3 DB has `Success Criteria: NONE` + `⚠ INCOMPLETE` marker + empty references | DB probe via `experiment get` |
| P4 | T4 pin_ratio in `.audit/` = 0 (dir absent); reinforce-only | `.audit` listing |
| P5 | T5-K single-parent: parent `Status: killed`; breach count ≥ 3 of 5 transitive-kill dimensions | DB probe for parent status + source `results.json` / `PAPER.md` / `MATH.md` read |

## 4. Assumptions / caveats (A-series)
- **A1.** "Present in repo" = grep-reachable in `*.py` under `pierre/`,
  `macro/`, `composer/`, `micro/models/` (excluding this runner).
  Seed-cohort probe reads on-disk `.safetensors` files under
  `micro/models/**/adapters/` for `seed_*` subtrees with size > 1 KB.
- **A2.** A 3-seed cohort at N=5 requires 15 distinct adapter
  checkpoints in a `seed_{0,1,2}/` convention. Lower is shortfall.
- **A3.** T1(5) seed-controlled merge probe requires literal cooccur
  of `seed\s*=\s*\d|seed[_-]?0|seed[_-]?1|seed[_-]?2` with one of
  `W_combined|w_combined|merge|compose|add_adapter` in the same
  file. A9 documents false-positive risk (generic `seed=` kwargs in
  dataloaders).
- **A4.** T2 uses conservative 5 s/sample for MMLU-Pro with
  generated-answer scoring on M5 Pro. Published MMLU-Pro eval
  times on comparable Apple Silicon are 3 – 7 s/Q. Not sensitive:
  the floor variant is scientifically incoherent with K1711
  (CV threshold inside CI noise) and K1712 (2 σ outlier test at
  N=3 seeds is underpowered regardless of Q count).
- **A5.** T3 reads the literal DB pretty-print. `Success Criteria:
  NONE` and `⚠ INCOMPLETE: success_criteria, references, kill_results
  (all untested)` are the operator-facing "missing" signals.
- **A6.** Parent `exp_model_room_model_gemma4_speed` is KILLED per
  live DB with K1688 FAIL, K1689 FAIL, K1690 PASS. This is the
  T5-K variant (parent-KILLED), single-parent form. 2nd in drain.
- **A7.** Memory `project_room_model.md` records: "SUPERSEDED for
  N>1 (killed 4×, Finding #571); K1690 bitwise-exact reversibility
  reusable only for N=1 hot-merge." Target runs at N=5 and
  compounds the breach with a seed-robustness claim.
- **A8.** Runner is pure stdlib + `experiment get` shell-out. Zero
  MLX, zero model load, zero HTTP bind. ≤ 3 s wall.
- **A9.** T1(5) seed-controlled merge grep may false-positive on
  dataloader `seed=` kwargs. Runner reports shortfall vs threshold
  literally and does NOT inflate — verdict over-determined by
  T1(1) ∨ T2 ∨ T3 ∨ T5-K without T1(5).
- **A10.** F-axis placement: (s4) T5-K parent-KILLED lineage,
  single-parent form. 2nd instance (first was iter 36
  `exp_model_loader_portability`). Sibling of F#651 (single-parent
  T5-K) — analyst may promote to F#651 scope addendum or register
  new F-id when cap lifts.
