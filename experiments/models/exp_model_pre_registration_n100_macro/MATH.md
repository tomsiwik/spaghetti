# MATH.md — exp_model_pre_registration_n100_macro (KILLED_PREEMPTIVE)

## 1. Hypothesis (as declared by target)
N=100 adapter composition on Gemma 4 E4B macro scale achieves per-domain
quality within 5 % of per-domain solo adapter on held-out eval, with no
domain catastrophically degraded (worst-case ≥ 80 % of solo), and the
Room Model `W_combined` pre-sum construction completes in < 60 s on
M5 Pro. The target frames itself as a macro scale-up of the SUPPORTED
micro T3.5 result (`exp_p1_t3_n100_composition`, numeric KC pass).

KC (pre-registered, locked by claim):
- K1708 — per-domain quality within 5 % of per-domain solo adapter on
  held-out eval at N=100.
- K1709 — no domain catastrophically degraded (worst-case ≥ 80 % of
  solo).
- K1710 — Room Model `W_combined` construction completes in < 60 s on
  M5 Pro.

## 2. Preempt theorem (defense-in-depth, 5-of-5 independent blocks)

**Theorem (preempt).** The empirical run is **impossible** or
**guaranteed-to-fail** iff at least **one** of the five blocks holds.
We show **four** hold independently (T1 ∧ T2 ∧ T3 ∧ T5) plus **one**
reinforces (T4). Any single block suffices. T5 is **T5-K variant**
(parent-KILLED) and further **double-T5-K** (both declared parents are
KILLED — first occurrence in drain).

### T1 — Artifact-absence block
Required artifacts (pre-reg, N=100 macro composition eval):

1. **100 pre-trained Gemma 4 E4B adapters** (one per domain), each
   passing solo eval at declared LoRA scale. The adapter cohort must
   be an actual on-disk set of `.safetensors` files under a
   `/adapters/` tree, with per-adapter provenance (domain label,
   training data split, checkpoint).
2. **Room Model `W_combined` construction routine** that pre-sums
   N adapter deltas into the base weight in < 60 s on M5 Pro (K1710).
   The parent `exp_model_room_model_gemma4_speed` already KILLED this
   routine at N=5 (69 tok/s vs 150 target, cos 0.9941 vs 0.999
   target — see §4 A6). Scaling from N=5 → N=100 multiplies dispatch
   and accumulates numerical drift; the routine does not exist in a
   passing state.
3. **Per-domain held-out eval harness at N=100** — 100 distinct
   domains × ≥ 50 held-out Q/domain with a scoring function
   calibrated per domain. Generic benchmarks (MMLU-Pro, GSM8K) do
   not map to 100 domains.
4. **Per-domain solo adapter eval baseline** — K1708 and K1709
   measure the composed adapter vs its **own solo** on the same
   held-out slice. Requires 100 solo runs, same sampling config,
   same scoring, same seeds. Infrastructure absent.
5. **Domain-specific routing / composition framework at N=100** —
   existing composition code caps at N=5 stacks (see iter 40/41/42
   drain: `pierre/` and `macro/` N=5 serve endpoints remain
   unshipped). N=100 routing requires a router (hash, xxHash,
   softmax, or equivalent) and per-token gate; drain history shows
   no functional N=100 routing code in repo.

Block fires if shortfall ≥ 3 of 5. Pre-analysis by code grep under
`pierre/`, `macro/`, `composer/`, `micro/models/` (excluding this
runner) plus an on-disk adapter-count check under
`micro/models/**/adapters/`:

- (1) 100 Gemma 4 E4B adapters on disk: **absent** — repo inventory
  shows ≲ 10 Gemma 4 E4B `.safetensors` adapter checkpoints across
  all experiments, nearly all from `exp_p1_t2_single_domain_training`
  (math/code/medical). No N=100 adapter cohort exists.
- (2) Room Model `W_combined` construction at N=100: **absent** —
  parent KILLED at N=5 (K1688 FAIL 69 vs 150 tok/s; K1689 FAIL cos
  0.9941 vs 0.999). A1 A6 document this transitively: if the
  N=5 routine fails, the N=100 routine (which reuses the same
  `W_combined +=` kernel) is worse, not better.
- (3) Per-domain held-out eval harness at N=100: **absent** — no
  `100_domains`, `domain_cohort_100`, `per_domain_heldout` module.
- (4) Per-domain solo baseline runner at N=100: **absent** — solo
  baselines exist for 3 T2.1 domains (math, code, medical); 100
  require re-training and re-eval infrastructure that is not
  written.
- (5) Composition / routing framework at N=100: **absent** —
  iter 40/41/42 confirmed N=5 serve remains unshipped; N=100 is a
  strictly larger unbuilt target.

Shortfall ≥ 5/5. **T1 blocks** (over-determined).

### T2 — Cost-bound block
Conservative N=100 macro composition eval on M5 Pro 48 GB, MLX,
Gemma 4 E4B:

- Base cold-load: 15 min.
- Room Model `W_combined` construction at N=100: target is < 60 s
  but parent-KILLED at N=5 takes substantially longer; budget at
  10 min conservative wall (K1710 would fail empirically; budget
  bounds below assume the construction even completes).
- 100 per-domain held-out evals × 50 Q/domain × 5 s/sample =
  25,000 s ≈ 416.7 min.
- 100 per-domain solo-baseline evals (same protocol) = another
  25,000 s ≈ 416.7 min.
- 100 solo adapter cold-loads × 10 s = 1,000 s ≈ 16.7 min.

Conservative total:
  `15·60 + 10·60 + 25000 + 25000 + 1000 = 900 + 600 + 25000 +
  25000 + 1000 = 52,500 s ≈ 875.0 min`
vs **120 min ceiling**. Block fires by > 7×.

Floor (smoke-size N=100, 10 Q/domain, 1 s/sample, no solo-baseline
re-eval — uses cached solo scores if they existed, which they
don't): `900 + 600 + 100·10·1 + 0 + 100·10 = 900 + 600 + 1000 +
0 + 1000 = 3,500 s ≈ 58.3 min` under ceiling, but K1708 and K1709
both degenerate: a 10-Q per-domain eval gives ± 31 pp half-width at
95 % CI, and the 5 % K1708 threshold is noise. Additionally, solo
scores do not exist for 97 of 100 domains, so the smoke variant is
structurally incoherent.

**T2 blocks.**

### T3 — Schema-incomplete block
DB record (verbatim from `experiment get exp_model_pre_registration_n100_macro`):
  `success_criteria: [] # MISSING`
  `⚠ INCOMPLETE: missing success_criteria`
  `references: []` (zero entries)

Zero `references` entries — the notes mention "T3.5 proved N=100 at
micro" in prose but no arxiv / finding / experiment id is registered
as a reference. F#502/F#646 antipattern: **12th occurrence** in this
drain (iter 43 was 11th). Stable, earned heuristic. **T3 blocks.**

### T4 — Audit-pin reinforcer
Macro experiment with no prior runner, no DB diff in last 72 h, no
`.audit` directory. Pin-ratio measured post-run; reinforce-only.
**T4 reinforces (does not block alone).**

### T5 — Source-scope breach block (T5-K variant, double-parent KILLED)
Target declares two parents in `depends_on`:
- `exp_model_room_model_gemma4_speed` — current DB `Status: killed`
  (K1688 FAIL 69 tok/s vs 150; K1689 FAIL cos 0.9941 vs 0.999;
  K1690 PASS).
- `exp_p1_t3_n25_composition` — current DB `Status: killed`
  (K1060 FAIL 0/5 adapters; K1061 FAIL MMLU regression; K1059 /
  K1062 PASS but only the orthogonality / disk-budget checks, not
  the behavioral KC).

**Both** declared parents are KILLED. This is the T5-K variant
(parent-KILLED) in its strongest form: **double-T5-K**. A parent
being KILLED means the ground-truth infrastructure / claim the
target would scale from has already failed its own KC. Scaling a
failed result to a larger N cannot flip the parent's empirical
failure; the target inherits the parents' killed status by
construction.

Transitive-kill breach dimensions (pre-reg ≥ 1 required for T5-K;
we count ≥ 3 for defense-in-depth):
  (A) **Room Model speed breach**. Parent measured 69 tok/s at
      N=5; target K1710 asks the same routine to complete < 60 s
      at N=100. N=100 construction is a superset of N=5
      (accumulating 20× more delta summations); parent-measured
      throughput falsifies target K1710 by transitivity.
  (B) **Room Model quality breach**. Parent measured cos 0.9941
      (routing ↔ W_combined); target K1708 asks for per-domain
      quality within 5 % of solo. Cos 0.9941 at N=5 already
      indicates the pre-sum drifts; at N=100 the drift compounds
      (every additional adapter adds numerical error in the
      combined kernel).
  (C) **N=25 adapter cohort breach**. Parent `exp_p1_t3_n25_
      composition` KILLED at N=5 (K1060 FAIL: 0 of 5 adapter
      `.safetensors` actually trained/loaded). Target
      extrapolates to N=100 from a parent that failed to deliver
      even 5 working adapters.
  (D) **N=25 MMLU-preservation breach**. Parent K1061 FAIL: MMLU
      regressed under composition at N=25. Target's K1708
      (per-domain quality within 5 %) implicitly requires
      base-preservation-equivalent; parent already falsifies at
      lower N.
  (E) **T3.5 non-declaration breach**. Target's `notes` reference
      `T3.5 / exp_p1_t3_n100_composition` (SUPPORTED micro, numeric
      KC only) but T3.5 is **not** declared in `depends_on`.
      Backfilling T3.5 as a parent is retrofit (violates the
      standard parent-declared-at-claim rule, see A8). Even if
      retrofitted, T3.5 is `micro` scale — per `scale: micro` it
      does not guarantee macro behavior (see Finding #478 ceiling
      and Finding #571 Room Model superseded for N>1).

Count = **5/5 breaches**. All five are **transitive-kill** breaches
under the T5-K variant (parent-killed) — not the supported-parent
scope-breach. **T5 (T5-K double) blocks** with wide margin. First
occurrence of double-T5-K in drain.

**Theorem conclusion.** Verdict is **4-of-5 independent blocks** (T1 ∧
T2 ∧ T3 ∧ T5-K-double) plus **1 reinforcing** (T4). Any single block
suffices. Target is unrunnable on `local-apple` / MLX / 48 GB M5 Pro
within a 120 min budget without operator action (train 100 adapters
on Gemma 4 E4B, rewrite the KILLED Room Model `W_combined` routine,
build a 100-domain eval harness, register success criteria and
references, and — on top of all that — empirically flip two
parent-KILLED results).

## 3. Predictions (pre-registered)

| ID | Prediction | Measurement |
|----|------------|-------------|
| P1 | T1 shortfall ≥ 3 of 5 required artifacts | code grep under pierre/, macro/, composer/, micro/models/ + on-disk adapter inventory under micro/models/**/adapters/ |
| P2 | T2 timing ≥ 120 min (conservative; floor scientifically incoherent) | arithmetic on N=100 per-domain eval protocol |
| P3 | T3 DB has `success_criteria: []` + `⚠ INCOMPLETE` marker + empty references | DB probe via `experiment get` |
| P4 | T4 pin_ratio in `.audit/` = 0 (dir absent); reinforce-only | `.audit` listing |
| P5 | T5-K double: both declared parents `Status: killed`; breach count ≥ 3 of 5 transitive-kill dimensions | DB probe for parent status + source `results.json` / `PAPER.md` / `MATH.md` read |

## 4. Assumptions / caveats (A-series)
- **A1.** "Present in repo" = grep-reachable in `*.py` under `pierre/`,
  `macro/`, `composer/`, `micro/models/` (excluding this runner).
  Adapter payload probe reads on-disk `.safetensors` files under
  `micro/models/**/adapters/` with size > 1 KB.
- **A2.** Adapter cohort probe requires ≥ 100 distinct Gemma-4-E4B-
  compatible `.safetensors` adapter checkpoints across the repo;
  anything below is a shortfall. Known upper bound from iter 42
  inventory is < 20 unique files.
- **A3.** T1(5) N=100 composition/routing probe requires literal
  cooccur of `N\s*=\s*100|100[_-]?adapter|cohort[_-]?100|domain[_-]?
  cohort` with one of `compose|route|stack|router` in the same file.
- **A4.** T2 uses conservative 5 s/sample for a 100-Q · 100-domain
  protocol. The real protocol (LLM-judged or per-domain open-ended)
  would be larger. Not sensitive: the floor variant is
  scientifically incoherent with K1708 and K1709 (CI too wide) and
  structurally incoherent (solo baselines absent for 97/100 domains).
- **A5.** T3 reads the literal DB pretty-print. The target's
  `notes` cites T3.5 but the `references:` list is empty; this is
  the operator-facing "missing" signal.
- **A6.** Both declared parents `exp_model_room_model_gemma4_speed`
  and `exp_p1_t3_n25_composition` are KILLED per live DB; this is
  the T5-K variant (parent-KILLED) in its strongest form — the
  double-parent-killed case is new in the drain.
- **A7.** A7/A8: the `notes` reference to T3.5
  (`exp_p1_t3_n100_composition`, SUPPORTED micro) is **not** a
  declared parent. Backfilling a parent from prose is retrofit.
  Even if treated as a parent, T3.5 is micro-scale and its
  KC are numeric-only (orthogonality, MMLU, routing accuracy, disk
  budget) — none of them check the behavioral claims K1708/K1709/
  K1710 require. Memory `project_room_model.md` further records
  Finding #571: Room Model SUPERSEDED for N>1; and
  `feedback_spectral_arc_closed.md` notes no more per-domain
  composition experiments at Room Model scale.
- **A8.** Runner is pure stdlib + `experiment get` shell-out. Zero
  MLX, zero model load, zero HTTP bind. ≤ 3 s wall.
- **A9.** F#502 12th-occurrence claim is cumulative drain count;
  runner reports the per-file `⚠ INCOMPLETE` literal from the DB,
  not a running counter. Counter is in LEARNINGS / scratchpad prose.
- **A10.** F-axis placement: (s4) T5-K parent-KILLED lineage,
  **double-T5-K sub-variant** (first in drain). Sibling or child of
  F#651 (single-parent T5-K, iter 36) is analyst's call; the
  double-parent variant is strictly stronger because the target
  scales from a FAILED N=5 parent and a FAILED Room Model speed
  parent simultaneously.
