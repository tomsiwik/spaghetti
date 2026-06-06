# MATH.md — exp_g4_lora_rank_importance_per_task (PREEMPT-KILL)

## Verdict: PREEMPT-KILL (F#666-pure standalone)

This experiment is preempt-killed before any code is run because the pre-registered KC set is **proxy-only with no behavioral target pair** while the experiment is **standalone** (`depends_on=[]`). Per Finding #666 + guardrail 1007 (TARGET-GATED KILL), KILL on a proxy-only KC set is forbidden — the dual-tail proxy structure permits a "PASS" outcome that does not certify behavioral benefit, and a "FAIL" outcome that does not certify behavioral loss. Both decision branches are unidentifiable as findings about the underlying claim.

This is the **~29th F#666-pure standalone preempt-KILL** in the drain window (running tally: F#703 + F#710 + F#714 + F#715 + F#719 + F#720 + F#722 + F#727 + F#728 + F#729 + F#730 + F#731 + F#732 + F#733 + F#734 + F#735 + F#736 + F#753 + F#754 + F#755 + F#756 + F#757 + ... ≈29). It is the **13th g4-ablation super-family sub-type** and the **1st rank-variance-bucket form** (structural-hyperparameter argmax-divergence; NEW proxy-bucket sub-form).

## §0 Platform / skills / model pins

Included for completeness — no platform code is executed.

- Platform skills: `/mlx-dev` + `/fast-mlx` (per `PLAN.md` Part 2). **Not invoked** — no MLX code written; honest disclosure per reviewer checklist item (m2).
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627 — proven adapter recipe). **Not loaded.**
- Adapter targets: would have been `v_proj + o_proj` per F#627 if runnable; with rank sweep over r ∈ {2, 4, 8, 16, 32}.
- Tasks: notes specify "task type" sweep but neither the task set nor the task-quality metric is defined in pre-reg. This ambiguity is itself a structural defect — the threshold `optimal rank variance > 4×` cannot be evaluated until "optimal rank" is operationalized via a task-quality argmax.
- Datasets: undefined in pre-reg.
- LORA_SCALE: would default to ≤8 per F#627/F#328/F#330; not relevant to the preempt theorem.

## §1 Preempt-KILL theorem

**Theorem (proxy-only KC set on standalone experiment is unidentifiable as a finding).** Let `E` denote experiment `exp_g4_lora_rank_importance_per_task` with kill criteria K = {K1941, K1942}:

- **K1941**: "All tasks need same rank (no task-specific rank sweet spot)" — structural claim about uniformity of `argmax_r M(r, task)` across tasks, where `M(·, task)` is some quality measure (per-task PPL, per-task downstream accuracy, per-task cosine similarity to teacher; **not specified in pre-reg**).
- **K1942**: "Optimal rank variance > 4× across tasks (one-size-fits-all rank is wasteful)" — same `argmax_r` distribution, computing `max(argmax) / min(argmax) > 4`.

**Step 1 — Both KCs are proxies.** Neither K1941 nor K1942 directly measures behavioral utility. They both reduce to a distribution over `argmax_r M(r, task_i)`, which:

1. Inherits the proxy/target status of `M`. If `M` is PPL, both KCs are *PPL-argmax-distribution* claims — proxy by F#666 (r≈0.08 PPL↔task-quality). If `M` is task accuracy, both KCs are *accuracy-argmax-distribution* claims — but the KCs do not measure accuracy itself, only the *index* of its maximum across the rank grid. Index-of-maximum is a **dimensionless structural statistic**, even when computed from a target metric.
2. Pre-reg fixes neither `M` nor the rank grid nor the task set. This degrees-of-freedom freedom permits post-hoc selection that can flip the KC verdict (antipattern: KC-after-data).

**Step 2 — Both KCs are forbidden-solo per guardrail 1007.** Per guardrail 1007 quoting F#666: "every proxy KC must be paired with a target-metric KC." K1941 and K1942 are both proxies (rank-argmax structural statistic), so the KC set has no target. KILL on either alone is forbidden.

**Step 3 — Standalone with no fallback target.** `depends_on=[]` and no parent target metric to inherit. Unlike compound experiments where a child KC may reference a parent target metric (e.g. F#699's quasi-target inheritance from MEMENTO parent), this experiment has no upstream target to anchor the rank-argmax statistic against.

**Step 4 — Decision-table analysis (per F#666 r≈0.08 PPL↔task-quality):**

| K1941 (uniform) | K1942 (variance > 4×) | Behavioral interpretation                                                              | F#666 verdict     |
| --------------- | --------------------- | -------------------------------------------------------------------------------------- | ----------------- |
| PASS (uniform)  | FAIL (variance ≤ 4×)  | argmax_r is uniform OR variance ≤ 4× — but neither implies "task-adaptive rank is unnecessary" behaviorally; could just be flat M(r) curves with noise picking ranks. | **inconclusive**  |
| PASS (uniform)  | PASS (variance > 4×)  | **CONTRADICTORY** — claims simultaneously uniform AND variance > 4×; KC pair self-incompatible.                                                         | **degenerate**    |
| FAIL (varied)   | FAIL (variance ≤ 4×)  | argmax differs across tasks but spread is small (e.g. r=4 for one, r=8 for another, max/min=2×). Behavioral utility unknown without a task-accuracy gain measurement. | **inconclusive**  |
| FAIL (varied)   | PASS (variance > 4×)  | Pre-reg "PASS" intent (researcher hopes to find this case) — argmax differs widely. Still a *structural* observation; says nothing about whether per-task rank actually buys behavioral utility. Could be 2 tasks × 2 ranks × noise → "4× variance" is noise floor, not signal. | **proxy-PASS without target — F#666 forbidden** |

Notice the K1941+K1942 truth table is **internally inconsistent**: K1941 (uniform) and K1942 (variance > 4×) cannot both be PASS without contradiction (variance > 4× means non-uniform). The KC pair forces an XOR-like dependency, leaving only 3 of 4 cells reachable, none of which produces a behaviorally-anchored finding.

**Step 5 — Threshold unanchored.** The 4× variance threshold is unanchored against:

- Common LoRA rank grids `{1, 2, 4, 8, 16, 32}` are themselves geometric (×2), so `max/min = 2× to 32×` natively. A "> 4×" threshold trivially fires whenever argmax spans 3 or more grid steps.
- F#742 prior result: rank-6 q_proj LoRA on Gemma 4 E4B has **C_20=0.335 weak structural concentration** — i.e. low-rank LoRA on q_proj already lacks a clean per-task rank optimum because the slab dim (256/512) ≫ rank. The argmax operation on a near-flat curve picks values at noise level, making the variance threshold dominated by sampling noise rather than task structure.
- The pre-reg notes "If tasks need different ranks, we can use task-adaptive rank allocation" — but the runnable falsification of this claim requires a *behavioral utility* measurement (task accuracy with task-adaptive vs fixed rank), not the *structural distribution* of argmax-rank values.

∴ K1941 and K1942 are forbidden-solo proxies on a standalone experiment. KILL is impermissible per F#666; SUPPORTED requires a target-metric pair which is absent. **QED.**

### §1.1 F#666 gating (negative)

- K1941 = **proxy** (structural rank-uniformity: distribution of argmax_r across tasks; index-of-maximum is dimensionless structural statistic).
- K1942 = **proxy** (structural rank-variance ratio: max(argmax)/min(argmax) across tasks).
- **No target KC.** KC set is **F#666-noncompliant — proxy-only forbidden-solo.**

### §1.2 Sub-axis classification — 13th g4-ablation sub-type, 1st rank-variance-bucket form

This experiment introduces a **NEW proxy-bucket form** within the F#666-pure standalone family: **structural-hyperparameter argmax-divergence** (a.k.a. "rank-variance-bucket"). Distinct from prior buckets:

| Proxy bucket                              | 1st observation                                            | Form                                                             |
| ----------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------- |
| Cosine-similarity bucket                  | F#720 (final-value)                                        | F#720 final-value, F#755 convergence-speed, F#756 tightness, F#757 cross-instance population-statistic dual-tail |
| Routing-accuracy bucket                   | F#703 + F#710 + F#736                                      | classification-acc, embedding-cache-acc-delta                    |
| Infrastructure-benchmark bucket           | F#714 + F#715 + F#736 + F#753                              | wall-clock latency, cache-staleness                              |
| Training-axis efficiency bucket           | F#756 (training-time savings)                              | training-time-vs-quality                                         |
| **Structural-hyperparameter argmax-divergence** | **(this) F#759 — argmax_r dispersion across tasks**  | **rank-uniformity + rank-variance-ratio**                        |

g4-ablation super-family sub-type tally (13 sub-types after this):
1. per-layer-cos
2. PPL-drift
3. canary-FNR
4. routing-collision
5. hash-ring-PPL
6. routing-family
7. gumbel-routing
8. perturbation-stability
9. SVD-rank-delta
10. SVD-denoise-PPL
11. init-method-comparison (F#723)
12. seed-determinism (F#757)
13. **rank-importance-per-task (this — rank-ablation sub-type)**

The g4-ablation super-family does NOT saturate — different sub-types address structurally different failure modes. (Hedgehog-ablation saturated at 7 per F#756 because the sub-types covered all reachable axes of training-procedure variation; g4-ablation has more orthogonal axes available.)

Pre-existing partial coverage:
- **F#742** (provisional): rank-6 q_proj LoRA on Gemma 4 E4B has **C_20=0.335 weak structural concentration** + **J̄=0.349 cross-domain functional specialization**. Implies low-rank LoRA on q_proj lacks a clean per-task rank optimum because slab dim ≫ rank. Direct anchor for "argmax_r is noise-dominated at low rank."
- **F#143** (killed): "Parallel rank accumulation no advantage at d=256" — 4× rank capacity ratio (32 vs 8) does not translate to quality advantage; magnitude dilution + data homogeneity cancel. Anchor: rank-utility scaling is NOT linear; a "rank optimum" may not exist at all for some task/architecture combinations.
- **F#522** (supported): TT-LoRA r=6 with MCQ classification loss recovers +14.5pp discriminative capacity. Anchor: "what rank does each task need" depends critically on the loss formulation, not just the rank — a confound K1941/K1942 don't disentangle.
- **F#686** (provisional): `exp_g4_adapter_class_composition_full` design includes K4 (rank-ablation) as a pair-anchored design; this experiment re-litigates the same axis without target anchoring. Direct redundancy with F#686's K4 once parent SUPPORTED.

## §2 Prior art (preempt rationale)

- **F#666** (defining): target-gated KC discipline — proxy-only KC set forbidden-solo.
- **F#703** (1st routing-acc-bucket F#666-pure standalone preempt-KILL).
- **F#710** (2nd routing-acc-bucket).
- **F#714** (1st infrastructure-benchmark bucket).
- **F#720** (1st cos-sim-bucket).
- **F#722** (4th Hedgehog-ablation hyperparameter-ablation).
- **F#735** (23rd F#666-pure with NEW variance-bound sub-flavor — closest prior on "variance-style proxy-bucket").
- **F#736** (24th F#666-pure with routing-acc + infra-bench co-fire).
- **F#753** (3rd routing-acc-bucket × 2nd infra-bench-bucket co-fire).
- **F#754** (24th drain-window: routing-acc-delta 4th + cache-staleness 3rd, routing-acc + infra-bench cross-pollination).
- **F#755** (25th: 6th Hedgehog-ablation curriculum).
- **F#756** (26th: 7th Hedgehog-ablation early-stopping; super-family SATURATES at 7).
- **F#757** (27th: 1st g4-ablation seed-determinism, 4th cos-sim-bucket cross-instance dual-tail).
- **F#758** (28th: 5th MEMENTO-cluster child, target-only-KC-panel-under-preempt-KILL CANONICALIZES at 3rd obs).
- **F#742** (provisional, direct prior): rank-6 q_proj on Gemma 4 E4B has weak per-task structural concentration; argmax_r is noise-dominated at low rank.
- **F#143** (killed): rank-capacity scaling non-linear at d=256 — "rank optimum" may not exist.
- **F#522** (supported): rank importance is loss-formulation-dependent; rank alone underdetermines behavioral utility.
- **F#686** (provisional): K4 (rank-ablation) of `exp_g4_adapter_class_composition_full` is the target-paired design that subsumes this experiment once parent SUPPORTED.

## §3 Predictions (registered, not measured)

All KC states are **"untested (preempt-blocked)"**:

| KC    | Claim                                                                | Kind  | Measurement status                          |
| ----- | -------------------------------------------------------------------- | ----- | ------------------------------------------- |
| K1941 | All tasks need same rank (no task-specific rank sweet spot)          | proxy | untested (preempt-blocked, F#666-pure)      |
| K1942 | Optimal rank variance > 4× across tasks (one-size-fits-all wasteful) | proxy | untested (preempt-blocked, F#666-pure)      |

Both KCs are F#666-noncompliant (proxy-only). Even if measured, their joint truth-table contains 3 inconclusive cells (PASS+FAIL contradictory, FAIL+FAIL inconclusive, PASS+FAIL inconclusive) and 1 "proxy-PASS without target — F#666 forbidden" cell. No reachable cell produces a behaviorally-anchored finding.

## §4 Unblock condition

Re-claimable when:

1. **Pair K1941/K1942 with a behavioral target KC.** Recommended: K1943 (target) "Task-adaptive rank allocation gives ≥ 5pp downstream task-accuracy gain over fixed-rank-r=6 baseline at iso-parameter-budget." This converts the experiment from "structural argmax distribution" to "behavioral utility of task-adaptive rank."
2. **Specify M(r, task) operationally**: choose either (a) downstream task accuracy (preferred, target-anchored) or (b) per-task PPL paired with a target-anchored downstream proxy. PPL alone insufficient per F#666 r≈0.08.
3. **Define rank grid and task set in pre-reg**: e.g. r ∈ {2, 4, 8, 16, 32}, tasks ∈ {GSM8K, MMLU, HumanEval} or similar — and freeze before training. Post-hoc rank-grid expansion is antipattern KC-after-data.
4. **Calibrate the 4× variance threshold against F#742 noise floor.** F#742 implies argmax_r at low rank on q_proj is noise-dominated (C_20=0.335); replication-style baseline (same task, same rank grid, multiple seeds) needed to establish noise-floor variance before claiming "4× across tasks" is signal.
5. **Or: subsume into `exp_g4_adapter_class_composition_full` K4 (rank-ablation)** once that parent reaches SUPPORTED — that design is target-paired and subsumes this question.

Re-register as `exp_g4_lora_rank_importance_per_task_v2` with the above corrections.

## §5 Follow-up

No `_impl` companion filed — preempt-structural kill is self-contained per F#666-pure precedent + reviewer.md §5. Recommended next action is target-paired re-register per §4.

## §6 Scope integrity

No silent objective swap (antipattern-t): this scaffold does NOT attempt:

- Substituting cosine similarity to teacher for `M(r, task)` and reporting argmax_r dispersion — would still be structural-only and still F#666-noncompliant; cos-sim is itself a proxy (F#720+F#755+F#756+F#757 cos-sim bucket precedent).
- Computing argmax_r on PPL alone and reporting "4× variance" as evidence of task-adaptive rank — substitutes PPL distribution structure for behavioral utility, weakly correlated per F#666.
- Picking a single task pair (e.g. medical vs code) post-hoc to maximize argmax-spread — antipattern KC-after-data.
- Reusing F#627 medical-only adapter as a stand-in for "all tasks" — single-task ≠ multi-task; would underdetermine rank-uniformity claim.
- Reading paper rank-ablation tables (e.g. LoRA paper, DoRA paper) and citing them as Gemma 4 stand-in — cross-architecture, cross-base-model extrapolation antipattern-m.

All five shortcuts would replace the behavioral utility claim with a proxy or substitute the architecture being measured.

## §7 Anti-pattern scan

- Composition-math: N/A (no composition).
- LORA_SCALE: N/A (no code).
- shutil.copy: N/A (no code).
- Hardcoded `"pass": True`: N/A (no code, `all_pass: false` written).
- Eval truncation producing base=0%: N/A (no eval).
- Proxy-model substitution: N/A (no code; would have used Gemma 4 E4B per F#627 if runnable).
- KC measures wrong object: K1941/K1942 measure argmax_r structural dispersion, NOT behavioral utility of task-adaptive rank — this IS the F#666-pure preempt rationale, not an antipattern in the scaffold.
- N=smoke reported as full: N/A (no N; `is_smoke: false`).
- Tautological routing: N/A (no routing).
- Thinking-mode truncation: N/A (no eval).
- File-existence cache: N/A (no code).
- KC-after-data: scaffold pre-registers preempt verdict before any data; no risk.
- Copy-paste scaffolding: scaffold derived from F#757 (`exp_g4_adapter_similarity_across_seeds` — closest sibling F#666-pure standalone in g4-ablation super-family) but variant-specific sections (rank-variance-bucket NEW form, 13th g4-ablation sub-type, F#742 prior-art anchor) rewritten, not copy-pasted.
