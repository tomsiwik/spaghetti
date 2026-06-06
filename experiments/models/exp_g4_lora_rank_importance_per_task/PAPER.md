# PAPER.md — exp_g4_lora_rank_importance_per_task

## Verdict: KILLED (preempt, F#666-pure standalone — ~29th drain-window)

This experiment was preempt-killed before any MLX code was written. The kill is structural, not empirical: the pre-registered KC set {K1941 (rank-uniformity), K1942 (rank-variance-ratio > 4×)} is **proxy-only with no behavioral target pair**, while the experiment is **standalone** (`depends_on=[]`). Per Finding #666 + guardrail 1007 (TARGET-GATED KILL), KILL on a proxy-only KC set is forbidden.

This is the **~29th F#666-pure standalone preempt-KILL** in the drain window, the **13th g4-ablation super-family sub-type** (rank-importance-per-task = NEW; rank-ablation sub-type), and the **1st rank-variance-bucket form** (structural-hyperparameter argmax-divergence; NEW proxy-bucket form distinct from prior cos-sim / routing-acc / infra-bench / training-axis-efficiency buckets).

## Prediction vs measurement

| KC    | Prediction                                                              | Kind  | Measurement  | Verdict   |
| ----- | ----------------------------------------------------------------------- | ----- | ------------ | --------- |
| K1941 | All tasks need same rank (uniform argmax_r M(r, task))                   | proxy | not measured | untested  |
| K1942 | argmax_r variance > 4× across tasks (one-size-fits-all rank wasteful)   | proxy | not measured | untested  |

**K1941 is "not measured" and structurally unidentifiable as a finding** because rank-uniformity is a *dimensionless structural statistic* on argmax_r M(r, task) — it inherits the proxy/target status of M, but the index-of-maximum operation strips behavioral information even when M is a target metric. PASS (uniform) does not imply "task-adaptive rank is unnecessary" behaviorally; flat M(r) curves with noise picking ranks (F#742 prior: rank-6 q_proj has C_20=0.335 weak concentration → argmax_r is noise-dominated) produce the same PASS without any task-uniformity signal.

**K1942 is "not measured" and similarly unidentifiable** because the 4× threshold is unanchored against:
1. Common LoRA rank grids `{1, 2, 4, 8, 16, 32}` are themselves geometric ×2; max/min trivially spans 2× to 32× without any task-structure signal.
2. F#742 noise floor at low rank on q_proj — argmax_r is dominated by sampling noise rather than task structure.
3. F#143 prior: 4× rank capacity ratio (32 vs 8) does NOT translate to quality advantage at d=256 (magnitude dilution + data homogeneity cancel).

**The KC pair is internally XOR-inconsistent**: K1941 (uniform argmax) and K1942 (variance > 4×) cannot both PASS — variance > 4× requires non-uniform argmax. The 4-cell truth table contains 1 contradictory cell, 2 inconclusive cells, and 1 "proxy-PASS without target — F#666 forbidden" cell. **No reachable cell produces a behaviorally-anchored finding.**

| K1941        | K1942        | Cell interpretation                                              | Verdict          |
| ------------ | ------------ | ---------------------------------------------------------------- | ---------------- |
| PASS (uniform) | PASS (>4×)   | CONTRADICTORY (uniform vs variance > 4× cannot coexist)           | **degenerate**   |
| PASS (uniform) | FAIL (≤4×)   | Uniform OR small spread — flat M(r) curves likely; behavioral utility unknown | **inconclusive** |
| FAIL (varied)  | FAIL (≤4×)   | argmax differs but spread ≤ 4× — could be 2× noise floor         | **inconclusive** |
| FAIL (varied)  | PASS (>4×)   | Researcher's hoped-for case — argmax differs widely, but proxy-PASS without target | **F#666 forbidden** |

Substituting cosine similarity to teacher for M (cos-sim bucket form) would still be structural-only; substituting PPL alone for M would substitute a weakly-correlated proxy (F#666 r≈0.08 PPL↔task-quality); picking task pairs post-hoc to maximize argmax-spread would be antipattern KC-after-data; reading paper rank-ablation tables (LoRA, DoRA) as Gemma 4 stand-ins would be cross-architecture extrapolation antipattern-m. Each shortcut would replace the missing target pair with a different proxy or substitute the architecture being measured.

## Assumptions

- F#742 (provisional) shows rank-6 q_proj LoRA on Gemma 4 E4B has weak structural concentration (C_20=0.335) — argmax_r at low rank is noise-dominated. This makes K1942's 4× variance threshold a sampling-noise threshold rather than a task-structure threshold without explicit noise-floor calibration.
- F#143 (killed) shows rank-capacity scaling is non-linear — a "rank optimum" may not exist for some task/architecture combinations. K1941's premise (a per-task rank sweet spot exists) is itself contestable before the KC fires.
- F#522 (supported) shows rank importance is loss-formulation-dependent (TT-LoRA r=6 + MCQ classification loss → +14.5pp discriminative capacity). K1941/K1942 don't disentangle this confound; "the rank task X needs" depends on which loss is computed.
- F#686 (provisional) `exp_g4_adapter_class_composition_full` K4 (rank-ablation) is the target-paired design that subsumes the question once parent SUPPORTED.
- F#666 gating: K1941 + K1942 are both proxies (rank-argmax structural statistics); no target paired. Standalone (`depends_on=[]`) provides no parent target to inherit. Forbidden-solo per guardrail 1007.
- Pre-reg defects beyond KC kind: M (the per-task quality metric) not specified; rank grid not specified; task set not specified. KC verdict flips on choice of any of the three — antipattern KC-after-data risk if filled in post-hoc. (This experiment is preempt-killed without measurement, so the antipattern does not actually fire.)

## Pattern continuation — F#666-pure standalone canonical 29th instance + NEW rank-variance-bucket

This is the **~29th F#666-pure standalone preempt-KILL** in the drain window:

| Cluster bucket                                  | Prior instances                                                                  |
| ----------------------------------------------- | -------------------------------------------------------------------------------- |
| Routing-accuracy bucket                         | F#703, F#710, F#736, F#754                                                        |
| Cosine-similarity bucket                        | F#720 (final-value), F#755 (convergence-speed), F#756 (tightness), F#757 (cross-instance dual-tail) |
| Infrastructure-benchmark bucket                 | F#714, F#715, F#736, F#753, F#754                                                 |
| Training-axis efficiency bucket                 | F#756 (training-time-vs-quality)                                                  |
| Hedgehog-ablation super-family (saturated at 7) | F#722, F#755, F#756, ... (saturated at F#756)                                     |
| g4-ablation super-family (12 sub-types)         | ..., F#723 (init-method), F#757 (seed-determinism)                                |
| **(this) Structural-hyperparameter argmax-divergence (NEW)** | **F#759 — rank-uniformity + rank-variance-ratio (1st obs of the bucket)** |

The new bucket is structurally distinct from prior buckets: it operates on the *index-of-maximum* of a quality metric across a hyperparameter grid, rather than on the metric value itself. Index-of-maximum is dimensionless and inherits the bucket's forbidden-solo property even if M is target-anchored, because the *argmax* operation discards the magnitude-of-improvement signal that target-anchoring would carry.

g4-ablation super-family sub-type tally: 13 (per-layer-cos, PPL-drift, canary-FNR, routing-collision, hash-ring-PPL, routing-family, gumbel-routing, perturbation-stability, SVD-rank-delta, SVD-denoise-PPL, init-method-comparison F#723, seed-determinism F#757, **rank-importance-per-task (this)**). Super-family does NOT saturate — different sub-types address structurally different failure modes (unlike Hedgehog-ablation which saturated at 7 because sub-types covered all reachable training-procedure axes).

## Sub-axis classification (proxy bucket; new form)

| Sub-axis                                              | Status at this iteration                                                                                |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| F#666-pure standalone running tally                   | ~29 (this is the next instance)                                                                          |
| g4-ablation super-family sub-types                     | 13 (this is 13th — rank-importance-per-task, NEW)                                                       |
| Cos-sim-bucket form count                              | 4 (no advance)                                                                                          |
| Routing-acc-bucket form count                          | 4 (no advance)                                                                                          |
| Infra-bench-bucket form count                          | 5 (no advance)                                                                                          |
| Training-axis efficiency-bucket form count             | 1 (no advance)                                                                                          |
| **Structural-hyperparameter argmax-divergence bucket** | **1 (NEW; this is 1st obs)**                                                                            |

## Related

- **F#666** — defining target-gated KC discipline.
- **Guardrail 1007** — TARGET-GATED KILL: never kill on proxy alone.
- **F#742** (provisional, direct anchor) — rank-6 q_proj LoRA on Gemma 4 E4B has weak structural concentration; argmax_r noise-dominated.
- **F#143** (killed) — rank-capacity scaling non-linear at d=256; "rank optimum" may not exist.
- **F#522** (supported) — rank importance is loss-formulation-dependent.
- **F#686** (provisional) — `exp_g4_adapter_class_composition_full` K4 (rank-ablation) is the target-paired version that subsumes this question once parent SUPPORTED.
- **F#723** — 11th g4-ablation sub-type (init-method-comparison) and target-paired template for v2.
- **F#757** — 12th g4-ablation sub-type (seed-determinism); also F#666-pure preempt-KILL precedent.
- **F#735** — closest prior on "variance-style proxy-bucket" sub-flavor.
- LoRA paper (Hu et al., arxiv:2106.09685) — original rank-utility analysis on GPT-3 fine-tuning. Cross-architecture; cannot anchor Gemma 4 E4B argmax-distribution.
- DoRA paper (Liu et al., arxiv:2402.09353) — rank-ablation comparison vs LoRA. Cross-architecture; same anchoring limitation.

## Unblock path

Re-claim this experiment when one of:

**Path A (target-paired re-register):** File `exp_g4_lora_rank_importance_per_task_v2` with:
1. K1941 + K1942 retained as structural proxies.
2. NEW K1943 (target): "Task-adaptive rank allocation gives ≥ 5pp downstream task-accuracy gain over fixed-rank-r=6 baseline at iso-parameter-budget."
3. M(r, task) operationally specified — downstream task accuracy preferred (target-anchored).
4. Rank grid + task set frozen in pre-reg (e.g. r ∈ {2, 4, 8, 16, 32}, tasks ∈ {GSM8K, MMLU, HumanEval}).
5. 4× variance threshold calibrated against F#742 noise floor via replication-style baseline (same task, same rank grid, multiple seeds — establish variance(argmax_r | task fixed, seed varies) before claiming variance(argmax_r | task varies) > 4× is task-signal).

**Path B (subsume into parent):** `exp_g4_adapter_class_composition_full` (F#686 provisional) reaches SUPPORTED at parent's `_impl`; K4 of that experiment is rank-ablation pair-anchored and subsumes the question. No new child needed.

## Follow-up filed

None — preempt-structural kill does not spawn an `_impl` companion. Recommended next action: close this pre-reg and choose Path A (v2 re-register) or Path B (subsume into F#686 K4).
