# MATH.md — exp_hedgehog_early_stopping_cos_plateau (PREEMPT-KILL, F#666-pure standalone, ~26th drain-window instance, 7th Hedgehog-ablation sub-type)

## Verdict: PREEMPT-KILL (KC-structural, F#666-pure standalone — Hedgehog-ablation 7th sub-type: training-stopping-criterion / early-stopping-ablation)

This experiment is preempt-killed before any code runs. The kill is **structural**: the pre-registered kill-criterion set K = {K1935, K1936} consists of two proxy metrics (inter-adapter cos-sim tightness + training-time savings ratio) with no paired target-metric KC. Under F#666 (guardrail 1007 — target-gated KILL discipline) neither KILL nor SUPPORTED is derivable regardless of empirical outcome.

This is a continuation of the F#666-pure standalone canonical pattern (F#700, F#701, F#703, F#705, F#706, F#707, F#708, F#710, F#711, F#714, F#715, F#716, F#720, F#722, F#728, F#729, F#730, F#731, F#732, F#734, F#735, F#736, F#753, F#754, F#755 — at least 25 prior). Specifically:
- **7th Hedgehog-ablation super-family sub-type**: after F#683-style axis-extension (1st), F#719/F#720 loss-variant-ablation (2nd), F#721 layer-selection-ablation (3rd), F#722 hyperparameter-ablation (4th: teacher-temperature), F#723 data-augmentation-ablation (5th), F#755 curriculum/training-procedure-ordering ablation (6th). This is the **1st training-stopping-criterion / early-stopping-ablation** sub-type instance — an *intra-training* stopping rule rather than an *inter-training* configuration sweep.
- **Closest precedents**: **F#722** (teacher-temperature sweep) and **F#755** (curriculum training) — both proxy-only Hedgehog-ablations on training-procedure axes, both preempt-killed. Same structural shape as this: training-procedure ablation with proxy-only KCs, no target-pair.
- **Sibling separator**: **F#723** (data-augmentation-ablation) avoided this kill by including K1877 = behavioral quality target paired with K1878 = cos-sim proxy → PROVISIONAL not killed.
- **Multi-bucket fire**: K1935 is the **3rd cos-sim-bucket form** (1st: F#720 final-value; 2nd: F#755 convergence-speed; 3rd: this — inter-training-stopping-point cos-sim *tightness/distance* form, "within 0.02 of full-training adapter"). K1936 is a **compute-cost/training-efficiency proxy** (training-time-savings ratio ≥ 30%) — adjacent to F#753's infrastructure-benchmark sub-flavor (routing latency) but on the *training* axis rather than *inference* axis.

## §0 Platform / skills / model pins

Included for reviewer checklist (m2) completeness. No platform code executes.
- Platform skills: `/mlx-dev` + `/fast-mlx` (per PLAN.md Part 2). **Not invoked** — no MLX code written.
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627). **Not loaded.**
- Adapter targets: N/A — Hedgehog cos-sim distillation per F#683 design (politeness behavior adapter, multi-layer cos-sim alignment to teacher); no LoRA injection or training in this run.
- Parent dependency: **none** (`depends_on: []`). NOT an F#669 preempt — although `exp_hedgehog_behavior_adapter_politeness_impl` is the implicit conceptual parent (open at P=1, status=open, never executed), the DB declares this experiment as standalone.

## §1 Preempt-KILL theorem (F#666-pure, training-stopping-criterion-ablation sub-flavor 1st instance)

**Theorem (KC-structural invalidity under target-gated KILL).** Let `E` denote experiment `exp_hedgehog_early_stopping_cos_plateau` with kill-criterion set K = {K1935, K1936}:
- K1935 := "Early-stopped adapter cos-sim within 0.02 of full-training adapter"
- K1936 := "Early stopping reduces training time < 30% (not significant)"

**Classification of K.**
- K1935 is a **proxy metric** — *inter-training-stopping-point* cos-sim tightness/distance. The Hedgehog framework operates on cos-sim against teacher per layer (Moudgil §3.1 baseline); K1935 measures the distance between two distillation outputs (early-stopped vs full-trained) on that *same* cos-sim space. Cos-sim-on-cos-sim is doubly proxy: the inner cos-sim (vs teacher) is a proxy for behavior (per F#666 + guardrail 1006, r≈0.08 PPL→task-quality); the outer "within 0.02" proximity bound is a proxy for *equivalence of adapter outputs*, not equivalence of adapter behavior. Two adapters can be near-cos-sim-identical yet diverge on LLM-judge politeness, MMLU subject preservation, or any downstream task (F#666 canonical: 40.2% per-sample classification acc + 0.0% target-gap shows cos-sim-equivalence and behavior-equivalence are separate axes). 3rd cos-sim-bucket form (after F#720 final-value, F#755 convergence-speed) — the *tightness/distance* form.
- K1936 is a **compute-efficiency proxy** — training-time-savings ratio. Training time is the wall-clock cost of producing the adapter; saving compute is desirable but does not measure *what the adapter does*. Without a behavioral target paired to K1936, "saves 30% compute" can be achieved trivially (stop after 0 steps; saves 100% compute; produces a useless adapter). The KC-text framing "< 30% (not significant)" reverses the polarity (kill if savings are *too small*) but does not introduce a behavioral floor. Compute-savings is the inference-side analog of routing latency (F#753 K1929 "any routing method > 10ms per query") — *infrastructure/efficiency* sub-flavor, but on the training axis.

Neither KC measures task accuracy, behavioral quality (politeness LLM-judge score per F#683 K1783), oracle-gap, generalization-gap on hard-tail (Hacohen-Weinshall 2019 arxiv:1904.03626 pattern), or any downstream-behavioral outcome. K is a 2-proxy, 0-target set.

**F#666 gating (guardrail 1007).** KILL requires **both** a failing proxy KC and a failing target KC. SUPPORTED requires **both** to pass. A verdict derived from a proxy-only KC set is tautological. Per F#714 / F#753 / F#754 / F#755 multi-proxy precedents, the analysis is per-KC then composed:

| K1935 | K1936 | V(K) under F#666                                                                              |
| ----- | ----- | --------------------------------------------------------------------------------------------- |
| PASS  | PASS  | Tautological SUPPORT — early-stopped adapter is "close enough" on cos-sim AND saves enough compute. But neither measures whether the adapter still works behaviorally on politeness. PASS-PASS could ship a cos-sim-tight, compute-cheap, behaviorally-degraded adapter. |
| PASS  | FAIL  | "Cos-sim tight but compute savings < 30%" — finding about training-curve plateau detection sensitivity, not a behavioral kill. Could indicate the plateau-detection threshold (50 steps in notes) fires too late. |
| FAIL  | PASS  | "Saves > 30% compute but cos-sim diverges > 0.02" — finding about over-aggressive early stopping, not a behavioral kill. Could be a *good* adapter via cluster-equivalence (F#666 canonical decoupling). |
| FAIL  | FAIL  | Both proxies disagree with the early-stopping hypothesis — still "finding about proxies, not kill" because no target measured. Early stopping could be cos-sim-distant yet behaviorally equivalent; or cos-sim-distant + low-compute-savings + behaviorally-superior (rare but possible: regularization-by-early-stopping, classic early-stopping rationale per Caruana et al. 2000). |

**No cell yields a valid F#666-compliant verdict.** K is unidentifiable at the F#666 layer. **QED.**

### §1.1 Cos-sim tightness ("within 0.02") inherits cos-sim-as-proxy

K1935 measures *‖cos_sim(early_stop, teacher) − cos_sim(full_train, teacher)‖_∞* (or analog) bounded by 0.02. Two pathological cases illustrate decoupling:

1. **K1935 PASS, behavior degraded**: early-stopped adapter is within 0.02 cos-sim of full-trained adapter on validation prompts but loses politeness coverage on hard-tail prompts (Hacohen-Weinshall 2019 generalization-gap pattern, arxiv:1904.03626). The plateau-detection metric saturates on easy prompts and stops training before the hard-tail signal arrives.
2. **K1935 FAIL, behavior preserved**: early-stopped adapter is > 0.02 cos-sim away from full-trained adapter (FAIL on K1935) but the *full-trained* adapter is over-fit to the teacher's idiosyncrasies, while the early-stopped adapter retains better generalization. Classic early-stopping-as-regularization rationale (Caruana, Lawrence, Giles 2000 NeurIPS; Prechelt 1998 "Early Stopping — But When?"). LLM-judge politeness equal or better at the early-stopped point. Cos-sim says "diverged"; behavior says "improved".

Both cases require a target-metric KC (LLM-judge politeness, behavioral generalization-gap on a held-out hard-tail) to disambiguate.

### §1.2 Training-time-savings is compute-cost-only proxy (1st training-efficiency-bucket form)

K1936 measures training-time savings ratio. Why is this not a target?
- Saving compute does not measure adapter quality. Stopping at iter=0 saves 100% compute; produces a noise adapter.
- The 30% threshold is arbitrary — neither anchored to an operational compute budget (e.g., "fits in a 30 min CI gate") nor to a behavioral floor (e.g., "saves N watt-hours while behavioral PASS").
- Training-time is the *production-side* analog of inference-time latency (F#753 K1929 "routing > 10ms"). The same F#666-pure pattern applies: efficiency without behavioral pairing is *infrastructure*, not *research*.
- F#702 precedent (latency + bitwise-exact equivalence) shows the canonical pattern for runnable efficiency metrics: pair the dynamic-process measurement with a behavioral invariant. K1936 has no such pair.

### §1.3 Plateau-detection circularity (cos-sim-on-cos-sim, training-instrumented)

The notes specify: "Monitor per-layer cos-sim during training. When it plateaus for 50 steps, stop." The stopping signal is itself the cos-sim convergence trace — cos-sim-driven stopping on a cos-sim-equivalence test (K1935). This is a **3rd cos-sim-on-cos-sim circularity**:
- F#755 (curriculum training, 6th Hedgehog-ablation) — cos-sim-difficulty-signal driving cos-sim-evaluation (cos-sim-on-cos-sim 1st instance, notes-level).
- F#NEW.3 (this) — cos-sim-plateau-stopping-signal driving cos-sim-equivalence-evaluation (cos-sim-on-cos-sim 2nd instance, *intra-training-trajectory*).
- F#720 (MSE loss-variant) — single cos-sim KC (no circularity, but same bucket).

Plateau detection on the same metric used for kill-criterion evaluation is a tighter version of F#755's circularity: F#755 used cos-sim difficulty to *order* training data, with cos-sim used at evaluation. This experiment uses cos-sim plateau to *halt* training, with cos-sim distance used at evaluation. The signal that decides when to stop is the same signal that decides whether the stopping was good — a near-tautology that can only be broken by a behavioral target outside the cos-sim space.

## §2 Prior art (preempt-KILL precedents and Hedgehog-ablation taxonomy)

- **F#666** (2026-04-19, conclusive): target-gated KILL discipline; guardrail 1007 enumerates classification accuracy, routing match rate, *PPL*, *cosine* explicitly as forbidden-solo proxies; canonical 40.2% proxy + 0.0% target gap.
- **F#683** (Hedgehog politeness behavior adapter — design): 5 target-gated KCs pre-registered (K1782 structural proxy paired with K1783 politeness-judge target). PROVISIONAL — implementation deferred. Defines the Hedgehog framework's behavioral target metric (LLM-judge politeness score) that this early-stopping experiment fails to inherit.
- **F#719** (Hedgehog cos-sim-vs-KL-div loss-variant): K1870 proxy + K1871 behavioral target → PROVISIONAL. **1st loss-variant Hedgehog-ablation sub-type**.
- **F#720** (Hedgehog MSE loss-variant): K1872 cos-sim only → killed triple-fire. **1st cos-sim-bucket (intra-loss-function-delta sub-variant), final-value form**.
- **F#721** (Hedgehog layer-selection-ablation): triple-fire preempt-KILL. **3rd Hedgehog-ablation sub-type: layer-selection-ablation**.
- **F#722** (Hedgehog teacher-temperature sweep): triple-fire preempt-KILL. **4th Hedgehog-ablation sub-type: hyperparameter-ablation**. Closest *inter-training-config* sibling.
- **F#723** (Hedgehog data-augmentation-ablation): K1877 target + K1878 proxy → PROVISIONAL. **5th sub-type: data-augmentation-ablation**. Demonstrates target-pair-runnable design.
- **F#755** (Hedgehog curriculum training, 2026-04-25): F#666-pure preempt-KILL. **6th sub-type: curriculum / training-procedure-ordering ablation**. 2nd cos-sim-bucket (convergence-speed form). 1st cos-sim-on-cos-sim notes-level circularity. Closest *intra-training-procedure* sibling.
- **F#754** (2026-04-25): 24th F#666-pure standalone, 4th routing-acc + 3rd infra-bench multi-bucket. Adjacent infrastructure-benchmark precedent.
- **F#753** (2026-04-25): 23rd F#666-pure standalone, 3rd routing-acc + 2nd infra-bench multi-bucket. Routing-latency precedent for the K1936 efficiency-bucket form (inference-side).
- **`mem-antipattern-f666-pure-standalone-preempt-kill`** (filed 2026-04-24, multiple escalations): claim-time detection rule; preempt-scaffold response.
- **Guardrail 1007** (PLAN.md): every proxy KC must be paired with a target-metric KC.
- **Guardrail 1006** (PLAN.md): "PPL does not predict task quality in this project (measured r≈0.08). Behavioral outcomes over metrics."

External early-stopping prior art (none cited in DB pre-reg, guardrail 1002 violation):
- Caruana, Lawrence, Giles 2000 (NeurIPS) "Overfitting in neural nets: Backpropagation, conjugate gradient, and early stopping" — classic early-stopping-as-regularization formulation; uses *held-out validation loss* as the stopping signal AND independently measures behavioral generalization on a test set (target-pair design).
- Prechelt 1998 "Early Stopping — But When?" — formal definition of plateau-detection variants (GL, PQ, UP); compares stopping criteria *with* behavioral generalization measurement.
- Wu et al. 2021 (arxiv:2010.13166) "Curriculum Learning for Knowledge Distillation" — distillation-context training-procedure ablation including student-task accuracy (target metric), runnable design template.

The runnable pattern from these precedents: stopping criterion (proxy) + behavioral generalization measurement (target). This experiment has only the proxy.

## §3 Predictions (registered, not measured)

| KC    | Claim                                                                           | Kind  | Sub-flavor                                                                          | Measurement status         |
| ----- | ------------------------------------------------------------------------------- | ----- | ----------------------------------------------------------------------------------- | -------------------------- |
| K1935 | Early-stopped adapter cos-sim within 0.02 of full-training adapter              | proxy | cos-sim-bucket tightness/distance form (3rd cos-sim-bucket instance)                | untested (preempt-blocked) |
| K1936 | Early stopping reduces training time < 30% (not significant)                    | proxy | training-efficiency / compute-cost (1st training-efficiency-bucket form, adjacent to F#753 inference-latency-bucket) | untested (preempt-blocked) |

No target-metric KC exists. K is structurally malformed per F#666.

KC text preserved verbatim from `experiment get exp_hedgehog_early_stopping_cos_plateau` output. No post-claim KC mutation (antipattern-u check: PASS).

## §4 Hygiene defects (noted, not load-bearing for kill)

Per `experiment get exp_hedgehog_early_stopping_cos_plateau`:

1. **`success_criteria: NONE`** — no SUPPORTED-condition declared (DB explicitly flags `⚠ INCOMPLETE: success_criteria`).
2. **`platform: —`** (null) — guardrail/hygiene defect (DB flags `⚠ INCOMPLETE: ... platform`).
3. **`references: []`** — guardrail 1002 violation (every new experiment MUST cite an arxiv paper or prior finding). Early-stopping has substantial prior art (Caruana et al. 2000 NeurIPS; Prechelt 1998; Hacohen-Weinshall 2019 arxiv:1904.03626 generalization framework; Wu et al. 2021 arxiv:2010.13166 distillation-context) — none cited.
4. **`experiment_dir: —`** (null until this iteration created the dir).

Four hygiene defects total. Crosses the AP-prereg-hygiene-multi-defect threshold (≥3 defects). However, F#666-pure structural defect alone is sufficient for kill independent of hygiene count (per F#703 invariant; same shape as F#722, F#754, F#755).

Notes field reads: "Monitor per-layer cos-sim during training. When it plateaus for 50 steps, stop. Reduces compute waste." — cos-sim-driven plateau detection on the same metric used for KC evaluation; reinforces F#666 violation (cos-sim-on-cos-sim circularity at the *intra-training-trajectory* level). The 50-step plateau threshold is unanchored — neither motivated by a published criterion (Prechelt PQ/UP variants) nor calibrated against a behavioral floor.

## §5 Unblock condition (re-claim requires KC-augmentation pre-registration)

Re-registration as a new experiment id (`exp_hedgehog_early_stopping_cos_plateau_behavioral` recommended) with the following fixes:

1. **Add a target-metric KC** pairing early-stopping-vs-full-training to a behavioral outcome on the parent Hedgehog axis. Candidate formulations:
   - **LLM-judge politeness equivalence target** (F#683 K1783 pattern, F#723 K1877 pattern): early-stopped adapter LLM-judge politeness score ≥ full-trained adapter politeness − 1pp on F#683 prompt set. Couples early-stopping claim to behavior. If this passes while K1935 fails, that's a finding (early stopping *regularizes* — behavior preserved despite cos-sim divergence). If both pass, early stopping is an honest compute-saver.
   - **Generalization-gap target** (Hacohen-Weinshall 2019 framework): early-stopped adapter shows ≤2pp degradation vs full-trained on held-out hard-tail prompts. Tests the early-stopping-as-regularization claim (Caruana et al. 2000).
   - **Pareto-quality target**: among early-stopping schedules that PASS K1935, best schedule's LLM-judge politeness ≥ full-trained − 0pp AND saves ≥30% compute. Cos-sim and compute-savings become *constraints*; behavior is the *verdict*.
2. **Add references**: F#666 (guardrail), F#683 (Hedgehog politeness target), F#722 (hyperparameter-ablation preempt), F#723 (data-augmentation runnable example), F#755 (curriculum-ablation preempt — sibling on training-procedure axis), Caruana et al. 2000 NeurIPS, Prechelt 1998, arxiv:1904.03626 (Hacohen-Weinshall 2019), arxiv:2010.13166 (Wu et al. 2021). Address guardrail 1002.
3. **Set `platform=local-apple`** (currently null).
4. **Populate `success_criteria`** mirroring the new target-metric PASS condition.
5. **Tighten notes**: state the *behavioral* outcome (LLM-judge politeness on hard tail) and decouple the stopping signal from the evaluation signal (e.g., stop on held-out validation loss plateau per Caruana 2000, evaluate on LLM-judge politeness — not cos-sim-on-cos-sim).
6. **Wait for parent F#683 (Hedgehog politeness adapter) to graduate from PROVISIONAL to SUPPORTED** before running early-stopping ablation. F#669-style child-on-unverified-parent risk: comparing early-stopped vs full-trained of an *unverified base method* has no anchor. Early-stopping vs full-training is meaningful only if full-training itself produces a working adapter; F#683 hasn't measured that yet.
7. **Anchor the 50-step plateau threshold**: replace with a published criterion (Prechelt PQ_α or GL_α with calibrated α) or pre-register a sweep over plateau-window sizes paired with a behavioral target.

Post-claim KC mutation is antipattern-u; edits must happen **before** re-claim. Recommendation: **close this pre-reg as structurally-malformed**; re-register `exp_hedgehog_early_stopping_cos_plateau_behavioral` after F#683 supported.

### §5.1 Pre-existing partial coverage

Prior findings provide partial coverage of training-stopping-criterion ablations:
- **F#722**: hyperparameter-ablation 4th sub-type — teacher-temperature sweep is a continuous *inter-training-config* hyperparameter ablation. Both are training-procedure ablations on Hedgehog distillation; F#722's preempt directly informs this one. Sibling on inter-config axis.
- **F#755**: curriculum-ordering 6th sub-type — *intra-training* training-procedure ablation. Closest sibling on procedural axis.
- **F#723**: data-augmentation 5th sub-type with target-pair runnable design. The K1877+K1878 design is the template the early-stopping experiment should adopt.
- **F#719/F#720**: loss-variant series shows that target-pair design avoids preempt (F#719), cos-sim-only design does not (F#720). Same lesson for early-stopping.
- **Caruana, Lawrence, Giles 2000** (NeurIPS): the canonical academic early-stopping ablation framework. Their evaluation includes test-set generalization (target metric), not just held-out-loss plateau (proxy).
- **Prechelt 1998**: formal plateau-detection criteria (GL, PQ, UP) — anchors the otherwise-arbitrary 50-step plateau threshold in the notes.
- **arxiv:2010.13166** (Wu et al., "Curriculum Learning for Knowledge Distillation"): distillation-specific training-procedure prior art with task-accuracy target.

Early-stopping-for-distillation is a real research question with substantial published prior art; this experiment's failure is structural (KC design + reference omission), not topical.

## §6 Follow-up

No `_impl` companion filed — preempt-structural KILL does NOT spawn `_impl` (per F#687/F#698/F#699/F#700/F#701/F#703/F#705/F#710/F#714/F#722/F#753/F#754/F#755 precedent + reviewer.md §5). Unblock is pre-registration-external (edit the DB entry to add target-pair KC + references + platform + behavioral signal-evaluation decoupling), not implementation-external.

`mem-antipattern-impl-follow-up-delegation` does not apply: that antipattern targets novel-mechanism PROVISIONAL, not preempt-structural KILL.
