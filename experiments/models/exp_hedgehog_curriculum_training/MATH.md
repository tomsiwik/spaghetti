# MATH.md — exp_hedgehog_curriculum_training (PREEMPT-KILL, F#666-pure standalone, ~25th drain-window instance, 6th Hedgehog-ablation sub-type)

## Verdict: PREEMPT-KILL (KC-structural, F#666-pure standalone — Hedgehog-ablation 6th sub-type: curriculum / training-procedure-ablation)

This experiment is preempt-killed before any code runs. The kill is **structural**: the pre-registered kill-criterion set K = {K1933, K1934} consists of two proxy metrics (relative cos-sim adapter quality + cos-sim convergence speed) with no paired target-metric KC. Under F#666 (guardrail 1007 — target-gated KILL discipline) neither KILL nor SUPPORTED is derivable regardless of empirical outcome.

This is a continuation of the F#666-pure standalone canonical pattern (F#700, F#701, F#703, F#705, F#706, F#707, F#708, F#710, F#711, F#714, F#715, F#716, F#720, F#722, F#728, F#729, F#730, F#731, F#732, F#734, F#735, F#736, F#753, F#754 — at least 24 prior). Specifically:
- **6th Hedgehog-ablation super-family sub-type**: after F#683-style axis-extension (1st), F#719/F#720 loss-variant-ablation (2nd), F#721 layer-selection-ablation (3rd), F#722 hyperparameter-ablation (4th: teacher-temperature), F#723 data-augmentation-ablation (5th). This is the **1st curriculum / training-procedure-ablation** instance — categorical/procedural rather than scalar hyperparameter.
- **Closest precedent**: **F#722** (Hedgehog teacher-temperature sweep) — both KCs proxy (cos-sim derived), killed as triple-fire preempt-KILL (F#666-pure + §5 intra-Hedgehog-temperature-delta + hygiene). Same structural shape: training-procedure ablation with proxy-only KCs, no target-pair.
- **Sibling separator**: **F#723** (data-augmentation-ablation) avoided this kill by including K1877 = behavioral quality target paired with K1878 = cos-sim proxy → PROVISIONAL not killed. Demonstrates that Hedgehog-ablation experiments with target-pair KCs are runnable; this one is not.

## §0 Platform / skills / model pins

Included for reviewer checklist (m2) completeness. No platform code executes.
- Platform skills: `/mlx-dev` + `/fast-mlx` (per PLAN.md Part 2). **Not invoked** — no MLX code written.
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627). **Not loaded.**
- Adapter targets: N/A — Hedgehog cos-sim distillation per F#683 design (politeness behavior adapter, multi-layer cos-sim alignment to teacher); no LoRA injection or training in this run.
- Parent dependency: **none** (`depends_on: []`). NOT an F#669 preempt — although `exp_hedgehog_behavior_adapter_politeness_impl` is the implicit conceptual parent (open at P=1, status=open, never executed), the DB declares this experiment as standalone.

## §1 Preempt-KILL theorem (F#666-pure, training-procedure-ablation sub-flavor 1st instance)

**Theorem (KC-structural invalidity under target-gated KILL).** Let `E` denote experiment `exp_hedgehog_curriculum_training` with kill-criterion set K = {K1933, K1934}:
- K1933 := "Curriculum training produces adapter > 3pp worse than random-order training"
- K1934 := "Curriculum training cos-sim convergence < random-order (worse)"

**Classification of K.**
- K1933 is a **proxy metric** — relative adapter quality delta. The Hedgehog framework operates on *cos-sim against teacher per layer* (Moudgil §3.1 baseline), so "adapter quality" at the experiment level defaults to cos-sim-against-teacher. Even if interpreted as PPL, PLAN.md guardrail 1006 establishes r≈0.08 PPL→task-quality correlation in this codebase: PPL is itself a proxy. Delta-of-proxies = proxy (per F#754 §1.1 invariant). No coupling to behavioral outcome (Hedgehog politeness target = LLM-judge politeness score, per F#683 K1783; not measured here).
- K1934 is a **proxy metric** — training-curve cos-sim convergence speed. Direct cos-sim measurement during training, explicitly cos-sim per KC text. F#720 precedent: a sole cos-sim KC at the cos-sim-bucket level was killed (1st cos-sim-bucket instance). K1934 inherits the 1st-cos-sim-bucket kill basis.

Neither KC measures task accuracy, behavioral quality (politeness LLM-judge score), oracle-gap, or any downstream-behavioral outcome. K is a 2-proxy, 0-target set.

**F#666 gating (guardrail 1007).** KILL requires **both** a failing proxy KC and a failing target KC. SUPPORTED requires **both** to pass. A verdict derived from a proxy-only KC set is tautological. Per F#714 / F#753 / F#754 multi-proxy precedents, the analysis is per-KC then composed:

| K1933 | K1934 | V(K) under F#666                                                                              |
| ----- | ----- | --------------------------------------------------------------------------------------------- |
| PASS  | PASS  | Tautological KILL — both proxies say curriculum is worse, but neither measures behavior. Curriculum could yield a worse cos-sim final adapter that nonetheless produces equivalent or better LLM-judge politeness via cluster-equivalence (F#666 canonical: 40.2% per-sample acc + 0.0% target gap). PASS-PASS would discard a potentially behavior-preserving training procedure. |
| PASS  | FAIL  | Mixed proxy outcome; F#666 rule "Proxy-FAIL + target-absent = finding about the proxy, not a kill" produces a finding about cos-sim convergence behavior, not a behavioral kill. |
| FAIL  | PASS  | Mixed proxy outcome; same finding-not-kill rule applies on the relative adapter quality proxy. |
| FAIL  | FAIL  | Both-fail proxy-only — under F#666 still "finding about proxies, not kill" because no target was measured. Curriculum could be cos-sim-equal-or-better yet behaviorally regress (e.g., over-fits early-easy examples and loses politeness diversity). |

**No cell yields a valid F#666-compliant verdict.** K is unidentifiable at the F#666 layer. **QED.**

### §1.1 Curriculum-vs-random delta is still a cos-sim/PPL proxy delta (1st curriculum-ablation instance)

K1933 measures *Δ(curriculum_quality, random_quality)* in cos-sim or PPL units. Two pathological cases illustrate decoupling:

1. **Δ ≤ 3pp PASS, behavior degraded**: curriculum order over-emphasizes easy (high-similarity) examples in early training, producing a high-cos-sim final adapter that reproduces teacher's *style* on easy prompts but loses behavioral coverage on hard ones. Cos-sim says "good"; LLM-judge politeness drops 5pp on the hard tail.
2. **Δ > 3pp FAIL, behavior preserved**: curriculum order yields slightly lower per-layer cos-sim but produces a more *generalizable* policy because hard examples late in training are seen with already-aligned base. LLM-judge politeness equal or better. Per F#723 data-augmentation precedent (target-pair design preserved the experiment), this is the regime where the procedure is actually useful.

Both cases require a target-metric KC (LLM-judge politeness, MMLU subject preservation, oracle-gap on a politeness benchmark) to disambiguate.

### §1.2 Cos-sim convergence speed is training-curve diagnostic, not behavior (1st curriculum-form of cos-sim-bucket re-instance)

K1934 measures rate of cos-sim improvement during training. Why is this not a target?
- Faster cos-sim convergence does not imply better generalization; could be over-fitting to the easy-first regime (degenerate to memorizing the easy block).
- Slower cos-sim convergence might indicate the curriculum is fighting the optimizer's natural preference, but the *final* adapter could still be behavior-preserving.
- F#720 precedent: K1872 (cos-sim only, MSE loss-variant) was preempt-killed as 1st cos-sim-bucket instance. K1934 is an intra-Hedgehog-cos-sim-bucket continuation: 2nd cos-sim-bucket instance of the form "cos-sim-during-training", but extracting *speed* rather than *final value*. Speed-of-cos-sim inherits cos-sim-as-proxy.

The F#702 precedent (latency + bitwise-exact equivalence) shows the canonical pattern for runnable training-curve metrics: pair the dynamic-process measurement with a behavioral invariant. K1934 has no such pair.

## §2 Prior art (preempt-KILL precedents and Hedgehog-ablation taxonomy)

- **F#666** (2026-04-19, conclusive): target-gated KILL discipline; guardrail 1007 enumerates classification accuracy, routing match rate, *PPL*, *cosine* explicitly as forbidden-solo proxies; canonical 40.2% proxy + 0.0% target gap.
- **F#683** (Hedgehog politeness behavior adapter — design): 5 target-gated KCs pre-registered (K1782 structural proxy paired with K1783 politeness-judge target; K1784 non-interference target; K1785 teacher-prompt ablation). PROVISIONAL — implementation deferred. Defines the Hedgehog framework's behavioral target metric (LLM-judge politeness score) that this curriculum experiment fails to inherit.
- **F#719** (Hedgehog cos-sim-vs-KL-div loss-variant): K1870 proxy + K1871 behavioral target → PROVISIONAL design-locked. **1st loss-variant Hedgehog-ablation sub-type**.
- **F#720** (Hedgehog MSE loss-variant): K1872 cos-sim only → killed triple-fire. **1st cos-sim-bucket (intra-loss-function-delta sub-variant)**.
- **F#721** (Hedgehog layer-selection-ablation): triple-fire preempt-KILL. **3rd Hedgehog-ablation sub-type: layer-selection-ablation** (cousin of axis-extension, loss-variant).
- **F#722** (Hedgehog teacher-temperature sweep): triple-fire preempt-KILL (F#666-pure + §5 intra-Hedgehog-temperature-delta + hygiene). **4th Hedgehog-ablation sub-type: hyperparameter-ablation** (cousin of layer-selection-ablation). KCs: both proxy. Closest structural sibling to this experiment.
- **F#723** (Hedgehog data-augmentation-ablation): K1877 target + K1878 proxy → PROVISIONAL. **5th Hedgehog-ablation sub-type: data-augmentation-ablation**. Demonstrates that Hedgehog-ablation experiments WITH target-pair KCs are runnable; F#722 vs F#723 is the canonical separator.
- **F#754** (2026-04-25): 24th F#666-pure standalone, 4th routing-acc + 3rd infra-bench multi-bucket. Most recent drain-window precedent.
- **`mem-antipattern-f666-pure-standalone-preempt-kill`** (filed 2026-04-24, escalated multiple times): claim-time detection rule; preempt-scaffold response.
- **Guardrail 1007** (PLAN.md): every proxy KC must be paired with a target-metric KC.
- **Guardrail 1006** (PLAN.md): "PPL does not predict task quality in this project (measured r≈0.08). Behavioral outcomes over metrics."

## §3 Predictions (registered, not measured)

| KC    | Claim                                                                            | Kind  | Sub-flavor                                                  | Measurement status                |
| ----- | -------------------------------------------------------------------------------- | ----- | ----------------------------------------------------------- | --------------------------------- |
| K1933 | Curriculum training produces adapter > 3pp worse than random-order training      | proxy | curriculum/training-procedure-ablation cos-sim/PPL delta    | untested (preempt-blocked)        |
| K1934 | Curriculum training cos-sim convergence < random-order (worse)                   | proxy | cos-sim-bucket convergence-speed (2nd cos-sim-bucket instance after F#720 final-value) | untested (preempt-blocked) |

No target-metric KC exists. K is structurally malformed per F#666.

KC text preserved verbatim from `experiment get exp_hedgehog_curriculum_training` output. No post-claim KC mutation (antipattern-u check: PASS).

## §4 Hygiene defects (noted, not load-bearing for kill)

Per `experiment get exp_hedgehog_curriculum_training`:

1. **`success_criteria: NONE`** — no SUPPORTED-condition declared (DB explicitly flags `⚠ INCOMPLETE: success_criteria`).
2. **`platform: —`** (null) — guardrail/hygiene defect (DB flags `⚠ INCOMPLETE: ... platform`).
3. **`references: []`** (none in DB output) — guardrail 1002 violation (every new experiment MUST cite an arxiv paper or prior finding). Curriculum learning has substantial prior art (Bengio et al. 2009 arxiv:0903.0738; Hacohen & Weinshall 2019 arxiv:1904.03626; Wu et al. 2021 arxiv:2010.13166 for distillation specifically) — none cited.
4. **`experiment_dir: —`** (null until this iteration created the dir).

Four hygiene defects total. Crosses the AP-prereg-hygiene-multi-defect threshold (≥3 defects). However, F#666-pure structural defect alone is sufficient for kill independent of hygiene count (per F#703 invariant; same shape as F#722, F#754).

Notes field reads: "Order training examples by teacher-student divergence. Start with easy (similar), end with hard (divergent). Tests curriculum learning." — describes a procedural ablation of training data ordering. The notes reify cos-sim against teacher as both the curriculum-difficulty signal AND the evaluation signal — making this a *cos-sim-driven curriculum on a cos-sim metric*, which is structurally tautological at the cos-sim layer (F#666 violation in framing reinforced).

## §5 Unblock condition (re-claim requires KC-augmentation pre-registration)

Re-registration as a new experiment id (`exp_hedgehog_curriculum_training_behavioral` recommended) with the following fixes:

1. **Add a target-metric KC** pairing curriculum-vs-random training to a behavioral outcome on the parent Hedgehog axis. Candidate formulations:
   - **LLM-judge politeness score delta** (F#683 K1783 pattern): curriculum-trained adapter LLM-judge politeness ≥ random-trained adapter LLM-judge politeness − 1pp at fixed prompt set. Couples curriculum claim to behavior; matches F#723 data-augmentation precedent.
   - **Generalization gap target**: curriculum-trained adapter on held-out hard prompts shows ≤2pp degradation vs random-trained. Tests the curriculum-learning literature claim (Bengio 2009, arxiv:0903.0738) directly.
   - **Pareto-quality target**: among curriculum schedules that PASS K1933 (cos-sim non-regression), best schedule's LLM-judge politeness ≥ random − 0pp. Cos-sim becomes a *constraint*, behavioral is the *verdict*.
2. **Add references**: F#666 (guardrail), F#683 (Hedgehog politeness target), F#722 (hyperparameter-ablation preempt), F#723 (data-augmentation-ablation runnable example), arxiv:0903.0738 (Bengio curriculum learning), arxiv:1904.03626 (Hacohen-Weinshall power of curriculum), arxiv:2010.13166 (Wu et al. curriculum-for-distillation). Address guardrail 1002.
3. **Set `platform=local-apple`** (currently null; DB hygiene fix).
4. **Populate `success_criteria`** mirroring the new target-metric PASS condition (e.g., "behavioral target KC PASS ∧ ≥1 curriculum schedule PASSES K1933").
5. **Tighten notes**: state the *behavioral* outcome the curriculum must preserve or improve (LLM-judge politeness on hard tail), not just "tests curriculum learning". Specify the difficulty signal (e.g., per-prompt teacher-output entropy) is *distinct from* the evaluation signal (LLM-judge), eliminating the cos-sim-on-cos-sim circularity.
6. **Wait for parent F#683 (Hedgehog politeness adapter) to graduate from PROVISIONAL to SUPPORTED** before running curriculum ablation. F#669-style child-on-unverified-parent risk: even with target-pair KCs, comparing two curricula of an unverified base method has no anchor. Curriculum-vs-random is meaningful only if random-order produces a working adapter; F#683 hasn't measured that yet.

Post-claim KC mutation is antipattern-u; edits must happen **before** re-claim. Recommendation: **close this pre-reg as structurally-malformed**; re-register `exp_hedgehog_curriculum_training_behavioral` after F#683 supported.

### §5.1 Pre-existing partial coverage

Prior findings provide partial coverage of curriculum-style training procedures:
- **F#722**: hyperparameter-ablation 4th sub-type — teacher-temperature sweep is a continuous hyperparameter analog of the discrete curriculum schedule. Both are training-procedure ablations on Hedgehog distillation; F#722's preempt directly informs this one.
- **F#723**: data-augmentation-ablation 5th sub-type with target-pair runnable design. Curriculum-on-existing-data and augmentation-of-existing-data are dual procedural axes; F#723's K1877+K1878 design is the template the curriculum experiment should adopt.
- **F#719/F#720**: loss-variant ablation series shows that a target-pair design avoids preempt (F#719) while cos-sim-only design does not (F#720). Same lesson applies to curriculum.
- **arxiv:2010.13166** (Wu et al., "Curriculum Learning for Knowledge Distillation"): the closest published prior art. Their evaluation includes student-task accuracy (target metric), not just cos-sim/KL convergence (proxy) — confirming the runnable design pattern.

Curriculum-learning-for-distillation is a real research question with published prior art; this experiment's failure is structural (KC design), not topical.

## §6 Follow-up

No `_impl` companion filed — preempt-structural KILL does NOT spawn `_impl` (per F#687/F#698/F#699/F#700/F#701/F#703/F#705/F#710/F#714/F#722/F#753/F#754 precedent + reviewer.md §5). Unblock is pre-registration-external (edit the DB entry to add target-pair KC + references + platform), not implementation-external.

`mem-antipattern-impl-follow-up-delegation` does not apply: that antipattern targets novel-mechanism PROVISIONAL, not preempt-structural KILL.
