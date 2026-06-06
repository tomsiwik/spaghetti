# MATH.md — exp_g4_adapter_similarity_across_seeds (PREEMPT-KILL, F#666-pure standalone, ~27th drain-window instance, 1st g4-ablation seed-determinism sub-type)

## Verdict: PREEMPT-KILL (KC-structural, F#666-pure standalone — g4-ablation super-family: 1st seed-determinism / reproducibility-ablation sub-type, dual-tail cos-sim-bucket form)

This experiment is preempt-killed before any code runs. The kill is **structural**: the pre-registered kill-criterion set K = {K1937, K1938} consists of two cos-sim proxy metrics covering opposite tails of the same pairwise-cos-sim distribution, with no paired target-metric KC. Under F#666 (guardrail 1007 — target-gated KILL discipline) and guardrail 1006 (cos-sim is explicitly a forbidden-solo proxy because cos-sim ↔ behavior is decoupled, r≈0.08 PPL→task-quality), neither KILL nor SUPPORTED is derivable regardless of empirical outcome.

This is a continuation of the F#666-pure standalone canonical pattern (F#700, F#701, F#703, F#705, F#706, F#707, F#708, F#710, F#711, F#714, F#715, F#716, F#720, F#722, F#728, F#729, F#730, F#731, F#732, F#734, F#735, F#736, F#753, F#754, F#755, F#756 — at least 26 prior). Specifically:
- **1st g4-ablation super-family seed-determinism / reproducibility-ablation sub-type**. The g4-ablation super-family already includes preempt instances on per-layer-cos-baseline (F#700), canary-FNR (F#706), PPL-drift (F#705), gumbel-routing-acc (F#710), perturbation-stability (F#711), SVD-rank-delta (F#712), SVD-denoise-PPL (F#716), routing-collision-rate (F#707), hash-ring-PPL (F#708), routing-family-equivalence (F#709). This adds the **seed-determinism / cross-instance reproducibility** sub-flavor — distinct from F#751's init-method-comparison (which compares *across init recipes*, not *across seeds within one recipe*).
- **4th cos-sim-bucket form**: K1937 + K1938 are dual-tail thresholds on the **cross-instance pairwise cos-sim population statistic** (5 adapters → C(5,2)=10 pairs). Bucket evolution: (1st) F#720 final-value single-pair, (2nd) F#755 convergence-speed single-trace, (3rd) F#756 tightness/distance ("within 0.02"), (**4th**) this — cross-instance population-statistic, dual-tail. Each form is a new failure to pair the cos-sim measurement to behavior.
- **Closest runnable sibling separator**: `exp_g4_adapter_initialization_comparison_v2` (open, P=2, depends on F#751 v1) has the *same shape question* (does the adapter weight matter, or does the task fully determine it?) but adds K1978 = "eval-PPL ratio worst/best > 1.10" as a behavioral-adjacent target paired with K1977 cos-sim proxy. v2 is the design template this experiment fails to adopt.
- **Pre-existing partial coverage**: F#169 already showed "Three init methods (random QR, Grassmannian AP, OSRM covariance-constrained) produce identical individual..." — init-method-invariance demonstrated. F#750 documents the shared-PRNG-key antipattern that confounds any seed-comparison if not handled. F#751 K1925 PROVISIONAL on init-method invariance with target-paired PPL spread 3.5%. The seed-determinism question is partially answered by these; the unique increment ("seeds within one init") is precisely what v2 will measure with PPL-paired KCs (F#751 v2 K1979 explicitly tests "seed-variance on PPL within one init > 5%").

## §0 Platform / skills / model pins

Included for reviewer checklist (m2) completeness. No platform code executes.
- Platform skills: `/mlx-dev` + `/fast-mlx` (per PLAN.md Part 2). **Not invoked** — no MLX code written.
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627). **Not loaded.**
- Adapter targets: N/A — would have been q_proj r=6 per F#751 v1 medical recipe (notes do not pin); no LoRA injection or training in this run.
- Parent dependency: **none** (`depends_on: []`). NOT an F#669 preempt — sibling F#751 (init-comparison v1, PROVISIONAL) and `exp_g4_adapter_initialization_comparison_v2` (open) cover overlapping ground but are not declared parents.

## §1 Preempt-KILL theorem (F#666-pure, dual-tail cos-sim-bucket cross-instance population-statistic, 1st seed-determinism g4-ablation)

**Theorem (KC-structural invalidity under target-gated KILL).** Let `E` denote experiment `exp_g4_adapter_similarity_across_seeds` with kill-criterion set K = {K1937, K1938}:
- K1937 := "Adapters from different seeds have pairwise cos > 0.80 (deterministic despite seed)"
- K1938 := "Adapters from different seeds have pairwise cos < 0.30 (completely seed-dependent)"

**Classification of K.**
- K1937 and K1938 are **dual-tail thresholds** on the same proxy quantity: mean (or per-pair) cosine similarity between A-matrices (and/or B-matrices, and/or ΔW = B@A) across 5 same-config different-seed adapter trainings. Both KCs are cos-sim **proxy** metrics. Per guardrail 1007 explicitly, cos-sim is a forbidden-solo proxy: F#666 canonical (40.2% per-sample classification acc + 0.0% target gap) shows cos-sim-equivalence and behavior-equivalence are separable axes. F#169 already showed init-method-invariance (cos-sim near identical across init recipes); F#751 K1925 showed PPL-invariance across init recipes; the present seed-axis question similarly cannot be resolved by cos-sim alone.
- The **dual-tail design** (PASS-K1937 covers deterministic regime, PASS-K1938 covers seed-dependent regime, intermediate region 0.30 ≤ cos ≤ 0.80 satisfies neither) is interesting *per se* — the experiment can fail to fire either KC and produce no verdict at all. But even when one KC fires, the verdict-mapping under F#666 is unidentifiable (see §1.1).

Neither KC measures task accuracy, behavioral quality (medical-domain PPL-ratio per F#627; LLM-judge quality per F#683 K1783), oracle-gap, generalization-gap on hard-tail (Hacohen-Weinshall 2019 arxiv:1904.03626 pattern), or any downstream-behavioral outcome. K is a 2-proxy, 0-target set.

### §1.1 Dual-tail F#666 verdict truth table (3-cell because middle band exists)

The dual-tail KC design partitions cos-sim space into three regions, not two. Verdict truth table:

| Mean pairwise cos | K1937 (cos > 0.80) | K1938 (cos < 0.30) | F#666 verdict |
|---|---|---|---|
| ≥ 0.80 (deterministic-tail PASS) | FAIL (kill on K1937) | PASS | Tautological "seeds don't matter" — but no behavioral evidence the produced adapters BEHAVE the same. Under F#666, cos-sim-equivalence ≠ behavior-equivalence (canonical 40.2% acc + 0.0% target gap). Two adapters with cos=0.95 may diverge on task accuracy via cluster-equivalence vs cluster-divergence at decision boundaries. Inadmissible under guardrail 1007. |
| 0.30 ≤ cos ≤ 0.80 (intermediate band) | FAIL | FAIL | Neither tail fires; experiment produces no verdict. Hyperparameter / threshold-arbitrariness defect — the 0.80 and 0.30 thresholds are unanchored to any prior result (F#169 was at-machine-precision near 0.0 for orthogonal init; F#562 showed Grassmannian QR produces cos≈4.77e-9 at init). The dual-tail design admits a "no verdict" cell. |
| < 0.30 (seed-dependent-tail PASS) | PASS | FAIL (kill on K1938) | Tautological "seeds matter completely" — but the adapters could STILL produce equivalent behavioral outcomes via cluster-equivalence (F#666 canonical: 60% misclassification yet 0% target gap). Cos-sim < 0.30 says A-matrices are nearly orthogonal as ΔW signatures, but ΔW signatures are *forward-pass-modulated* by the frozen base; behavior depends on the *output* not the *parameter alignment*. Inadmissible under guardrail 1007. |

**No cell yields a valid F#666-compliant verdict.** K is unidentifiable at the F#666 layer. **QED.**

### §1.2 Cross-instance population-statistic (4th cos-sim-bucket form)

K1937 + K1938 measure pairwise-cos-sim on a population of N=5 same-config different-seed adapters (10 pairs). This is a new cos-sim-bucket sub-form distinct from prior preempts:

| Cos-sim-bucket form | Finding | KC text | Population shape |
|---|---|---|---|
| 1st: final-value single-pair | F#720 (Hedgehog MSE-vs-cos loss) | K1872 cos < 0.70 between two adapters | one inter-loss-variant pair |
| 2nd: convergence-speed single-trace | F#755 (curriculum training) | K1934 cos convergence < random-order | one inter-curriculum-trace |
| 3rd: tightness/distance "within X" | F#756 (early-stopping plateau) | K1935 cos within 0.02 of full-trained | one inter-stopping-point delta |
| **4th: cross-instance population-statistic, dual-tail** | **THIS** | K1937 + K1938 dual-tail on pairwise cos across N=5 same-config different-seed adapters | **C(5,2)=10 pairs, summarized as mean/median** |

The new form treats the cos-sim measurement as a *distributional* property over a multi-instance population, using thresholds on the marginal distribution rather than on a single delta. None of the prior cos-sim-bucket forms used a population statistic — F#720/F#755/F#756 each measured a single inter-instance delta. The dual-tail design is a 1st-of-its-kind structural feature in the cos-sim-bucket: K1937 PASS and K1938 PASS are mutually exclusive (the cos cannot be > 0.80 and < 0.30 simultaneously); K1937 FAIL and K1938 FAIL is *possible* (intermediate band) and creates the no-verdict cell above.

### §1.3 Why F#751 v2 is the runnable separator

`exp_g4_adapter_initialization_comparison_v2` (open at P=2, depends on F#751 v1) has the same general shape question — "is the adapter weight reproducible across configurations?" — but the v2 KC set is target-paired:

| KC | Text | Kind |
|---|---|---|
| K1977 | Cross-init final A-matrix \|cos\| < 0.20 across all 3 init pairs | proxy (cos-sim) |
| K1978 | Final eval-PPL ratio worst/best > 1.10 (10% spread) | **target-adjacent** (eval-PPL spread is behavior-proximate; F#627 medical-domain recipe; PPL is still r≈0.08 vs task-quality but it's the standard adapter-effectiveness metric) |
| K1979 | Seed-variance on PPL within one init > 5% | **target-adjacent** (variance bound for identifiability) |

K1977 alone would be F#666-violating; K1977+K1978+K1979 forms a Pareto-quality target structure (cos-sim proxy + PPL behavior-adjacent + variance identifiability). This is the runnable design pattern; `exp_g4_adapter_similarity_across_seeds` should adopt the same KC structure.

Even tighter: F#751 v2 K1979 directly answers the seed-determinism question in PPL terms — "does seed-variance on PPL within one init exceed 5%?" If F#751 v2 reports K1979 PASS or FAIL with measurement, the seed-similarity claim is *already answered* in PPL terms. The cos-sim claim in `exp_g4_adapter_similarity_across_seeds` would still need a behavior-target pair to be admissible under F#666, but the bulk of the seed-vs-task question is closed by F#751 v2.

## §2 Prior art (preempt-KILL precedents and g4-ablation taxonomy)

- **F#666** (2026-04-19, conclusive): target-gated KILL discipline; guardrail 1007 enumerates classification accuracy, routing match rate, *PPL*, *cosine* explicitly as forbidden-solo proxies; canonical 40.2% proxy + 0.0% target gap.
- **F#169** (killed): "Three init methods (random QR, Grassmannian AP, OSRM covariance-constrained) produce identical individual..." — direct prior coverage on the init-vs-task-determinism axis (init-method, not seed). Demonstrates B-matrices compensate for any A-matrix configuration; the task-determines-the-adapter hypothesis has substantial prior support at the init-method level.
- **F#562** (supported): Grassmannian intra-column orthogonality at native Gemma 4 dims (cos≈4.77e-9 at init).
- **F#627** (supported): medical-domain LoRA q_proj r=6 scale=6 baseline recipe at Gemma 4 E4B; the canonical training configuration that this experiment would inherit.
- **F#750** (provisional, 2026-04-25): Antipattern: shared PRNG key across init-variant comparisons yields correlated starting matrices (illusory cross-init similarity). **Directly relevant** — any seed-comparison experiment must use distinct top-level seeds per training, not split-of-shared-master-key. Pre-reg notes do not specify seed-handling protocol; if implemented naively (shared `mx.random.key(SEED)` split across N variants), result would inherit the F#750 confound.
- **F#751** (provisional, 2026-04-25): Adapter init-invariance at Gemma 4 E4B d=2560 r=6 q_proj — PROVISIONAL (target K1925 PASS at PPL spread 3.5%; proxy K1924 FAIL by PRNG confound). Sibling on the reproducibility-axis.
- **`exp_g4_adapter_initialization_comparison_v2`** (open, P=2, depends on F#751): Cross-init clean-seed v2 with K1977/K1978/K1979 target-paired KC structure. Runnable design template.
- **F#700** (killed): exp_g4_per_layer_cos_baseline preempt-KILL. **1st F#666-pure standalone canonical**, also g4-ablation cos-sim variance.
- **F#720** (killed): 1st cos-sim-bucket (final-value form, single inter-pair).
- **F#755** (killed): 2nd cos-sim-bucket (convergence-speed form, single trace).
- **F#756** (killed): 3rd cos-sim-bucket (tightness/distance form, single inter-stopping-point delta).
- **`mem-antipattern-f666-pure-standalone-preempt-kill`** (filed 2026-04-24, multiple escalations): claim-time detection rule; preempt-scaffold response.
- **Guardrail 1007** (PLAN.md): every proxy KC must be paired with a target-metric KC.
- **Guardrail 1006** (PLAN.md): "PPL does not predict task quality in this project (measured r≈0.08). Behavioral outcomes over metrics."

External reproducibility-statistics prior art (none cited in DB pre-reg, guardrail 1002 violation):
- Hu et al. 2021 arxiv:2106.09685 (LoRA paper) — original LoRA defines low-rank adapters but does not analyze cross-seed reproducibility; ablation falls under "follow-up needed".
- Bouthillier et al. 2019 arxiv:1909.10314 ("Unreproducible Research is Reproducible") — formal framework for measuring random-seed-induced variance vs hyperparameter-induced variance in deep learning; defines metric-level vs decision-level reproducibility (the latter = behavior-equivalence test, what F#666 demands).
- Madaan et al. 2024 arxiv:2402.01906 ("LoRA Learns Less and Forgets Less") — reports cross-seed stability of LoRA fine-tuning at the *task-accuracy* level (target metric), the runnable template.

## §3 Predictions (registered, not measured)

| KC    | Claim                                                                    | Kind  | Sub-flavor                                                                          | Measurement status         |
| ----- | ------------------------------------------------------------------------ | ----- | ----------------------------------------------------------------------------------- | -------------------------- |
| K1937 | Adapters from different seeds have pairwise cos > 0.80 (deterministic)   | proxy | cos-sim-bucket cross-instance population-statistic, dual-tail upper-tail (4th cos-sim-bucket form) | untested (preempt-blocked) |
| K1938 | Adapters from different seeds have pairwise cos < 0.30 (seed-dependent)  | proxy | cos-sim-bucket cross-instance population-statistic, dual-tail lower-tail (4th cos-sim-bucket form) | untested (preempt-blocked) |

No target-metric KC exists. K is structurally malformed per F#666. Dual-tail design admits 3-cell verdict map (deterministic-tail / intermediate / seed-dependent-tail), all three inadmissible.

KC text preserved verbatim from `experiment get exp_g4_adapter_similarity_across_seeds` output. No post-claim KC mutation (antipattern-u check: PASS).

## §4 Hygiene defects (noted, not load-bearing for kill)

Per `experiment get exp_g4_adapter_similarity_across_seeds`:

1. **`success_criteria: NONE`** — no SUPPORTED-condition declared (DB explicitly flags `⚠ INCOMPLETE: success_criteria`).
2. **`platform: —`** (null) — guardrail/hygiene defect (DB flags `⚠ INCOMPLETE: ... platform`).
3. **`references: []`** — guardrail 1002 violation (every new experiment MUST cite an arxiv paper or prior finding). The reproducibility / random-seed-variance literature is substantial (Bouthillier 2019 arxiv:1909.10314; Madaan 2024 arxiv:2402.01906; sibling F#169, F#562, F#627, F#750, F#751) — none cited.
4. **`experiment_dir: —`** (null until this iteration created the dir).
5. **Threshold pair unanchored**: 0.80 and 0.30 are both unanchored to any prior measurement. F#169 / F#562 results suggest cos-sim across orthogonal-init adapters is at machine precision (≈10⁻⁹); within one init across seeds is unmeasured but plausibly very high (> 0.95). The "intermediate band" 0.30–0.80 likely never fires in practice, making the dual-tail design a single-tail in disguise (K1937 will likely fire) — but the structural F#666 violation persists either way.
6. **Sample size N=5 unmotivated**: 10 pairs is a small population; no power analysis; no variance bound (cf. F#751 v2 K1979 which adds explicit variance bound).
7. **No seed-handling protocol**: pre-reg notes do not specify whether to use `mx.random.key(seed)` per variant or split-of-shared-key. Naive implementation would reproduce F#750 antipattern. (Even with distinct keys, the F#666 violation remains.)

Seven hygiene defects total. Crosses the AP-prereg-hygiene-multi-defect threshold (≥3 defects). However, F#666-pure structural defect alone is sufficient for kill independent of hygiene count (per F#703 invariant; same shape as F#722, F#754, F#755, F#756).

Notes field reads: "Train 5 adapters with same config, different seeds. If high similarity, the task determines the adapter. If low, initialization matters a lot." — the framing ("task vs init determines the adapter") is *behavioral*, but the operationalization (cos-sim threshold) is structural. The notes' own framing reveals the F#666 gap: "the task determines the adapter" is a *behavioral* claim that requires a *behavioral* measurement (do the adapters produce equivalent task outputs?) — cos-sim as a parameter-space identity test is a proxy.

## §5 Unblock condition (re-claim requires KC-augmentation pre-registration)

Re-registration as a new experiment id (`exp_g4_adapter_similarity_across_seeds_behavioral` recommended) with the following fixes:

1. **Add a target-metric KC** pairing seed-similarity to a behavioral outcome on the parent G4 axis. Candidate formulations:
   - **Eval-PPL spread target** (F#751 v2 K1978/K1979 pattern): cross-seed eval-PPL ratio worst/best ≤ 1.05 on F#627 medical recipe. Pairs the cos-sim claim with the standard adapter-effectiveness measure. If K1937 PASS (cos > 0.80) AND PPL-spread target PASS, "task-determines-adapter" is supported behaviorally. If K1937 PASS but PPL-spread FAIL, that's a finding about cos-sim being a misleading reproducibility proxy.
   - **Behavioral output equivalence** (F#666 canonical pattern): on N=20 medical-domain held-out prompts, % of prompts where all 5 seed-adapters produce > 0.95 BLEU-overlap or LLM-judge-equivalent output. A behavioral identity test that complements cos-sim parameter identity.
   - **Pareto-quality target**: among seed-pairs that PASS K1937 OR K1938, max-pair PPL-divergence < 5% AND median-pair behavioral-equivalence > 0.90.
2. **Add references**: F#666 (guardrail), F#169 (init-method coverage), F#562 (Grassmannian orthogonality), F#627 (medical recipe), F#750 (PRNG-key antipattern), F#751 (init-comparison v1 PROVISIONAL), `exp_g4_adapter_initialization_comparison_v2` (sibling design template), Bouthillier 2019 arxiv:1909.10314, Madaan 2024 arxiv:2402.01906, Hu 2021 arxiv:2106.09685. Address guardrail 1002.
3. **Set `platform=local-apple`** (currently null).
4. **Populate `success_criteria`** mirroring the new target-metric PASS condition.
5. **Anchor threshold pair**: replace dual-tail 0.30/0.80 with thresholds calibrated against F#169 / F#562 prior measurements (e.g., "pairwise cos > 0.95" if checking task-determinism; "pairwise cos < machine-precision-bound + headroom" if checking seed-dependence). Or drop dual-tail in favor of variance bound (Bouthillier 2019 framework).
6. **Specify seed-handling protocol**: pre-register that each of N=5 trainings uses a distinct top-level `mx.random.key(seed_i)` (not split-of-shared-key), to avoid the F#750 antipattern.
7. **Increase N for statistical power**: N=5 → 10 pairs is low; consider N=10 → 45 pairs and report distribution (mean, std, percentiles) rather than single-threshold pass/fail.
8. **Wait for `exp_g4_adapter_initialization_comparison_v2` to complete first**: v2's K1979 (seed-variance on PPL within one init > 5%) directly measures the seed-determinism question in PPL terms. Result of v2 K1979 is a precondition for whether this experiment is worth re-registering — if K1979 PASS (low variance), behavioral seed-determinism is demonstrated and the cos-sim follow-up is a reproducibility-hygiene check; if K1979 FAIL (high variance), cos-sim follow-up should be tied to the variance source.

Post-claim KC mutation is antipattern-u; edits must happen **before** re-claim. Recommendation: **close this pre-reg as structurally-malformed**; await `exp_g4_adapter_initialization_comparison_v2` result (K1979 in particular) and re-register `exp_g4_adapter_similarity_across_seeds_behavioral` only if the v2 result motivates further investigation.

### §5.1 Pre-existing partial coverage

Prior findings provide substantial partial coverage of the seed-determinism question:
- **F#169**: Init-method-invariance on adapters (cos-sim near identical across init recipes; B compensates for A). Establishes the "task determines adapter" prior at the init-method level.
- **F#562**: Grassmannian QR init produces cos≈4.77e-9 at init — establishes a baseline for what "near zero" means in cos-sim terms (machine precision, not 0.30).
- **F#627**: Canonical Gemma 4 E4B medical-recipe LoRA q_proj r=6 scale=6 — the training config this experiment would inherit; behavioral PPL baseline available for direct comparison.
- **F#750**: PRNG-shared-key antipattern — directly informs seed-handling protocol for any cross-seed comparison.
- **F#751 v1 + v2**: Init-comparison K1925 PROVISIONAL (PPL spread 3.5% across init); v2 K1977+K1978+K1979 will provide the target-paired template + direct seed-variance measurement.
- **Bouthillier 2019 arxiv:1909.10314**: Formal framework for random-seed reproducibility — distinguishes metric-level vs decision-level reproducibility (the latter = behavior-equivalence, what F#666 demands).
- **Madaan 2024 arxiv:2402.01906**: LoRA cross-seed stability at the task-accuracy level (target-metric runnable template).

Adapter cross-seed reproducibility is a real research question; this experiment's failure is structural (KC design + reference omission + threshold-anchoring + seed-protocol-specification), not topical.

## §6 Follow-up

No `_impl` companion filed — preempt-structural KILL does NOT spawn `_impl` (per F#687/F#698/F#699/F#700/F#701/F#703/F#705/F#710/F#714/F#722/F#753/F#754/F#755/F#756 precedent + reviewer.md §5). Unblock is pre-registration-external (edit the DB entry to add target-pair KC + references + platform + threshold-anchoring + seed-protocol-specification), not implementation-external.

`mem-antipattern-impl-follow-up-delegation` does not apply: that antipattern targets novel-mechanism PROVISIONAL, not preempt-structural KILL.
