# PAPER — exp_g4_adapter_similarity_across_seeds (KILLED — preempt-structural, F#666-pure standalone, 1st g4-ablation seed-determinism sub-type, 4th cos-sim-bucket form)

## Verdict: KILLED (KC-structural preempt, F#666-pure standalone, ~27th drain-window instance, 1st g4-ablation super-family seed-determinism / reproducibility-ablation sub-type, 4th cos-sim-bucket form: cross-instance population-statistic with dual-tail thresholds)

This experiment is preempt-killed on structural grounds before any code executes. The verdict is deterministic from the KC-set shape, not from measurement.

## Summary

Pre-registered kill-criterion set K = {K1937, K1938} contains two cos-sim proxy metrics covering opposite tails of the same pairwise-cos-sim population statistic, with no paired target-metric KC:

- **K1937** ("Adapters from different seeds have pairwise cos > 0.80 (deterministic despite seed)") — *cross-instance pairwise cos-sim population statistic, upper-tail threshold of dual-tail design*. Cos-sim is a forbidden-solo proxy under guardrail 1007: F#666 canonical decoupling (40.2% per-sample classification acc + 0.0% target gap) shows cos-sim-equivalence and behavior-equivalence are separable axes. Two adapters with cos=0.95 may diverge on task accuracy via cluster-equivalence vs cluster-divergence at decision boundaries.
- **K1938** ("Adapters from different seeds have pairwise cos < 0.30 (completely seed-dependent)") — same cross-instance pairwise cos-sim population statistic, *lower-tail threshold of dual-tail design*. Adapters with cos < 0.30 (near-orthogonal as parameter signatures) could STILL produce equivalent behavioral outcomes via cluster-equivalence (F#666 canonical: 60% misclassification yet 0% target gap).

Under F#666, a proxy-only KC set has no valid verdict regardless of empirical outcome. The dual-tail KC design partitions cos-sim space into three regions, all three inadmissible:
- Deterministic-tail PASS (K1937 PASS, K1938 FAIL) → tautological "task determines adapter" without behavioral evidence.
- Intermediate band (K1937 FAIL, K1938 FAIL) → no verdict cell — the experiment produces neither PASS nor KILL signal.
- Seed-dependent-tail PASS (K1937 FAIL, K1938 PASS) → tautological "init/seed determines adapter" without behavioral evidence.

This is the **1st g4-ablation super-family seed-determinism / reproducibility-ablation sub-type** instance — distinct from F#751's init-method-comparison (which compares *across init recipes*, not *across seeds within one recipe*). The pattern is well-established post taxonomy refactor at F#714: ≥26 prior F#666-pure standalone instances exist (F#700, F#701, F#703, F#705, F#706, F#707, F#708, F#710, F#711, F#714, F#715, F#716, F#720, F#722, F#728, F#729, F#730, F#731, F#732, F#734, F#735, F#736, F#753, F#754, F#755, F#756, …).

**g4-ablation super-family taxonomic placement (this experiment introduces the 1st seed-determinism sub-type):**

| # | Sub-type | Canonical finding(s) | KC design | Outcome |
|---|---|---|---|---|
| 1 | per-layer-cos-baseline | F#700 | proxy-only | KILLED |
| 2 | PPL-drift | F#705 (o1_removal_naive) | proxy-only | KILLED |
| 3 | canary-FNR | F#706 | proxy-only | KILLED |
| 4 | routing-collision-rate | F#707 | proxy-only | KILLED |
| 5 | hash-ring-PPL | F#708 | proxy-only | KILLED |
| 6 | routing-family-equivalence | F#709 | inter-variant-delta | KILLED |
| 7 | gumbel-routing-acc | F#710 | proxy-only | KILLED |
| 8 | perturbation-stability | F#711 (gs_random_perm) | proxy-only | KILLED |
| 9 | SVD-rank-delta | F#712 | inter-rank-delta | KILLED |
| 10 | SVD-denoise-PPL | F#716 | proxy-only | KILLED |
| 11 | init-method-comparison | F#751 | **target-paired (K1924+K1925)** | **PROVISIONAL** |
| **12** | **seed-determinism / reproducibility-within-init-recipe** | **THIS (F#NEW)** | **proxy-only dual-tail** | **KILLED preempt-structural** |

Pattern: F#751 (init-method-comparison) used target-pairing (K1925 PPL-spread paired with K1924 cos-sim) and graduated to PROVISIONAL — the runnable design for reproducibility-axis ablations. This experiment uses the same axis (reproducibility) but proxy-only KCs and is preempt-killed.

**Cos-sim-bucket form taxonomic placement (this experiment is the 4th form):**

| # | Form | Canonical finding | KC text | Population shape |
|---|---|---|---|---|
| 1 | final-value single-pair | F#720 | K1872 cos < 0.70 between two adapters | one inter-loss-variant pair |
| 2 | convergence-speed single-trace | F#755 | K1934 cos convergence < random-order | one inter-curriculum-trace |
| 3 | tightness/distance single-delta | F#756 | K1935 cos within 0.02 of full-trained | one inter-stopping-point delta |
| **4** | **cross-instance population-statistic, dual-tail** | **THIS** | **K1937 + K1938 dual-tail on pairwise cos across N=5 same-config different-seed adapters** | **C(5,2)=10 pairs, summarized as mean/median, dual-tail thresholds creating 3-cell verdict map** |

The 4th form is structurally novel within the cos-sim-bucket: it treats cos-sim as a *distributional* property over a multi-instance population using thresholds on the marginal distribution, rather than as a single inter-instance delta. The dual-tail design admits a no-verdict intermediate band — 1st no-verdict-cell-by-construction precedent in cos-sim-bucket.

## Prediction vs measurement

| KC    | Claim                                                                       | Kind  | Sub-flavor                                                                          | Verdict                                |
| ----- | --------------------------------------------------------------------------- | ----- | ----------------------------------------------------------------------------------- | -------------------------------------- |
| K1937 | Adapters from different seeds have pairwise cos > 0.80 (deterministic)      | proxy | cos-sim-bucket cross-instance population-statistic, dual-tail upper-tail (4th form) | UNTESTED (preempt-blocked, F#666-pure) |
| K1938 | Adapters from different seeds have pairwise cos < 0.30 (seed-dependent)     | proxy | cos-sim-bucket cross-instance population-statistic, dual-tail lower-tail (4th form) | UNTESTED (preempt-blocked, F#666-pure) |

No measurement was taken. No model was loaded; no dataset opened; no LoRA adapter constructed; no N=5 trainings executed; no pairwise cos-sim computed; no population-statistic summarized; no threshold tested. The verdict derives from `F#666 proxy-only KC set` + `no target-metric pair` ⇒ inadmissible-for-all-outcomes (3-cell truth table in MATH.md §1.1).

## Why this is not runnable as-is

Even if the cross-seed similarity comparison were executed, every cell of the {K1937 (cos > 0.80), K1938 (cos < 0.30)} dual-tail design maps to an inadmissible verdict under F#666:

| Mean pairwise cos | K1937 | K1938 | Verdict under F#666 |
|---|---|---|---|
| ≥ 0.80 | PASS | FAIL | Tautological "seeds don't matter / task determines adapter" — but no behavioral evidence the produced adapters BEHAVE the same. Cos-sim-equivalence ≠ behavior-equivalence (F#666 canonical: 40.2% per-sample acc + 0.0% target gap). Could ship reproducibility claim that fails downstream. |
| 0.30 ≤ cos ≤ 0.80 | FAIL | FAIL | No verdict cell — neither tail fires. Hyperparameter-arbitrariness: the 0.80 and 0.30 thresholds are unanchored. F#562 measurements suggest cos-sim across orthogonal-init adapters is at machine precision (~10⁻⁹); within one init across seeds is unmeasured but plausibly very high (>0.95). Intermediate band may never fire in practice. |
| < 0.30 | FAIL | PASS | Tautological "init/seed determines adapter completely" — but adapters could STILL produce equivalent behavioral outcomes via cluster-equivalence at the decision boundary (F#666 canonical decoupling). Cos-sim < 0.30 is a parameter-space identity test, not a behavioral identity test. Cluster-equivalence at the output is possible despite parameter-space orthogonality. |

The F#666 rule operates on KC *kind*, not measurement value. Proxy-only structure is the disease; measurement outcome is the symptom.

### Pathological-case illustrations of decoupling

1. **K1937 PASS + behavior diverges** (cluster-divergence at decision boundary): all 10 pairs have cos > 0.80 but on a held-out medical-domain prompt set, 30% of prompts produce different output tokens across seeds (rouge < 0.85). Parameter-space tightness does not imply behavior-space tightness — the LoRA composition with the frozen base maps the small parameter delta into a non-trivial output delta on prompts near decision boundaries. Cos-sim says "deterministic"; behavior says "seed-dependent on hard prompts". The reproducibility claim ("task determines adapter") would be wrong in production.
2. **K1938 PASS + behavior preserved** (cluster-equivalence despite parameter orthogonality): all 10 pairs have cos < 0.30 (near-orthogonal A-matrices) but all 5 adapters produce LLM-judge-equivalent output on 95% of medical prompts. The full-rank ΔW = B@A has many parameter-space realizations that yield the same forward function class — same LoRA-effect-on-output via different path through parameter space. Cos-sim says "seed-determines-everything"; behavior says "task-determines-the-output-class" through a many-to-one parameterization. F#169 already demonstrated this for init-methods (B compensates for any A configuration).
3. **K1937 FAIL + K1938 FAIL** (intermediate band): mean cos = 0.55. Neither kill criterion fires; experiment produces no verdict despite measurement. Hygiene defect by design — the dual-tail thresholds leave a substantive cos-sim region without a structural decision rule.

Each pathology requires a behavioral target KC (F#751 v2 K1979 PPL-spread within-init pattern, F#666-style behavioral-equivalence test, or Madaan 2024 task-accuracy cross-seed stability) to disambiguate.

## Hygiene defects

| Defect                  | Status                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------- |
| F#666 violation         | Present (K1937 + K1938 are both cos-sim proxies; cos-sim explicitly forbidden-solo by guardrail 1007) |
| `success_criteria` MISSING | Present (DB explicitly flags `⚠ INCOMPLETE: success_criteria`)                     |
| `platform: —`           | Present (DB shows `platform: —` null; `⚠ INCOMPLETE: ... platform`)                  |
| `references: []`        | Present (guardrail 1002 violation; no arxiv or finding citation despite extensive prior coverage F#169 / F#562 / F#627 / F#750 / F#751 / Bouthillier 2019 / Madaan 2024 / Hu 2021) |
| Threshold pair unanchored | 0.80 and 0.30 are both unanchored to any prior measurement; F#169 / F#562 baselines suggest the intermediate band may never fire in practice |
| Sample size N=5 unmotivated | 10 pairs is low; no power analysis; no variance bound (cf. F#751 v2 K1979 explicit variance bound) |
| Seed-handling protocol unspecified | Pre-reg notes do not specify whether to use `mx.random.key(seed)` per variant or split-of-shared-key; naive implementation would reproduce F#750 confound |
| No target pair for behavioral claim in notes | Notes framing ("task determines adapter") is *behavioral* but operationalization (cos-sim threshold) is *structural* — internal contradiction between framing and measurement |

Hygiene-defect count = 8. Crosses AP-prereg-hygiene-multi-defect (≥3) threshold. F#666-pure structural defect alone is sufficient for kill independent of hygiene count.

## Taxonomic comparison with drain-window precedents

| Dimension              | F#751 (init-method, target-paired)        | F#720 (cos-sim final-value, 1st cos-bucket) | F#755 (cos-sim convergence-speed, 2nd) | F#756 (cos-sim tightness, 3rd) | This (cross-instance population, 4th)         |
| ---------------------- | ----------------------------------------- | ------------------------------------------- | -------------------------------------- | ------------------------------ | ---------------------------------------------- |
| Parent dep             | none (v1)                                 | none                                        | none                                   | none                           | none                                           |
| KC count               | 2                                         | 1                                           | 2                                      | 2                              | 2 (dual-tail same quantity)                    |
| KC kinds               | **target + proxy**                        | proxy-only                                  | proxy-only                             | proxy-only                     | **proxy-only dual-tail**                       |
| F#666 violation        | no                                        | yes                                         | yes                                    | yes                            | yes                                            |
| Reproducibility-axis   | init-method                               | n/a                                         | n/a                                    | n/a                            | **seed-within-one-init**                       |
| Cos-sim-bucket form    | n/a (target-paired)                       | final-value single-pair                     | convergence-speed single-trace         | tightness/distance single-delta | **cross-instance population-statistic dual-tail** |
| Verdict-cell count     | 2 (PASS/FAIL)                             | 2                                           | 2                                      | 2                              | **3 (deterministic / intermediate / seed-dep)** |
| No-verdict cell exists | no                                        | no                                          | no                                     | no                             | **yes (intermediate band 0.30–0.80)**          |
| Verdict                | PROVISIONAL (K1925 PASS, K1924 confound)  | KILLED (preempt-structural)                 | KILLED (preempt-structural)            | KILLED (preempt-structural)    | **KILLED (preempt-structural)**                |
| `_impl` follow-up      | none                                      | none                                        | none                                   | none                           | none                                           |

The invariant: `depends_on: []` + cos-sim-only KC set ⇒ preempt-KILL, independent of cos-sim-bucket form (final-value, convergence-speed, tightness/distance, cross-instance population-statistic), KC count, hygiene count, axis (intra-training, inter-config, intra-trajectory, inter-instance), or verdict-cell count.

## Caveats

- This is the 1st g4-ablation seed-determinism sub-type and the 4th cos-sim-bucket form. The space of "obvious cos-sim-only KC formulations" is closing fast: final-value, convergence-speed, tightness/distance, cross-instance population-statistic now all observed. Researchers attempting a 5th cos-sim-bucket form (e.g., temporal cos-sim drift, cos-sim conditional on input distribution) should pre-register with target-pair KCs to avoid preempt.
- The dual-tail KC design is structurally distinct from prior cos-sim KCs and creates a no-verdict intermediate band. This is the 1st no-verdict-cell-by-construction precedent in cos-sim-bucket — a hygiene defect that compounds the F#666 violation.
- Sibling `exp_g4_adapter_initialization_comparison_v2` (open at P=2) addresses overlapping ground (reproducibility-axis) with target-paired KC structure (K1977 cos proxy + K1978 PPL-ratio target + K1979 PPL-variance-within-init target). v2 K1979 directly answers the seed-determinism question in PPL terms — bulk of this experiment's research question is closed by v2 result. Recommendation: gate any re-registration on v2 completion.
- F#169 already established init-method-invariance ("B compensates for any A configuration"); the seed-determinism question is a within-init-recipe extension that may have similar structural answer (B compensates for seed variation in A) — but this would still be a *parameter-space* finding, requiring behavioral pairing per F#666.
- Threshold pair (0.80, 0.30) is unanchored. F#562 measurements show Grassmannian QR init produces cos≈4.77e-9 at init (machine precision); F#751 final-cos values 0.977–0.9995 (pre-PRNG-fix) suggest cross-seed within-init is plausibly >0.95. The intermediate band may never fire in practice, making the dual-tail design an arbitrary upper-tail threshold (0.80) in disguise.
- Pre-reg notes do not specify seed-handling protocol. Per F#750, naive shared-PRNG-key implementation produces correlated starting matrices and confounds any cross-seed similarity claim. Even with the F#666 fix, an F#750 confound would remain unless distinct top-level seeds are pre-registered.
- Base model `mlx-community/gemma-4-e4b-it-4bit` not loaded; no adapters trained; no MLX code executed.

## Follow-up (recommended)

If a cross-seed adapter similarity ablation on Gemma 4 E4B is still a research question of interest *after* `exp_g4_adapter_initialization_comparison_v2` completes (v2 K1979 directly measures within-init seed-variance on PPL), register `exp_g4_adapter_similarity_across_seeds_behavioral` with target-gated KCs:

```yaml
kill_criteria:
  - K_proxy_cos_high_anchored:
      Pairwise cos > 0.95 (anchored to F#562 baseline) on >= 8/10 pairs of N=5 same-config different-seed adapters
  - K_target_ppl_spread:
      Eval-PPL ratio worst/best <= 1.05 on F#627 medical-domain held-out batch (target-paired with cos-sim claim, F#751 v2 K1978/K1979 pattern)
  - K_target_behavioral_equivalence:
      On N=20 medical held-out prompts, >= 90% produce LLM-judge-equivalent or rouge-L > 0.85 across all 5 seeds (behavioral identity test, F#666 canonical pattern)
  - K_target_pareto_quality:
      Among seed-pairs that PASS K_proxy_cos_high_anchored, max-pair PPL-divergence < 5% AND median-pair behavioral-equivalence > 0.90
references:
  - F#666 (target-gated KILL discipline)
  - F#169 (init-method-invariance prior coverage)
  - F#562 (Grassmannian orthogonality baseline -- anchors threshold)
  - F#627 (medical recipe -- training config)
  - F#750 (PRNG-key antipattern -- mandates distinct top-level seeds)
  - F#751 v1 + v2 (init-comparison sibling, target-paired template)
  - exp_g4_adapter_initialization_comparison_v2 (open, MUST complete first)
  - arxiv:1909.10314 (Bouthillier 2019 -- formal reproducibility framework)
  - arxiv:2402.01906 (Madaan 2024 -- LoRA cross-seed task-accuracy stability, target-pair runnable template)
  - arxiv:2106.09685 (Hu 2021 -- LoRA original, ablation context)
platform: local-apple
seed_handling_protocol: |
  Use distinct mx.random.key(seed_i) for each of N=5 trainings, where seed_i in {42, 137, 314, 1337, 2718}.
  NOT split-of-shared-master-key (F#750 confound).
sample_size: N=10 (45 pairs) for statistical power; report distribution (mean, std, percentiles)
success_criteria: [K_target_ppl_spread PASS AND K_target_behavioral_equivalence PASS AND >=1 pair PASSES K_target_pareto_quality]
notes: "Cross-seed adapter reproducibility on Gemma 4 E4B q_proj r=6 medical recipe (F#627). Cos-sim threshold anchored to F#562 baseline. Target-paired with PPL-spread + behavioral-equivalence. Gated on exp_g4_adapter_initialization_comparison_v2 K1979 result first."
parent_dependency: exp_g4_adapter_initialization_comparison_v2 must reach status=supported before this can claim
```

This closes the F#666 gap (target-pair present), addresses guardrail 1002 (references comprehensive), anchors the threshold (F#562 baseline), specifies the seed-protocol (F#750-compliant), increases statistical power (N=10), and satisfies the F#669-style child-anchor requirement (parent v2 must be supported).

## Unblock condition (no rerun of this pre-reg)

See MATH.md §5. Pre-reg must be edited before any re-claim to add target-metric KCs, references, populate success_criteria, set platform=local-apple, anchor threshold pair, and specify seed-handling protocol. Post-claim KC mutation is antipattern-u; recommendation is to **close this pre-reg as structurally-malformed** and use the follow-up template above instead, gated on `exp_g4_adapter_initialization_comparison_v2` reaching supported.
