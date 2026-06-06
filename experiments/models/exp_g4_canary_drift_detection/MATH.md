# exp_g4_canary_drift_detection — MATH.md (F#666-pure standalone preempt-KILL)

## §0 Base model / adapters / required skills disclosure

- **Base model:** Gemma 4 (target; unloaded — no compute performed).
- **Adapters:** none constructed (preempt-KILL scaffold, no training run).
- **Required platform skills (per PLAN.md Part 2):** `/mlx-dev`, `/fast-mlx` — **Not invoked. No MLX code written.** Canonical preempt disclosure per F#700/F#701/F#703/F#705 precedent; `run_experiment.py` imports `json` + `pathlib` only.

## §1 Theorem: V(K1581) is unidentifiable under F#666

**Pre-reg KC set (verbatim, from `experiment get exp_g4_canary_drift_detection`):**
- K1581: "FNR <= 5% on synthetic-corrupted adapter" — result: `untested`.

**Classification per guardrail 1007:** FNR is a False Negative Rate on a binary detection task — textbook "classification accuracy" per guardrail 1007's explicit enumeration ("classification accuracy, routing match rate, PPL, cosine, clustering purity"). K1581 measures detection-classifier performance on a SYNTHETIC test distribution (adapter is artificially corrupted, canary must flag it). The behavioral target — detection of adapters that cause user-visible task-accuracy drops — is NOT measured. K1581 is the sole KC; it pairs with no target-metric KC (task accuracy, behavioral quality, oracle-gap, post-canary-pass PASS@k preservation). `depends_on: []` — no parent supplies a target anchor; parent finding F#156 (Canary FNR=2.0% + mechanistic `Degradation ~ f(rho)*g(cos)` linkage) is an anchor in spirit but not operationalized in this pre-reg.

**Truth table (exhaustive, 2¹ = 2 outcome classes):**

| K1581 (FNR ≤ 5% on synthetic) | Interpretation under F#666 | Verdict identifiable? |
|-------------------------------|----------------------------|-----------------------|
| PASS | Tautological SUPPORT: detector works on SYNTHETIC corruption distribution. Cannot claim "canary protects users from real quality degradation" without behavioral anchor. Synthetic-corruption distribution may not match real-composition-induced corruption (the deployment case F#156 names as `rho=0.89` production regime). Reviewer applies antipattern-t. | No |
| FAIL | "A finding about the proxy, not a kill" per F#666 — synthetic-FNR > 5% does not establish that canary fails to protect users. Synthetic corruption may be unrealistically subtle (testing harder than production) or unrealistically aggressive (testing easier than production). Either way, behavioral quality drop is not established. | No |

**QED.** Every outcome class is unidentifiable. Running the experiment wastes compute and yields a verdict the reviewer must overturn. F#666 (target-gated KILL) governs: preempt-KILL before compute.

## §2 Prior art & governing findings

- **F#666** (conclusive, 2026-04-19, `exp_softmax_router_scaling`) — target-gated KILL discipline. Proxy-PASS alone is tautological; proxy-FAIL alone is "a finding about the proxy, not a kill". Pair proxy KCs with target-metric KCs.
- **F#700 / F#701 / F#703 / F#705** — F#666-pure standalone preempt-KILL precedents. `reviewer.md §5` clause `KILL (preempt-structural — F#666-pure standalone)` promoted at 3rd instance (2026-04-24); 4th (F#705) confirmed lexical expansion (PPL-as-proxy) with no re-promote.
- **F#156** (supported, 2026-03-28, `exp_canary_quality_detection`) — parent motivation. Canary FNR=2.0% CI[1.9%, 2.1%], paired with cosine-gating anti-correlation r=−0.41 AND mechanistic linkage "Degradation ~ f(rho)*g(cos)" that connects FNR to actual quality drop via `rho` (adapter-perturbation magnitude) and `cos` (cosine with base). The mechanistic formula is the target-metric anchor in the parent — it is NOT inherited by this pre-reg, which measures FNR on synthetic alone without rho/cos/task-accuracy pairing.
- **F#594** (killed, 2026-xx-xx, N=2 null-space LoRA composition) — cites canary as insufficient in isolation; "Composed null_A + null_B ... K1667 activation-probe advantage +15pp" demonstrated that canary-passing adapters can still leak information detectable by other probes. Operational analogue to F#666-pure failure mode: proxy-PASS without target-PASS is unidentifiable.

## §3 Pre-reg KC verbatim

```
kill_criteria:
  - id: 1581
    text: "FNR <= 5% on synthetic-corrupted adapter"
    result: untested
success_criteria: []  # empty (hygiene defect)
references: []        # empty (hygiene defect; notes cite F#156 informally)
platform: local-apple # present
depends_on: []
```

Hygiene defect count: 2 (success_criteria empty, references empty). Below the 3+ threshold for `mem-antipattern-prereg-hygiene-multi-defect`. F#666-pure-standalone applies independently of hygiene count.

## §4 Unblock path

**Do NOT patch K1581 via `experiment update` — KC mutation post-claim is antipattern-u.**

Re-register a new pre-reg `exp_g4_canary_drift_target_paired` with a paired target KC set derived from F#156's operational form:

- **K1 (target, load-bearing, behavioral):** On a held-out set of real N=25 compositions that DEGRADE HumanEval PASS@1 by ≥ 3pp vs N=1 baseline, canary TPR ≥ 95%. This is the actual deployment claim — detect adapters that hurt users.
- **K2 (target, load-bearing, false-positive bound):** On a held-out set of N=25 compositions that DO NOT degrade HumanEval PASS@1 (drop ≤ 1pp), canary FPR ≤ 10%. Prevents canary from blocking safe deployments.
- **K3 (proxy, sanity, conditional on K1 PASS):** FNR ≤ 5% on synthetic-corrupted adapter (the original K1581). Retained as sanity, NOT load-bearing.
- **K4 (mechanistic anchor, inherited from F#156):** Measure correlation between canary score and (rho·cos) metric; Pearson r ≥ 0.5.

KILL requires K1 FAIL (missing real corruptions) OR K2 FAIL (too many false alarms). SUPPORTED requires K1 PASS + K2 PASS. Proxy-only (K3/K4) outcomes are not verdicts per F#666.

## §5 No `_impl` companion

Preempt-structural KILL does not spawn `_impl` per F#687/F#698/F#699/F#700/F#701/F#703/F#705 precedent and `reviewer.md §5` F#666-pure clause. Unblock is pre-reg-external: a new pre-reg with paired target KCs, not a follow-up `_impl`.

## §6 Scope of this filing

- This is a **scaffold**, not an experiment. No training, no inference, no MLX computation. `run_experiment.py` imports `json + pathlib` only; `main()` writes a graceful-failure `results.json` with `verdict="KILLED"`, K1581 `untested`, preempt-reason `F666_PURE_PREEMPT_KILL`.
- Behavioral claim not measured. Parent F#156 remains `supported` and is untouched by this filing.

## §7 Taxonomic placement (drain-window row)

| Row | Experiment | Proxy flavor | §5 clause |
|-----|------------|--------------|-----------|
| 1 | F#700 `exp_g4_per_layer_cos_baseline` | cos-sim | promoted |
| 2 | F#701 `exp_adapter_orthogonality_audit` | pairwise-cos + eff-rank | promoted |
| 3 | F#703 `exp_followup_tfidf_medical_unaliased` | routing weighted-acc | promoted |
| 4 | F#705 `exp_g4_o1_removal_naive` | PPL | already promoted, lexical-expansion |
| **5** | **F#TBD `exp_g4_canary_drift_detection`** (this filing) | **FNR (classification-accuracy on synthetic)** | already promoted, near-canonical (guardrail 1007 explicit enumeration) |

Delta at row 5: first drain-window instance where the pure-proxy metric is **FNR on a detection classifier** — near-canonical to guardrail 1007's "classification accuracy" enumeration. Prior 4 rows exercised derived proxies (cos-sim, rank, routing-acc, PPL). Row 5 confirms the clause applies to the canonical named case. Potential taxonomy-refactor trigger (analyst decision) given 5+ instances, but clause remains operationally correct without refactor.

Distinctions:
- **vs F#669-family preempt-structural**: parent-orthogonal. `depends_on: []`; no parent needs to reach `supported`. F#156 is supported.
- **vs F#702 hygiene-patch PROVISIONAL**: F#702 had target-metric KCs making the experiment runnable despite hygiene defects. This pre-reg has no target KC — no hygiene patch rescues it.
- **vs tautological-inter-variant-delta (F#704)**: F#704 had a target metric but tautological framing. This pre-reg has a pure-proxy metric — different failure mode.
- **vs parent F#156**: F#156 measured FNR-on-synthetic BUT paired it with mechanistic linkage formula `f(rho)*g(cos)` connecting proxy to behavior. Parent was target-anchored implicitly via mechanism; this pre-reg is not.

## §8 Well-formed follow-up pre-reg template

```yaml
id: exp_g4_canary_drift_target_paired
title: "Gemma 4 canary protects against real N=25 composition-induced quality drops (TPR+FPR paired)"
priority: 2
scale: micro
platform: local-apple
depends_on: [exp_canary_quality_detection]  # F#156 prereq
success_criteria:
  - "K1 PASS + K2 PASS → canary is deployment-ready on Gemma 4 at N=25"
kill_criteria:
  - id: TBD-K1
    text: "On held-out N=25 comps with HumanEval PASS@1 drop >= 3pp, canary TPR >= 95%"
    metric: canary_tpr_on_degrading_comps
    type: target
    result: untested
  - id: TBD-K2
    text: "On held-out N=25 comps with HumanEval PASS@1 drop <= 1pp, canary FPR <= 10%"
    metric: canary_fpr_on_safe_comps
    type: target
    result: untested
  - id: TBD-K3
    text: "FNR <= 5% on synthetic-corrupted adapter (sanity, conditional on K1)"
    metric: canary_fnr_synthetic
    type: proxy
    result: untested
  - id: TBD-K4
    text: "Pearson correlation between canary score and (rho·cos) >= 0.5"
    metric: canary_rho_cos_correlation
    type: mechanistic_anchor
    result: untested
references:
  - F#156  # parent: canary FNR=2% with rho*cos mechanistic linkage
  - F#594  # cautionary: canary-passing adapters can still leak via other probes
  - F#666  # governing: target-gated KILL discipline
tags: [g4-gemma4, canary, detection, paired-kc, f666-compliant]
```

— End MATH.md —
