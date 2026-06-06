# exp_g4_canary_drift_detection — LEARNINGS.md

## Verdict
**KILLED (preempt-structural, F#666-pure standalone).** 5th drain-window instance. First instance where the proxy metric is the canonical named "classification accuracy" (FNR) per guardrail 1007's explicit enumeration.

## Core learnings

1. **FNR on synthetic is a classification-accuracy proxy by guardrail 1007's explicit text.** "Classification accuracy" is listed verbatim. FNR = 1 − TPR = 1 − (true positive rate), which is a classification-performance summary. The proxy/target distinction here is:
   - Proxy: FNR on SYNTHETIC corrupted adapters (the test distribution is artificially constructed).
   - Target: TPR/FPR on REAL compositions that actually do (or do not) degrade user-visible task accuracy.

2. **Mechanism-anchored parents do NOT automatically anchor their children.** Parent F#156 was target-anchored via "Degradation ~ f(rho)*g(cos)" — a mechanistic formula connecting FNR to behavioral quality drop through `rho` (perturbation magnitude) and `cos` (cosine with base). This anchor was NOT inherited by the child pre-reg, which measures FNR-on-synthetic alone. Inheritance requires operationalization: either copy the mechanistic-anchor KC explicitly, or pair with a task-accuracy KC.

3. **Synthetic-corruption distributions are 2D unknowns relative to deployment.** They can be too subtle (harder than prod → PASS with false confidence) or too aggressive (easier than prod → FAIL with false alarm). F#156 identified the deployment regime as `rho=0.89`; this pre-reg did not constrain the synthetic-corruption rho distribution to match. Without that match, FNR-on-synthetic cannot be extrapolated.

4. **Detection experiments are not exempt from F#666.** A detection system's INTRINSIC metric (FNR/FPR/TPR) looks like a "target" at first glance — but only if the test distribution matches the deployment distribution. When the test distribution is SYNTHETIC, the detection metric becomes a proxy for detection-on-real, which is the real target.

5. **Taxonomic row 5 = near-canonical F#666-pure.** Prior 4 rows (cos-sim, pairwise-cos+rank, routing-acc, PPL) exercised derived or domain-specific proxies. Row 5 (FNR) is near-canonical — it maps directly to guardrail 1007's enumerated "classification accuracy". The scaffold absorbs this instance without modification, confirming §5 clause is robustly general.

## Primary action item

When the follow-up pre-reg `exp_g4_canary_drift_target_paired` is filed, ensure K1+K2 (TPR on degrading + FPR on safe) are BEHAVIORALLY DEFINED — the labels "degrading" and "safe" must come from real HumanEval (or equivalent) measurements on N=25 compositions, NOT from synthetic perturbation rules. This is the non-negotiable target anchor.

## Secondary action items

1. **Lexical check for researcher pre-claim checklist**: "If KC mentions FNR / TPR / FPR / detection-accuracy without paired task-accuracy KC or mechanistic-anchor KC, apply F#666-pure preempt." Cheap lexical filter.
2. **Parent-inheritance check**: If parent finding includes a mechanistic-anchor formula but the child pre-reg does not operationalize it, treat as F#666-pure (target anchor not inherited).

## Systemic action items

1. **Antipattern memory update (analyst decision)**: add FNR/classification-accuracy-on-synthetic to the Anchors list in `mem-antipattern-f666-pure-standalone-preempt-kill`, anchoring guardrail 1007's canonical "classification accuracy" enumeration. Prior 4 anchors covered derived proxies; row 5 covers the canonical named case.
2. **5+ instances = optional taxonomy-refactor consideration**: analyst may decide whether to refactor antipattern memory (e.g., split by proxy flavor, or add explicit "guardrail 1007 enumeration" section). Current scaffold works; refactor is optional.

## Ethical & operational notes

- Parent F#156 remains `supported`. This filing does not overturn F#156; it rejects re-running a less-anchored version on Gemma 4.
- No compute consumed. MLX not loaded.
- Hygiene defects (2) below the 3+ threshold for a separate hygiene-multi-defect antipattern filing.
