# PAPER — exp_followup_tfidf_medical_unaliased (KILLED — preempt-structural, F#666-pure, 3rd instance)

## Verdict: KILLED (KC-structural preempt, F#666-pure standalone, 3rd drain-window instance)

This experiment is preempt-killed on structural grounds before any code executes. Verdict is deterministic from the KC-set shape, not from measurement.

## Summary

Pre-registered kill-criterion set K = {K1569} contains a single proxy classification-accuracy metric ("unaliased N=25 TF-IDF routing achieves ≥85% weighted accuracy") with no paired target-metric KC. Under Finding #666 (guardrail 1007 — target-gated KILL discipline), a proxy-only KC set has no valid verdict regardless of empirical outcome: proxy-PASS-alone is tautological (routing accuracy does not imply downstream utility — F#666 canonical result showed 40.2% proxy acc + 0.0% target gap), proxy-FAIL-alone cannot KILL (F#666 explicit: "Proxy-FAIL + target-absent = finding about the proxy, not kill").

This is the **3rd F#666-pure standalone preempt-KILL** in the current drain window (after F#700, F#701). The promoted antipattern memory `mem-antipattern-f666-pure-standalone-preempt-kill` (filed 2026-04-24) specifies 3rd-instance escalation: analyst should add explicit F#666-pure-standalone preempt clause to `reviewer.md §5`.

A secondary structural problem strengthens the kill: the pre-reg `notes` field cites "killed_07.md exp_p1_t4_tfidf_routing_v2 self-inflicted break" but the parent is actually `status=SUPPORTED` with K1238 PASS (N=25 weighted acc **84.2%** on disjoint splits with hard-negative `clinical_knowledge`, `virology`, `biology`). The premise "aliasing was the lift" is factually wrong — the parent already did not alias medical↔clinical_knowledge. Even absent F#666, the motivation is disproven.

## Prediction vs measurement

| KC    | Claim                                                          | Kind  | Verdict                                      |
| ----- | -------------------------------------------------------------- | ----- | -------------------------------------------- |
| K1569 | Unaliased N=25 TF-IDF routing achieves >=85% weighted accuracy | proxy | UNTESTED (preempt-blocked, F#666-pure)       |

No measurement was taken. No TF-IDF classifier was trained. No routing accuracy was computed. The verdict derives from `F#666 proxy-only KC set` + `no target-metric pair` ⇒ tautological-for-both-outcomes (see truth table in MATH.md §1).

## Hygiene fixes (not applicable)

Unlike F#702 (hygiene-patch PROVISIONAL where KCs were target-metric and the experiment was runnable), F#666 here bars any patch: the KC structure itself is malformed. The fix is pre-registration-level (add target KC + correct motivation), not code-level.

For reference, hygiene defects observed on this pre-reg:

| Defect                 | Status                                                                     |
| ---------------------- | -------------------------------------------------------------------------- |
| F#666 violation        | Present (K1569 is proxy-only; forbidden solo per guardrail 1007)           |
| success_criteria: []   | Present                                                                    |
| references: []         | Present (guardrail 1002 violation)                                         |
| platform: local-apple  | **Set** (one hygiene defect fewer than F#700/F#701)                        |
| motivation accuracy    | **Wrong** (claims parent was killed by aliasing; parent is SUPPORTED)      |

Hygiene count = 2 (not 3). This is below the AP-prereg-hygiene-multi-defect threshold; `AP-F666-pure-standalone` alone applies.

## Taxonomic comparison with drain-window precedents

| Dimension              | F#700 (1st F#666-pure)           | F#701 (2nd F#666-pure)           | This (3rd F#666-pure)                | F#702 (1st hygiene-patch)            |
| ---------------------- | -------------------------------- | -------------------------------- | ------------------------------------ | ------------------------------------ |
| Parent dep             | none                             | none                             | none                                 | none                                 |
| KC count               | 1 (K1856)                        | 2 (K1857, K1858)                 | 1 (K1569)                            | 2 (K1909, K1910)                     |
| KC kind                | proxy-only (variance)            | proxy-only (cosine + rank)       | proxy-only (routing match rate)      | **target-metric** (latency + tokens) |
| F#666 violation        | yes                              | yes                              | yes                                  | no (runnable under F#666)            |
| hygiene defects        | 3 (SC + ref + platform)          | 3 (SC + ref + platform)          | 2 (SC + ref; platform set)           | 3 (SC + ref + platform)              |
| motivation accuracy    | (unchecked)                      | (unchecked)                      | **wrong** (parent is SUPPORTED)      | ok                                   |
| verdict                | KILLED (preempt-structural)      | KILLED (preempt-structural)      | **KILLED (preempt-structural)**      | PROVISIONAL (design-lock)            |
| `_impl` follow-up      | none                             | none                             | none                                 | yes                                  |

The invariant that makes `AP-F666-pure-standalone` a clean antipattern: `depends_on: []` + proxy-only KC set ⇒ preempt-KILL, independent of hygiene count or motivation accuracy.

## Caveats

- K is structurally malformed per F#666 — no measurement can produce a valid verdict. Running the classifier would waste compute and risk antipattern-t (structurally invalid verdict) if results were marked PASS/FAIL in isolation.
- Secondary semantic corroboration: parent SUPPORTED at 84.2% with the "aliasing" hard-negatives already in place. Premise of this follow-up is disproven independently.
- Base model `mlx-community/gemma-4-e4b-it-4bit` not loaded; no adapters injected; no MLX code executed.

## Follow-up (recommended)

If routing-utility at N=25 on Gemma 4 E4B is still a research question of interest, register `exp_followup_tfidf_routing_n25_behavioral_pair` with:

```
K_proxy  = routing weighted acc ≥85% at N=25 (disjoint splits, no aliasing)
K_target = end-to-end MMLU-Pro subject-domain accuracy within 3pp of oracle-adapter baseline
references = F#666, F#251, F#257, exp_p1_t4_tfidf_routing_v2 (parent SUPPORTED)
platform = local-apple
success_criteria = [both KCs PASS]
notes = corrected motivation (parent is SUPPORTED; this extends with target-metric pair)
```

This closes the F#666 gap and couples routing accuracy to downstream utility.

## Unblock condition (non-rerun of this pre-reg — KC-augmentation-only)

See MATH.md §4. Pre-reg must be edited to (a) add a target-metric KC, (b) add references, (c) populate success_criteria, (d) correct motivation text. Post-claim KC mutation is antipattern-u, so edits must happen before any re-claim. Recommendation is to close this pre-reg and use the follow-up template above instead.
