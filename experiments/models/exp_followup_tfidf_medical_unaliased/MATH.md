# MATH.md — exp_followup_tfidf_medical_unaliased (PREEMPT-KILL, F#666-pure, 3rd instance)

## Verdict: PREEMPT-KILL (KC-structural, F#666-pure standalone — 3rd drain-window instance)

This experiment is preempt-killed before any code is run. The kill is **structural**: the pre-registered kill-criterion set K = {K1569} consists of a single proxy classification-accuracy metric (routing weighted accuracy) with no target-metric pairing. Under F#666 (guardrail 1007) neither KILL nor SUPPORTED is derivable regardless of empirical outcome.

This is the **third** F#666-pure standalone preempt-KILL in the drain window (after F#700 `exp_g4_per_layer_cos_baseline` and F#701 `exp_adapter_orthogonality_audit`). Per the promoted antipattern memory `mem-antipattern-f666-pure-standalone-preempt-kill` the 3rd instance triggers escalation to **reviewer.md §5 explicit clause** (analyst action this iteration).

## §0 Platform / skills / model pins

Included for completeness per reviewer checklist item (m2), even though no platform code executes.
- Platform skills: `/mlx-dev` + `/fast-mlx` (per PLAN.md Part 2). **Not invoked** — no MLX code written; canonical preempt-form disclosure.
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627). **Not loaded.**
- Adapter targets: N/A — routing experiments test classifier accuracy over prompt/domain text; no LoRA injection, no adapter composition.
- Parent dependency: **none** (`depends_on: []`). This is NOT an F#669 preempt.
- Prior experiment that would have been re-run without aliasing: `exp_p1_t4_tfidf_routing_v2` (status=**SUPPORTED**, N=25 weighted acc 84.2% with hard negatives including `clinical_knowledge`). **Not re-run.**

## §1 Preempt-KILL theorem (F#666-pure, 3rd instance)

**Theorem (KC-structural invalidity under target-gated KILL).** Let `E` denote experiment `exp_followup_tfidf_medical_unaliased` with kill-criterion set K = {K1569}:
- K1569 := "Unaliased N=25 TF-IDF routing achieves >=85% weighted accuracy (else aliasing was the lift)"

**Classification of K.**
- K1569 is explicitly a **proxy metric** — F#666 guardrail 1007 lists *classification accuracy* and *routing match rate* by name as forbidden-solo proxies. Weighted routing accuracy is precisely "routing match rate" under a weighting scheme. The F#666 result narrative is explicit on routing: "routing experiments must pair classification-accuracy KC with target-metric oracle-gap KC; otherwise threshold-on-proxy fails while target metric is achieved via different mechanism."

Neither task accuracy, behavioral quality, oracle-gap, nor any downstream-behavioral outcome is measured. K is a proxy-only set (1 proxy, 0 targets).

**F#666 gating (guardrail 1007).** KILL requires **both** a failing proxy KC and a failing target KC. SUPPORTED requires **both** to pass. A verdict derived from a proxy-only KC set is tautological:
- Proxy-PASS (≥85% accuracy, no aliasing) = tautological SUPPORT — does **not** imply selected adapters produce useful downstream output. F#666 canonical case demonstrated 40.2% per-sample proxy FAIL yet 0.0% target gap (full behavioral success), so proxy PASS is not behaviorally sufficient.
- Proxy-FAIL (<85% accuracy, no aliasing) = cannot KILL per F#666. F#666 rule explicitly: "Proxy-FAIL + target-absent = finding about the proxy, not a kill." The finding would be "TF-IDF weighted accuracy without `medical↔clinical_knowledge` aliasing is X%" — a proxy diagnostic, not a behavioral kill.

**Corollary (standalone F#666-pure preempt).** Let V(K) be the verdict derivable from K = {K1569}. For every assignment in {PASS, FAIL} to K1569:

| K1569 | V(K) under F#666                                                                             |
| ----- | -------------------------------------------------------------------------------------------- |
| PASS  | Proxy-SUPPORT with no target pair → tautological (F#666 canonical counter-example: routing can pass target while proxy fails, so proxy-pass-alone is not sufficient) |
| FAIL  | Proxy-only FAIL; per F#666, "Proxy-FAIL + target-absent = finding about the proxy, not kill" |

**Neither cell yields a valid verdict under F#666.** K is unidentifiable. **QED.**

**Semantic corroboration via parent-SUPPORTED status.** The parent experiment `exp_p1_t4_tfidf_routing_v2` is already `status=supported` with K1238 PASS: N=25 weighted accuracy **84.2%** on disjoint splits with hard negatives (including `clinical_knowledge`, `virology`, `biology`). The current experiment's 85% threshold is +0.8 percentage points above the parent's measured value. Two independent issues compound on F#666:

1. The parent **already does not alias** medical↔clinical_knowledge — per parent notes "includes hard-negative MMLU subjects (clinical_knowledge, virology, biology)." So the premise of this follow-up ("the prior was broken by aliasing") is factually wrong: the prior was clean; the purported "self-inflicted break" is a mis-remembered killed_07.md entry. The pre-reg's stated motivation does not match DB ground truth.
2. Even with valid motivation, a ~1% threshold delta on a proxy metric is not a behavioral finding. F#666 guardrail says routing acc alone cannot support or kill; it must be paired with oracle-gap or downstream task accuracy.

### §1.1 Secondary structural defects

Per `experiment get exp_followup_tfidf_medical_unaliased`:

1. **`success_criteria: []`** — empty. No SUPPORTED-condition declared. Independent of F#666, this blocks any SUPPORTED verdict.
2. **`references: []`** — violates guardrail 1002 ("Every new experiment MUST cite an arxiv paper or prior finding"). The notes field says "killed_07.md exp_p1_t4_tfidf_routing_v2 self-inflicted break" but the parent is actually `status=supported`; no formal citation of F#666 (the guardrail this experiment violates), F#251/F#257 (TF-IDF + logistic routing prior art), or arxiv routing papers.
3. **`platform: local-apple`** — set (one hygiene defect fewer than F#700/F#701).

Hygiene-defect count = 2 (vs 3 in F#700/F#701). This is **not** a match for the promoted `AP-prereg-hygiene-multi-defect` antipattern (which requires 3+ hygiene defects); it IS a match for `AP-F666-pure-standalone` (which keys only on the proxy-only KC structure, hygiene is incidental).

## §2 Prior art

- **F#666** (target-gated KILL discipline, guardrail 1007): KILL requires proxy+target, SUPPORTED requires proxy+target; routing experiments specifically must pair classification-accuracy with oracle-gap / downstream-task-accuracy.
- **F#700** (2026-04-24): 1st F#666-pure standalone preempt-KILL (`exp_g4_per_layer_cos_baseline`, K1856 cos-sim variance).
- **F#701** (2026-04-24): 2nd F#666-pure standalone preempt-KILL (`exp_adapter_orthogonality_audit`, K1857 cosine + K1858 effective rank). Promotion trigger reached.
- **This experiment (3rd instance)**: escalation trigger — analyst should edit `reviewer.md §5` to add an explicit F#666-pure-standalone preempt clause separate from the F#669 family.
- **`exp_p1_t4_tfidf_routing_v2`** (2026-04-11, `status=supported`): Parent/predecessor with K1238 PASS at N=25 weighted acc 84.2% on disjoint splits with hard negatives including `clinical_knowledge`. The 85% threshold of K1569 is 0.8pp tighter than the parent's measured value. Disproves the "aliasing was the lift" premise.
- **Guardrail 1002**: experiments MUST cite a paper/finding.
- **Guardrail 1007** (F#666): target-gated KILL.
- `mem-antipattern-f666-pure-standalone-preempt-kill` (filed 2026-04-24 after F#701): explicit detection rule at claim-time + preempt-scaffold response.

## §3 Predictions (registered, not measured)

| KC    | Claim                                                                             | Kind  | Measurement status                                |
| ----- | --------------------------------------------------------------------------------- | ----- | ------------------------------------------------- |
| K1569 | Unaliased N=25 TF-IDF routing achieves >=85% weighted accuracy                    | proxy | untested (preempt-blocked, F#666-pure structural) |

No target-metric KC exists. K is structurally malformed per F#666.

KC text is preserved verbatim from DB (checked byte-for-byte against `experiment get` output). No post-claim KC mutation.

## §4 Unblock condition

Re-claim requires **KC-augmentation** (pre-registration modification before re-claim):

1. Add at least one target-metric KC pairing K1569 to a behavioral outcome. Candidate formulations:
   - "At N=25 routing hits ≥85% weighted accuracy **AND** end-to-end task accuracy (MMLU-Pro subject classification with correct adapter selection → generation) within 3pp of oracle-selected-adapter baseline." — directly couples routing acc to downstream utility.
   - Or: "Routing weighted acc ≥85% **AND** Spearman |r|≥0.4 between per-sample routing confidence and generation-quality-delta-vs-oracle." — ties proxy to behavioral anchor.
2. Add references: at minimum F#666 (the violated guardrail), F#251/F#257 (prior TF-IDF + logistic routing findings), parent `exp_p1_t4_tfidf_routing_v2`. Ideally cite arxiv 2504.10957 (Zhong et al. task arithmetic) or 2310.14840 (arrow routing) for the downstream-task-accuracy link.
3. `platform=local-apple` already set — no fix needed.
4. Populate `success_criteria` (mirror of KC pass condition).
5. Correct the `notes` field: the parent is **SUPPORTED**, not killed; the "self-inflicted break" premise is factually wrong and must be removed or the motivation reformulated.

These edits must happen **before** re-claim (KC-pre-registration rule; post-claim KC mutation is antipattern-u per reviewer checklist).

**Note on research value.** The parent already SUPPORTED N=25 weighted acc 84.2% on disjoint splits with hard-negative clinical_knowledge. The marginal 0.8pp threshold delta on a proxy metric, even if measured, is behaviorally meaningless: per F#666 canonical result, a routing mechanism can exhibit 40.2% per-sample proxy accuracy yet achieve 0.0% target gap via semantic-cluster routing. Without the target-metric pair, nothing about routing utility is learned.

Recommendation: **close this pre-reg as structurally-malformed**, do not resurrect. If routing-utility questions remain, re-register under a fresh experiment id with target-gated KC + behavioral outcome (routing-confidence × generation-quality correlation, or end-to-end adapter-selection→task-accuracy).

## §5 Follow-up

No `_impl` companion filed — preempt-structural KILL does NOT spawn `_impl` (per F#687/F#698/F#699/F#700/F#701 precedent + reviewer.md §5). Unblock is pre-registration-external (requires editing the DB entry, not writing an impl).

No `mem-antipattern-impl-follow-up-delegation` obligation: that antipattern applies to novel-mechanism PROVISIONAL only. Preempt-structural KILL is explicitly excluded.

## §6 Scope integrity

No silent objective swap (antipattern-t): this scaffold does NOT:
- Run the TF-IDF classifier and mark K1569 PASS/FAIL in isolation (would produce an F#666-tautological verdict).
- Invent a target-metric KC post-claim (would be antipattern-u, post-claim KC mutation).
- Swap to an easier proxy (e.g. top-2 accuracy, or N<25) — preserves the F#666 violation.
- Substitute a different dataset (e.g. re-alias medical↔clinical_knowledge and compare) — would misrepresent the pre-reg.

KC text preserved verbatim from DB. Results.json encodes K1569 with `result=untested` and preempt reason. No silent simplification.

## §7 Sub-case taxonomy — 3rd instance escalation

Drain-window taxonomy after this iteration:

| Sub-case                                      | Parent status       | KC-structure        | Finding                  | Count |
| --------------------------------------------- | ------------------- | ------------------- | ------------------------ | ----- |
| F#669 classic (parent-unverified, F#666-ok)   | PROVISIONAL         | target-gated        | F#669/F#687/F#699        | 3     |
| F#669 + F#666 compound                        | PROVISIONAL         | proxy-only          | F#698                    | 1     |
| **F#666-pure standalone**                     | **none**            | **proxy-only**      | **F#700 + F#701 + this** | **3** |
| (runnable, F#666-compliant)                   | none / SUPPORTED    | target-gated        | regular KILL/SUPPORT     | N/A   |

**Row 3 reaches 3 instances.** Per promoted antipattern memory `mem-antipattern-f666-pure-standalone-preempt-kill`: "Escalation 3rd instance → reviewer.md §5 explicit clause" — analyst action this iteration.

Distinction from F#700/F#701:
- F#700/F#701: 3 hygiene defects (empty success_criteria + empty references + null platform) + F#666 violation
- This experiment: 2 hygiene defects (success_criteria + references empty; platform **set**) + F#666 violation
- Common invariant: **F#666-pure KC set + standalone (no `depends_on`)**. Hygiene defects are incidental, confirming `AP-F666-pure-standalone` keys on KC structure not hygiene.

Also distinct from F#702 (1st hygiene-patch PROVISIONAL): that case had **target-metric** KCs (K1909 wall-clock latency, K1910 bitwise-exact token equivalence), so F#666 did not bar verdict; here K1569 is purely proxy and F#666 bars any verdict.

## §8 Followup suggestion (non-binding — for analyst / future researcher)

If routing-utility at N=25 on Gemma 4 E4B remains an open research question, a well-formed follow-up would be:

```
title: exp_followup_tfidf_routing_n25_behavioral_pair
kill_criteria:
  K_proxy: routing weighted accuracy ≥85% at N=25 (disjoint splits, no aliasing)
  K_target: end-to-end MMLU-Pro subject-domain accuracy within 3pp of oracle-adapter baseline
          (measures whether routing choices translate to downstream utility)
references: F#666, F#251, F#257, exp_p1_t4_tfidf_routing_v2 (parent SUPPORTED)
platform: local-apple
success_criteria: [both KCs PASS]
```

Analyst may register this after closing the current pre-reg; explicit handover below.
