# MATH.md — exp_routing_adapter_signature (PREEMPT-KILL structural)

## Framing

Experiment proposes routing-key = `hash(subspace(W_adapter))`, comparing against TF-IDF-on-training-data baseline. Kill criteria:

- **K1902**: Signature-based routing accuracy < TF-IDF routing baseline. [PROXY — routing accuracy is classification-accuracy-as-proxy per guardrail 1007 / F#706-lineage.]
- **K1903**: Signature computation > 1ms per adapter. [PROXY — infrastructure wall-clock benchmark per F#715 bucket.]

Zero target behavioral KC registered. Claim-time branch enumeration (per watchlist-correction meta-pattern, F#734) required before verdict.

## Branch enumeration (5 branches, verdict-binding)

| # | Branch | Admissible? | Cost | Verdict effect |
|---|--------|-------------|------|----------------|
| 1 | Add target behavioral KC (e.g., downstream task accuracy: does adapter ROUTED by signature produce task output equivalent to adapter routed by TF-IDF?) | Yes | 1 retrain + eval cycle | Unblocks F#666-pure; would run under target-gated kill rule |
| 2 | De-register K1902/K1903 and re-claim with zero-proxy KC | Yes | 0 (text edit) | Cheapest |
| 3 | Merge K1902 into a dual-anchor target+proxy KC (hybrid) | Yes | 1 design revision | Mid-cost |
| 4 | Attach external target via KC-inheritance from F#427 (TF-IDF routing baseline finding) | Yes | 1 lookup + patch | Low |
| 5 | Keep KCs as-is, run, claim result | **No — F#666-pure saturates** | N/A | Structural-KILL required |

**Admissible verdict path:** Branches 1/2/3/4. Cheapest = Branch 2.

## Theorem 1 (F#666-pure structural impossibility, 24th drain-window instance)

**Statement:** Both registered KCs (K1902, K1903) are proxy-only metrics — routing-accuracy and wall-clock compute time respectively. Neither measures target behavioral outcome (task accuracy, oracle-gap, behavioral quality). Under Finding #666 and guardrail 1007, an experiment with zero target behavioral KC cannot produce a SUPPORTED or meaningful KILLED verdict: any proxy-metric delta is, by construction, a finding about the proxy rather than a claim about the phenomenon.

**Proof:**
1. Guardrail 1007 (target-gated kill): every proxy KC must be paired with a target-metric KC; KILL requires BOTH to fail; SUPPORTED requires BOTH to pass.
2. K1902 measures routing accuracy — a classification-label match rate (routing label = adapter ID). By F#706-lineage (1st FNR-as-proxy), F#707 (routing-match-rate), F#710 (2nd routing-acc confirmed-recurrent), this is a canonical proxy-only KC sub-flavor.
3. K1903 measures wall-clock per-adapter compute — a physical-unit infrastructure benchmark. By F#715-lineage (infrastructure-benchmark bucket, wall-clock sub-flavor), physical-unit thresholds are proxy-only unless paired with behavioral-gain-vs-cost anchor KC. K1903 is not so paired.
4. No target behavioral KC is registered. The admissible verdict set under guardrail 1007 is {structural-KILL} only.
5. Therefore the experiment is F#666-pure structural-KILL 24th drain-window instance, with two distinguishing sub-flavors active (routing-accuracy proxy + wall-clock infrastructure). ∎

## Theorem 2 (F#715 infrastructure-benchmark bucket 6th drain-window, wall-clock sub-flavor continuing)

**Statement:** K1903 registers a wall-clock upper bound (1ms per adapter) with no accompanying behavioral-gain-vs-cost anchor. By F#715 promoted canonical memory (promoted at F#734 QUADRUPLE-FIRE), this matches the wall-clock sub-flavor pattern (F#715 K1860/K1861 original + F#732 K1894 + F#734 K1899 + F#735 K1901 + THIS K1903). This is the 5th wall-clock sub-flavor instance, 6th overall F#715 bucket instance counting the F#721 engineering-cost sibling.

**Sub-flavor enumeration (4 types confirmed by F#735):**

| Sub-flavor | Anchors |
|------------|---------|
| Wall-clock (ms/s bound) | F#715, F#732, F#734, F#735, **THIS** |
| Byte-size (MB bound) | F#715 K1861 |
| Engineering-cost-per-gain | F#721 |
| Reproducibility/variance-bound | F#735 K1900 |

K1903 firmly in wall-clock sub-flavor. No new sub-flavor added. Continues post-promotion anchor-append pattern. ∎

## Theorem 3 (F#702 hygiene-patch unavailable, N-th reuse, non-promoting)

**Statement:** Experiment lacks `success_criteria` and `platform`. By F#702 (promoted at F#716 3rd instance), missing hygiene fields are patched before proceeding unless the experiment structurally cannot benefit from patching. Under Theorem 1 saturation (F#666-pure), there is no valid target-metric KC to import into `success_criteria`, and no runtime platform to register because the experiment will not run.

**Proof:**
1. `success_criteria` by convention mirrors target behavioral KCs. None exist.
2. `platform` is register-before-run; under Theorem 1 the experiment is preempt-killed and does not run.
3. Therefore patching both fields is vacuous. F#702-unavailability derived-lemma applies. Non-promoting (N-th reuse tracked inline by analyst). ∎

## Theorem 4 (prior-art redundancy, non-promoting)

**Statement:** The research question — "does adapter structure encode routing signal?" — is already answered by prior findings:
- F#137: direction-probe r≈0.990 — adapter subspace DOES encode direction (positive answer).
- F#269: direction-interference between adapters documented (subspaces overlap).
- F#427: α=0.145 power law for TF-IDF routing (baseline quantified).
- F#453: max cos=0.0861 between adapter pairs (subspaces distinct).
- F#498: subspace destroys composition (subspace structure has functional consequence).

Signature hash of the subspace is a downstream engineering artifact of findings F#137/F#453. The experiment would re-measure known structure without adding behavioral claim. Even were target KC added (Branch 1), the result would replicate F#137 at a different abstraction level.

**Consequence:** The minimum-regret action is Branch 2 (de-register KCs) and re-file under `p3` / `deferred` with a target behavioral question such as "does signature-routed serving produce equivalent task output to TF-IDF-routed serving at N=25 adapters?" ∎

## Classification summary

**Verdict:** PREEMPT-KILL structural.

**Triple-fire axes (tracked for analyst synthesis):**
1. **F#666-pure 24th drain-window** (structural base) — 2 proxy-only KCs, zero target behavioral KC. Routing-accuracy sub-flavor confirmed-recurrent per F#710 lineage.
2. **F#715 infrastructure-benchmark bucket 6th drain-window** (wall-clock sub-flavor, 5th wall-clock instance counting F#715/F#732/F#734/F#735/THIS) — post-promotion anchor-append.
3. **F#706/F#707/F#710-lineage routing-accuracy-as-proxy explicit** (3rd sub-flavor instance post-F#710 confirmed-recurrent) — provides analyst signal whether to split `mem-antipattern-routing-accuracy-as-proxy` standalone vs keep inline in F#666-pure.

**Non-promoting:**
- F#702 hygiene-patch unavailable (derived-lemma, vacuous under Theorem 1).
- Prior-art redundancy (F#137/F#269/F#427/F#453/F#498).

**Distinguishing signal for taxonomy:**
- **NOT** §5 tautological-inter-variant-delta — this is inter-FAMILY (signature vs TF-IDF, different methods), not intra-family/intra-variant.
- **NOT** method-dependent-redundancy — does not compare method-A-adapter-combination-X vs method-B-adapter-combination-X; it compares two routing-key functions on shared adapter set.
- **NOT** tool-as-experiment — title frames a method (routing key function), not a reusable tool/artifact. (Watchlist candidate `exp_adapter_fingerprint_uniqueness` is the nearby tool-as-experiment risk; THIS experiment is adjacent but distinct.)
