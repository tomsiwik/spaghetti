# exp_g4_hash_ring_remove_n25 — PAPER.md

**Verdict: KILLED (preempt, F#666-pure standalone)**

## Abstract

The pre-reg K1583 — "mean PPL <= 3%, max <= 5%" — is a single proxy-metric KC with no paired target-metric KC and `depends_on: []`. Per guardrail 1007 (and explicitly per F#666), PPL is a proxy; running the experiment is guaranteed to produce an unidentifiable verdict. This filing is a preempt-KILL scaffold (no compute) per `reviewer.md §5 KILL (preempt-structural — F#666-pure standalone)` clause and precedents F#700/F#701/F#703/F#705/F#706/F#707.

This is the **7th drain-window F#666-pure standalone instance** and the **2nd PPL-as-proxy instance** (after F#705). Notable sub-pattern: parent F#133 (`exp_hash_ring_remove_expert`) itself uses PAIRED KC design (K1 PPL + K2 neighbor accuracy) — the child stripped K2 and kept only the K1 PPL. This is **template-regression**, distinct from F#705's stale-caveat pattern (where parent F#161's *secondary advice* went stale post-guardrail-1007). 2nd template-regression instance triggers candidate antipattern memory filing.

## Prediction vs Measurement

| KC | Prediction | Measurement | Result |
|----|------------|-------------|--------|
| K1583 (mean PPL drop ≤3% AND max ≤5%, N=25→24 hash-ring remove) | not measured — proxy-only KC unidentifiable under F#666 | not measured — no compute executed | **untested** |

No measurements were taken. MLX was not loaded. Gemma 4 was not loaded. No adapters were constructed. No routing simulation ran.

## Why this is KILLED (structural, not mechanism)

Exhaustive 2¹ truth table over K1583 ∈ {PASS, FAIL} (both sub-thresholds collapse to one PPL-axis verdict):

| K1583 outcome | F#666 interpretation | Identifiability |
|---------------|---------------------|-----------------|
| PASS (mean ≤3%, max ≤5%) | Tautological SUPPORT. PPL-r≈0.08 with task quality means a 3% drop bound proves nothing about HumanEval / MMLU / behavioral equivalence. Reviewer applies antipattern-t. | Unidentifiable |
| FAIL (mean >3% OR max >5%) | Per F#666: "proxy-FAIL + target-absent = a finding about the proxy, not a kill". Does not establish behavioral quality drop; may reflect PPL's known 0.08-correlation distortion. | Unidentifiable |

Both outcomes are unidentifiable. The KC structure itself — not the mechanism — guarantees an ambiguous verdict. This is the F#666-pure standalone signature.

## Taxonomic row (drain-window position 7)

| # | Experiment | Pattern | Proxy flavor | Date | §5 clause status |
|---|------------|---------|--------------|------|------------------|
| 1 | F#700 `exp_g4_per_layer_cos_baseline` | F#666-pure | cos-sim | 2026-04-24 | promoted |
| 2 | F#701 `exp_adapter_orthogonality_audit` | F#666-pure | pairwise-cos + eff-rank | 2026-04-24 | promoted |
| 3 | F#703 `exp_followup_tfidf_medical_unaliased` | F#666-pure | routing weighted-acc | 2026-04-24 | promoted |
| 4 | F#705 `exp_g4_o1_removal_naive` | F#666-pure | **PPL (1st)** | 2026-04-24 | no re-promote |
| 5 | F#706 `exp_g4_canary_drift_detection` | F#666-pure | FNR (canonical guardrail 1007 "classification accuracy") | 2026-04-24 | no re-promote |
| 6 | F#707 `exp_g4_xxhash_routing_n25` | F#666-pure | R / collision-rate (canonical guardrail 1007 "routing match rate" dual) | 2026-04-24 | no re-promote |
| **7** | **`exp_g4_hash_ring_remove_n25` (this filing)** | **F#666-pure** | **PPL (2nd) — mean+max sub-threshold variant** | **2026-04-24** | **no re-promote** |

Delta at row 7:
- **2nd PPL-as-proxy instance** — lexical-expansion within an already-anchored proxy flavor; not a new canonical guardrail 1007 anchor (rows 5/6 covered classification-accuracy / routing-match-rate canonicals).
- **2nd template-regression sub-pattern instance** — promotion threshold for sub-pattern is 3rd; 2nd triggers candidate antipattern memory filing per analyst non-blocking note.
- **Taxonomy-refactor trigger remains live** (was triggered at row 5 per analyst notes); 7 instances is the analyst-flagged "revisit at 7th+" inflection. Three refactor options on file: (a) super-category consolidation with F#669-family, (b) proxy-flavor split sub-categories, (c) "guardrail 1007 enumeration" sub-section.

## Sub-pattern: template-regression (2nd instance)

| Instance | Parent | Parent state | Child regression |
|----------|--------|--------------|------------------|
| 1st (F#705) | F#161 | supported with stale caveat ("PPL validation needed") pre-dating guardrail 1007 | Child built KC from stale parent caveat |
| **2nd (this filing)** | **F#133** | **supported with PAIRED KC design itself (K1 PPL + K2 neighbor accuracy, both measured)** | **Child stripped K2 pairing, kept only K1 PPL** |

Both produce F#666-pure children, but the upstream causal structure differs. 2nd instance crosses the candidate-antipattern threshold; analyst should consider promoting to a memory entry (3rd instance promotes to formal antipattern per F#704 / F#669 promotion convention).

## Unblock path

Re-register as `exp_g4_hash_ring_remove_n25_target_paired` mirroring the **parent F#133 PAIRED KC template**:

- **K1 (target, load-bearing):** HumanEval PASS@1 drop ≤ 1.0pp after hash-ring remove, N=25 → 24.
- **K2 (target, sibling F#133 template):** ≥ 95% neighbor accuracy on removed expert's hash neighborhood (parent's K2 scaled to N=25).
- **K3 (proxy, conditional):** mean PPL drop ≤ 3%, max ≤ 5% (sanity only; not load-bearing).

KILL requires K1 FAIL + (K2 FAIL or K3 FAIL). SUPPORTED requires K1 PASS + K2 PASS + K3 PASS. See MATH.md §8 for the full yaml template.

**Do NOT patch K1583 via `experiment update`** — KC mutation post-claim is antipattern-u.

## Parent unaffected

- **F#133** (`exp_hash_ring_remove_expert`, supported, 2026-03-15) — direct parent. Uses PAIRED KC design (K1 PPL + K2 neighbor accuracy, both measured at N=8). Status unchanged. This filing's existence does NOT call F#133 into question; F#133's design is in fact the well-formed template that the unblock path mirrors.

## No `_impl` companion

Preempt-structural KILL excludes `_impl` per F#687/F#698/F#699/F#700/F#701/F#703/F#705/F#706/F#707 + `reviewer.md §5` F#666-pure clause. Unblock is pre-reg-external.

## Skills invocation disclosure

`/mlx-dev` and `/fast-mlx`: **Not invoked. No MLX code written.** `run_experiment.py` imports `json + pathlib` only. Canonical preempt form per F#700/F#701/F#703/F#705/F#706/F#707.

## Assumptions

- The `audit-2026-04-17` tag (no `-rerun` suffix) indicates audit-lineage marking only, not a "KNOWN-BUGGY code requires fix-before-rerun" flag (per researcher.md workflow step 3 — only `audit-2026-04-17-rerun` triggers RECOVERY_PLAN.md application). No `run_experiment.py` existed in the experiment dir; this filing creates the preempt scaffold from scratch. Decision logged: assume tag is lineage-only; if reviewer disagrees, the verdict still stands (preempt-structural is independent of audit-rerun status).

— End PAPER.md —
