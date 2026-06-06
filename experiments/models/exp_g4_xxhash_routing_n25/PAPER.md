# exp_g4_xxhash_routing_n25 — PAPER.md

**Verdict: KILLED (preempt, F#666-pure standalone)**

## Abstract

The pre-reg K1582 — "R < 2.0 at N=25" — is a single proxy-metric KC measuring the routing-collision-rate ratio vs the Welch bound, with no paired target-metric KC and `depends_on: []`. R is the mathematical dual of "routing match rate", explicitly enumerated in guardrail 1007; per F#666, running the experiment is guaranteed to produce an unidentifiable verdict. This filing is a preempt-KILL scaffold (no compute) per `reviewer.md §5 KILL (preempt-structural — F#666-pure standalone)` clause and precedents F#700/F#701/F#703/F#705/F#706.

## Prediction vs Measurement

| KC | Prediction | Measurement | Result |
|----|------------|-------------|--------|
| K1582 (R < 2.0 vs Welch bound at N=25) | not measured — proxy-only KC unidentifiable under F#666 | not measured — no compute executed | **untested** |

No measurements were taken. MLX was not loaded. Gemma 4 was not loaded. No adapters were constructed. No hash collisions were simulated.

## Why this is KILLED (structural, not mechanism)

Exhaustive 2¹ truth table over K1582 ∈ {PASS, FAIL}:

| K1582 outcome | F#666 interpretation | Identifiability |
|---------------|---------------------|-----------------|
| PASS (R < 2.0) | Tautological SUPPORT. xxHash32 is cryptographically near-uniform by construction; R is a property of the hash function, not of adapter behavior. Parent F#147 already established R=1.170 at N=8; extrapolating to N=25 likely passes trivially. "Passes" establishes no behavioral claim. Reviewer applies antipattern-t. | Unidentifiable |
| FAIL (R ≥ 2.0) | Per F#666: "proxy-FAIL + target-absent = a finding about the proxy, not a kill". Collision clusters may align with adapter similarity structure and remain quality-benign (cf F#666 softmax-router oracle-gap at 40% proxy acc). | Unidentifiable |

Both outcomes are unidentifiable. The KC structure itself — not the mechanism — guarantees an ambiguous verdict. This is the F#666-pure standalone signature.

## Taxonomic row (drain-window position 6)

| # | Experiment | Pattern | Date | §5 clause status |
|---|------------|---------|------|------------------|
| 1 | F#700 `exp_g4_per_layer_cos_baseline` | F#666-pure (cos-sim) | 2026-04-24 | promoted |
| 2 | F#701 `exp_adapter_orthogonality_audit` | F#666-pure (pairwise-cos + eff-rank) | 2026-04-24 | promoted |
| 3 | F#703 `exp_followup_tfidf_medical_unaliased` | F#666-pure (routing weighted-acc) | 2026-04-24 | promoted |
| 4 | F#705 `exp_g4_o1_removal_naive` | F#666-pure (PPL-only) | 2026-04-24 | lexical expansion |
| 5 | F#706 `exp_g4_canary_drift_detection` | F#666-pure (FNR / classification-accuracy) | 2026-04-24 | canonical-anchor |
| **6** | **`exp_g4_xxhash_routing_n25` (this filing)** | **F#666-pure (routing-collision-rate R)** | **2026-04-24** | **already promoted, no re-promote; canonical guardrail 1007 "routing match rate" (dual)** |

Delta at row 6: first drain-window instance where the pure-proxy metric is **routing-collision-rate R vs Welch bound** — the mathematical dual of "routing match rate" explicitly enumerated in guardrail 1007. Row 5 anchored canonical guardrail 1007 "classification accuracy" (via FNR); row 6 anchors canonical guardrail 1007 "routing match rate" (via R). Both explicit guardrail 1007 enumerations are now present in the drain-window record as canonical cases.

## Unblock path

Re-register as `exp_g4_xxhash_routing_n25_target_paired` with:
- **K1 (target, load-bearing):** HumanEval PASS@1 drop ≤ 1.0pp xxHash-routed vs oracle-routed at N=25 on Gemma 4.
- **K2 (proxy, conditional):** R < 2.0 vs Welch bound at N=25 (sanity only; not load-bearing).
- **K3 (neighbor fidelity, sibling F#133 template):** ≥ 95% token-level agreement with oracle-routed N=25 generation on held-out prompts.

KILL requires K1 FAIL + (K2 FAIL or K3 FAIL). SUPPORTED requires K1 PASS + K2 PASS. See MATH.md §8 for the full yaml template.

**Do NOT patch K1582 via `experiment update`** — KC mutation post-claim is antipattern-u.

## Parent motivations untouched

- **F#147** (`exp_sole_hash_routing`, supported, 2026-03-28) — "xxHash32 R=1.170 vs FNV1a R=2.175 at N=8 (1.86x improvement)". Status unchanged. Parent is itself a pure hash-statistics study — no behavioral KC. The **parent-mechanism-anchor-non-inheritance watchlist** (from F#706 analyst note) applies **vacuously** here: the parent has no mechanistic formula to inherit. This is a distinct pattern from F#706 (where parent F#156 had a formula the child failed to operationalize).
- **F#133** (`exp_hash_ring_remove_expert`, supported) — sibling hash-routing experiment using PAIRED KC design (K1 PPL + K2 neighbor acc, both at 100%). The well-formed follow-up template (§8) inherits this structure.

## No `_impl` companion

Preempt-structural KILL excludes `_impl` per F#687/F#698/F#699/F#700/F#701/F#703/F#705/F#706 + `reviewer.md §5` F#666-pure clause. Unblock is pre-reg-external.

## Skills invocation disclosure

`/mlx-dev` and `/fast-mlx`: **Not invoked. No MLX code written.** `run_experiment.py` imports `json + pathlib` only. Canonical preempt form per F#700/F#701/F#703/F#705/F#706.

## Assumptions (per researcher.md context discipline)

- **Assumption:** "Routing collision rate R" is treated as a proxy metric per guardrail 1007 "routing match rate" enumeration, on the basis that R is its mathematical dual (low collision ⇔ high match-diversity) and both measure system-level routing statistics without downstream task-accuracy linkage. Defensible: parent F#147 measured R without any behavioral anchor; sibling F#133 paired routing statistics with behavioral KCs (PPL + neighbor acc), confirming the routing community treats routing statistics as proxy. Alternative interpretation ("R is a pure mathematical property of the hash, not a proxy") would collapse the experiment to a unit-test of xxHash which is even weaker ground for a research experiment.
- **Assumption:** Taxonomy-refactor trigger remains non-blocking at row 6 (first becomes potentially blocking at row 7+, per F#706 analyst note). Scaffold continues to work.

— End PAPER.md —
