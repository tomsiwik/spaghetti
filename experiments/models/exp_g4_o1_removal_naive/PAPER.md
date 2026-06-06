# exp_g4_o1_removal_naive — PAPER.md

**Verdict: KILLED (preempt, F#666-pure standalone)**

## Abstract

The pre-reg K1580 — "max PPL drift ≤ 0.2% after remove, N=25 → 24" — is a single proxy-metric KC with no paired target-metric KC and `depends_on: []`. Per guardrail 1007 (and explicitly per F#666), PPL is a proxy; running the experiment is guaranteed to produce an unidentifiable verdict. This filing is a preempt-KILL scaffold (no compute) per `reviewer.md §5 KILL (preempt-structural — F#666-pure standalone)` clause and precedents F#700/F#701/F#703.

## Prediction vs Measurement

| KC | Prediction | Measurement | Result |
|----|------------|-------------|--------|
| K1580 (PPL drift ≤ 0.2% after N=25→24 remove) | not measured — proxy-only KC unidentifiable under F#666 | not measured — no compute executed | **untested** |

No measurements were taken. MLX was not loaded. Gemma 4 was not loaded. No adapters were constructed.

## Why this is KILLED (structural, not mechanism)

Exhaustive 2¹ truth table over K1580 ∈ {PASS, FAIL}:

| K1580 outcome | F#666 interpretation | Identifiability |
|---------------|---------------------|-----------------|
| PASS (PPL drift ≤ 0.2%) | Tautological SUPPORT. PPL-r≈0.08 with task quality means 0.2% drift proves nothing about behavioral preservation. Reviewer applies antipattern-t. | Unidentifiable |
| FAIL (PPL drift > 0.2%) | Per F#666: "proxy-FAIL + target-absent = a finding about the proxy, not a kill". Does not establish behavioral quality drop. | Unidentifiable |

Both outcomes are unidentifiable. The KC structure itself — not the mechanism — guarantees an ambiguous verdict. This is the F#666-pure standalone signature.

## Taxonomic row (drain-window position 4)

| # | Experiment | Pattern | Date | §5 clause status |
|---|------------|---------|------|------------------|
| 1 | F#700 `exp_g4_per_layer_cos_baseline` | F#666-pure (cos-sim proxy-only) | 2026-04-24 | promoted |
| 2 | F#701 `exp_adapter_orthogonality_audit` | F#666-pure (pairwise-cos + eff-rank) | 2026-04-24 | promoted |
| 3 | F#703 `exp_followup_tfidf_medical_unaliased` | F#666-pure (routing weighted-acc) | 2026-04-24 | promoted |
| **4** | **`exp_g4_o1_removal_naive` (this filing)** | **F#666-pure (PPL-only)** | **2026-04-24** | **already promoted, no re-promote** |

Delta at row 4: first drain-window instance where the pure-proxy metric is PPL (prior 3 were cosine-family / routing-accuracy). Expands the F#666-pure lexicon to confirm PPL-only KC triggers the clause as expected per guardrail 1007 text.

## Unblock path

Re-register as `exp_g4_o1_removal_target_paired` with:
- **K1 (target, load-bearing):** HumanEval PASS@1 drop ≤ 1.0pp after N=25 → 24 removal on Gemma 4.
- **K2 (proxy, conditional):** PPL drift ≤ 0.2% (sanity only; not load-bearing).
- **K3 (neighbor fidelity, sibling F#133 template):** ≥ 95% token-level agreement with N=25 generation on held-out prompts.

KILL requires K1 FAIL + (K2 FAIL or K3 FAIL). SUPPORTED requires K1 PASS + K2 PASS. See MATH.md §8 for the full yaml template.

**Do NOT patch K1580 via `experiment update`** — KC mutation post-claim is antipattern-u.

## Parent motivations untouched

- **F#161** (`exp_attention_layer_removal_safety`, supported, 2026-03-15) — "naive subtraction sufficient at cos<0.01". Status unchanged. F#161 caveat "status supported not proven until PPL validation" predates guardrail 1007; modern bar is behavioral pair, not PPL alone.
- **F#417** (`exp_p1_t0_grassmannian_gemma4`, supported, 2026-04-09) — Grassmannian QR algebraically exact on Gemma 4. Status unchanged.
- **F#133** (`exp_hash_ring_remove_expert`, supported) — sibling template using PAIRED KC design (K1 PPL + K2 neighbor acc, both at 100%). The well-formed follow-up template (§8) inherits this structure.

## No `_impl` companion

Preempt-structural KILL excludes `_impl` per F#687/F#698/F#699/F#700/F#701/F#703 + `reviewer.md §5` F#666-pure clause. Unblock is pre-reg-external.

## Skills invocation disclosure

`/mlx-dev` and `/fast-mlx`: **Not invoked. No MLX code written.** `run_experiment.py` imports `json + pathlib` only. Canonical preempt form per F#700/F#701/F#703.

— End PAPER.md —
