# exp_g4_o1_removal_naive — MATH.md (F#666-pure standalone preempt-KILL)

## §0 Base model / adapters / required skills disclosure

- **Base model:** Gemma 4 (target; unloaded — no compute performed).
- **Adapters:** none constructed (preempt-KILL scaffold, no training run).
- **Required platform skills (per PLAN.md Part 2):** `/mlx-dev`, `/fast-mlx` — **Not invoked. No MLX code written.** Canonical preempt disclosure per F#700/F#701/F#703 precedent; `run_experiment.py` imports `json` + `pathlib` only.

## §1 Theorem: V(K1580) is unidentifiable under F#666

**Pre-reg KC set (verbatim, from `experiment get exp_g4_o1_removal_naive`):**
- K1580: "max PPL drift <= 0.2% after remove, N=25 -> 24" — result: `untested`.

**Classification per guardrail 1007:** PPL is explicitly named in guardrail 1007 as a proxy metric ("classification accuracy, routing match rate, PPL, cosine, clustering purity"). Guardrail 1006 anchors this at r≈0.08 Pearson between PPL and task quality in this codebase. K1580 is the sole KC; it pairs with no target-metric KC (task accuracy, behavioral equivalence, oracle-gap, PASS@k). `depends_on: []` — no parent supplies a target anchor.

**Truth table (exhaustive, 2¹ = 2 outcome classes):**

| K1580 (PPL drift ≤ 0.2%) | Interpretation under F#666 | Verdict identifiable? |
|--------------------------|----------------------------|-----------------------|
| PASS | Tautological SUPPORT: proxy-only PASS cannot claim "quality preserved" without behavioral anchor. Antipattern-t: reviewer will KILL-on-tautology. | No |
| FAIL | "A finding about the proxy, not a kill" per F#666 — PPL drift alone does not establish behavioral quality drop; may reflect PPL's known 0.08-correlation distortion, not real degradation. | No |

**QED.** Every outcome class is unidentifiable. Running the experiment wastes compute and yields a verdict the reviewer must overturn. F#666 (target-gated KILL) governs: preempt-KILL before compute.

## §2 Prior art & governing findings

- **F#666** (conclusive, 2026-04-19, `exp_softmax_router_scaling`) — target-gated KILL discipline. Proxy-PASS alone is tautological; proxy-FAIL alone is "a finding about the proxy, not a kill". Pair proxy KCs with target-metric KCs.
- **F#700 / F#701 / F#703** — F#666-pure standalone preempt-KILL precedents at the 3-instance promotion threshold. `reviewer.md §5` clause `KILL (preempt-structural — F#666-pure standalone)` promoted 2026-04-24.
- **F#161** (supported, 2026-03-15, `exp_attention_layer_removal_safety`) — parent motivation. Naive subtraction sufficient at cos<0.01; status "supported not proven until PPL validation". Caveat predates guardrail 1007; modern bar is PPL + behavioral pair, not PPL alone.
- **F#417** (supported, 2026-04-09, `exp_p1_t0_grassmannian_gemma4`) — parent motivation. Grassmannian QR algebraically exact at Gemma 4 dimensions; pairwise |A_iᵀA_j| ≤ 1.06e-15.
- **F#133** (supported, 2026-03-xx, `exp_hash_ring_remove_expert`) — directly analogous removal experiment; used PAIRED KC design (K1 PPL mean −2.23% + K2 neighbor accuracy 100%). Sibling precedent demonstrates the well-formed template this pre-reg fails to match.

## §3 Pre-reg KC verbatim

```
kill_criteria:
  - id: 1580
    text: "max PPL drift <= 0.2% after remove, N=25 -> 24"
    result: untested
success_criteria: []  # empty (hygiene defect)
references: []        # empty (hygiene defect; notes cite F#161 + F#417 informally)
platform: local-apple # present
depends_on: []
```

Hygiene defect count: 2 (success_criteria empty, references empty). Below the 3+ threshold for `mem-antipattern-prereg-hygiene-multi-defect`. F#666-pure-standalone applies independently of hygiene count.

## §4 Unblock path

**Do NOT patch K1580 via `experiment update` — KC mutation post-claim is antipattern-u.**

Re-register a new pre-reg `exp_g4_o1_removal_target_paired` with a paired target KC:
- **K1 (target, load-bearing):** HumanEval PASS@1 (or MMLU subset accuracy) drop ≤ 1.0% after removal, N=25 → 24 on Gemma 4. Preserves the behavioral claim ("removal doesn't degrade").
- **K2 (proxy, sanity, conditional on K1 PASS):** PPL drift ≤ 0.2% on held-out validation. Does not override K1.
- **K3 (neighbor fidelity, sibling F#133 template):** ≥ 95% token-level agreement with N=25 generation on the held-out set.

KILL requires K1 FAIL + K2 FAIL (or K3 FAIL). SUPPORTED requires K1 PASS + K2 PASS. Proxy-only outcomes are non-verdicts per F#666.

## §5 No `_impl` companion

Preempt-structural KILL does not spawn `_impl` per F#687/F#698/F#699/F#700/F#701/F#703 precedent and `reviewer.md §5` F#666-pure clause. Unblock is pre-reg-external: a new pre-reg with paired target KCs, not a follow-up `_impl`.

## §6 Scope of this filing

- This is a **scaffold**, not an experiment. No training, no inference, no MLX computation. `run_experiment.py` imports `json + pathlib` only; `main()` writes a graceful-failure `results.json` with `verdict="KILLED"`, K1580 `untested`, preempt-reason `F666_PURE_PREEMPT_KILL`.
- Behavioral claim not measured. Parent motivations (F#161 + F#417) remain `supported` and are untouched by this filing.

## §7 Taxonomic placement (drain-window row)

| Row | Experiment | Antipattern | §5 clause |
|-----|------------|-------------|-----------|
| 1 | F#700 `exp_g4_per_layer_cos_baseline` | F#666-pure standalone | §5 clause (promoted) |
| 2 | F#701 `exp_adapter_orthogonality_audit` | F#666-pure standalone | §5 clause (promoted) |
| 3 | F#703 `exp_followup_tfidf_medical_unaliased` | F#666-pure standalone | §5 clause (promoted) |
| **4** | **F#TBD `exp_g4_o1_removal_naive`** (this filing) | **F#666-pure standalone** (PPL-only, single KC) | §5 clause (already promoted, no re-promote at 4th) |

Distinction:
- **vs F#669-family preempt-structural**: parent-orthogonal. `depends_on: []`; no parent needs to reach `supported`.
- **vs F#702 hygiene-patch PROVISIONAL**: F#702 had target-metric KCs making the experiment runnable despite hygiene defects. This pre-reg has no target KC — no hygiene patch rescues it.
- **vs tautological-inter-variant-delta (F#704)**: F#704 had a target metric but tautological framing. This pre-reg has a pure-proxy metric — different failure mode.

## §8 Well-formed follow-up pre-reg template

```yaml
id: exp_g4_o1_removal_target_paired
title: "Gemma 4 N=25→24 Grassmannian adapter removal: task-accuracy drop ≤ 1pp (behavioral)"
priority: 2
scale: micro
platform: local-apple
depends_on: [exp_p1_t0_grassmannian_gemma4]  # F#417 prereq
success_criteria:
  - "K1 PASS + K2 PASS + K3 PASS → removal is O(1) safe on Gemma 4 at N=25"
kill_criteria:
  - id: TBD-K1
    text: "HumanEval PASS@1 drop ≤ 1.0pp after remove, N=25→24"
    metric: humaneval_pass_at_1
    type: target
    result: untested
  - id: TBD-K2
    text: "PPL drift ≤ 0.2% on held-out validation (sanity, conditional on K1)"
    metric: ppl_drift
    type: proxy
    result: untested
  - id: TBD-K3
    text: "≥95% token-level agreement with N=25 generation on held-out prompts"
    metric: token_agreement_rate
    type: target
    result: untested
references:
  - F#161  # parent: naive subtraction at cos<0.01
  - F#417  # parent: Grassmannian QR exact on Gemma 4
  - F#133  # sibling template: PAIRED KC design for removal
  - F#666  # governing: target-gated KILL discipline
tags: [g4-gemma4, composition, removal, paired-kc, f666-compliant]
```

— End MATH.md —
