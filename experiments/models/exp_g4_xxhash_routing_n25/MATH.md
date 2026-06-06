# exp_g4_xxhash_routing_n25 — MATH.md (F#666-pure standalone preempt-KILL)

## §0 Base model / adapters / required skills disclosure

- **Base model:** Gemma 4 (target; unloaded — no compute performed).
- **Adapters:** none constructed (preempt-KILL scaffold, no routing run, no inference).
- **Required platform skills (per PLAN.md Part 2):** `/mlx-dev`, `/fast-mlx` — **Not invoked. No MLX code written.** Canonical preempt disclosure per F#700/F#701/F#703/F#705/F#706 precedent; `run_experiment.py` imports `json` + `pathlib` only.

## §1 Theorem: V(K1582) is unidentifiable under F#666

**Pre-reg KC set (verbatim, from `experiment get exp_g4_xxhash_routing_n25`):**
- K1582: "R < 2.0 at N=25" — result: `untested`.

**Classification per guardrail 1007:** R is the routing-collision-rate ratio vs Welch bound (a pure statistical property of the hash function at N=25 experts). Guardrail 1007 explicitly names **"routing match rate"** in the proxy enumeration: *"classification accuracy, routing match rate, PPL, cosine, clustering purity"*. Routing-collision-rate is the mathematical dual of routing-match-rate (low collision = high match-diversity); both measure system-level routing statistics with no linkage to behavioral outcome on downstream tasks. K1582 is the sole KC; it pairs with no target-metric KC (task accuracy, behavioral quality, oracle-gap, PASS@k). `depends_on: []` — no parent supplies a target anchor.

**Truth table (exhaustive, 2¹ = 2 outcome classes):**

| K1582 (R < 2.0 at N=25) | Interpretation under F#666 | Verdict identifiable? |
|-------------------------|----------------------------|-----------------------|
| PASS | Tautological SUPPORT: hash mathematically has low collision ratio by construction (xxHash is cryptographically near-uniform). "Passes" = the hash's own statistical property, not adapter behavior. Antipattern-t: reviewer will KILL-on-tautology. | No |
| FAIL | "A finding about the proxy, not a kill" per F#666 — R ≥ 2.0 does not establish that adapter routing produces poor downstream outputs; collision clusters may align with adapter similarity structure and remain quality-benign (cf F#666 softmax-router oracle-gap at 40% proxy acc). | No |

**QED.** Every outcome class is unidentifiable. Running the experiment wastes compute and yields a verdict the reviewer must overturn. F#666 (target-gated KILL) governs: preempt-KILL before compute.

## §2 Prior art & governing findings

- **F#666** (conclusive, 2026-04-19, `exp_softmax_router_scaling`) — target-gated KILL discipline. Proxy-PASS alone is tautological; proxy-FAIL alone is "a finding about the proxy, not a kill". Pair proxy KCs with target-metric KCs.
- **F#700 / F#701 / F#703 / F#705 / F#706** — F#666-pure standalone preempt-KILL precedents. `reviewer.md §5` clause `KILL (preempt-structural — F#666-pure standalone)` promoted 2026-04-24 at 3rd instance. Rows 4/5/6 = lexical expansion (PPL, FNR, routing-collision-rate).
- **F#147** (supported, 2026-03-28, `exp_sole_hash_routing`) — parent motivation. "xxHash32 R=1.170 vs FNV1a R=2.175 at N=8 (1.86x improvement)". Pure hash-statistics study — no behavioral KC, no downstream accuracy measurement. Pre-reg for this filing inherits only the proxy metric (R), not the behavioral anchor (which parent also lacks). Parent-mechanism-anchor-non-inheritance watchlist applies **vacuously**: the parent has no mechanistic formula to inherit.
- **F#133** (supported, 2026-03-xx, `exp_hash_ring_remove_expert`) — sibling hash-routing experiment used PAIRED KC design (K1 PPL mean −2.23% + K2 neighbor accuracy 100%). Precedent demonstrates the well-formed template this pre-reg fails to match.
- **Welch bound reference:** Welch 1974 — lower bound on collision probability for N codes in finite alphabet. R = (observed collision prob) / (Welch bound); R ≥ 1 always. "R < 2.0" is an arbitrary threshold with no linkage to task performance.

## §3 Pre-reg KC verbatim

```
kill_criteria:
  - id: 1582
    text: "R < 2.0 at N=25"
    result: untested
success_criteria: []  # empty (hygiene defect)
references: []        # empty (hygiene defect; notes cite F#147 informally)
platform: local-apple # present
depends_on: []
```

Hygiene defect count: 2 (success_criteria empty, references empty). Below the 3+ threshold for `mem-antipattern-prereg-hygiene-multi-defect`. F#666-pure-standalone applies independently of hygiene count.

## §4 Unblock path

**Do NOT patch K1582 via `experiment update` — KC mutation post-claim is antipattern-u.**

Re-register a new pre-reg `exp_g4_xxhash_routing_n25_target_paired` with a paired target KC:
- **K1 (target, load-bearing):** HumanEval PASS@1 (or MMLU subset accuracy) drop ≤ 1.0pp at N=25 xxHash-routed vs N=25 oracle-routed on Gemma 4. Preserves the behavioral claim ("hash routing doesn't degrade vs oracle").
- **K2 (proxy, sanity, conditional on K1 PASS):** R < 2.0 vs Welch bound at N=25. Does not override K1.
- **K3 (neighbor fidelity, sibling F#133 template):** ≥ 95% token-level agreement with oracle-routed N=25 generation on held-out prompts.

KILL requires K1 FAIL + (K2 FAIL or K3 FAIL). SUPPORTED requires K1 PASS + K2 PASS. Proxy-only outcomes are non-verdicts per F#666.

## §5 No `_impl` companion

Preempt-structural KILL does not spawn `_impl` per F#687/F#698/F#699/F#700/F#701/F#703/F#705/F#706 precedent and `reviewer.md §5` F#666-pure clause. Unblock is pre-reg-external: a new pre-reg with paired target KCs, not a follow-up `_impl`.

## §6 Scope of this filing

- This is a **scaffold**, not an experiment. No training, no inference, no routing simulation, no MLX computation. `run_experiment.py` imports `json + pathlib` only; `main()` writes a graceful-failure `results.json` with `verdict="KILLED"`, K1582 `untested`, preempt-reason `F666_PURE_PREEMPT_KILL`.
- Behavioral claim not measured. Parent F#147 remains `supported` and is untouched by this filing.

## §7 Taxonomic placement (drain-window row)

| Row | Experiment | Antipattern | §5 clause |
|-----|------------|-------------|-----------|
| 1 | F#700 `exp_g4_per_layer_cos_baseline` | F#666-pure (cos-sim) | §5 clause (promoted) |
| 2 | F#701 `exp_adapter_orthogonality_audit` | F#666-pure (pairwise-cos + eff-rank) | §5 clause (promoted) |
| 3 | F#703 `exp_followup_tfidf_medical_unaliased` | F#666-pure (routing weighted-acc) | §5 clause (promoted) |
| 4 | F#705 `exp_g4_o1_removal_naive` | F#666-pure (PPL-only) | §5 clause (lexical expansion) |
| 5 | F#706 `exp_g4_canary_drift_detection` | F#666-pure (FNR / classification-accuracy) | §5 clause (canonical-anchor) |
| **6** | **F#TBD `exp_g4_xxhash_routing_n25`** (this filing) | **F#666-pure (routing-collision-rate R)** | §5 clause (already promoted, no re-promote at 6th) |

Distinction:
- **vs F#669-family preempt-structural**: parent-orthogonal. `depends_on: []`; no parent needs to reach `supported`.
- **vs F#702 hygiene-patch PROVISIONAL**: F#702 had target-metric KCs making the experiment runnable despite hygiene defects. This pre-reg has no target KC — no hygiene patch rescues it.
- **vs tautological-inter-variant-delta (F#704)**: F#704 had a target metric but tautological framing. This pre-reg has a pure-proxy metric — different failure mode.

**Taxonomic novelty (row 6):** First drain-window instance where the proxy is **routing-collision-rate R** (vs Welch bound). Expands F#666-pure lexicon to canonical guardrail 1007 "routing match rate" via its mathematical dual (low collision ⇔ high match-diversity). Row 5 was canonical guardrail 1007 "classification accuracy" (FNR/TPR/FPR). Row 6 is canonical guardrail 1007 "routing match rate". Both explicit guardrail 1007 enumerations now anchored in the drain-window record.

**Taxonomy-refactor trigger status:** active since row 5 (per F#706 scratchpad), still non-blocking at row 6. Revisit at 7th+ instance or when a proxy flavor appears that doesn't map cleanly to guardrail 1007 enumeration.

## §8 Well-formed follow-up pre-reg template

```yaml
id: exp_g4_xxhash_routing_n25_target_paired
title: "xxHash32 routing at N=25 on Gemma 4: HumanEval PASS@1 drop ≤ 1pp vs oracle routing (behavioral)"
priority: 2
scale: micro
platform: local-apple
depends_on: [exp_sole_hash_routing]  # F#147 prereq
success_criteria:
  - "K1 PASS + K2 PASS + K3 PASS → xxHash32 routing is behaviorally equivalent to oracle at N=25"
kill_criteria:
  - id: TBD-K1
    text: "HumanEval PASS@1 drop ≤ 1.0pp xxHash-routed vs oracle-routed, N=25"
    metric: humaneval_pass_at_1
    type: target
    result: untested
  - id: TBD-K2
    text: "R < 2.0 vs Welch bound at N=25 (sanity, conditional on K1)"
    metric: collision_ratio_welch
    type: proxy
    result: untested
  - id: TBD-K3
    text: "≥95% token-level agreement with oracle-routed N=25 generation on held-out prompts"
    metric: token_agreement_rate
    type: target
    result: untested
references:
  - F#147  # parent: xxHash32 best hash at N=8
  - F#133  # sibling template: PAIRED KC design for hash-routing
  - F#666  # governing: target-gated KILL discipline
tags: [g4-gemma4, routing, hash, paired-kc, f666-compliant]
```

— End MATH.md —
