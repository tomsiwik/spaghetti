# exp_g4_hash_ring_remove_n25 — MATH.md (F#666-pure standalone preempt-KILL)

## §0 Base model / adapters / required skills disclosure

- **Base model:** Gemma 4 (target; unloaded — no compute performed).
- **Adapters:** none constructed (preempt-KILL scaffold, no training run, no routing).
- **Required platform skills (per PLAN.md Part 2):** `/mlx-dev`, `/fast-mlx` — **Not invoked. No MLX code written.** Canonical preempt disclosure per F#700/F#701/F#703/F#705/F#706/F#707 precedent; `run_experiment.py` imports `json` + `pathlib` only.

## §1 Theorem: V(K1583) is unidentifiable under F#666

**Pre-reg KC set (verbatim, from `experiment get exp_g4_hash_ring_remove_n25`):**
- K1583: "mean PPL <= 3%, max <= 5%" — result: `untested`.

**Classification per guardrail 1007:** PPL is explicitly named in guardrail 1007 as a proxy metric ("classification accuracy, routing match rate, PPL, cosine, clustering purity"). Guardrail 1006 anchors this at r≈0.08 Pearson between PPL and task quality in this codebase. K1583 is the sole KC; both thresholds (mean ≤3%, max ≤5%) are sub-conditions on the same proxy metric. No paired target-metric KC (task accuracy, behavioral equivalence, oracle-gap, PASS@k). `depends_on: []` — no parent supplies a target anchor.

**Truth table (exhaustive, 2¹ = 2 outcome classes — both sub-thresholds collapse to one PPL-axis verdict):**

| K1583 (mean PPL drop ≤3% AND max ≤5%) | Interpretation under F#666 | Verdict identifiable? |
|----------------------------------------|----------------------------|-----------------------|
| PASS | Tautological SUPPORT: proxy-only PASS cannot claim "removal preserves quality" without behavioral anchor. PPL-r≈0.08 with task quality means a 3% drop bound proves nothing about HumanEval / MMLU / behavioral equivalence. Antipattern-t: reviewer will KILL-on-tautology. | No |
| FAIL | "A finding about the proxy, not a kill" per F#666 — PPL drop alone does not establish behavioral quality drop; may reflect PPL's known 0.08-correlation distortion, not real degradation of routing or task accuracy. | No |

**QED.** Every outcome class is unidentifiable. Running the experiment wastes compute and yields a verdict the reviewer must overturn. F#666 (target-gated KILL) governs: preempt-KILL before compute.

## §2 Prior art & governing findings

- **F#666** (conclusive, 2026-04-19, `exp_softmax_router_scaling`) — target-gated KILL discipline. Proxy-PASS alone is tautological; proxy-FAIL alone is "a finding about the proxy, not a kill". Pair proxy KCs with target-metric KCs.
- **F#700 / F#701 / F#703 / F#705 / F#706 / F#707** — F#666-pure standalone preempt-KILL precedents. `reviewer.md §5` clause `KILL (preempt-structural — F#666-pure standalone)` promoted 2026-04-24 at the 3-instance threshold. F#705 was the first PPL-as-proxy instance (`exp_g4_o1_removal_naive`, 2026-04-24); this filing is the second.
- **F#133** (supported, 2026-03-15, `exp_hash_ring_remove_expert`) — **direct parent**. Used PAIRED KC design: K1 PPL mean −2.23%, max −4.53% **AND K2 neighbor accuracy 100%**. The parent's own caveats explicitly call out that "Per-expert quality metric (cosine alignment) captures direction but not magnitude — an attenuated expert would show high cosine but degraded quality" and "Code K1 aggregate check is wrong (reports KILL because stress tests with cos=0.3-0.5 exceed threshold; PAPER.md correctly reports conditional pass)". Parent also notes: "PPL validation … not measured" but pairs with neighbor-acc + structural checks. This child pre-reg drops the K2 neighbor pairing and keeps only the K1 PPL — a **template-regression** from the parent's own well-formed design.

**Template-regression observation (sub-pattern):** F#705 was filed under F#161, which had a *caveat* ("status supported not proven until PPL validation") that pre-dated guardrail 1007 — child took stale parent guidance. This filing is filed under F#133, whose PAIRED KC design is *itself* the well-formed template; the child *strips* the K2 neighbor pairing. Distinct sub-pattern from F#705: not "parent caveat went stale" but "parent design ignored". See §7 below.

## §3 Pre-reg KC verbatim

```
kill_criteria:
  - id: 1583
    text: "mean PPL <= 3%, max <= 5%"
    result: untested
success_criteria: []  # empty (hygiene defect)
references: []        # empty (hygiene defect; notes cite F#133 informally)
platform: local-apple # present
depends_on: []
tags: [routing, audit-2026-04-17, g4-gemma4]
```

Hygiene defect count: 2 (success_criteria empty, references empty). Below the 3+ threshold for `mem-antipattern-prereg-hygiene-multi-defect`. F#666-pure-standalone applies independently of hygiene count.

Note: `audit-2026-04-17` tag (no `-rerun` suffix) marks audit lineage but does not flag KNOWN-BUGGY code requiring fix-before-rerun (per researcher.md workflow step 3). No `run_experiment.py` exists in the experiment dir; this filing creates the preempt scaffold from scratch.

## §4 Unblock path

**Do NOT patch K1583 via `experiment update` — KC mutation post-claim is antipattern-u.**

Re-register a new pre-reg `exp_g4_hash_ring_remove_n25_target_paired` mirroring the **parent F#133 PAIRED KC template**:

- **K1 (target, load-bearing):** HumanEval PASS@1 (or MMLU subset accuracy) drop ≤ 1.0pp after hash-ring removal at N=25 → 24 on Gemma 4. Preserves the behavioral claim ("removal at N=25 doesn't degrade").
- **K2 (target, sibling F#133 template):** ≥ 95% neighbor accuracy on the removed expert's hash neighborhood (parent's K2, scaled to N=25).
- **K3 (proxy, sanity, conditional on K1 + K2 PASS):** mean PPL drop ≤ 3%, max ≤ 5% on held-out validation. Does not override K1/K2.

KILL requires K1 FAIL + (K2 FAIL or K3 FAIL). SUPPORTED requires K1 PASS + K2 PASS + K3 PASS. Proxy-only outcomes (K3 alone) are non-verdicts per F#666.

## §5 No `_impl` companion

Preempt-structural KILL does not spawn `_impl` per F#687/F#698/F#699/F#700/F#701/F#703/F#705/F#706/F#707 precedent and `reviewer.md §5` F#666-pure clause. Unblock is pre-reg-external: a new pre-reg with paired target KCs, not a follow-up `_impl`.

## §6 Scope of this filing

- This is a **scaffold**, not an experiment. No training, no inference, no MLX computation, no routing simulation. `run_experiment.py` imports `json + pathlib` only; `main()` writes a graceful-failure `results.json` with `verdict="KILLED"`, K1583 `untested`, preempt-reason `F666_PURE_PREEMPT_KILL`.
- Behavioral claim not measured. Parent F#133 remains `supported` and is untouched by this filing.

## §7 Taxonomic placement (drain-window row)

| Row | Experiment | Antipattern | Proxy flavor | §5 clause |
|-----|------------|-------------|--------------|-----------|
| 1 | F#700 `exp_g4_per_layer_cos_baseline` | F#666-pure standalone | cos-sim | §5 clause (promoted) |
| 2 | F#701 `exp_adapter_orthogonality_audit` | F#666-pure standalone | pairwise-cos + eff-rank | §5 clause (promoted) |
| 3 | F#703 `exp_followup_tfidf_medical_unaliased` | F#666-pure standalone | routing weighted-acc | §5 clause (promoted) |
| 4 | F#705 `exp_g4_o1_removal_naive` | F#666-pure standalone | **PPL (1st)** | §5 clause (no re-promote) |
| 5 | F#706 `exp_g4_canary_drift_detection` | F#666-pure standalone | FNR (canonical guardrail 1007 "classification accuracy") | §5 clause (no re-promote) |
| 6 | F#707 `exp_g4_xxhash_routing_n25` | F#666-pure standalone | R / collision-rate (canonical guardrail 1007 "routing match rate" dual) | §5 clause (no re-promote) |
| **7** | **F#TBD `exp_g4_hash_ring_remove_n25`** (this filing) | **F#666-pure standalone** | **PPL (2nd) — mean+max sub-threshold variant** | **§5 clause (no re-promote)** |

**Sub-pattern flag (template-regression — 2nd instance, candidate antipattern memory):**

| Sub-pattern instance | Parent | Parent state | Child regression |
|----------------------|--------|--------------|------------------|
| 1st (F#705) | F#161 | supported with stale caveat ("PPL validation needed") pre-dating guardrail 1007 | Child built KC from stale caveat |
| **2nd (this filing)** | **F#133** | **supported with PAIRED KC design itself (K1 PPL + K2 neighbor-acc)** | **Child stripped K2 pairing, kept only K1 PPL** |

Both sub-patterns produce F#666-pure children, but the upstream causal structure differs:
- **stale-caveat regression (F#705)**: parent's *secondary advice* went stale.
- **paired-template stripping (this filing)**: parent's *primary KC design* was ignored.

Promotion threshold for sub-pattern: at 3rd instance per F#704 / F#669 promotion convention. **2nd instance triggers candidate antipattern memory filing** (analyst non-blocking).

Distinction from other patterns:
- **vs F#669-family preempt-structural**: parent-orthogonal. `depends_on: []`; no parent needs to reach `supported`.
- **vs F#702 hygiene-patch PROVISIONAL**: F#702 had target-metric KCs making the experiment runnable despite hygiene defects. This pre-reg has no target KC — no hygiene patch rescues it.
- **vs tautological-inter-variant-delta (F#704)**: F#704 had a target metric but tautological framing. This pre-reg has a pure-proxy metric — different failure mode.
- **vs F#706/F#707 (canonical guardrail 1007 enumerations)**: F#706 (FNR/classification accuracy) and F#707 (R/routing match rate) both anchor *first-time-canonical* guardrail 1007 enumerations. This filing is **2nd PPL instance** — lexical-expansion within an already-anchored proxy flavor; not a new canonical anchor.

## §8 Well-formed follow-up pre-reg template (mirrors parent F#133 paired design)

```yaml
id: exp_g4_hash_ring_remove_n25_target_paired
title: "Gemma 4 N=25→24 hash-ring expert removal: task-acc + neighbor-fidelity + PPL (paired)"
priority: 2
scale: micro
platform: local-apple
depends_on: [exp_p1_t0_grassmannian_gemma4]   # F#417 prereq for adapter construction
success_criteria:
  - "K1 PASS + K2 PASS + K3 PASS → hash-ring remove is O(1) safe on Gemma 4 at N=25"
kill_criteria:
  - id: TBD-K1
    text: "HumanEval PASS@1 drop ≤ 1.0pp after hash-ring remove, N=25 → 24"
    metric: humaneval_pass_at_1
    type: target
    result: untested
  - id: TBD-K2
    text: "≥ 95% neighbor accuracy on removed expert's hash neighborhood (sibling F#133 template, scaled N=25)"
    metric: neighbor_accuracy
    type: target
    result: untested
  - id: TBD-K3
    text: "mean PPL drop ≤ 3%, max ≤ 5% on held-out validation (sanity, conditional on K1+K2)"
    metric: ppl_drop
    type: proxy
    result: untested
references:
  - F#133  # parent: PAIRED KC template (PPL + neighbor accuracy)
  - F#161  # related: naive subtraction at cos<0.01
  - F#417  # parent: Grassmannian QR exact on Gemma 4
  - F#666  # governing: target-gated KILL discipline
  - F#705  # precedent: 1st PPL F#666-pure preempt-KILL (sibling shape)
tags: [g4-gemma4, composition, removal, paired-kc, f666-compliant]
```

— End MATH.md —
