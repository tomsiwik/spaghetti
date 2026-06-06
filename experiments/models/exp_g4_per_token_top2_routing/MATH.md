# MATH.md — Per-token top-2 routing on Gemma 4 N=25 (precondition probe)

## Experiment Type
Verification / precondition probe — Finding #58 port from BitNet → Gemma 4 E4B 4-bit at N=25.

## Motivation
Finding #58 (`exp_bitnet_per_token_routing`, supported 2026-03-23, K1 PASS):
per-token top-2 avg PPL 13.65 vs uniform 15.85 (+13.9%) with router accuracy
91.7% at sequence level. Top-1 was *worse* than uniform in Finding #58 (−10.9%,
overshoot) — the mechanistic claim is that top-2 **dilutes adapter overshoot**
when individual adapters degrade base PPL on their own domain.

Audit note (`audit-2026-04-17`) requests porting the per-token top-2
mechanism to Gemma 4 E4B 4-bit at N=25. DB-tracked `KC #1578`:

> `routed_PPL < 0.95 * exclusive_PPL on 5 domains`

## Prior Math

**Theorem A (Finding #58, arxiv:2006.16668 MoE selection variance):**
For `N` experts with routing probabilities `p ∈ Δ^{N-1}`, top-K mixture
`y_K = Σ_{i ∈ TopK(p)} (p_i / Σ_{j∈TopK(p)} p_j) · f_i(x)` has variance
`Var(y_K) ≤ Var(y_1) / K` under i.i.d. overshoot noise. Top-2 halves
overshoot variance when overshoot is uncorrelated across experts.

**Theorem B (Finding #310, exp_hidden_state_probe_router):** On Qwen, ridge
regression over per-token hidden states routes at 98.3% token accuracy and
matches oracle PPL. The per-token routing signal *exists* in hidden states;
it is linearly decodable.

**Theorem C (Finding #305, exp_mixed_domain_per_token_routing):** Per-token
routing on full-sequence forwards is **null** (+0.0%): when all experts see
the same KV-cache, the top-2 weighted mix collapses to oracle single because
the KV cache homogenizes token-wise decisions. Per-token routing only wins
under **segment isolation** or **independent forwards per expert**.

**Theorem D (Finding #583, TF-IDF routing on Gemma 4 N=25):**
Exclusive nearest-centroid TF-IDF routing on Gemma 4 at N=25 with hard
negatives: 85.0% weighted accuracy (KILLED at 88% KC threshold). Baseline
for `exclusive_PPL` is achievable **if** N=25 adapters exist, but routes
~15% of samples to the wrong adapter.

## Theorem 1 (port claim for K1578)

Let `𝒜 = {A_1, …, A_25}` be trained Gemma-4 E4B 4-bit LoRA adapters on
`v_proj+o_proj`, rank `r=6`, one per domain `d_i ∈ D_25` (5 real NLP + 20
MMLU-Pro subjects). Let `PPL_base(x)` be Gemma 4 base perplexity on a
domain-labelled eval batch `x ∈ D_i`.

Let `π_TFIDF : x → {1,…,25}` be the nearest-centroid TF-IDF router
(Finding #583, 85.0% N=25 accuracy).

Let `π_top2_token : h_t → (i, j, α_t)` be a per-token router producing a
top-2 selection with mixing coefficient `α_t ∈ [0,1]`, e.g. a ridge
classifier over hidden states at layer `ℓ*` (Finding #310).

Define, per domain `D_i`:

    exclusive_PPL_i := PPL_base ∘ (merge A_{π_TFIDF(x)}) on x ∈ D_i
    routed_PPL_i    := PPL_base ∘ (merge α_t·A_i + (1−α_t)·A_j) on x ∈ D_i

**Claim (K1578):** On 5 pre-selected Gemma-4 domains,

    routed_PPL_i < 0.95 · exclusive_PPL_i  for i = 1, …, 5.

## Preconditions (pre-registered, routing K1578 to KILLED if any fails)

The claim is measurable only if all three hold. Preconditions are pre-registered
**before** any data collection to lock the routing; adding/relaxing a P-criterion
post-hoc is a verdict-consistency antipattern (see PLAN.md §1 check 5).

### P1 — Gemma 4 trained adapters

At least 5 Gemma-4-e4b-it-4bit LoRA adapters on `v_proj+o_proj`, `r=6`, with
valid `.safetensors` weights (not stub configs), covering 5 distinct domains.

**Target paths (upstream `exp_p1_t2_single_domain_training`, T2.1):**
- `adapters/math/adapters.safetensors`
- `adapters/code/adapters.safetensors`
- `adapters/medical/adapters.safetensors`

(Plus 2 more from `exp_p0_n25_vproj_composition` config stubs — requires
retraining if missing.)

### P2 — Per-token routing mechanism

A trained per-token router (ridge classifier over Gemma 4 hidden states, or
equivalent) with validated accuracy on `D_25` hold-out. Finding #310 exists
only for Qwen hidden states; no Gemma 4 counterpart is in the DB.

**Target artifact:** `micro/models/exp_g4_hidden_state_probe/router.safetensors`
or equivalent. Falls back to TF-IDF per sequence (which defeats the per-token
claim entirely and collapses the test to a trivial equality).

### P3 — Exclusive TF-IDF baseline at N=25

Upstream measurement of `exclusive_PPL_i` on Gemma 4 at N=25 via TF-IDF
routing + per-sample adapter merge. Finding #583 established TF-IDF accuracy
(85.0%) but **did not** measure exclusive_PPL — the adapters never existed
(P1 failure propagates).

**Target upstream:** any of
- `exp_g4_tfidf_ridge_n25_clean/results.json` (dir missing)
- `exp_p1_t4_tfidf_routing_gemma4/results.json` (ran; adapters missing;
  verdict KILLED)
- `exp_g4_tfidf_routing_no_alias/results.json` (open, never run).

### Pre-registered routing for K1578

```
if all(P1, P2, P3):    # measure
    run(exclusive_PPL, routed_PPL)
    K1578 = (routed < 0.95 * exclusive on all 5 domains)
else:                  # unmeasurable precondition KILL
    verdict = "KILLED"
    K1578  = {"status": "FAIL",
              "reason": "unmeasurable: P1/P2/P3 missing",
              "measured": None}
```

**KC-discipline note:** no K1578 threshold is modified here. The 0.95× cut is
inherited unchanged from the DB KC text. Precondition failure yields
*unmeasurable* (FAIL), never silently upgraded to PASS.

## Standing-rule precedents (this loop)

This is the **7th** precondition-probe KILL in the audit-2026-04-17 cohort.
Prior instances with identical pattern (missing Gemma 4 adapter weights +
audit-rerun tag):

1. `exp_followup_sft_behavioral_lora_scale_5` (Finding #600) — QR ≥ 0.90
2. `exp_followup_ss_rn_path_valid_sft` (Finding #602) — |Δacc| ≤ 5pp
3. `exp_followup_orthogonal_projection_scale_control` — theoretical-refutation
4. `exp_followup_answer_conditioned_ppl` (Finding #603) — K1567 measured-KILL
5. `exp_followup_format_compat_peft_required` (Finding #604) — SUPPORTED
6. `exp_followup_ss_rn_path_valid_sft` pre-registered routing (reused)

The standing rule: audit-rerun experiments that depend on upstream adapter
training **blocked** by KILLED parents route K1 to FAIL-unmeasurable, not
to smoke or synthetic padding.

## Unblock path

1. Rerun `exp_p1_t2_single_domain_training` at LORA_SCALE=5 (Finding #586
   scale-safety bound) to regenerate `adapters/{math,code,medical}/adapters.safetensors`.
2. Train 2 additional Gemma 4 adapters (`finance`, `legal`) at matched recipe
   on disjoint corpora.
3. Train a per-token ridge router on Gemma 4 hidden states (replicate
   Finding #310 recipe: layer ℓ*=mid, λ=0.1, target accuracy ≥ 95%).
4. Re-run this probe; if all P1/P2/P3 PASS, the routed/exclusive PPL
   measurement path runs in the `all_preconditions_pass` branch.

Heavy training (3 × 1000-step Gemma 4 LoRA runs, ~45 min/domain × 5 domains
≈ 4h MLX) is deliberately out of scope for Ralph's 30 min/hat budget; this is
why the precondition probe is the honest routing.

## Kill criteria (pre-registered)

- **K1578** (DB): `routed_PPL_i < 0.95 · exclusive_PPL_i` on 5 Gemma-4
  domains. Measured iff P1∧P2∧P3. Otherwise FAIL-unmeasurable.

## Assumptions

- MLX-LM 0.31.0 (latest tested) with Gemma 4 support.
- TF-IDF router and per-token router outputs are *not* substitutable; Finding
  #305 (Theorem C) ensures top-2 per-token on shared KV cache is null, so
  falling back to TF-IDF per-sequence for the "per-token" mechanism **cannot**
  produce a PASS without violating the experiment's own premise.
- 0.95× threshold is a strict 5% improvement — not a statistical bound, a
  point estimate. On PPL with N_eval = 200/domain, 5% corresponds to roughly
  2.5σ per-domain under Finding #310 noise levels; 5/5 domains at 2.5σ is a
  hard bar.

## References

- arXiv:2006.16668 — GShard (top-K MoE)
- Finding #58 — `exp_bitnet_per_token_routing` (BitNet, supported)
- Finding #310 — `exp_hidden_state_probe_router` (Qwen ridge router)
- Finding #305 — `exp_mixed_domain_per_token_routing` (null-under-shared-KV)
- Finding #431 — TF-IDF routing scales (86.1% @ N=25 Gemma 4)
- Finding #583 — TF-IDF N=25 KILLED at 88% under hard negatives (85.0%)
- Finding #600/602/603/604 — precondition-probe KILL cohort
