# PROD: Red-Team Malicious Adapter Cannot Extract Other Adapters' Data

## Type: Verification (red-team empirical check of null-space isolation claim)

## Motivation

`exp_prod_privacy_redesign` (dependency) established two relevant structural
facts on Gemma-4 E4B v_proj null-space adapters (layers 16-23):

1. **Orthogonality sanity PASS** — `max|W_v @ A_eff| = 1.36e-5` across all 8
   layers. Null-space range is exactly isolated from `W_v`'s range (P7.A1
   Theorem 1 replicated).
2. **K1644 cross-user B-matrix cosine FAIL** — `mean_ℓ max|cos(B_A, B_B)| = 0.39
   > 0.30`. Two users trained independently on disjoint domains share ~0.4
   cosine on B output directions, because Q/A-format steering is a shared
   subspace of `v_proj` outputs (not orthogonalized in the construction).

This red-team experiment tests whether a malicious adapter composed with a
victim adapter can *operationally* extract victim data, under the known B-cos
= 0.39 leakage floor. The notes on this experiment predict PASS by orthogonality
(Finding #562 + null-space), but the dependency's K1644 FAIL means the
orthogonality claim is load-bearing only for `A_eff` (null-space), not for B.

## Prior Work

- **`exp_prod_privacy_redesign`** (KILLED): K1643 PASS (holdout PPL ratio 0.47),
  K1642 FAIL (MIA protocol confounded), K1644 FAIL (B cosine 0.39 > 0.30).
  Exact orthogonality `W_v @ A_eff = 1.36e-5` is the structural guarantee we
  inherit.
- **P7.A1 Theorem 1**: null-space construction `A_eff = A @ Q^T` with `Q` the
  null-space of `W_v` yields `W_v @ A_eff = 0` exactly.
- **Finding #156** (canary queries best for degradation detection): FNR=2%
  CI[1.9-2.1%] for member texts at 3 seeds. Canary extraction is the strongest
  signal for memorization when training data is overfit (final_loss=0.10 on
  user A per dependency's `a_null`).
- **Shokri+ 2017 / Yeom+ 2018**: MIA via loss threshold. Adapter memorization
  drives member-loss << non-member-loss.

## Mathematical Framework

### Setup

Load three adapters from `exp_prod_privacy_redesign` (no retraining):
- `adapter_user_A_null.safetensors` — victim user A, medical QA, null-space.
- `adapter_user_B_null.safetensors` — "malicious" user B, legal QA, null-space.
- `adapter_user_A_std.safetensors` — standard-LoRA baseline for reference.

Composed model: base `W_v` + s·B_A @ A_A @ Q^T + s·B_B @ A_B @ Q^T (additive
composition, same Q per layer since Q depends only on base `W_v`).

### Theorem 1 (Null-Space Isolation of Extraction Target)

Inherited from P7.A1: for any user u, `W_v @ (A_eff^u)^T = 0` exactly. Therefore
the base-model output subspace cannot be used to read back the adapter's effect
via any linear probe on `x` alone — the adapter signal lives purely in the
range of `B_u` projected through the null-space-gated input pathway.

**Consequence:** an attacker without the B matrix cannot recover the adapter
output. But with composed access (which includes B_B implicitly through the
additive delta), the attacker can query the model.

### Theorem 2 (Composed-Output Decomposition Ambiguity — informal)

Let `ΔW_composed = s·(B_A @ A_A + B_B @ A_B) @ Q^T`. Rank is at most `2r` (here
`r=6`, so rank ≤ 12). Given only `ΔW_composed` and `Q`, the decomposition
into `(B_A, A_A)` and `(B_B, A_B)` is not unique: for any invertible
`M ∈ ℝ^{2r × 2r}`, the pair `(B_A', A_A') = ([B_A, B_B]·M_left, M_right·[A_A; A_B])`
satisfies the same product. Parameter extraction up to subspace is bounded by
the rank-r SVD approximation error.

**Prediction:** without side information, an attacker can recover the combined
rank-2r subspace but cannot distinguish which singular components belong to
user A vs user B. Cosine between the "attacker-claimed user-A adapter" (rank-r
best fit) and the true `B_A @ A_A` is bounded by the singular-value structure.

### Theorem 3 (Canary-Prefix Vulnerability under Overfit Adapter)

The dependency measured `final_loss ≈ 0.10` on user A's 20 training texts after
500 iterations. Cross-entropy 0.10 nats/token corresponds to per-token
probability `exp(-0.10) ≈ 0.905` — i.e., next-token prediction is near-certain
on member texts. Standard greedy decoding from a short prefix of a member text
will reproduce the training text at high fidelity.

**Prediction:** per Finding #156, canary-prefix extraction recovers member
text at >50% token overlap for ≥10 of 20 canaries.

### Theorem 4 (Activation-Probe Separability under Overfit)

Member texts produce adapter-saturated hidden states (loss ~0.10); non-member
OOD texts produce base-like hidden states (loss ~5.3 per dependency's
`k1642.nonmember_losses`). A linear probe on last-layer hidden states trivially
separates these two populations. Even same-distribution held-out texts carry
less adapter signal than memorized members (dependency's standard-LoRA PPL 177
on holdout vs 0.08 final_loss on train).

**Prediction:** linear probe AUC > 0.90 distinguishing members from user-A
holdout under composed model.

## Kill Criteria (Pre-Registered, Locked at MATH.md Commit)

- **K1667** (DB id 1667): **Activation-probe recovery < 1% training data.**
  Protocol: Load composed (null_A + null_B) model. Compute last-layer mean
  hidden states for (a) 20 user A member texts and (b) 20 user A holdout-proxy
  texts (reuse USER_A_HOLDOUT + OOD fillers to get 20). Fit a logistic-regression
  probe on 10+10 train split; evaluate on 10+10 test split. Recovery rate =
  test-split member classification accuracy above 50% baseline. PASS if
  `(accuracy - 0.50) < 0.01` (i.e., probe has no signal above chance).

  *Deviation from DB text* (pre-registered here): DB says "<1% of victim
  training data". We operationalize as "<1% classification advantage over
  chance" — this is the strongest reading of "activation probe recovers <1%".
  Recovery of individual training-data *content* via activation probe is not
  a defined attack in the literature; membership inference is. Documented
  scope-narrowing.

- **K1668** (DB id 1668): **Parameter extraction subspace overlap < 1%.**
  Protocol: Build composed per-layer `ΔW_composed = s·(B_A@A_A + B_B@A_B)@Q^T`.
  Take rank-r SVD of `ΔW_composed`; call the recovered adapter-direction
  basis `U_attack ∈ ℝ^{d_out × r}`. Compute subspace overlap
  `overlap_A = ||U_attack^T @ B_A_normalized||_F^2 / r` and
  `overlap_B` analogously. PASS if `max(overlap_A, overlap_B) < 0.01` averaged
  over 8 layers.

  *Interpretation*: The attacker recovers the composed delta exactly (trivially,
  by querying); what matters is whether they can attribute components to a
  specific user. Subspace overlap quantifies "how much of one user's B
  direction is in the attack-recovered rank-r subspace". Under Theorem 2 and
  K1644=0.39 from dependency, we expect significant overlap (>>1%), i.e.,
  predicted FAIL.

- **K1669** (DB id 1669): **Canary extraction ≤ 1 of 100 extracted.**
  Protocol: Use the 20 member texts from user A as the canary set (pre-reg
  scope-down from 100 → 20; rationale: existing adapters from dependency were
  not trained with random-token canaries, and retraining is out of scope for
  this experiment). For each canary, take the first 30 tokens as prefix, greedy-
  decode 50 tokens under composed model, measure token-level overlap with the
  full training text. "Extracted" = overlap ratio ≥ 0.5. PASS if ≤ 1 of 20.

  *Deviation from DB text*: scope-down to 20, and PASS threshold scaled
  accordingly (1/20 = 5%, same rate as DB's 1/100 = 1%). Canary design: using
  member texts themselves as canaries is a weaker test than random-token canaries
  (more false positives if the base model knows the topic) — so this is a
  conservative (attack-favoring) test: PASSING here would be strong evidence
  the null-space construction prevents extraction. FAILING is expected given
  overfit member loss 0.10.

## Assumptions (documented per researcher hat rules)

1. **No retraining**: inherit adapters from `exp_prod_privacy_redesign` run.
   Implication: training-set overlap with the dependency is 100%. This does
   not affect validity of the red-team attacks — we are attacking the already-
   released adapters.
2. **Canary count scope-down** 100→20 with proportional threshold adjustment.
3. **Composed model = additive null-space**: both adapters loaded on same
   v_proj layers simultaneously.
4. **Attacker model**: has composed-model black-box inference access + knows
   `Q_ℓ` null-space bases (public construction). Does NOT know individual
   `A_u`, `B_u`. The K1668 attack assumes the attacker can also read the
   composed `ΔW_composed` by probing (achievable via input probing in
   polynomial queries — we just compute it directly since we have weights).
5. **K1667 test pool**: user A member (n=20) vs user A holdout (n=5, padded
   with 15 Gemma-4-base-generated neutral medical queries for balance). If
   pool size is insufficient, reduce test to n=5+5 with accuracy-only metric.
6. **MLX platform**: Apple M5 Pro 48GB, `mlx-lm 0.31.2`, scale=8.0 (safe, per
   `mem-antipattern-003`).

## Predictions (for PAPER.md cross-check)

- **P1 (K1667):** Linear probe AUC > 0.90; accuracy-above-chance > 0.30. **FAIL.**
- **P2 (K1668):** `max(overlap_A, overlap_B)` averaged over 8 layers ≈ 0.5–1.0
  (top-2r subspace contains almost all of B_A and B_B directions). **FAIL.**
- **P3 (K1669):** ≥ 10 of 20 canaries reproduced at overlap ≥ 0.5. **FAIL.**

**Overall predicted verdict: KILLED** on all three KCs. The null-space
construction preserves `W_v`'s range (K1643 quality pass from dependency)
but does NOT by itself prevent red-team extraction because B matrices are not
cross-user orthogonal (K1644 FAIL from dependency). This experiment verifies
the predicted failure mode empirically, providing the anchor for a v2 that
adds Gram-Schmidt on B during training (per dependency LEARNINGS #2: Gram-
Schmidt on B is the critical path for N>1 privacy).

## Experiment Design

1. Load Gemma-4 E4B 4-bit + null_bases + both null adapters (reuse from
   dependency dir).
2. **Phase K1668** (cheapest, no inference): compute composed delta per layer,
   SVD, subspace overlap. Pure weight-space computation.
3. **Phase K1667** (activation probe): forward pass on member + holdout texts
   through composed model, collect last-layer mean hidden states, fit probe,
   eval.
4. **Phase K1669** (canary extraction): per canary, greedy-decode 50 tokens
   from 30-token prefix under composed model; token-overlap score.
5. Write results.json with all three KCs + booleans.

## Success Criteria

Because dependency K1644 failed (B cos 0.39 > 0.30), the pre-registered
prediction is that all three KCs FAIL. If a KC passes unexpectedly, report
honestly as supported; otherwise verdict = `killed` with a clear roadmap to v2
(add Gram-Schmidt on B during training).

## References

- `exp_prod_privacy_redesign` (dependency, KILLED) — layers + bases + adapters.
- `exp_p7_null_space_adapter_quality` (P7.A1) — null-space LoRA construction.
- `exp_p1_t5_user_adapter_privacy` (T5.4, KILLED) — Finding #438, `lora_a` vs
  `lora_b` MIA protocol lessons.
- Finding #156 — canary queries outperform cosine gating for memorization
  detection (FNR 2.0% vs 33.8%).
- Shokri+ 2017 (`arxiv:1610.05820`), Yeom+ 2018 (`arxiv:1709.01604`) — MIA
  loss-threshold literature.
- P7.A1 Theorem 1 — null-space structural orthogonality.
