# PROD: Privacy Redesign via Null-Space Isolation — PAPER

## Verdict: KILLED

2 of 3 pre-registered kill criteria failed at full N (500 iters, 8 layers,
scale=8, r=6). K1642 and K1644 failed due to architectural gaps the
pre-registration itself partially predicted; K1643 passed decisively.

## Summary

Null-space reparameterization of v_proj LoRA on Gemma-4 E4B preserves W_v by
construction (orthogonality confirmed at 1.36e-5 across all 8 non-shared layers).
At macro scale (500 training iters per user × 2 users) the structural property
holds exactly as P7.A1 predicted. Null-space quality actually **exceeds**
standard LoRA on held-out user-domain text (K1643 PASS, ratio 0.47). Both
privacy-specific KCs failed, but for different structural reasons:

- **K1642 (MIA)** measured domain adoption, not per-example memorization, because
  the pre-registered protocol used OOD non-members. 20/20 recovery at τ*=3.42
  is the expected behavior of *any* trained domain adapter (members in-domain,
  non-members OOD). The OOD-non-member approach, inherited from T5.4
  LEARNINGS as the "fix" for same-distribution non-members, is itself flawed:
  it confounds style learning with memorization.
- **K1644 (cross-user B cosine)** landed at 0.39 vs threshold 0.30. Pre-registered
  threshold relaxation (from DB's 0.05 to behavioral 0.30) was not enough. B
  matrices concentrate on shared Q/A-format steering directions that don't
  isolate by domain without explicit orthogonalization (T5.4 recommendation).

## Prediction vs Measurement

| KC | Prediction (MATH.md) | Measurement (full N) | Outcome |
|---|---|---|---|
| K1642 MIA recovery ≤ 5% | "Likely KILLED if adapter overfits (500 iters); PASS if under-fit" | 20/20 (100%) | **FAIL** — overfit + MIA measures wrong object |
| K1643 PPL_null / PPL_std ≤ 1.05 | "Likely PASS (d_null=2048 ≫ r=6 capacity)" | 0.4748 (null 2.1× better) | **PASS** |
| K1644 mean max\|cos(B_A, B_B)\| < 0.30 | "≈ 0.10–0.25, PASS" | 0.3942 | **FAIL** by ~10pp |
| Orth sanity < 1e-4 | Exact by Theorem 1 | max 1.36e-5 | **PASS** |

## Measured numbers (from `results.json`)

- **Base model**: `mlx-community/gemma-4-e4b-it-4bit`, mlx-lm 0.31.2
- **Config**: rank=6, scale=8.0, lr=1e-4, iters=500, 8 non-shared layers (16–23)
- **Runtime**: 3.7 min end-to-end on M5 Pro 48 GB
- **Standard LoRA (user_A)**: 76,800 trainable params, final_loss=0.0819,
  holdout PPL=177.14
- **Null-space LoRA (user_A)**: 61,440 trainable params (20% fewer), final_loss=0.1002,
  holdout PPL=84.11
- **Null-space LoRA (user_B)**: 122,880 trainable params (layer 23 has d_out=1024
  vs 512 for 16–22), final_loss=0.0794
- **K1642 details**: member_loss mean=0.101 min=0.062; non-member (legal)
  mean=5.33 min=3.42. τ*=3.42. 20/20 members below τ*.
- **K1644 per-layer max\|cos\|**: {16: 0.398, 17: 0.375, 18: 0.373, 19: 0.375,
  20: 0.403, 21: 0.434, 22: 0.377, 23: 0.419}. Mean 0.3942.
- **Orthogonality**: layer-by-layer max\|W_v @ A_eff\| in [1.03e-5, 1.36e-5].

## What this run established (substantive, despite KILL)

1. **Null-space LoRA has a regularization effect under small-data training.**
   Standard LoRA reached final_loss 0.08 (memorization) but holdout PPL=177
   (severe overfitting). Null-space adapter reached final_loss 0.10 but holdout
   PPL=84 (2.1× better generalization). The null-space constraint acts as a
   structural prior that limits overfitting when training data is small. This is a
   stronger finding than the pre-registered K1643 threshold — at small user
   corpora (20 QA), null-space is **preferable** to standard LoRA on quality alone.
   Noted as a candidate Finding for the Analyst.

2. **Structural orthogonality (P7.A1 Theorem 1) replicated on different target
   layers and different training data.** Layers 16–23 (not 24–41) confirmed as
   the operational set on Gemma-4 E4B. Every future experiment touching
   Gemma-4 v_proj/k_proj should target this range.

3. **B-matrix output-direction cosine is NOT geometrically isolated without
   explicit orthogonalization.** Even with distinct domain training (medical
   vs legal), trained B_A and B_B share ~0.4 cosine on v_proj output
   directions — well above the mathematical prediction of 1/√512 ≈ 0.044 for
   independent isotropic vectors. The shared answer-format steering dominates.
   This confirms T5.4 LEARNINGS and reopens the Gram-Schmidt-on-B recommendation.

## Why K1642 failed structurally (not just numerically)

The pre-registered MIA uses OOD non-members (user_A medical members vs user_B
legal texts as non-members). This was inherited from T5.4 LEARNINGS as the "fix"
for same-distribution non-members that made delta=0pp "mathematically
impossible". In retrospect:

- **Same-distribution non-members** conflate memorization with nothing (no signal).
- **OOD non-members** conflate memorization with domain learning (signal from
  domain, not memorization).

Neither protocol isolates the per-example-memorization signal the DB KC text
asks about ("adapter weights contain no recoverable tokens from training data").
The correct MIA protocol requires:

- A large pool (N≥100) of same-domain QA;
- Random split into train (e.g., 20) and held-out-same-domain (e.g., 80);
- Loss-threshold distinguisher at FPR=5% on held-out-same-domain;
- TPR-at-5%-FPR is the recovery fraction.

With 20 train and 5 held-out-same-domain (our USER_A_HOLDOUT), the split is too
small to fit a reliable τ*. The corpus has to be ≥100 for the protocol to be
statistically meaningful. This is infeasible at the current corpus scale; a v2
experiment needs generated/sourced corpora of sufficient size.

**This is antipattern #6 firing on K1642** ("KC measures wrong object"). Per
researcher hat rule 5.6, if antipattern #6 applies, do not mark supported. We
therefore complete as `killed`, documenting the structural issue.

## Why K1644 failed (architectural)

Theorem 2 predicts `std(cos) ≈ 1/√d_out ≈ 0.044` for independent isotropic B
columns. Observed `mean max|cos| = 0.39`, i.e., ~9× the isotropic prediction.
This means trained B matrices are **not** independent random vectors — they
concentrate on a shared low-dimensional response-format subspace (probably
related to "produce an answer to a Q/A formatted query" features that both
users' training data shared).

Fix (out of scope here, for v2):
- Add a Gram-Schmidt / Procrustes step that orthogonalizes B_B against B_A
  during B's training (the recipe T5.4 LEARNINGS proposed).
- OR pre-compute an orthonormal basis spanning `d_out` per user, assign disjoint
  subspaces for each user's B.

## Assumptions as actually satisfied (cross-check MATH.md §Assumptions)

1. N=2 user pairs — satisfied. Extrapolation to N=10 still out of scope.
2. Cross-user cosine threshold 0.30 — used; FAILED at 0.39.
3. K1643 quality metric on held-out USER_A — used. **PASS**.
4. Full N (not smoke) — confirmed, `is_smoke=false` in results.json.
5. Training data: 20 medical + 20 legal — used.
6. Scale=8.0 — used (not 20, per mem-antipattern-003).
7. mlx-lm 0.31.2 — confirmed.

## Antipattern pre-flight (PLAN.md §1 checklist)

1. Composition bug: N/A (no composition).
2. Tautological routing: N/A (no routing).
3. LORA_SCALE=20: avoided (scale=8.0).
4. KC swap: no edit to MATH.md post-run (`git diff` is empty on MATH.md since
   the pre-registration commit).
5. KC measures wrong object: **K1642 yes** (see above) → verdict=killed, not
   supported. K1643 pre-registration matches quality claim. K1644 pre-registration
   matches geometric-isolation claim.
6. Smoke-as-full: avoided (`is_smoke=false`).
7. `shutil.copy` as new adapter: N/A.
8. Hardcoded `"pass": True`: audited — all `pass` fields computed from measurements.
9. Thinking-mode truncation: N/A.
10. Eval-template truncation producing base=0%: N/A (PPL eval, no generation).
11. Proxy-model-substituted-for-target: N/A (macro Gemma-4 E4B as targeted).

## Follow-up path (v2, not implemented here)

1. Build a 100+ same-domain QA corpus for proper MIA.
2. Implement Gram-Schmidt on B during user_B training (orthogonalize against
   user_A's B columns). Predict: mean max|cos| ≤ 0.05 by construction.
3. Re-pre-register K1642 as "TPR-at-5%-FPR on held-out-same-domain ≤ 5%".
4. Keep null-space structural isolation (K1299 / orthogonality sanity) — it is
   the settled load-bearing result.

## Files

- `MATH.md` — pre-registration (git-clean since commit; no post-hoc KC edits).
- `run_experiment.py` — 8-phase pipeline; full run 3.7 min on M5 Pro.
- `results.json` — all measurements; verdict=KILLED.
- `null_bases.safetensors`, `adapter_user_A_std.safetensors`,
  `adapter_user_A_null.safetensors`, `adapter_user_B_null.safetensors` —
  training artifacts for replication.
