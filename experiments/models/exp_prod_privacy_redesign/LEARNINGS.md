# LEARNINGS — exp_prod_privacy_redesign

## Core Finding
Null-space LoRA on Gemma-4 E4B preserves W_v exactly (max|W_v @ A_eff| = 1.36e-5
across layers 16–23) AND generalizes 2.1× better than standard LoRA on held-out
user text (holdout PPL 84 vs 177). Privacy claim is NOT established: pre-registered
MIA protocol (K1642) structurally cannot measure per-example memorization, and
cross-user B-matrix isolation (K1644) fails at 0.39 without explicit Gram-Schmidt.

## Why
- **P7.A1 Theorem 1 replicated** on new training data / distinct layer set → the
  structural orthogonality guarantee is now load-bearing and composable (K1299
  and K1641 independently measured same exact-by-construction property).
- **Null-space is a small-data regularizer, not just a quality-preserver.** At
  N=20 QA, standard LoRA final_loss=0.08 but overfits (holdout PPL 177);
  null-space final_loss=0.10 but PPL 84. Fewer trainable params (61k vs 77k) +
  range-space excision → lower-capacity prior → better generalization on small
  user corpora. Stronger than pre-reg K1643 threshold; propagate as Finding.
- **K1642 structural failure**: OOD non-members confound domain learning with
  memorization (20/20 members below τ* because members are in-domain at loss
  0.1, non-members OOD at loss 5.3). Same-distribution non-members confound
  memorization with nothing. Proper MIA requires pool N≥100 same-domain, random
  train/holdout split, TPR-at-5%-FPR on held-out-same-domain.
- **K1644 structural failure**: B matrices trained independently share ~0.4
  cosine on v_proj outputs because Q/A-format steering is a dominant shared
  subspace. Theorem 2 JL prediction (1/√512 ≈ 0.044) does not apply — training
  concentrates directions; need explicit Gram-Schmidt on B_B against B_A.

## Implications for Next Experiment
1. **Do NOT rerun OOD-non-member MIA.** V2 requires ≥100-example same-domain
   corpus, random split, TPR-at-FPR metric. Infeasible without data scale-up.
2. **Gram-Schmidt on B is now the critical path for N>1 privacy claims.** Both
   T5.4 LEARNINGS and this run converge: without orthogonalizing B_B against
   B_A during training, output-direction leakage is ~9× the isotropic floor.
3. **Null-space-as-regularizer** is a standalone Finding worth testing at
   scale: compare standard LoRA vs null-space LoRA holdout quality across
   N∈{20, 50, 100, 200} training QA. Predict null-space advantage shrinks as N
   grows (capacity-bound regime).
4. **Pre-registration discipline**: MATH.md must be committed BEFORE experiment
   runs (REVIEW non-blocking issue). Current dir untracked — OK for KILL,
   blocks any future PROCEED.
5. **Settled**: Layers 16–23 are the operational non-shared set for Gemma-4 E4B
   v_proj null-space work. Stop re-deriving the layer range.
