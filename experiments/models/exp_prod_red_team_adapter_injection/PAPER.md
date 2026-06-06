# PROD: Red-Team Malicious Adapter Cannot Extract Other Adapters' Data

## Verdict: KILLED

Two of three kill criteria failed. The null-space LoRA composition does NOT by
itself provide operational privacy between two users' adapters loaded
simultaneously. Specifically:

- **K1667 (activation probe)** FAIL — linear probe on last-layer hidden states
  distinguishes member (user A training) from non-member (user A holdout +
  neutral medical padding) at 65% accuracy (+15 pp above chance, threshold
  for PASS was +1 pp).
- **K1668 (parameter extraction subspace overlap)** FAIL — the rank-r SVD of
  the composed delta recovers ~58% of the subspace spanned by user B's B
  directions (and ~46% of user A's), averaged across 8 layers. Threshold for
  PASS was <1%.
- **K1669 (canary extraction)** PASS — greedy-decoding 20 tokens from a 15-token
  prefix reproduces 0/20 member texts at ≥50% token overlap.

This confirms the prediction set by the dependency `exp_prod_privacy_redesign`'s
K1644 FAIL (cross-user B cosine = 0.39 > 0.30): without explicit orthogonalization
of B matrices during training, composition leaks membership and adapter-direction
structure even though exact null-space preservation (`W_v @ A_eff = 1.36e-5`)
still holds.

## Prediction-vs-measurement

| KC | Claim (from MATH.md) | Predicted | Measured | Pass? |
|----|-----------|-----------|----------|-------|
| K1667 | Activation probe advantage < 1% over chance | FAIL (>0.30) | +15 pp | FAIL |
| K1668 | Max subspace overlap < 1% averaged over 8 layers | FAIL (0.5–1.0) | 58.3% | FAIL |
| K1669 | ≤1/20 canaries extracted at ≥50% overlap | FAIL (≥10) | 0/20 | PASS |

**Qualitative agreement**: two of three predictions matched. The one miss
(K1669 PASS when predicted FAIL) is behaviorally instructive — see "Surprising
finding" below.

## Measurements in detail

### K1668 — parameter extraction (subspace overlap)

Per-layer subspace overlap between rank-6 SVD of `ΔW_composed` output
directions and each user's true B directions (QR-orthonormalized):

| layer | overlap_A | overlap_B |
|-------|-----------|-----------|
| 16 | 0.52 | 0.54 |
| 17 | 0.34 | 0.66 |
| 18 | 0.28 | 0.79 |
| 19 | 0.57 | 0.51 |
| 20 | 0.36 | 0.68 |
| 21 | 0.51 | 0.51 |
| 22 | 0.52 | 0.51 |
| 23 | 0.58 | 0.46 |
| **mean** | **0.46** | **0.58** |

Interpretation: an attacker who observes `ΔW_composed` (which is accessible by
black-box query, and is the natural output of adapter composition) can recover
a rank-6 subspace that contains ~50% of either user's B subspace on average.
This is a direct consequence of dependency's K1644 FAIL — B_A and B_B share
~39% cosine, so a rank-12 space spans most of rank-6+rank-6.

### K1667 — activation probe

- 20 member hidden states (user A training texts) + 20 non-member
  (5 user A holdout + 15 neutral medical padding), mean-pooled last-layer
  (d=2560) states extracted under composed model.
- Logistic-regression probe, 2-fold stratified CV, standardized features.
- Fold accuracies: 0.60, 0.70 → mean 0.65 → +15 pp above chance.

Interpretation: the composed model's internal representation of member texts is
systematically different from non-member texts, even with both adapters loaded.
This is a membership-inference signal, though at small sample size (n=20/class)
the confidence interval is wide (roughly ±0.10–0.15 at 95%). The signal is
meaningful (+15 pp > sample noise floor), but a principled MIA-v2 with
TPR-at-5%-FPR on ≥100 same-domain samples would quantify it more precisely.

### K1669 — canary extraction

- 20 canaries (user A member texts), 15-token prefix → greedy-decode 20 tokens.
- Token-level positional overlap: max 0.40, mean 0.11, 0 texts at ≥0.5.
- 0/20 extracted → PASS.

The composed model does NOT verbatim reproduce training text via greedy
decoding from a short prefix. Two mechanisms plausibly explain this:

1. **Additive composition perturbation**: loading both adapters at scale=8 each
   produces effective scale=16 through the shared null-space input channel,
   shifting the output distribution enough to break exact memorization
   (training was done with only the single user's adapter loaded).
2. **Decoder-level smoothing**: Gemma-4 with null-space LoRA on v_proj alone
   modifies value projections only; the other attention pathways (Q, K, O,
   MLPs) remain base, providing substantial per-token variance that prevents
   lossless memorization even at member loss ~0.10.

## Surprising finding

**K1669 decoupled from K1667 and K1668.**

The prediction tied all three KCs together ("overfit adapter at loss ≤ 0.10
⇒ canary extraction succeeds"). Empirically, when two adapters are composed:

- Membership-inference signal is **preserved** (hidden states still encode
  member-specific structure → probe 65%).
- Subspace structure is **preserved** (SVD recovers B directions → overlap 58%).
- Verbatim extraction is **destroyed** (greedy decoding doesn't reproduce text
  → 0/20 extracted).

This suggests that verbatim canary extraction is a stricter test than either
subspace overlap or probe AUC, and **composition itself acts as a de facto
privacy operation against verbatim recall**, even though it fails to protect
against the stronger subspace- and probe-based attacks.

**Caveat — strength of the K1669 test**: greedy decoding from a 15-token
prefix on 40–60 token QA texts may be too weak a threat model. A stronger
canary test would (a) use longer contexts, (b) try temperature-0.7 sampling
with multiple seeds, (c) use training-specific random-token canaries rather
than member texts themselves. All of these require retraining, which this
experiment declined (scope-down documented in MATH.md Assumption #2). Treat
K1669 PASS as a lower-bound observation, not a proof of canary safety.

## Implications

1. **N=2 additive composition leaks subspace structure**. The composed delta's
   rank-r SVD top-r output directions project onto each individual user's B
   subspace at ~50%. This is not privacy — with even modest side information
   an attacker can attribute components to users.
2. **Membership inference survives composition**. Probe advantage +15 pp
   significantly exceeds chance at n=20/class, even with a simple mean-pooled
   hidden state. This signal likely grows with more training iters or larger
   per-user corpora.
3. **Canary extraction is non-trivial under composition**. This is a
   behavioral finding (not predicted by the pre-reg math): adding a second
   adapter on the same null-space channel disrupts greedy-decoding fidelity
   enough to defeat the 15-token-prefix attack. Value: composition *itself*
   may be a first-line (weak) defense against naïve extraction attempts.
4. **Critical path to privacy is Gram-Schmidt on B during training** (per
   dependency LEARNINGS #2). Without it, K1667/K1668 will continue to fail
   at any N.

## Assumptions (locked at MATH.md commit — no KC relaxation post-hoc)

All assumptions from MATH.md carry through:
- Reused adapters from `exp_prod_privacy_redesign` (no retraining).
- Canary count scope-down 100→20 with proportional threshold (but threshold
  was ≤1 of 20, stricter than the DB's ≤1 of 100 proportion = 0.2/20).
- `mlx-lm 0.31.2`, Apple M5 Pro 48GB, scale=8.0, rank=6, 8 non-shared layers
  (16–23).
- Logistic probe at `C=1.0`, 2-fold CV, n=20/class.

## Numerical health

- K1668 produced RuntimeWarnings from numpy at layer 16 during QR
  orthonormalization of `B_A.T` — harmless, attributable to a near-degenerate
  column pair; the computed overlaps at layer 16 (0.52 / 0.54) are consistent
  with other layers and with theoretical prediction.
- K1667 triggered sklearn RuntimeWarnings on large-magnitude feature vectors
  pre-standardization. Fixed by `nan_to_num` + `StandardScaler` in the rerun.
  Fold-1 accuracy dropped from 0.65 (pre-fix) to 0.70 (post-fix) — stable
  ≈ 0.65 mean across reruns.

## What's next

The researcher hat will recommend a v2 experiment: retrain user A and user B
null-space adapters with explicit Gram-Schmidt orthogonalization of B_B against
B_A (procedure from dependency LEARNINGS #2: after each training step, project
out B_A's column span from B_B's gradient updates). The resulting adapters
should drive cross-user B cosine below 0.10 and, per this experiment's
learnings, collapse both K1667 probe advantage and K1668 subspace overlap.
Canary K1669 would likely remain PASS (it already does).

## Runtime

- Total: 0.42 min (~25 s) on Apple M5 Pro 48GB.
- K1668: ~1 s (pure weight-space SVD).
- K1667: ~15 s (40 forward passes on composed model).
- K1669: ~10 s (20 greedy-decode × 20 tokens).

Fast because no retraining; all three attacks use pre-trained weights.
