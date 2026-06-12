# MATH — exp_pgolf_random_frame_compose

**Claim.** (1) At micro scale, a from-scratch byte-level GPT whose dense linears are FROZEN random
matrices plus trained low-rank corrections is competitive with a fully-trained dense control
(replication of PGolf "Random Linear Adapter", ref #707, 1.1971 BPB with 37.5% params/block saved).
(2) When two domain adapters are trained over FROZEN output frames, making the frames
disjoint-orthonormal blocks (DFA, bet dfa-init) cuts composition interference vs sharing one frame.

This is the from-scratch companion to `exp_bet_dfa_r1_n2_composition` (frozen-Gemma scale, where
post-hoc projection recovered only 17.6% of the gap — F-pending). Here the frame is present AT TRAIN
TIME, the one degree of freedom R1 could not test.

## Setup

Substrate (new, this experiment): byte-level GPT in MLX (mlx==0.31.1), tied embeddings (ref #711
anchor), pre-LN, d=256, L=4, H=4, ctx=256, batch=32, AdamW. Two text domains:

- **prose** = tinyshakespeare (downloaded once, cached; 1.1 MB)
- **code** = concatenated Python stdlib sources (local, deterministic glob, truncated to 4 MB)

Pretrain corpus = 50/50 byte-interleave; BPB = CE_nats / ln 2 per byte on held-out val windows.

## Theorem 1 (frame irrelevance at micro scale — replication)

Let W₀ ∈ R^{d×d} be a frozen random matrix (entries N(0, 1/d)) and Δ = B A a trained rank-r
correction, r = 32, applied to every attention/MLP linear. The function class
{x ↦ (W₀ + BA)x} composed with trained embeddings/norms covers, for each layer, an
r·2d-dimensional affine subspace of weight space anchored at a near-isometry (random W₀ is
well-conditioned w.h.p. at d=256, Marchenko–Pastur). Ref #707 measured this class reaching
1.1971 BPB, competitive with dense at equal budget. **Prediction P1:** at equal steps (1500) and
equal data, `bpb_random_frame − bpb_dense ≤ 0.08`.

**K2320 (pre-registered kill):** random-frame arm worse than dense control by **> 0.08 BPB** at
equal training budget → replication fails → `killed`.

## Theorem 2 (disjoint frozen frames compose with less interference)

Phase 2 adapters: per linear layer, δᵢ(x) = Bᵢ(Aᵢx), with **Bᵢ frozen** (the frame), Aᵢ trained,
Aᵢ zero-init (δᵢ = 0 at start). Two arms, identical except frame geometry:

- **shared:** B₁ = B₂ = Q[:, :r] for one random orthonormal Q (same seed) — both domains' deltas
  live in the SAME r-dim output subspace.
- **disjoint:** B₁ = Q[:, :r], B₂ = Q[:, r:2r] with Q orthonormal ⇒ B₁ᵀB₂ = 0 exactly — composed
  delta outputs are orthogonal by construction, every layer.

Composition is Σᵢ Bᵢ(Aᵢx) (never (ΣB)(ΣA)); adapter scale = 1.0 ≤ 8. r = 16, 2r = 32 ≤ d.

Interference per domain i: `I_i = bpb_composed(val_i) − bpb_solo_i(val_i)`; arm interference
`I = (I_prose + I_code)/2`. In the shared arm, B A₁ and B A₂ write into the same subspace, so
their summed delta perturbs each domain's tuned output directly (worst case ‖δ₁+δ₂‖ up to additive
in the same coordinates). In the disjoint arm cross-terms vanish in output space: domain i's
output coordinates along Bᵢ receive zero contribution from adapter j≠i. **Prediction P2:**
`I_disjoint ≤ 0.8 · I_shared` (≥20% reduction).

**K2321 (pre-registered kill):** `(I_shared − I_disjoint)/I_shared < 0.20` → DFA mechanism absent
at micro scale → `killed`.

**Validity gate (pre-registered, not a goalpost move):** K2321 is only decidable if interference is
actually present: `I_shared ≥ 0.02 BPB`. If `I_shared < 0.02`, K2321 is `inconclusive` and the
overall verdict is `provisional` (no interference to cut — the experiment cannot speak).

## Budget (M5 Pro, MLX)

2 pretrains × 1500 steps + 4 adapter trainings × 800 steps, d=256/L=4/ctx=256/batch=32,
compiled train step, eval uncompiled. Estimated 10–25 min wall-clock. `is_smoke: false`.

## Honest risk

Same gap that killed A-orthogonality and capped R1 at 17.6% recovery: param/output-space
disjointness may not buy functional non-interference once layers are stacked (layer ℓ+1 mixes the
disjoint coordinates immediately). K2321 is designed to expose exactly that for minutes of compute.
