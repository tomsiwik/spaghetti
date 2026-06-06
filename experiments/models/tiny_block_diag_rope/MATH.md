# Block-Diagonal Attention: RoPE Position Invariance Proof

## Type: Proof Verification (Type 1)

## A. Failure Mode Identification

**Claimed failure (Finding #314):** Block-diagonal attention with K=2 segments
shows 8.9% PPL gap vs segment-isolated evaluation. The gap was attributed to
RoPE position encoding: segment B at positions [b, b+1, ..., b+L-1] receives
different rotary embeddings than segment B in isolation at positions [0, 1, ..., L-1].

**Re-diagnosis:** The gap was misattributed. RoPE attention is
relative-position-invariant. The actual failure mode was a **code-path
discrepancy** in Finding #314's implementation: the block-diagonal forward used
manual layer-by-layer LoRA computation while segment-isolated used the standard
model forward path.

## B. The Right Question

Not "how do we reset RoPE per segment?" but rather:

**"Is the attention output for within-segment tokens position-invariant under
block-diagonal causal masking?"**

If yes, RoPE reset is unnecessary and the implementation gap is a software bug,
not a fundamental limitation.

## C. Prior Mathematical Foundations

### Theorem (RoPE Relative Position Property)

From Su et al. (arXiv 2104.09864), Section 3.4:

RoPE encodes position p by applying rotation matrix R(p) to query/key vectors.
For queries at position i and keys at position j:

```
(R(p_i) q)^T (R(p_j) k) = q^T R(p_i)^T R(p_j) k = q^T R(p_j - p_i) k
```

The attention score depends only on the **relative position** (p_j - p_i), not
on absolute positions p_i and p_j individually.

**Proof:** R(p) is a block-diagonal rotation matrix with 2x2 blocks:

```
R_m(p) = [[cos(p * theta_m), -sin(p * theta_m)],
           [sin(p * theta_m),  cos(p * theta_m)]]
```

where theta_m = base^(-2m/d). Since R_m is a rotation matrix:
R_m(p_i)^T R_m(p_j) = R_m(-p_i) R_m(p_j) = R_m(p_j - p_i).

This holds for each 2x2 block independently, hence for the full rotation. QED.

### Corollary: Value Stream Independence

Values V are NOT rotated by RoPE. The attention output is:

```
O_i = sum_j alpha_{ij} V_j,    where alpha_{ij} = softmax(q_i^T R(j-i) k_j / sqrt(d))
```

The weights alpha_{ij} depend only on relative positions. V_j is position-free.
Therefore O_i is invariant to the absolute position offset of the segment.

## D. Proof of Guarantee

**Theorem 1.** (Block-Diagonal Position Invariance) Let M be a transformer with
RoPE positional encoding and block-diagonal causal mask M_bd with segments
S_1 = [0, b) and S_2 = [b, b+L). For any token at position p in segment S_2
(p >= b), let p' = p - b be the corresponding position in segment-isolated
evaluation. Then:

```
logit(token_p | M_bd, S_2 at positions [b..b+L)) = logit(token_{p'} | M_std, S_2 at positions [0..L))
```

up to floating-point precision.

*Proof.* By structural induction on transformer layers.

**Base case (embedding):** The embedding layer e(token) is position-independent.
For the same token, the embedding is identical regardless of absolute position.

**Inductive step:** Assume hidden state h_l at layer l is identical for all
tokens in S_2 under both regimes (up to position relabeling). We show h_{l+1}
is also identical.

**(i) Attention sub-layer:**

1. Projections: Q = q_proj(RMSNorm(h_l)), K = k_proj(RMSNorm(h_l)),
   V = v_proj(RMSNorm(h_l)). All are position-independent linear maps applied
   to position-identical hidden states. Q, K, V are identical.

2. RoPE application: In M_bd, Q_p gets rotation R(p). In M_std, Q_{p'} gets
   R(p'). These differ. However, for any pair (i, j) both in S_2:
   - M_bd: score(i, j) = Q_i^T R(i)^T R(j) K_j = Q_i^T R(j-i) K_j
   - M_std: score(i', j') = Q_{i'}^T R(i')^T R(j') K_{j'} = Q_{i'}^T R(j'-i') K_{j'}
   Since Q_i = Q_{i'} (by inductive hypothesis), K_j = K_{j'}, and
   (j-i) = (j'-i') = (j-i) (within-segment relative positions are preserved):
   score(i,j) = score(i',j').

3. Block-diagonal mask: Under M_bd, token i in S_2 cannot attend to any token
   in S_1. Under M_std, there are no tokens outside S_2. The attention set is
   identical in both regimes.

4. Value aggregation: O_i = sum_{j in S_2} alpha_{ij} V_j. Since alpha and V
   are identical, O is identical.

5. Post-attention: attn_sub_norm, o_proj are position-independent. Residual
   connection preserves equality.

**(ii) MLP sub-layer:**

RMSNorm, gate_proj, up_proj, down_proj, relu2, ffn_sub_norm are all
position-independent transformations. Since input h_{l,post_attn} is identical
(from step i), MLP output is identical.

**(iii) LoRA adapter sub-layer:**

If LoRA adapters are applied (RuntimeLoRA: y = base(x) + alpha * (x @ A) @ B),
the LoRA computation is position-independent: it depends only on the hidden
state x, not on position. Since x is identical by the inductive hypothesis,
the LoRA output is identical.

Therefore h_{l+1} is identical, completing the induction.

**Final step:** The final RMSNorm + LM head are position-independent.
Logits are identical. QED.

**Corollary 1.** Per-segment RoPE position reset is unnecessary for
block-diagonal attention isolation. The standard RoPE application already
produces position-invariant outputs for within-segment tokens.

## E. Quantitative Predictions

| # | Prediction | Source | Kill criterion |
|---|-----------|--------|----------------|
| P1 | PPL gap (bd vs isolated) < 0.5% | Theorem 1 (floating-point noise only) | K816: gap < 5% (PASS if < 0.5%) |
| P2 | Max per-token NLL diff < 0.5 | Theorem 1 + bf16 precision | |
| P3 | bd fair gap < 5% (segment-level quality) | Theorem 1 (position invariance → bd = isolated) | K817: bd fair gap < 5% |
| P4 | Segment A diff = 0.0 (replicates #314) | Segment A has same positions in both regimes | |
| P5 | Segment B diff = segment A diff (within noise) | Theorem 1: position invariance | Falsifies #314's claim |

**The key prediction is P5:** If Theorem 1 holds, segment B's diff from
isolated should be the SAME as segment A's diff (both just floating-point
noise from different code paths). Finding #314 reported seg A diff = 0.000 and
seg B diff = 0.020, attributing the seg B difference to RoPE. Our prediction is
that with a UNIFIED code path (pierre adapters + standard forward), both
segments show negligible diff.

## F. Assumptions & Breaking Conditions

1. **No absolute position embeddings.** BitNet-2B-4T uses RoPE only (no learned
   absolute position embedding, no ALiBi). If a model used absolute position
   embeddings, Theorem 1 would fail.

2. **Block-diagonal mask correctly implemented.** If any cross-segment attention
   leaks through, the within-segment isolation breaks.

3. **Same adapter for both regimes.** The comparison assumes the same adapter
   weights. Different adapters would trivially produce different outputs.

4. **Floating-point determinism.** Different code paths (manual layer loop vs
   model.__call__) may produce different floating-point results due to
   operation ordering. This is noise, not a real gap.

## G. Worked Example (d=4, 2 tokens per segment)

Let d=4, so RoPE has 2 rotation blocks. Positions: segment A = [0,1],
segment B starts at position 2.

Token at position 2 in block-diagonal (segment B, first token):
- Q = q_proj(h), K = k_proj(h)  (same h as position 0 by embedding equality)
- Q_rot = R(2) Q, K_rot = R(2) K
- Only attends to itself (block-diagonal + causal): score = Q^T R(0) K = Q^T K
- O = alpha * V (where alpha = softmax(Q^T K / sqrt(d)) = 1.0 for self-attention)

Token at position 0 in segment-isolated:
- Q = q_proj(h), K = k_proj(h)  (same h)
- Q_rot = R(0) Q, K_rot = R(0) K
- Only attends to itself: score = Q^T R(0) K = Q^T K
- O = 1.0 * V

Identical output. The R(2) rotation cancels in the score computation.

## H. Boundary Token Analysis (Post-Experiment Addition)

The experiment revealed that Finding #314's 8.9% gap was entirely from including
the boundary prediction in the PPL average, not from RoPE.

**The boundary token:** At position boundary-1, the model predicts the first
token of segment B using the last token of segment A. This is a cross-domain
prediction with no relevant context. Empirically, boundary NLL = 13-23
(mean 17.07), compared to within-segment mean NLL ~ 1.5-3.

**Quantitative impact:** For segments of length L=128, the boundary contributes:
- 1 prediction out of 2L-1 = 255 total predictions (0.39% of tokens)
- But boundary NLL ~ 17 while mean NLL ~ 2, so it shifts mean NLL by
  (17 - 2) / 255 = 0.059, which shifts PPL by exp(0.059) - 1 = 6.1%

This matches the observed gap: fair gap 0.2% vs full gap 7.3%.

The boundary token is an IRREDUCIBLE cost of multi-domain sequences. It exists
regardless of attention strategy (block-diagonal, standard causal, or otherwise).
It should be excluded from quality comparisons between attention strategies.

## I. Complexity & Architecture Connection

No additional computation beyond mask construction. Block-diagonal masking
requires O(T^2) additive mask construction (or O(T) with segment indexing).
RoPE reset adds NO computation since it is proven unnecessary.

The relevant architecture component is the standard attention + RoPE pathway
in Llama-family models (BitNet-2B-4T is Llama architecture with BitLinear layers).

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   **RoPE attention scores depend only on relative position (j-i), not absolute
   position, by the rotation group property R(a)^T R(b) = R(b-a).**

2. Which existing theorem(s) does the proof build on?
   **Su et al. (2021), Section 3.4: RoPE relative position property. Standard
   rotation matrix algebra: R(-a)R(b) = R(b-a).**

3. What specific numbers does the proof predict?
   **P1: PPL gap < 0.5%. P2: Max NLL diff < 0.5. P5: Seg B diff = Seg A diff
   (both near 0, not 0.020 vs 0.000 as in #314).**

4. What would FALSIFY the proof (not just the experiment)?
   **A position-dependent mechanism other than RoPE in the architecture (e.g.,
   absolute position embedding, position-dependent normalization). Or a flaw in
   the rotation group property.**

5. How many hyperparameters does this approach add?
   **0. Block-diagonal masking + standard RoPE is the complete solution.**

6. Hack check: Am I adding fix #N to an existing stack?
   **No. This experiment REMOVES a proposed fix (RoPE reset) by proving it
   unnecessary. The block-diagonal mask alone is sufficient.**
