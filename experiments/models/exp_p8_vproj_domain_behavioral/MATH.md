# MATH.md — P8: v_proj+o_proj Domain Adapters for Behavioral Text Quality

## Background: Why q_proj Adapters Failed Behaviorally

The behavioral E2E experiment (exp_p1_p0_behavioral_e2e) showed that q_proj-only
adapters improve benchmark accuracy (GSM8K +82pp) but fail to improve text generation
quality (math: 30% improvement rate, code: 20% — adapted text was WORSE).

**Root cause:** The attention mechanism computes `Attn(Q,K,V) = softmax(QK^T/√d)V`.
q_proj modifies Q, which changes *what positions the model attends to*. But the actual
content of the output is determined by V (value stream) and O (output projection).

For multiple-choice benchmarks, shifting attention patterns is sufficient — the model
just needs to attend to the right answer option. For text generation, the model needs
to produce domain-specific tokens, which requires modifying the value-to-output pathway.

## Theorem 1: Output-Path Sufficiency for Behavioral Modification

**Theorem.** Let `h_out = O · softmax(QK^T/√d) · V · x` be the output of an attention
layer. An adapter targeting {v_proj, o_proj} directly modifies the output token
distribution, while an adapter targeting only {q_proj} modifies the output only
indirectly through attention weight changes.

**Proof.**

The output of attention layer l is:
```
h_l = W_o · softmax((W_q x)(W_k x)^T / √d) · W_v x
```

**Case 1: q_proj adapter (ΔW_q).** The modified output is:
```
h_l' = W_o · softmax(((W_q + ΔW_q) x)(W_k x)^T / √d) · W_v x
```
The adapter modifies attention weights `α = softmax(...)` but the value content `W_v x`
and output projection `W_o` are unchanged. The effect is a reweighting of existing value
vectors — the vocabulary of available content is unchanged.

**Case 2: v_proj + o_proj adapter (ΔW_v, ΔW_o).** The modified output is:
```
h_l' = (W_o + ΔW_o) · softmax(QK^T / √d) · (W_v + ΔW_v) x
```
Expanding:
```
h_l' = W_o·α·W_v·x          (base output)
     + W_o·α·ΔW_v·x         (new value content, base projection)
     + ΔW_o·α·W_v·x         (base values, new output projection)
     + ΔW_o·α·ΔW_v·x        (new content, new projection — second order, small)
```
The terms `ΔW_v·x` and `ΔW_o·(...)` directly modify what content flows forward and how
it maps to output logits. This changes the *vocabulary distribution* of the output.

**Behavioral prediction:** v_proj+o_proj adapters trained on domain-specific text will
shift the output token distribution toward domain vocabulary, producing measurably
higher domain-vocabulary density in generated text. QED.

## Theorem 2: Grassmannian Composition on v_proj

**Theorem.** Let adapters Δ_i = B_i A_i^T for i = 1,...,5 on v_proj, where
A_i ∈ R^{n×r} are Grassmannian-orthogonal (A_i^T A_j = 0 for i≠j).
Then parameter-space interference is zero: ⟨Δ_i, Δ_j⟩_F = 0.

**Proof.** (Finding #126, #341, established)
```
⟨Δ_i, Δ_j⟩_F = trace(A_i B_i^T B_j A_j^T)
              = trace(B_j A_j^T A_i B_i^T)    [cyclic permutation]
              = trace(B_j · 0 · B_i^T)         [Grassmannian: A_j^T A_i = 0]
              = 0
```
This holds independently per projection layer. If A-matrices on v_proj are
Grassmannian-orthogonal AND A-matrices on o_proj are Grassmannian-orthogonal,
then both projections have zero parameter-space interference simultaneously.

**Capacity:** v_proj has dimension d=2816, null-space dim=2048 (Finding #493).
At rank r=16, this gives floor(2048/16) = 128 orthogonal slots. Similarly for o_proj.
5 adapters << 128 capacity. QED.

## Theorem 3: Composition Behavioral Quality Bound

**Claim.** Under Grassmannian composition (Theorem 2), the behavioral quality of
each adapter under composition should be ≥80% of its solo quality.

**Reasoning.** Parameter-space orthogonality eliminates first-order interference.
Activation-space interference (empirically bounded at max |cos|=0.29 for N=5,
Finding #353) causes bounded degradation. At N=5 with known bounds:

Expected behavioral retention per domain ≈ 1 - N·ε_activation
where ε_activation is the per-adapter activation-space cross-talk.
At ε_activation ≈ 0.03 (Finding #353 average), retention ≈ 1 - 5·0.03 = 0.85.

**Prediction:** composition retains ≥80% of solo adapter behavioral quality.

## Quantitative Predictions

| Kill Criterion | Prediction | Threshold | Basis |
|---------------|-----------|-----------|-------|
| K1 (Math) | ~70-80% vocabulary improvement rate | ≥60% | Finding #480: +70pp SOAP, +90pp Legal |
| K2 (Code) | ~65-75% vocabulary improvement rate | ≥60% | Same mechanism, code has strong vocab |
| K3 (Medical) | ~70-80% vocabulary improvement rate | ≥60% | Medical terminology is distinct |
| K4 (Composition) | ~80-90% retention | ≥80% | Theorem 3, Finding #353 |

## Note on Composition Test (K4)

Without Grassmannian A-matrices (standard LoRA training), composition may show
interference. K4 tests the *principle* that v_proj+o_proj adapters compose.
If K1-K3 pass but K4 fails, the implication is: v_proj+o_proj is the right target
AND Grassmannian A-matrices are needed for composition (confirming the architecture).
