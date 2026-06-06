# P5.C0: Standing Committee Adapter — Results

## Summary

**Status: KILLED (1/3 pass)**

A permanent reasoning adapter ("standing committee") composed with a domain
format adapter via module-disjoint LoRA shows zero parametric cosine (K1278 PASS)
but severe functional interference: reasoning degrades 10pp and format degrades
60pp versus domain-only. Module-disjoint LoRA guarantees parametric orthogonality
but NOT functional independence.

## Prediction vs Measurement

| Metric | Prediction (MATH.md) | Measured | Match? |
|--------|---------------------|----------|--------|
| K1276: Composed vs domain reasoning | ≥ +3pp | **−10pp** (70% vs 80%) | **FAIL** |
| K1277: Format degradation | < 2pp | **60pp** (40% vs 100%) | **FAIL** |
| K1278: cos(committee, domain) | = 0.0 | **0.0** (by construction) | **PASS** |
| Cross-interaction magnitude | O(ε²) ≈ 2×10⁻⁶ | Dominant (60pp format loss) | **WRONG** |

## Full Accuracy Table

| Configuration | Reasoning | Format |
|---------------|-----------|--------|
| Base (no adapter) | 10% | 100% |
| Committee only (q_proj + o_proj) | 70% | 60% |
| Domain only (v_proj + down_proj) | 80% | 100% |
| Composed (committee + domain) | 70% | 40% |

## Analysis

### 1. Base Model Reasoning Confound

Base Gemma 4 E4B scores only 10% reasoning — not because it can't reason, but because
it generates `<|channel>thought` tokens that consume the max_tokens budget (120 tokens)
before reaching the answer. The model is reasoning internally but never outputs the
final answer within the generation window.

The domain adapter (v_proj + down_proj) somehow suppresses thinking tokens and produces
direct answers (80% accuracy). The committee adapter (q_proj + o_proj) teaches explicit
CoT ("Let me solve this step by step...") which is shorter than the thinking channel
and reaches the answer more often (70%).

### 2. Functional Interference Through Attention

The core finding: **module-disjoint LoRA achieves cos=0 in parameter space but
creates severe functional interference through the attention mechanism.**

The attention computation couples Q and V:
```
Attn(x) = softmax(Q·K^T / √d) · V
```

- Committee adapter changes Q (q_proj) and the output projection (o_proj)
- Domain adapter changes V (v_proj) and the MLP (down_proj)
- The changed Q-attention patterns operate on changed V-values

When composed, the committee's modified attention patterns select different V-values
than intended by the domain adapter. This is NOT an O(ε²) perturbation — it's a
first-order multiplicative interaction through the softmax.

### 3. Degeneration Patterns

The composed model exhibits three failure modes:

**a) Repetition loops**: Format prompts produce "MEET MEET MEET..." or "ME ME ME..."
— the committee's q_proj changes create an attention fixation where the model attends
to the same token position repeatedly.

**b) Pattern contamination**: Format tasks produce "MEETING\nStep 1: Calculate..."
mixing the committee's CoT template with the domain's structured output.

**c) Degenerate steps**: "Step 1: Define the answer to answer\nStep 2: Answer" — the
CoT skeleton activates but has no reasoning content, producing vacuous structure.

### 4. Why MATH.md's O(ε²) Bound Was Wrong

Theorem 2 bounded the cross-interaction as O(||ΔW_r|| · ||ΔW_d||) and estimated this
at O(2×10⁻⁶) based on ||ΔW||/||W|| ≈ r/d. This analysis was flawed because:

1. **Softmax is not linear**: The attention weight softmax(Q·K^T/√d) is a highly
   nonlinear function of Q. A small ΔQ can shift the argmax of attention, redirecting
   all information flow. The perturbation isn't small in the functional sense.

2. **Autoregressive amplification**: Each generation step depends on all previous tokens.
   A single attention-pattern shift in one layer compounds across all subsequent layers
   and generation steps, creating the observed repetition loops.

3. **The bound applies per-token, not per-generation**: Even if each individual
   token's perturbation is O(ε²), the error accumulates across the generation sequence,
   growing linearly with sequence length.

### 5. Training Details

| Adapter | Modules | Params | Loss first→last | Decrease | Latency |
|---------|---------|--------|-----------------|----------|---------|
| Committee | q_proj, o_proj | 229K | 2.078 → 0.238 | 88.5% | ~110ms/step |
| Domain | v_proj, down_proj | 229K | 2.016 → 1.078 | 46.5% | ~110ms/step |

Committee adapter converges much faster (88% vs 47% loss decrease), likely because
CoT patterns are more coherent than varied format templates. Both adapters are 1.96 MB.

## Impossibility Structure

Module-disjoint composition fails because the attention mechanism creates **obligatory
multiplicative coupling** between Q and V. There is no module partition of a transformer
layer where one subset controls "reasoning" and the other controls "format" without
attention coupling them. Specifically:

```
y = softmax((W_Q + ΔQ)x · ((W_K)x)^T / √d) · (W_V + ΔV)x
```

Even with ΔQ and ΔV in different modules, the output y depends on the product of
terms involving both changes. The softmax's argmax sensitivity means small ΔQ changes
can redirect 100% of the attention weight, amplifying ΔV's effect unpredictably.

**What would make it work**: Same-module composition with proven subspace isolation
(e.g., Grassmannian A-matrix initialization on the SAME modules), where both adapters
share the module but their weight updates live in orthogonal subspaces of the same
weight matrix. This avoids the Q×V coupling by keeping both adapters in the same
functional pathway.

## Behavioral Assessment

The composed model produces degenerate text (repetition loops, vacuous CoT steps,
pattern contamination). This is not a marginal metric failure — it's a fundamental
behavioral breakdown. The system is worse than either adapter alone on both tasks.

## Total Runtime

17.15 minutes (5 model loads + 2 training rounds + 4 evaluation rounds).
