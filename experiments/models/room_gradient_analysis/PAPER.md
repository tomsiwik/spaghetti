# Room Model: Adapter Gradient Analysis

## Hypothesis

The spatial gradients of LoRA adapter B-matrices (∇H: discrete differences between adjacent matrix elements, treated as edge detection on a heightmap) encode domain-level behavioral information. Specifically, gradient similarity between two domain adapters should correlate with their behavioral similarity.

**Rationale**: In the Room model formulation (W_combined = Σ ΔW_i), each adapter's B-matrix spatial structure may reflect domain-specific patterns. High gradient magnitude indicates rapid changes in the learned representation, which could signal domain differentiation. If gradient patterns align with behavioral differences, we can use ∇H as a free routing signal without computing a separate router network.

## Prediction

**Before Running Experiment:**
- Correlation coefficient r between gradient similarity and behavioral similarity should satisfy: |r| ≥ 0.3
- This represents a moderate positive correlation: adapters with similar gradient profiles should have similar behavioral outputs

## Measurement Results

Experiment executed on 5 domain adapters from bitnet_sft_generation_v3:
- Domains: medical, code, math, legal, finance
- Analysis scope: 10 pairwise comparisons, 20 modules analyzed per comparison

### Gradient-Behavior Correlation Matrix

| Adapter Pair | Gradient Similarity | Behavioral Similarity | Relationship |
|--------------|--------------------|-----------------------|--------------|
| medical-code | 0.6558 | 0.10 | High gradient, low behavior |
| medical-math | 0.5845 | 0.20 | Moderate gradient, low behavior |
| medical-legal | 0.6173 | 0.30 | Moderate-high gradient, moderate behavior |
| medical-finance | 0.6127 | 0.20 | Moderate gradient, low behavior |
| code-math | 0.5852 | 0.50 | Moderate gradient, high behavior |
| code-legal | 0.6876 | 0.10 | High gradient, low behavior |
| code-finance | 0.7034 | 0.10 | High gradient, low behavior |
| math-legal | 0.5650 | 0.15 | Moderate gradient, low behavior |
| math-finance | 0.5564 | 0.20 | Moderate gradient, low behavior |
| legal-finance | 0.9197 | 0.40 | Very high gradient, moderate behavior |

### Correlation Coefficient

**Measured r = 0.1985**  
**Threshold requirement: |r| ≥ 0.3**  
**Result: FAIL**

The actual correlation of r = 0.1985 is substantially below the 0.3 threshold. This represents only a weak positive relationship between gradient similarity and behavioral similarity.

## Module-Level Insights

Top 5 modules with highest gradient variance across domains (i.e., most domain-differentiating):
1. `model.layers.0.self_attn.o_proj.lora_b` — variance: 1.85e-06
2. `model.layers.2.self_attn.v_proj.lora_b` — variance: 1.85e-06
3. `model.layers.27.self_attn.v_proj.lora_b` — variance: 1.61e-06
4. `model.layers.27.self_attn.o_proj.lora_b` — variance: 1.40e-06
5. `model.layers.1.self_attn.v_proj.lora_b` — variance: 1.16e-06

**Observation**: Attention output projection modules (o_proj) and value projection modules (v_proj) show the highest variance in gradient magnitude across domains. This suggests that domains primarily differentiate in how they modulate attention flow, not in early token processing.

## Conclusion

### Kill Criteria Assessment

**K825: Gradient similarity uncorrelated with behavioral similarity (r < 0.3)**  
**Status: KILLED** ✗

The hypothesis predicts that gradient patterns should correlate with behavioral differences. The measured correlation of r = 0.1985 is statistically insufficient to establish this relationship, falling 33% short of the 0.3 threshold.

### Failure Analysis

The weak correlation reveals that **spatial structure in B-matrices does not encode domain-level behavioral information**. This suggests:

1. **Gradient patterns are not routing signals**: The edge-detection metaphor breaks down. Adapter B-matrices don't exhibit structured spatial patterns that reflect domain semantics.

2. **Behavior emerges from full matrix magnitude, not local differences**: The correlation in Table 1 shows several counterexamples:
   - `legal-finance`: r_grad = 0.9197 (nearly identical gradients) but r_behav = 0.40 (moderate behavioral similarity)
   - `code-math`: r_grad = 0.5852 but r_behav = 0.50 (only moderate gradient agreement despite strong behavioral similarity)

3. **Gradient is not domain-invariant**: If ∇H were a reliable routing signal, we would expect systematic clustering—domains that behave similarly would have similar gradients. Instead, we see noise.

### Mathematical Implication

The Room model formulation W_combined = Σ ΔW_i remains sound, but the routing problem cannot be solved via spatial gradient analysis. A learned router or attention-based gating is necessary to achieve domain-aware composition.

### Recommendation

Future work should:
- Explore full-matrix cosine similarity (not gradients) as a routing signal
- Investigate whether routing emerges implicitly from attention mechanisms rather than explicit adapter analysis
- Consider that domain-specific behavior may be encoded in magnitude, sign patterns, or spectral properties rather than spatial structure
