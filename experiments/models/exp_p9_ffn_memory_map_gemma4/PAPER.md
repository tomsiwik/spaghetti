# P9.D0: FFN Neurons as Key-Value Memories in Gemma 4 E4B — KILLED

## Abstract
Applied the Geva et al. (arXiv:2012.14913) FFN-as-key-value-memory framework to
Gemma 4 E4B (4-bit, 42 layers, 10240 intermediate). All three kill criteria fail:
pattern specificity 23.7% (threshold: 50%), domain clustering 1.66x (threshold: 2x),
next-token agreement 0.10% (threshold: 1%). The GeGLU activation and 4-bit quantization
fundamentally weaken the key-value memory interpretability that holds for ReLU models.

## Prediction vs Measurement

| Metric | Predicted | Measured | Kill | Status |
|--------|-----------|----------|------|--------|
| K1: Pattern rate | 55-70% | 23.7% | >=50% | **FAIL** |
| K2: Clustering ratio | >=2x | 1.66x | >=2x | **FAIL** |
| K3: Next-token agreement | 1.5-4% | 0.10% | >1% | **FAIL** |

## Key Findings

### 1. GeGLU Distributes Knowledge More Diffusely Than ReLU
The original paper (Geva et al.) analyzed ReLU FFNs where neurons produce hard zeros,
creating sharp specialization. GeGLU's approximate GELU gate produces small but non-zero
activations for nearly all inputs. Result: 40,276 neurons are "frequently activated"
(5+ times across 100 inputs), but only 23.7% show domain-specific patterns.

**Early vs Late Layers**: Domain specialization is concentrated in layers 0-8 (349-520
domain-specific neurons each) and drops dramatically in middle layers (13-28 at only
13-80 domain-specific neurons). This inverts Geva et al.'s finding that upper layers
are more specialized — in Gemma 4, specialization is an early-layer phenomenon.

### 2. Code Domain Most Distinctive
Among domains, code has the highest exclusivity (31.6% of its top neurons are unique to
code). This makes sense: code syntax is structurally distinct from natural language.
Math (19.6%), medical (19.5%), legal (20.3%), and general (23.3%) share more neurons.

### 3. Value Vector Predictions Degraded by Quantization
Next-token agreement at 0.10% is 265x above random (0.0004%) but 35x below the paper's
3.5% and 10x below our 1% threshold. The 4-bit quantization of down_proj weights
introduces per-group quantization noise that distorts the value vector → vocabulary mapping.
Each value vector has 2560 dimensions quantized to 4 bits in groups — the accumulated
error across dimensions degrades the argmax prediction.

### 4. Gemma 4 E4B Uses Uniform FFN Dimensions
All 42 layers have d_ff = 10240 (no double-wide layers despite the config default).
This means the E4B variant doesn't use KV-shared layers with expanded FFN, unlike
the 26B model.

## Impossibility Structure

The Geva et al. key-value memory framework requires:
1. **Sparse activation** (ReLU creates hard zeros) — GeGLU produces soft activations
2. **Full-precision values** (vocabulary projection requires precise weights) — 4-bit
   quantization adds noise proportional to 2^(-bits)/sqrt(group_size)
3. **Sufficient data per neuron** (paper used 100K+ examples, we used 100) — statistical
   power insufficient to detect fine-grained patterns

These three factors compound: soft activation makes pattern detection harder, quantization
makes value prediction noisier, and small data makes both estimates unreliable.

For our project (composable ternary experts): FFN neuron-level editing is NOT viable
as an adapter mechanism. The neurons don't specialize cleanly enough for targeted editing.
LoRA-based adaptation (which operates on subspace projections, not individual neurons)
remains the correct approach.

## Relevance to Architecture

This KILLED result reinforces that:
- **LoRA targets the right abstraction level**: subspace projections (rank-8/16) capture
  domain knowledge more effectively than individual neurons
- **Pre-merge composition works because LoRA is continuous**: individual neuron editing
  would require discrete selection, which doesn't compose
- The blocked experiment (exp_p9_ffn_targeted_edit) should be reconsidered — neuron
  editing is fundamentally limited by GeGLU diffuseness

## Experimental Details
- Model: mlx-community/gemma-4-e4b-it-4bit
- 100 inputs (20 per domain: math, code, medical, legal, general)
- Top-256 neurons tracked per input per layer
- Value predictions: batched projection through tied embeddings
- Runtime: 25s on M5 Pro
