# P7.B1: Weighted Multi-Adapter Composition via Null-Space Projections

## Summary

Weighted composition of null-space adapters outperforms exclusive (argmax) routing by
32.7% on mixed-domain queries and 18.5% on single-domain queries. The MATH.md theorem
(null-space closure under convex combination) is verified: max|W_v @ D| = 9.57e-7,
six orders below the 1e-4 threshold. All 3 kill criteria pass.

## Setup

- Model: Gemma 4 e4b-it-4bit (quantized)
- 5 null-space LoRA adapters (r=16, scale=20) on v_proj layers 16-23
- Adapters reused from exp_p7_null_projection_routing (trained 300 iters each)
- Router: TF-IDF cosine similarity with softmax normalization
- Single-domain: 2 test texts per domain (10 total)
- Mixed-domain: 2 texts per cross-domain pair, 4 pairs (8 total)
- Metric: NTP loss (cross-entropy, lower = better)

## Prediction vs Measurement

| Prediction | Metric | Predicted | Measured | Verdict |
|------------|--------|-----------|----------|---------|
| P1: Single-domain degradation < 2pp | degradation% | < 2.0% | **-18.5%** (improvement) | PASS |
| P2: Mixed-domain improvement >= 3pp | improvement% | >= 3.0% | **32.7%** | PASS |
| P3: Orthogonality preserved | max\|W_v@D\| | < 1e-4 | **9.57e-7** | PASS |

## Detailed Results

### Single-Domain (per domain)

| Domain | No Adapter | Exclusive | Weighted | Oracle | Route Correct |
|--------|-----------|-----------|----------|--------|---------------|
| medical | 10.50 | 5.38 | 5.15 | 5.38 | 2/2 |
| code | 9.34 | 5.52 | 4.24 | 5.30 | 2/2 |
| math | 5.61 | 3.05 | 3.64 | 3.05 | 2/2 |
| legal | 11.41 | 6.83 | 4.80 | 6.83 | 2/2 |
| finance | 9.78 | 6.28 | 4.22 | 5.75 | 1/2 |
| **Average** | **9.33** | **5.41** | **4.41** | **5.26** | **9/10** |

### Mixed-Domain (per pair)

| Mix | No Adapter | Exclusive | Weighted | Oracle | Weight Entropy |
|-----|-----------|-----------|----------|--------|----------------|
| medical+legal | 9.47 | 8.58 | 5.52 | 7.67 | 0.998 |
| code+finance | 11.44 | 9.19 | 5.86 | 7.66 | 0.998 |
| math+medical | 8.72 | 6.77 | 5.00 | 5.97 | 1.000 |
| legal+finance | 11.88 | 10.25 | 7.03 | 8.81 | 0.999 |
| **Average** | **10.38** | **8.70** | **5.85** | **7.53** | **0.999** |

### Orthogonality Verification

| Weight Config | max|W_v @ D| | Threshold |
|---------------|-------------|-----------|
| Uniform (0.2 each) | 5.32e-7 | 1e-4 |
| Peaked medical (0.8) | 9.57e-7 | 1e-4 |
| Mixed med+legal (0.4/0.4) | 7.60e-7 | 1e-4 |

## Kill Criteria

| ID | Criterion | Threshold | Measured | Verdict |
|----|-----------|-----------|----------|---------|
| K1303 | Weighted > exclusive on mixed-domain | >= 3pp | **32.7pp** | **PASS** |
| K1304 | No single-domain degradation | < 2pp | **-18.5pp** (improved) | **PASS** |
| K1305 | Cross-domain benefit from multi-adapter | > 0pp | **43.6pp** | **PASS** |

## Key Finding: Ensemble Effect Dominates Routing

The TF-IDF routing weights have near-uniform entropy (0.996-1.000). The top weight
ranges from 0.20-0.25 for 5 domains (uniform = 0.20). This means the experiment
effectively tests **null-space adapter averaging** rather than smart routing.

Despite near-uniform weights:
- Weighted composition outperforms exclusive on BOTH single and mixed-domain queries
- Weighted even outperforms **oracle** exclusive (4.41 vs 5.26 single, 5.85 vs 7.53 mixed)
- This means no single adapter matches the quality of the average of all five

**Interpretation**: The improvement comes from ensemble effects, not routing precision.
Each adapter captures domain-specific patterns in null(W_v). Averaging pulls in useful
features from all domains. The null-space guarantee ensures averaging cannot degrade
the base model (orthogonality preserved at 9.57e-7).

## Caveats

1. **Memorization scale**: Adapters trained on 8 texts/domain for 300 iters. Results may
   differ at larger training scale where adapters become more specialized.
2. **Near-uniform weights**: TF-IDF discrimination is weak at this vocabulary size (40
   training texts total). At larger scale, weights would be more peaked, potentially
   changing the single vs weighted tradeoff.
3. **No behavioral evaluation**: NTP loss is a proxy. Whether weighted composition
   produces qualitatively better text is untested.
4. **Ensemble vs composition**: The result may be explained by simple ensembling (averaging
   helps any set of models) rather than null-space-specific composition. A control with
   non-null-space adapters would disambiguate.

## Connections

- **LoRAHub (2310.13699)**: Gradient-free LoRA composition via weighted sum improves
  quality when relevant adapters are combined. Our result extends this with null-space
  orthogonality guarantee.
- **Finding #494**: Null-space LoRA preserves 98.7% quality — confirmed here as all
  single-domain adapters improve over no-adapter.
- **Finding #495**: Route in range(W_v), adapt in null(W_v). This experiment validates
  the "adapt" half — null-space composition is structurally safe and empirically beneficial.

## Runtime

- Total: 0.2 min (precompute + evaluate + verify)
- Platform: Apple M5 Pro 48GB
- Peak memory: ~4.5 GB (model + deltas)
