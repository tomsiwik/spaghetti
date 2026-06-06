# Parameter-Matched Monolithic Ablation: SOLE vs Rank-80 Monolithic LoRA

## Hypothesis

**exp_bitnet_monolithic_rank_matched**: A rank-80 monolithic LoRA (~108M params) trained
on the union of all 5 domain datasets for 2000 steps will NOT beat SOLE routed (5 x r=16,
same ~108M total) on >60% of per-domain metrics.

**Kill criterion**: rank-80 monolithic beats SOLE routed on >60% of per-domain metrics (3+/5).

## Motivation

The prior SOLE-vs-monolithic experiment (exp_bitnet_sole_vs_monolithic) showed SOLE routed
wins 4/5 domains. However, that comparison has a 5x parameter asymmetry: SOLE uses
5 x r=16 = 108M total params vs monolithic r=16 = 21.6M params. A reviewer would dismiss
the result as "more parameters = better."

This experiment eliminates the confound by training a rank-80 monolithic LoRA that exactly
matches SOLE's total parameter count (108,134,400 each).

## Method

- **Base model**: BitNet-2B-4T (ternary, d=2560, 30 layers)
- **Monolithic**: 1 x r=80 ternary LoRA (QAT+STE), 108,134,400 trainable params
- **SOLE**: 5 x r=16 ternary LoRA (from prior experiment), 108,134,400 total params
- **Training**: 2000 steps, lr=1e-4, seq_len=128, seed=42, shuffled union data
- **7 target modules/layer**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Domains**: medical, code, math, legal, creative
- **Platform**: Apple Silicon (MLX), $0 compute

Parameter match verified: ratio = 1.00x exactly.

## Results

### Per-Domain Perplexity (lower is better)

| Domain   | Base PPL | SOLE r=16 | Mono r=16 | Mono r=80 | Winner  | Gap vs SOLE |
|----------|----------|-----------|-----------|-----------|---------|-------------|
| Medical  | 15.80    | 8.00      | 8.36      | 8.25      | SOLE    | -3.0%       |
| Code     | 3.52     | 2.76      | 2.84      | 2.82      | SOLE    | -2.1%       |
| Math     | 4.74     | 3.12      | 3.27      | 3.23      | SOLE    | -3.5%       |
| Legal    | 25.52    | 17.95     | 19.67     | 19.04     | SOLE    | -5.8%       |
| Creative | 3.60     | 3.17      | 3.00      | 2.98      | MONO-80 | +6.3%       |
| **Avg**  | **10.64**| **7.00**  | **7.43**  | **7.27**  | **SOLE**| **-3.7%**   |

### Key Findings

1. **SOLE wins 4/5 domains** even at matched total parameters. The specialization
   advantage persists when the parameter confound is removed.

2. **Rank scaling helps but isn't enough**: Mono r=80 improves -2.2% over mono r=16
   across all domains (more capacity → better performance). But SOLE's domain-specialized
   routing still wins by -3.7% on average.

3. **Creative domain exception persists**: Monolithic wins creative writing at both r=16
   and r=80, likely due to beneficial cross-domain transfer (creative benefits from
   code/math patterns). Gap widens from +5.6% (r=16) to +6.3% (r=80).

4. **Legal shows largest SOLE advantage** (-5.8%): highly specialized domain vocabulary
   benefits most from dedicated expert training without interference.

5. **Training converged**: loss 1.85 → 1.46 over 2000 steps (converged by 5% threshold).

### Rank Scaling Analysis

| Domain   | r=16 → r=80 improvement |
|----------|------------------------|
| Medical  | -1.3%                  |
| Code     | -0.7%                  |
| Math     | -1.1%                  |
| Legal    | -3.2%                  |
| Creative | -0.8%                  |
| **Avg**  | **-2.2%**              |

Rank scaling provides diminishing returns: 5x more parameters yields only 2.2% average
improvement. Domain-specific routing (SOLE) achieves 3.7% improvement with the same params.

## Kill Criteria Assessment

- **K1**: Mono r=80 wins 1/5 domains (threshold: >=3 to kill) → **PASS**

## Verdict: SUPPORTED

SOLE's specialization advantage is genuine and not an artifact of parameter asymmetry.
At exactly matched total parameters, SOLE routed outperforms monolithic on 4/5 domains
by an average of 3.7%.

**What this means for the paper**: The "more parameters" objection is definitively
addressed. SOLE's value is both quality (better per-domain PPL) AND operational
(modularity, no forgetting, incremental updates).

## Caveats

1. **Single seed** (42) — multi-seed would strengthen the claim
2. **Same training steps** (2000) for both — rank-80 may need more steps to converge
   (though loss did converge by the 5% threshold)
3. **Creative exception** is consistent and may indicate a class of tasks where monolithic
   is preferred (cross-domain transfer tasks)
4. **PPL-only metric** — downstream task accuracy could show different patterns
5. **Same training data shuffle seed** — different shuffles could affect results

## Runtime

- Model load + unpack: ~1s
- Training (2000 steps, r=80): ~20 min
- Evaluation (5 domains): ~2 min
- Total: ~22 min on Apple Silicon
