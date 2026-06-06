# End-to-End Cost Accounting: Mathematical Framework

## Notation

| Symbol | Definition | Unit |
|--------|-----------|------|
| C_total | True total cost per expert | $ |
| C_gen | Data generation cost (teacher API) | $ |
| C_train | GPU training cost (pure training) | $ |
| C_load | Model loading overhead per expert | $ |
| C_bench | Benchmark evaluation cost per expert | $ |
| C_setup | One-time setup amortized over N | $ |
| C_idle | GPU idle/waste amortized over N | $ |
| C_ortho | Orthogonality check cost | $ |
| C_gs | Gram-Schmidt projection cost | $ |
| C_merge | Pre-merge composition cost | $ |
| C_gate | Quality gate decision cost | $ |
| r_gpu | GPU rental rate | $/hr |
| t_train | Pure training time per expert | min |
| t_load | Model loading time per expert | min |
| N | Number of experts | - |

## Cost Decomposition

The true cost per expert is:

```
C_total = C_gen + C_train + C_load + C_bench + C_setup/N + C_idle/N + C_ortho + C_gs + C_merge + C_gate
```

### 1. Data Generation

For Groq batch API with teacher model T:

```
C_gen = price_per_token(T) * tokens_per_example * n_examples
```

At 1000 examples of ~500 tokens each with Llama 3.3 70B:
```
C_gen(70B) = $0.355/expert   (measured from pilot-50 log)
C_gen(8B)  = $0.020/expert   (Groq 8B batch pricing)
```

### 2. Training Cost

```
C_train = (t_train / 60) * r_gpu
```

For 300 steps, rank-16, all-modules on 4090:
```
C_train = (12.5 / 60) * $0.34 = $0.071
```

### 3. Model Loading Overhead

The subprocess-per-expert design reloads the base model each time:
```
C_load = (t_load / 60) * r_gpu = (2.5 / 60) * $0.34 = $0.014
```

This is avoidable with persistent process + adapter hot-swap.

### 4. Benchmark Evaluation

Sequential load-and-evaluate per expert:
```
C_bench = (t_bench / 60) * r_gpu = (2.4 / 60) * $0.34 = $0.014
```

With vLLM hot-swap: t_bench drops from 2.4 min to ~10s:
```
C_bench(vLLM) = (10/3600) * $0.16 = $0.0004
```

### 5. CPU-Only Operations (Zero Marginal Cost)

These operations run on local CPU with negligible cost:

| Operation | Time | FLOPs | Cost |
|-----------|------|-------|------|
| Orthogonality check | <8ms | ~80M | $0.00 |
| GS projection (N=50) | ~6s/expert | O(N*r*d) | $0.00 |
| Pre-merge (per expert) | ~0.5s | ~52G | $0.00 |
| Quality gate | <1ms | O(1) | $0.00 |

### 6. Amortized Setup

```
C_setup = (C_taxonomy + C_model_download + C_deps) / N
       = ($0.02 + $0.07) / 50
       = $0.002
```

Decreases as 1/N -- negligible at N >= 100.

## Kill Criteria Formalization

**K1:** C_total <= $1.00

**K2:** Define training cost as C_train + C_load (all GPU time attributable to training):
```
overhead = C_total - (C_train + C_load)
K2: overhead / (C_train + C_load) <= 3.0
```

## Scaling Laws

Cost per expert at scale N with teacher T and GPU G:

```
C_total(N, T, G) = C_gen(T) + (t_total(G)/60) * r(G) + C_setup/N + C_idle/N
```

As N -> infinity:
```
C_total -> C_gen(T) + (t_total(G)/60) * r(G)
```

The asymptotic cost is dominated by:
- Teacher API cost: scales linearly with n_examples, not N
- GPU training time: scales with steps * model_size, not N

## Numerical Verification

| Scenario | C_gen | C_gpu | C_amort | C_total |
|----------|-------|-------|---------|---------|
| Pilot-50 (actual) | $0.354 | $0.099 | $0.025 | $0.477 |
| N=500, 70B teacher | $0.354 | $0.099 | $0.002 | $0.455 |
| N=500, 8B teacher | $0.020 | $0.099 | $0.002 | $0.119 |
| Optimal (8B+A5000+vLLM) | $0.020 | $0.040 | $0.001 | $0.061 |

## Key Insight

The cost structure is **teacher-dominated** at 70B:

```
C_gen / C_total = 74.1%    (70B teacher)
C_gen / C_total = 16.8%    (8B teacher)
```

Switching from 70B to 8B teacher reduces cost by 75% ($0.48 -> $0.12).
The quality-cost tradeoff of teacher size is the most impactful lever.

## Assumptions

1. Training time ~15 min/expert is from pilot-50 on 4090 (300 steps, batch=8, seq=1024)
2. Model loading time ~2.5 min estimated from typical 7B 4-bit load time on 4090
3. Benchmark time ~2 hr for 50 experts from orchestrate.sh estimates
4. GPU idle waste of $1.16 from difference between reported $22 total and accounted costs
5. Groq pricing as of 2026-03: 70B batch at ~$0.355/1000 examples, 8B at ~$0.02/1000
6. CPU operations truly zero cost (run on local machine, not rented GPU)
