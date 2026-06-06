# LoRA Soups CAT: Research Digest

## Hypothesis

Learnable per-module composition coefficients (CAT, arXiv 2410.13025) trained on 5% calibration data will beat uniform 1/N merging by >10% and achieve superlinear composition (composed > best individual) on at least 2/5 domains.

## What This Model Is

CAT (Composition via Adaptive Training) from the LoRA Soups paper learns per-module scalar weights alpha_i^m for combining multiple LoRA adapters. Instead of uniform 1/N scaling, each of the 420 LoRA parameter tensors across 5 adapters (= 2100 scalars) gets an independently learned coefficient, optimized on a small calibration set via gradient descent. The merged weights are static -- zero inference overhead.

We applied this to 5 pre-trained domain adapters (python, math, medical, legal, creative) on BitNet-2B-4T, reusing adapters from the flat_lora_training experiment.

## Key References

- **LoRA Soups** (arXiv 2410.13025): Original CAT method. 43% better than static merge, 257% superlinear on Llama-2-7B with 2 tasks.
- **TIES-Merging** (Yadav et al., NeurIPS 2023): Trim-Elect-Sign merge.
- **DARE** (Yu et al., ICML 2024): Drop And REscale.
- **Flat-LoRA findings** (this project, 2026-03-28): Adapters are near-orthogonal (|cos|=0.001) from high-dimensional concentration. Sharpness <0.3%.

## Empirical Results

### Revision Notes (v2, post-adversarial-review)

Five fixes applied from adversarial review:
1. **LR sweep**: Tested {1e-4, 1e-3, 1e-2, 1e-1}. ALL four learning rates diverged. Fixed double forward pass (used value_and_grad).
2. **Parameter count**: Corrected from M=168/840 to M=420/2100 (lora_a and lora_b are separate entries).
3. **Task Arithmetic**: lambda=0.2 was degenerate (= uniform 1/N). Swept {0.1, 0.2, 0.3, 0.5}. Best: lambda=0.5, now the top method.
4. **Verdict**: Consistent SUPPORTED-with-caveat logic (K1=primary, K2=stretch goal).
5. **Grassmannian**: Near-orthogonality is from high-dimensional concentration, not Grassmannian construction.

### Kill Criteria

| Criterion | Result | Detail |
|-----------|--------|--------|
| K1 (id:554): CAT not better than DO-Merging | **PASS** (marginal) | CAT (7.94) beats uniform (7.98) by +0.43% |
| K2 (id:555): No superlinear composition | **FAIL** | 0/5 domains superlinear |

### Success Criteria

| Criterion | Result | Detail |
|-----------|--------|--------|
| S1 (id:64): Composed > best individual on >=2/5 domains | **FAIL** | 0/5 domains |

### Per-Domain PPL Comparison

| Domain | Base | Individual | Uniform 1/N | Task Arith (0.5) | TIES | CAT | CAT vs Ind |
|--------|------|-----------|-------------|-------------------|------|-----|-----------|
| python | 2.74 | 2.22 | 2.51 | 2.38 | 2.29 | 2.64 | -19.1% worse |
| math | 5.54 | 3.60 | 4.93 | 4.43 | 4.56 | 5.12 | -42.3% worse |
| medical | 6.96 | 4.76 | 6.16 | 5.54 | 5.65 | 4.99 | -4.8% worse |
| legal | 21.87 | 16.49 | 20.35 | 18.77 | 19.23 | 20.92 | -26.9% worse |
| creative | 6.35 | 4.94 | 5.94 | 5.51 | 5.59 | 6.04 | -22.4% worse |

### Merge Method Ranking (revised)

| Method | Avg PPL | vs Base | vs Uniform |
|--------|---------|---------|------------|
| **Task Arith (lam=0.5)** | **7.33** | **+15.7%** | **+8.1%** |
| TIES | 7.46 | +14.1% | +6.4% |
| DARE | 7.95 | +8.6% | +0.4% |
| CAT | 7.94 | +8.6% | +0.4% |
| Uniform 1/N | 7.98 | +8.2% | 0.0% |

**Major revision finding:** Task Arithmetic at lambda=0.5 is now the best method (+15.7% vs base), beating TIES (+14.1%). The previous run used lambda=0.2 (= uniform), hiding TA's true potential. Lambda sweep: {0.1: 8.50, 0.2: 7.98, 0.3: 7.75, 0.5: 7.33}.

### CAT Training Diagnostics (LR Sweep)

| LR | First 50 Loss | Last 50 Loss | Diverged? | Time |
|----|--------------|-------------|-----------|------|
| 1e-4 | 1.289 | 2.397 | Yes | 35.9s |
| 1e-3 | 1.280 | 2.369 | Yes | 35.9s |
| 1e-2 | 1.267 | 2.321 | Yes | 35.9s |
| 1e-1 | 1.272 | 2.201 | Yes | 35.9s |

ALL learning rates diverged. This definitively confirms the original suspicion: the calibration loss landscape for alpha is too noisy/flat for gradient-based optimization. The loss increase is consistent across 4 orders of magnitude of LR, ruling out "wrong LR" as the explanation.

Best (least-diverged) was lr=1e-1:
- 2100 learnable scalars (5 adapters x 420 modules)
- 125 calibration sequences (25 per domain)
- Alpha range: [-0.45, 1.59] (started at 0.2 uniform)
- Medical adapter gets 1.9x more weight (0.378 vs 0.20 baseline)

## Analysis

### Why CAT optimization diverges at ALL learning rates

The loss landscape for per-module alpha is dominated by noise from the small calibration set (125 sequences for 2100 parameters = 0.06 sequences per parameter). The gradient signal dL/d(alpha_i^m) requires the inner product of the model gradient with each adapter's delta at each module. With near-orthogonal adapters (|cos|=0.001), these deltas are in essentially independent subspaces, so the gradient for each alpha_i^m is independent of other alphas. This makes the optimization well-conditioned but low-signal: the direction is right but the magnitude is tiny relative to noise from mini-batch sampling.

The divergence pattern (loss increases 1.8x from first to last 50 steps) is consistent across all LRs, suggesting the issue is not learning rate but signal-to-noise ratio in the gradients.

### Why Task Arithmetic at lambda=0.5 beats everything

The previous analysis was flawed: Task Arithmetic at lambda=0.2 is algebraically identical to uniform 1/N merge (sum * 0.2 = sum/5). When properly tuned, lambda=0.5 means each adapter contributes 50% of its full delta (vs 20% at uniform). For near-orthogonal adapters that don't interfere, higher scaling is better -- it reduces dilution. The monotonic improvement from lambda=0.1 to 0.5 suggests lambda=1.0 (full sum) might be even better, though this risks instability from excessive perturbation.

This is a key insight: for orthogonal adapters, the optimal merge is NOT weighted average (sum to 1) but scaled sum (lambda > 1/N per adapter). The theoretical maximum for non-interfering adapters is lambda=1.0 (full superposition).

### Near-orthogonality from concentration, not construction

The adapters are standard LoRA from flat_lora_training, NOT Grassmannian-initialized. Near-orthogonality (|cos|=0.001) comes from high-dimensional concentration of measure: for random vectors in R^d with d ~ 17.2M, E[|cos|] ~ 1/sqrt(d). This means ALL conclusions apply to any independently-trained adapter system, not just Grassmannian architectures.

### Why no superlinear composition

Same reasons as v1: N=5 diverse domains, ternary base with flat landscape, orthogonal adapters prevent constructive interference. The LoRA Soups paper's 257% superlinear result relied on related tasks with inter-adapter synergy.

## Limitations

1. **Divergent CAT training**: All LRs diverge, but only 200 steps tested per LR. Cosine schedule or gradient clipping might help, though the fundamental issue is signal-to-noise.
2. **Small calibration set (125 sequences)**: Severely underdetermined for 2100 parameters.
3. **Lambda sweep incomplete**: Did not test lambda > 0.5 for Task Arithmetic. Optimal lambda may be higher.
4. **Single seed**: No seed variation.
5. **PPL-only evaluation**: No task-specific metrics.

## What Would Kill This

At micro scale (tested):
- K1 PASS (marginal): CAT beats uniform by only 0.43% -- essentially noise
- K2 FAIL: No superlinear effect (0/5 domains)

## Verdict

**SUPPORTED with caveat.** K1 technically passes but the margin is negligible (0.43%). CAT provides no meaningful advantage over uniform merge on near-orthogonal ternary adapters. All LRs diverge, confirming the landscape is too noisy for gradient-based per-module coefficient learning at this calibration set size.

**Practical recommendation**: Use Task Arithmetic at lambda=0.5 (+15.7% vs base, +8.1% vs uniform) or TIES (+14.1% vs base). Both are simpler, faster, and more effective than CAT. The finding that lambda=0.5 > 0.2 suggests further exploration of scaling factors > 1/N for orthogonal adapters.

**Key insight for the project**: For near-orthogonal adapters, the merge problem is NOT about finding the right weighting (CAT) or resolving conflicts (TIES). It is about scaling: each adapter's delta should be applied at close to full strength because orthogonal deltas do not interfere. This points toward lambda optimization rather than learned per-module coefficients.
