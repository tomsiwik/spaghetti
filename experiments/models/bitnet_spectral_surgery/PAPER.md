# Spectral Surgery on BitNet-2B LoRA Adapters: Research Digest

## Hypothesis

Spectral Surgery (arXiv 2603.03995) -- SVD decomposition of trained LoRA followed by gradient-guided singular value reweighting -- can improve adapter quality as a training-free post-hoc Evolve quality gate.

**Falsifiable claim**: Spectral-refined adapters will show lower PPL and higher KR-Test scores than unrefined adapters on at least 3/5 domains, with surgery completing in under 5 minutes per adapter.

## What This Experiment Is

We applied the Spectral Surgery algorithm to 5 existing BitNet-2B-4T LoRA adapters (python, math, medical, legal, creative) trained in the bitnet_2b_real_composition experiment. The algorithm decomposes each LoRA layer's effective update via SVD, estimates per-singular-value sensitivity using gradients on a 128-example calibration set, then reweights the singular values to suppress detrimental directions and amplify beneficial ones.

The goal was to test whether this training-free refinement could serve as a quality gate in the Evolve pipeline -- improving adapter quality without retraining.

## Key References

- **Spectral Surgery** (arXiv 2603.03995): The source algorithm. Reports +4.4pp on CommonsenseQA for Llama-3.1-8B/Qwen3-8B LoRA adapters.
- **LoRA** (arXiv 2106.09685): Original low-rank adaptation method.
- **retrain_evolve LEARNINGS**: Our prior experiment showing PPL and KR-Test diverge for short-trained LoRA adapters.

## Empirical Results

### VERDICT: KILLED (K1=FAIL, K2=FAIL, K3=FAIL)

### K1: PPL Improvement (FAIL - 1/5 domains improved)

| Domain | Baseline PPL | Refined PPL | Delta |
|--------|-------------|-------------|-------|
| python | 2.22 | 2.21 | +0.4% |
| math | 3.60 | 3.61 | -0.1% |
| medical | 4.74 | 4.76 | -0.4% |
| legal | 16.53 | 16.59 | -0.4% |
| creative | 4.92 | 4.95 | -0.6% |

Only python showed marginal improvement (+0.4%). Four domains got slightly worse. The changes are within noise floor (all <1% magnitude).

### K2: KR-Test Improvement (FAIL - 0/5 domains improved)

| Domain | Baseline KR | Refined KR | Delta |
|--------|------------|------------|-------|
| python | 0.914 | 0.914 | +0.000 |
| math | 0.953 | 0.953 | +0.000 |
| medical | 0.971 | 0.971 | +0.000 |
| legal | 0.895 | 0.895 | +0.000 |
| creative | 0.980 | 0.980 | +0.000 |

Zero KR-Test movement across all 5 domains. The surgery has no effect on factual discrimination.

### K3: Speed (FAIL - max 467.5s, threshold 300s)

| Domain | Surgery Time (s) |
|--------|-----------------|
| python | 451.1 |
| math | 440.5 |
| medical | 420.5 |
| legal | 453.8 |
| creative | 467.5 |

All adapters exceed the 5-minute threshold. The bottleneck is gradient computation (128 forward+backward passes through the 2.4B model), not SVD/reweighting.

### Composition Impact

| Metric | Before Surgery | After Surgery | Delta |
|--------|---------------|---------------|-------|
| Avg composed PPL | 7.96 | 8.00 | -0.4% |
| Mean |cos| | 0.0010 | 0.0032 | +3.2x |

Surgery slightly worsened composition and increased inter-adapter cosine similarity by 3.2x (from 0.001 to 0.0032 -- still well below the 0.05 threshold, but the wrong direction).

### Base Model KR-Test (Context)

| Domain | Base KR | Adapter KR | Delta |
|--------|---------|------------|-------|
| python | 0.943 | 0.914 | -0.029 |
| math | 0.953 | 0.953 | +0.000 |
| medical | 0.971 | 0.971 | +0.000 |
| legal | 0.868 | 0.895 | +0.026 |
| creative | 0.980 | 0.980 | +0.000 |

Notable: the python adapter actually REDUCES KR-Test vs base (learns domain style but hurts discrimination), while legal is the only adapter that improves KR-Test. This is consistent with the retrain_evolve finding that LoRA learns style before facts.

## Why Surgery Failed: Three Root Causes

### 1. Already-Efficient Spectrum

The original paper targets adapters with "inefficient spectra" -- where beneficial effects concentrate in a small subset of singular directions while the rest accumulate noise. Our adapters were trained for only 200 iterations (short training). Short-trained LoRA adapters have spectra that are already concentrated: the training hasn't run long enough to build up noise in unused directions. There is nothing to "surgically remove."

This is the opposite of the paper's setting (long-trained adapters on standard Llama/Qwen with full fine-tuning datasets).

### 2. Nuclear-Norm Constraint Makes Surgery Zero-Sum

The L1 normalization constraint (||sigma'||_1 = ||sigma||_1) ensures that any suppression of one singular value must be compensated by amplification of another. When the spectrum is already efficient, this constraint prevents net improvement -- it merely shuffles energy between directions without creating new beneficial signal.

### 3. Gradient Noise at Micro Scale

With 128 calibration examples on a 2.4B parameter model, gradient estimates are noisy. The sensitivity scores g_k may not reliably distinguish truly detrimental from beneficial directions. The original paper uses 128 examples on Llama-3.1-8B / Qwen3-8B, but those models were fine-tuned for many more steps with larger learning rates, producing clearer gradient signal.

## Limitations

1. **Scale mismatch**: The original paper tested on Llama-3.1-8B and Qwen3-8B with standard (FP16) LoRA. Our ternary base model may produce different gradient landscapes.

2. **Training duration**: Our adapters (200 iterations, 1 epoch) may be too undertrained for spectral surgery to help. The paper's adapters were trained with standard recipes (multiple epochs, learning rate scheduling).

3. **Hyperparameters**: We used eta_sup=1.0, eta_amp=0.5 (reasonable defaults based on the paper's description). The optimal values might differ for ternary models.

4. **KR-Test limitations**: Our cross-item KR-Test at n=35-50 pairs has limited statistical power. However, seeing exactly 0.000 delta across all 5 domains strongly suggests no effect, not just insufficient power.

5. **Calibration data**: We used the same domain training data as calibration. Cross-domain calibration might produce different results.

## What Would Kill This (Already Killed)

- K1: PPL not better on majority of domains -- **CONFIRMED (1/5)**
- K2: KR-Test not better on majority of domains -- **CONFIRMED (0/5)**
- K3: Surgery takes >5 min per adapter -- **CONFIRMED (420-468s)**

## Implications for the Project

1. **Spectral Surgery is not viable as an Evolve quality gate** for our short-trained ternary LoRA adapters. The mechanism addresses a problem (inefficient spectra) that our adapters do not have.

2. **The retrain-from-scratch approach remains the best Evolve primitive**: since the adapters already have efficient spectra, the only way to improve them is to train on better/more data (as confirmed by exp_bitnet_retrain_evolve).

3. **Quality gate should focus on evaluation, not refinement**: PPL + KR-Test non-regression + cosine < 0.05 (the revised gate from retrain_evolve LEARNINGS) is sufficient. Post-hoc refinement adds cost without benefit.

4. **Interesting negative on composition**: surgery increases inter-adapter cosine by 3.2x. The SVD re-factorization rotates adapter subspaces toward shared directions, which is harmful for composition. If spectral surgery were ever used, it would need a composition-aware constraint.
