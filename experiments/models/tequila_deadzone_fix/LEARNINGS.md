# Learnings: exp_tequila_deadzone_fix

## Core Finding

Tequila-style dynamic adaptive biases do NOT reduce the zero fraction in ternary weights at micro scale (32% unchanged), but provide a free -6.7% PPL improvement via bias fusion. The 32% zero fraction is a statistical equilibrium of STE ternary quantization with Gaussian-like weight distributions, determined by erf(1/(2√2)) ≈ 0.31.

## Why This Happened (Literature-Grounded)

The zero fraction stability is explained by three interacting mechanisms identified in the literature:

1. **Alpha-coupling equilibrium** (identified by our adversarial review): The deadzone threshold α/2 = mean(|W|)/2 is a function of the weight distribution itself. If reactivation gradients cause some dead weights to grow, α increases proportionally, raising the threshold and trapping other weights. For a roughly Gaussian distribution, the CDF below mean(|W|)/2 yields ~31% — exactly what we observe. This is a self-reinforcing equilibrium that no bias-based mechanism can break.

2. **Locally flat objective regions** (Tequila paper; Sparse-BitNet): Dead weights contribute nothing to the forward pass through the main computation path. Even with the reactivation bias providing gradient signal, STE passes the gradient unchanged through the quantization, but the quantized weight remains zero. The shadow weight updates don't change the discrete forward computation until they cross the threshold — and the threshold moves with the distribution.

3. **Stochastic pull-back** (STE literature): Weights that briefly escape the deadzone via noise lack consistent optimization signal to stay outside it. Without a mechanism that preferentially pushes weights *toward* the threshold boundary, escape is a random walk that the threshold-coupling neutralizes.

**Why Tequila works at full scale but not at micro scale**: The original Tequila paper (arxiv 2509.23809, Figure 8) reports visible weight distribution improvement at 1B-3B scale with 10B tokens of training. At our micro scale (64M params, 2M tokens, 2000 steps), the training budget is insufficient for the reactivation gradient to accumulate enough signal to overcome the alpha-coupling equilibrium. The bias compensation (PPL improvement) is the fast effect; actual deadzone escape is the slow effect requiring orders of magnitude more training.

## Confirming Evidence

- **Tequila (arxiv 2509.23809)**: Confirms deadzone trapping as a fundamental STE failure mode. At full scale (1B-3B), reports weight distribution shift toward BF16-like profiles (Figure 8). Our micro-scale result is consistent: the mechanism works but requires sufficient training budget.
- **Sparse-BitNet (arxiv 2603.05168)**: Reports 42% natural sparsity in ternary weights, consistent with our 32% observation (different threshold computation yields different equilibrium point). Confirms zeros are structural, not a training artifact.
- **Ternary Weight Networks (TWN)**: The original TWN paper derives optimal threshold Δ* ≈ 0.75·mean(|W|) assuming Gaussian weights. Our observation of ~31% zeros at threshold mean(|W|)/2 is consistent with the Gaussian CDF prediction.
- **Sherry (Tencent/AngelSlim)**: Uses adaptive error compensation (λ_t·X·W residual) — mathematically similar to Tequila's bias but applied during training, not post-hoc. Confirms that continuous residual signals improve PPL without necessarily changing sparsity patterns.

## Contradicting Evidence

- **Tequila at full scale**: The original paper shows actual weight distribution improvement (not just PPL), suggesting that with sufficient training (10B tokens vs our 2M), the reactivation gradient CAN overcome the alpha-coupling equilibrium. **Our negative K1 result may be scale-dependent**, not mechanism-dependent.
- **Trained Ternary Quantization (TTQ)**: Uses two independent asymmetric scaling coefficients (W_p, W_n) with dynamic thresholds. Because the maximum absolute values change during training, the threshold shifts, allowing natural weight migration across deadzone boundaries. TTQ demonstrates that dynamic thresholds can enable escape — contradicting our finding that escape is impossible, but through a different mechanism (threshold modification, not bias compensation).

**Discrepancy explanation**: Our experiment tested bias compensation at fixed-threshold quantization. Methods that successfully reduce zeros modify the quantization function itself (dynamic thresholds, asymmetric scaling, or sufficient training to shift the weight distribution). The contradiction is in the mechanism class, not the physics: bias-based approaches compensate; threshold-modifying approaches actually reduce zeros.

## Alternative Approaches (What We Could Try Instead)

1. **Asymmetric quantization (TTQ-style)**: Learn separate positive/negative scaling factors per layer. The asymmetry shifts the effective deadzone boundary dynamically during training. This directly attacks the alpha-coupling equilibrium by making the threshold trainable rather than a fixed function of mean(|W|). Reference: TTQ (Zhu et al., 2016).

2. **Learned Step Size Quantization (LSQ)**: Make α a trainable parameter updated via backpropagation. While the literature notes this alone doesn't fully solve deadzone trapping (the STE noise problem remains), it breaks the alpha-coupling by decoupling the threshold from the weight distribution. Reference: LSQ (Esser et al., 2020).

3. **Zero-point reassignment (SEQ-style)**: Instead of mapping deadzone weights to literal zero, map them to a non-zero value (α·b). This ensures every weight contributes to the forward pass, eliminating the locally-flat-objective problem entirely. Tradeoff: breaks ternary hardware efficiency by reintroducing dense multiplications. May be acceptable on MLX where ternary kernels aren't specialized.

4. **Tequila + longer training**: Simply running Tequila for 100x more tokens (200M+) on our architecture may be sufficient for the reactivation gradient to overcome the alpha-coupling equilibrium. The original paper's success at 10B tokens suggests this is a training budget issue, not a mechanism failure.

5. **Hybrid Tequila + Sherry**: Combine Tequila's continuous reactivation with Sherry's 3:4 structured sparsity. Instead of fighting the zeros, embrace structured sparsity (exactly 75% zeros per block) while using reactivation for the remaining weights. This turns the deadzone from a bug into a feature.

6. **ADMM-based iterative optimization**: Use Alternating Direction Method of Multipliers to jointly optimize scaling factors and thresholds. Borrows from sparse signal processing. Reference: used in pruning literature for joint optimization.

## Implications for Next Experiments

1. **Don't abandon Tequila — increase training budget**: The K1 FAIL is likely a scale artifact. If we care about deadzone reduction, test Tequila at 50M+ tokens before concluding the mechanism doesn't work.

2. **Integrate the bias fusion NOW**: The -6.7% PPL win is free (zero inference cost). Add `TequilaBitLinear` as the default BitLinear recipe for all future experiments. This is a pure win regardless of whether zeros are reduced.

3. **Explore threshold-modifying approaches**: For actual deadzone reduction, the literature points strongly toward dynamic/learnable thresholds (TTQ, LSQ) rather than bias compensation. A micro-experiment comparing TTQ-style asymmetric α against standard BitLinear would test this directly.

4. **The 32% zero equilibrium is a feature, not a bug, for composition**: If zeros are structural and stable, they represent a natural sparsity pattern. For Composable Ternary Experts, consistent zero patterns across adapters could enable structured sparse composition (skip computation for positions that are zero in all adapters).

5. **Alpha-coupling is the core mechanism to break**: Any future deadzone reduction experiment should explicitly target the alpha-coupling equilibrium (α = mean(|W|) creates a self-reinforcing zero fraction). Methods that decouple α from the weight distribution are the right direction.
