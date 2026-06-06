# LEARNINGS: Brainstacks Null-Space SVD Isolation on Ternary Adapters

## Finding: Null-Space Projection Works on Ternary Adapters with Scale-Dependent Forgetting

**Status:** SUPPORTED (K687 PASS, K688 FAIL, K689 PASS)

### What We Learned

1. **Subspace separation is excellent and matches theory.** Mean pairwise cosine of principal directions = 0.026, matching the theoretical prediction K/d = 64/2560 = 0.025. Five domains with K=64 each occupy well-separated 2.5% subspaces in hidden space. This validates the core Brainstacks assumption for ternary adapters.

2. **Ternary noise causes scale-dependent forgetting.** High-scale adapters (medical/code/math at scale=20.0) show negligible forgetting (< 0.003). Low-scale adapters (legal at scale=4.0) show 2.5% forgetting. The quantization noise is proportional to α = mean(|B|), which is constant across scales — so at lower adapter scales, the noise-to-signal ratio increases. This is the MATH.md prediction (leakage ≤ α√K) manifesting as scale-dependent vulnerability.

3. **Gradient norm preservation degrades linearly at ~1.2% per prior domain.** Finance (4 prior domains) preserves 95.2%. This is slightly better than the theoretical prediction of 1 - N_prior × K/d = 1 - 4 × 0.025 = 90%, likely because the domains' actual subspace overlap reduces the total excluded volume.

4. **Max cosine ~0.977 reveals shared instruction-following directions.** While mean cosine is low (0.026), the max cosine between any pair is ~0.977. This indicates 1-2 directions shared across ALL domain pairs — likely the "### Instruction:" / "### Response:" template format. The Brainstacks null-space projection will incorrectly project out these shared directions. A fix: compute projectors from domain-SPECIFIC deltas (subtract the shared component first).

### Limitations

- **Only 50 validation samples per domain** (val files have 50 entries). With K=64 > n_samples=50, the energy ratio is trivially 100%. A proper test needs ≥200 samples to validate whether K=64 captures sufficient energy in a higher-rank delta matrix.
- **Val loss forgetting ≠ leakage norm.** K688 measures val loss delta but MATH.md predicts leakage norm. These are related but not directly comparable. Future work should measure both.
- **No training-time projection tested.** This experiment applies null-space projection post-hoc to pre-trained adapters. The Brainstacks protocol constrains adapter TRAINING within the null space, which would produce different (likely better) results.

### Implications for Architecture

1. **Null-space projection is viable for our deployment track** — the 3 high-scale domains (medical, code, math) that matter most show near-zero forgetting. Legal and finance need scale-matching fixes.

2. **Scaling to 25 domains:** With K=64 each, 25 domains would use 25×64/2560 = 62.5% of hidden space. Gradient preservation would drop to ~70%. Either reduce K per domain or use a hierarchical projection scheme.

3. **Combine with routing for defense-in-depth:** Null-space projection provides structural guarantees (eliminates direction interference for high-scale adapters). Entropy-gated routing provides dynamic guarantees (skip irrelevant experts). Together, they address both the 20% direction interference AND the 80% capacity interference identified in Finding #270.

### What Would Make This Conclusive

- Train adapters WITH null-space projection constraints (Brainstacks §3.5 protocol, not post-hoc)
- Use ≥400 validation samples for meaningful energy analysis
- Measure behavioral forgetting (task accuracy), not just val loss
- Test at K=32 vs K=64 vs K=128 to find the optimal tradeoff
