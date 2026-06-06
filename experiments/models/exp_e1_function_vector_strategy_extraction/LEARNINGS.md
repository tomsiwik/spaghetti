# E1 Learnings: Function Vector Strategy Extraction

## Core Finding

Mean-difference activation extraction (Todd et al. 2310.15213) fails for strategy vectors. Cross-strategy cosine similarity = 0.99 — the method captures "system prompt present vs absent" (format signal), not strategy-specific content. Target KC K2018 = 0pp GSM8K delta confirms zero behavioral transfer.

## Why

Problem-solving strategies are broad behavioral modes, not narrow input-output functions. The format signal (instruction-following mode activation) dominates strategy signal by ~100x in activation magnitude. Mean-difference conflates both; format wins.

This is a category error: Function Vectors work for discrete transformations (capitalize, translate) where the function IS the dominant activation change. Strategies are perturbations on top of a shared instruction-following mode.

## Implications for Next Experiment

1. **E11 (Linear Strategy Extraction) inherits this failure** if it uses mean-difference. Must use contrastive extraction (strategy_A - strategy_B) to cancel format signal, or residual subtraction (subtract mean-of-all-strategies first).

2. **E6 (First Strategy Adapter) should not extract from activations at all** — train the adapter directly on strategy-eliciting data. The activation-extraction path is dead for strategies.

3. **E14 (Grassmannian Activation Orthogonality)** is unaffected — it operates in weight space, not activation space. But any activation-space validation step must account for format signal dominance.

4. **General principle (Finding #801)**: When extracting behavioral directions from activations, the extraction method must be contrastive against the closest non-target condition, not against a neutral baseline. Neutral baselines let shared confounds (format, attention pattern, positional encoding shifts) dominate.
