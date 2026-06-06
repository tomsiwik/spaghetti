# E22-full: Adapter Poisoning Robustness — Full Run

## Verdict: KILLED

Both KCs fail. Clean-only composition destroys model at 35 layers, making poisoning analysis meaningless.

## Setup
- Model: gemma-4-e4b-it-4bit (Gemma 4 E4B)
- 35 layers (42 - 7 global attention), 5 clean adapters + 1 poison, 100 QA
- Poison multipliers: [1, 3, 5, 10, 15, 20]
- v_proj, rank=6, lora_scale=6

## Prediction vs Measurement

| Metric | Predicted | Measured |
|--------|-----------|----------|
| Base accuracy | ~85% | 84% |
| Grass clean-only | ~85% | **2%** |
| Random clean-only | ~85% | **1%** |
| Grass worst drop | 15-25pp | **82pp** |
| Random worst drop | 60-80pp | 78-81pp |
| Protection margin | 30-55pp | **-3 to +2pp** |

## Kill Criteria Results

| KC | Threshold | Measured | Result |
|----|-----------|----------|--------|
| K2059 | < 30pp grass drop | 82pp | **FAIL** |
| K2060 | > 2pp margin | 2.0pp (≤, not >) | **FAIL** |

## Sweep Detail

| Mult | Grass acc | Rand acc | Grass drop | Rand drop | Margin |
|------|-----------|----------|------------|-----------|--------|
| 1× | 2% | 3% | 82pp | 81pp | -1pp |
| 3× | 3% | 3% | 81pp | 81pp | 0pp |
| 5× | 2% | 5% | 82pp | 79pp | -3pp |
| 10× | 4% | 6% | 80pp | 78pp | -2pp |
| 15× | 4% | 5% | 80pp | 79pp | -1pp |
| 20× | 7% | 5% | 77pp | 79pp | +2pp |

## Root Cause Analysis

The failure is structural, not a bug:

1. **Clean-only composition destroys the model at 35 layers.** 5 synthetic adapters × 35 layers = 175 rank-6 ΔW perturbations. Each ΔW = B_i @ A_i where B_i ∝ W @ A_i^T. At 35 layers, total perturbation magnitude overwhelms the base model. Accuracy drops from 84% → 2%.

2. **Smoke's 3-layer result was sampling artifact.** At 3 layers (3/42 = 7% of model), perturbation is small enough to be tolerable. At 35 layers (83% of model), the same construction is catastrophic. F#821's 55pp protection margin was entirely an artifact of insufficient scale.

3. **Grassmannian provides zero protection at scale.** All sweep results show margins of -3 to +2pp — pure noise. When both conditions are already at floor (~2-7% accuracy), there's no signal to differentiate. The input-space feature isolation mechanism (E22 smoke's explanation) is real per-layer but irrelevant when the total perturbation destroys model coherence.

4. **Same pattern as E14-full.** E14-full showed Grassmannian decorrelation benefit = 0.0018 (noise) at scale. E22-full shows protection margin = noise at scale. Both phenomena that appeared in smoke (3 layers) vanish at full scale.

## Findings

**F#821 falsified at scale**: Grassmannian poisoning protection (55pp margin) was a 3-layer artifact. At 35 layers, margin is noise-level (-3 to +2pp).

**New finding**: Synthetic adapter composition (B ∝ W @ A^T) is inherently destructive at high layer count. Any experiment using this construction at >10 layers produces floor-level accuracy, making all differential measurements meaningless.

## Implications

1. E22 smoke's PROVISIONAL cannot be upgraded — the smoke result does not replicate at scale.
2. Grassmannian has no demonstrated safety benefit at scale (E14-full killed activation decorrelation, E22-full kills poisoning protection).
3. Future composition experiments must either (a) use trained adapters or (b) limit to ≤10 layers if using synthetic construction.
