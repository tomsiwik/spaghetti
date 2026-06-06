# E4: Activation Arithmetic Composition — Learnings

## Core Finding
Strategy forcing is counterproductive for reasoning on Gemma 4 E4B. Three independent mechanisms (E1 mean-diff, E6 Hedgehog, E4 ActAdd) all degrade GSM8K accuracy. The failure is in the *concept* of explicit reasoning-strategy injection, not the *method*.

## Why
The model's default reasoning—shaped during pretraining and instruction tuning—is already near-optimal for GSM8K at this scale. Explicit decomposition prompting alone costs -15pp (10% vs 25% base), proving the issue is upstream of any injection mechanism. Strategy-specific information exists in early layers (L3-9, cos 0.19-0.63) but converges to a shared reasoning mode by L24+ (cos >0.87). Overriding this convergence disrupts the model's internal optimization.

## Key Sub-Findings

1. **Contrastive extraction partial fix**: cos 0.76 vs E1's 0.99. Early layers (L3-9) discriminate strategies; late layers don't. Confirms E1 #801 prescription: contrastive baselines cancel format signal.

2. **K2024 tautological-proxy (F#666)**: KC measured "change" not "improvement" — passed vacuously with -5pp degradation. Future KCs for steering must threshold on *signed* delta, not absolute.

3. **Three-experiment convergence**: E1 (extraction), E6 (distillation), E4 (injection) — all fail for reasoning strategies. The strategy-adapter path for reasoning is empirically closed on Gemma 4 E4B.

## Implications for Next Experiments

- **E11**: Contrastive extraction works at early layers (L3-9). Design should use second-order contrastive (strategy_A - strategy_B) and target layers ≤12. But must accept that even perfect extraction may not help if injection is counterproductive.
- **E5/E9**: Self-discovered strategies and composable CoT must let the model choose its own reasoning, not force external strategies. Consider preference-based (DPO/RLHF) over direct injection.
- **E7-v2**: Cannot exist as "validate strategy transfer" when strategy forcing itself is harmful. Needs fundamental redesign: transfer of *surface* behaviors (style, format, persona) not reasoning strategies.
- **Hedgehog domain adapters** (JS/Python/Rust, P=3): Still viable — coding style is surface behavior, not reasoning. E6 showed Hedgehog works for surface behaviors (politeness F#783).
- **General principle**: Locus matters. Input-processing behaviors (style, format) can be steered. Generation-time behaviors (reasoning strategies) resist external forcing because the model's internal strategy selection is already optimized.

## Finding Registry
- **#807**: ActAdd contrastive steering degrades GSM8K by -5pp. Explicit decompose prompt degrades by -15pp. Strategy forcing counterproductive for reasoning on Gemma 4 E4B. Converges E1 #801, E6 #804.
