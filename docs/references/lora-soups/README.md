# LoRA Soups: Merging LoRAs for Practical Skill Composition Tasks

**Source**: https://arxiv.org/abs/2410.13025

**Authors**: Akshara Prabhakar, Yuanzhi Li, Karthik Narasimhan, Sham Kakade, Eran Malach, Samy Jelassi

**Venue**: COLING 2025 Industry Track

**Code**: https://github.com/aksh555/LoRA-Soups

## Key Insight

Learnable Concatenation (CAT) -- per-layer trainable scalar weights for combining independently trained LoRA adapters -- outperforms data mixing by 12% and static merging by 43% on binary skill composition tasks (e.g., math + coding).

## Relevance to Our Work

Most directly comparable prior art to SOLE. Both compose independently-trained LoRA adapters at inference time. Key differences:

1. LoRA Soups requires optimization of per-layer weights; SOLE uses unit weights (zero cost)
2. LoRA Soups is restricted to binary (k=2) composition; SOLE scales to N>>2
3. LoRA Soups has no orthogonality analysis; SOLE provides structural guarantee
4. LoRA Soups has no evolution mechanism; SOLE supports clone-and-compete

Our positioning: LoRA Soups is a composition technique; SOLE is an architecture. They are complementary -- CAT could be an optional optimization step within SOLE.

## Key Findings from the Paper

- CAT beats data mixing by 12% average and model merging baselines by 43% on GSM-Hard
- Demonstrates "super-linear" improvements from binary skill merging
- First work showing model merging > data mixing for binary skill composition
- Limited to 2-skill composition; authors acknowledge scaling challenges

## Related Papers

- Ostapenko et al., "Towards Modular LLMs" (arXiv:2405.11157) -- library + Arrow routing
- Task-Aware LoRA Composition (arXiv:2602.21222) -- vector DB retrieval routing
