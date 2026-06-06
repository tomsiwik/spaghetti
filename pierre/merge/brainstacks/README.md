# Brainstacks (MLX)

MLX implementation of [Brainstacks: Cross-Domain Cognitive Capabilities via Frozen MoE-LoRA Stacks for Continual LLM Learning](https://arxiv.org/abs/2604.01152) (Abu Ayyash, 2026).

## What This Implements

The five core components of Brainstacks:
1. **MoELoRADelta** (§3.1) — MoE-LoRA with N=4 experts, top-2 noisy routing, rank-16, rsLoRA scaling
2. **StackedMoELoRALayer** (§3.2) — Additive composition of frozen + active adapter stacks
3. **Residual Boosting** (§3.3) — Inner loop training multiple sequential stacks per domain
4. **Null-Space Projection** (§3.5) — SVD-based orthogonal subspace isolation for zero forgetting
5. **Meta-Router** (§3.6) — Outcome-based sigmoid gating for cross-domain composition

## File Structure

```
brainstacks/
├── configs/base.yaml        # All hyperparameters with paper citations
├── src/
│   ├── model.py             # MoELoRADelta, StackedMoELoRALayer, NullSpaceProjector, MetaRouter
│   ├── loss.py              # Task + aux loss, router BCE + confidence margin
│   └── train.py             # Algorithm 1 (inner loop) + Algorithm 2 (outer loop)
├── REPRODUCTION_NOTES.md    # Unspecified choices, scope decisions
└── README.md
```

## Key Finding

> "Domain stacks encode transferable cognitive primitives — instruction-following clarity, numerical reasoning, procedural logic, chain-of-thought structure — rather than domain-specific knowledge, with medical prompts optimally routing to chat+math stacks in 97% of cases despite zero medical data in those stacks." (§5.1)

## Citation

```bibtex
@article{abuayyash2026brainstacks,
  title={Brainstacks: Cross-Domain Cognitive Capabilities via Frozen MoE-LoRA Stacks for Continual LLM Learning},
  author={Abu Ayyash, Mohammad R.},
  journal={arXiv preprint arXiv:2604.01152},
  year={2026}
}
```
