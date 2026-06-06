# MoE-Freeze: Research Digest

## What This Model Is

MoE-Freeze (`moe_freeze`) is a Mixture-of-Experts language model with an **expert lifecycle
mechanism** designed for continual learning. It extends the base `MoEGPT` model by adding a
`on_domain_switch()` hook that fires whenever the training distribution changes. On each
switch, the model:

1. **Freezes** the most specialized expert (highest weight norm) in each layer, committing it
   as a stable memory of the past domain.
2. **Recycles** the least specialized unfrozen expert (lowest weight norm), reinitializing it
   from scratch to provide capacity for the new domain.

The model is a minimal, interpretable instantiation of lifecycle-based continual learning:
no replay buffer, no knowledge distillation, no task labels beyond the switch signal itself.

---

## Lineage in the Arena

```
gpt  ->  moe  ->  moe_freeze
```

- `gpt`: standard decoder-only transformer with MLP feed-forward blocks.
- `moe`: replaces each MLP with a `MoELayer` (N experts, top-k routing, load-balance loss).
- `moe_freeze`: subclasses `MoEGPT`, overrides `on_domain_switch()` with freeze-recycle logic.

No architectural changes are made to the forward pass. The lifecycle acts entirely through
parameter management: which expert parameters are trainable and which are held fixed.

---

## Key References

**Expert lifecycle and task-specific experts**

- Aljundi et al., "Expert Gate: Lifelong Learning with a Network of Experts" (CVPR 2017).
  Proposes training one expert per task, gated by an autoencoder-based task detector. The
  present model approximates this idea within a single MoE layer rather than a growing network.

- Rusu et al., "Progressive Neural Networks" (2016). Column-per-task architecture; lateral
  connections allow reuse without overwriting. MoE-Freeze is a compressed, fixed-capacity
  analogue: experts play the role of columns, freezing plays the role of lateral isolation.

**Selective weight freezing**

- Mallya & Lazebnik, "PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning"
  (CVPR 2018). Prune-then-freeze approach to packing multiple tasks into one network. MoE-Freeze
  uses norm-based selection rather than magnitude pruning, and operates at the expert level
  rather than the individual weight level.

**Continual learning surveys**

- Kirkpatrick et al., "Overcoming Catastrophic Forgetting in Neural Networks" (PNAS 2017).
  EWC penalizes changes to important weights via Fisher information. MoE-Freeze instead
  prevents changes by hard freezing, avoiding the need to compute or maintain importance scores.

- De Lange et al., "A Continual Learning Survey" (TPAMI 2022). Taxonomy covers regularization,
  replay, and architectural methods. MoE-Freeze is an architectural method (parameter isolation).

---

## The Homeostatic Cleanup Metaphor

The lifecycle is best understood as **homeostatic cleanup**, not as an optimizer. It does not
improve a model's ability to learn any single domain. Instead it:

- Prevents the overwriting of already-learned structure (stability).
- Clears out low-quality expert slots rather than leaving them as dead weight (plasticity reset).

Measured effect on CIFAR-100 with ResNet experts: forgetting reduced from 7.3% (static MoE)
to 6.8% (lifecycle MoE), with final accuracy essentially unchanged (28.2% vs 29.1%). The
lifecycle redistributes capacity more defensively; it does not expand total capacity.

This is consistent with the biological metaphor: synaptic homeostasis trims unused connections
and consolidates strong ones, preventing runaway potentiation, without adding new information.

---

## Empirical Results Context

From the CIFAR-100 benchmark (ResNet experts, 10 tasks x 10 classes, ~712K params/expert):

| Method           | Final Accuracy (FA) | Forgetting |
|------------------|---------------------|------------|
| Static MoE       | 28.2%               | 7.3%       |
| Lifecycle MoE    | 29.1%               | 6.8%       |
| Single network   | ~4-5%               | ~95%       |

Key observations:

- The lifecycle provides a modest but consistent reduction in forgetting (-0.5 pp).
- Both MoE variants vastly outperform single-network fine-tuning, confirming expert isolation
  as the dominant source of continual learning benefit.
- The lifecycle fires at all scales tested: MLP (58 params), CNN (64K), ResNet (712K).

From the ViT-B/16 benchmark (frozen backbone, linear head experts, 10 tasks, 76.9K params/expert):

- With N_experts = 10 (one per task): lifecycle fires no events, matches static 78.2% / 0% forgetting.
- With N_experts = 5 (constrained): lifecycle preserves T0 at 77% (vs 20% for static) but T5
  drops from 78% to 20%. Net forgetting is not reduced; capacity is redistributed, not increased.

---

## The Preservation-Coverage Trade-off

The central limitation of expert-freezing strategies under a fixed budget:

> When N_experts < N_tasks, freezing an expert for one task steals capacity from future tasks.
> Preservation of early tasks comes at the direct cost of coverage of later tasks.

Formally: with `N` experts and `M` tasks, the system has `N - min(N, M)` experts remaining
for the last task. If `M > N`, the last `M - N` tasks share only the unfrozen, recycled experts.

This is not a flaw in the implementation but a fundamental constraint of fixed-capacity
parameter isolation. It is analogous to the limited number of progressive network columns
before memory cost becomes prohibitive.

**Implication for design**: lifecycle mechanisms are most valuable when there is slack —
when the number of available experts exceeds the number of domains encountered. In the
common case where this does not hold, architectural expansion (growing the expert pool) or
replay-based methods are needed to break the trade-off.

---

## When Lifecycle Helps vs. Hurts

| Regime                       | Effect                                              |
|------------------------------|-----------------------------------------------------|
| N_experts >= N_tasks (slack) | Each domain claims a frozen expert; forgetting reduced |
| N_experts = N_tasks          | Marginal benefit; last domain has no frozen protection |
| N_experts < N_tasks          | Zero-sum or negative; early-task preservation steals late-task capacity |
| Multiple generations (MNIST) | Strong benefit; experts cycle through many tasks, lifecycle fires repeatedly |

The MNIST benchmark (6 tasks, N_experts >= 6 per layer) showed: lifecycle E (round-robin +
lifecycle) beats static C by 12.8% accuracy (95.3% vs 82.5%), with redundancy reduced from
0.46 to 0.26 and knowledge precision up 358% over generations.

The ViT/CIFAR-100 constrained benchmark (N=5, M=10) confirmed the zero-sum regime: just-in-time
freeze preserved T0 but degraded T5 by equal magnitude.

---

## Implementation Notes

- `on_domain_switch(domain)` is called by the training loop at each domain boundary. The
  `domain` string is received but not used; the metric is computed from parameter state alone.
- The freeze guard `if (li, best) not in self._frozen_experts` prevents double-freezing an
  already-frozen expert, which would be a no-op but is explicit for correctness.
- The recycle guard `worst != best` prevents recycling an expert in the same step it was
  frozen, which could lose knowledge before it is ever used.
- `mx.eval()` after `ExpertMLP()` construction forces immediate materialization of the
  recycled expert's parameters in MLX's lazy evaluation model.
