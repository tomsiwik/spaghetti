# LEARNINGS — T0.4: Q-Only Adapters on K=V Global Layers

**Finding #412 (supported) | 2026-04-09**

## Core Finding

Q-only LoRA adapters on Gemma4 global attention (K=V) provide algebraically guaranteed KV cache sharing: k_proj has no adapter → K is base-only → V=K is base-only → KV cache is adapter-independent for any number of simultaneous users. Q-only quality_ratio=1.24, outperforming Q+K (ratio=1.0 baseline) on retrieval task.

## Why

K1001/K1002 = 0.0 exactly (algebraic, dimension-independent). Q+K LoRA modifies K which modifies V (since V=K), creating conflicting optimization targets and splitting capacity. Q-only concentrates all capacity on query representation — better suited to query-centric domain tasks. KV sharing requires zero implementation coordination; it is structurally enforced.

## Implications for Next Experiment

P1 adapter architecture is fixed: q_proj only, targeting NoPE dims [128:512], algebraically guaranteeing (1) position-invariance (T0.3) and (2) KV cache sharing for N-user multi-tenant serving. T0 foundation is complete — proceed to T1 experiments (Grassmannian at Gemma4 dims, or PLE injection).
