# LEARNINGS: exp_p7_null_projection_routing — KILLED

## Core Finding
Null-space A-matrix projection is structurally incapable of domain routing. Accuracy = 20% (chance), all inputs route to the highest-magnitude adapter (legal). Domain information lives in range(W_v) by construction, making null-space routing mathematically impossible.

## Why It Failed
V = range(W_v) and V⊥ = null(W_v) are orthogonal complements. Domain-discriminative features are in V (the base model uses them for value computation). Routing signal operates entirely in V⊥. Therefore <routing_signal, domain_info> = 0 — no normalization or training can overcome this structural orthogonality. This is distinct from Finding #295 (B-projection overlap): A-projection fails because null(W_v) contains no domain signal at all, not because of overlap.

## Architectural Clarification
Two concerns are mathematically complementary and cannot be unified:
- **Route** in range(W_v) or hidden states (where domain info lives)
- **Adapt** in null(W_v) (where inter-adapter interference is zero)

The Room Model "routing IS the matmul" holds for standard LoRA but fails for null-space LoRA. Future routing must use hidden states or range-space features, not null projections.

## Secondary Confirmation
All 5 null-space adapters converge (loss 0.008–0.029) in <1 min each. Training stability of null-space LoRA confirmed again (cf. Finding #494).

## Next Experiment Direction
Use hidden state activations (full h_t or range(W_v) projection) as the routing signal, while keeping adapter isolation in null(W_v). These are separate modules operating in complementary subspaces.

## References
- Finding #494: Null-space LoRA quality works (98.7%) — adapts correctly, cannot route
- Finding #493: v_proj null_dim=2048, domain info confirmed absent from null space
- Finding #295: B-projection routing fails (overlap); A-projection routing fails (domain-blind)
- arXiv:2212.04089 (Task Arithmetic): task vectors in full param space, not null subspace
