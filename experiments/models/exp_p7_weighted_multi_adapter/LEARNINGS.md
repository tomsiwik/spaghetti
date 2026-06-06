# LEARNINGS: P7.B1 — Weighted Multi-Adapter Composition

## Core Finding
Averaging null-space adapters (even with near-uniform TF-IDF weights) beats exclusive
argmax routing by 32.7% on mixed-domain queries and 18.5% on single-domain queries.
Null-space orthogonality preserved at machine precision (9.57e-7 < 1e-4 threshold).

## Why
Null-space closure under convex combination is trivially guaranteed (subspace closure
under linear combination — see MATH.md). The practical benefit comes from ensemble
effects: no single adapter matches the average of all five at memorization scale.
TF-IDF weights collapse to near-uniform (entropy 0.996–1.000), so "weighted routing"
is effectively "average all adapters." LoRAHub (2310.13699) shows the same pattern.

## Key Caveat: Ensemble, Not Routing
Individual adapters are generic regularizers (not domain-specialized) at 8 texts/300
iters. Oracle routing picks wrong domains in ~50% of cases. The benefit is likely
generic ensembling, NOT null-space-specific composition. A non-null-space control is
needed to separate the two effects.

## Implications for Next Experiment
The structural guarantee (safe averaging) is durable. The routing narrative is not yet
supported. Next priority: (1) train domain-specialized adapters at larger scale to get
peaked routing weights, OR (2) run a non-null-space averaging control to isolate whether
null-space structure contributes beyond generic ensembling. Finding #495 guidance stands:
route in range(W_v), adapt in null(W_v) — but routing quality requires domain-specialized
adapters to matter.
