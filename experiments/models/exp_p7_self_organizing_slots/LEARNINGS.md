# LEARNINGS: exp_p7_self_organizing_slots

## Core Finding
Self-organizing adapter positions in null space is structurally impossible. null(W_v) ⊥ range(W_v) by construction, so no function of null-space coordinates can carry domain information.

## Why
Domain adaptation ΔW = BA operates entirely in range(W_v). Any signal derived from null-space positions has zero mutual information with domain features: I(f(A_i); domain) = 0. Three independent P7 findings (#495, #498, #500) confirm this from routing accuracy, clustering structure, and quality prediction angles.

## P7 Line Closure
Null space serves exactly ONE purpose: interference prevention via orthogonal isolation (Grassmannian QR packing). It is not an information source for routing, quality, or domain detection.

## Implications for Next Experiment
All routing and quality signals must be sourced from range(W_v) or external features (TF-IDF, embeddings, B-matrix projections). The closed-form Grassmannian QR initialization is already optimal for interference prevention — no learning needed there. Next work should focus on range-space routing or composition scaling (P1 priorities: Fisher-Rao merging, ridge regression routing).
