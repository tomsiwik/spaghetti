# bet: simplex-routing — per-query composition search (GATED on dfa-init R2)

**Thesis.** Once adapters share a compatible basis (dfa-init), the static K=7 merge is strictly
dominated by a **per-query simplex search**: for each query, search the convex weights
w ∈ Δᴺ over the bank (a few forward probes on a short prefix) instead of guessing one global mix.
Related: Soup-of-Experts (arXiv:2502.01804), Arrow/PHATGOOSE-style zero-shot routing.

**Why gated:** without the shared basis, per-query weights re-inherit the interference ceiling —
this is the lesson of 248 composition experiments. Do NOT open before dfa-init R2 passes.

**Pierre branch:** `bet/simplex-routing` · **Baseline to beat:** best static merge + per-token top-1 routing (current pierre v3 router).

## Ladder
- **R1 — oracle headroom** (days). On the DFA bank: grid-search w per query offline (oracle).
  **Gate:** oracle-per-query ≥ static-best +5pp — proves the headroom exists before building a router.
  **Kill:** oracle ≤ static +2pp (per-query weights buy nothing in the shared basis → bet dies cheaply).
- **R2 — cheap probe router** (1 week). Predict w from a short-prefix probe (logit features, no
  trained router net). **Gate:** captures ≥50% of the oracle gap at ≤15% latency overhead.
- **R3 — SHIP:** probe router into pierre's serving path on `bet/simplex-routing`; league-score
  vs main and vs jury-decode (the branches now compete head-to-head on the same suite).
