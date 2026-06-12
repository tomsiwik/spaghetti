# bet: jury-decode — out-reason frontier by spending test-time compute it can't

**Thesis.** Small + verifier-guided search beats much larger models on **checkable** tasks
(rStar-Math 7B+MCTS > o1-preview, arXiv:2501.04519; 1B > 405B under compute-optimal test-time
allocation, arXiv:2502.06703), bounded by verifier quality (arXiv:2411.17501). Pierre's unique
asset: a bank of domain adapters hot-swappable in <1ms — a **decorrelated jury** of verifiers,
where a single frontier verifier has correlated blind spots.

**Why this can be a breakthrough:** it's the one axis where an M5 Pro can spend something the
frontier cannot — unlimited cheap verifier diversity per token budget. Composition stops being
a merge problem and becomes a *search* asset.

**Pierre branch:** `bet/jury-decode` · **Baseline to beat:** solo adapter greedy decode (math 0.85 on the current harness slice).

## Ladder
- **R1 — verifier gain** (days). Best-of-N (N=8) GSM8K with ONE adapter-as-verifier scoring
  candidates vs self-consistency vs greedy. **Gate:** verifier-BoN > self-consistency by ≥3pp at
  equal token budget. **Kill:** verifier ranks no better than random (AUC ≤0.55 on correct-vs-wrong).
- **R2 — decorrelation** (days). Jury of 3 adapters (math/python/medical heads) vs the single best
  verifier, same budget. **Gate:** jury > best single verifier by ≥2pp, and error-overlap between
  jury members measurably below single-verifier self-overlap (the mechanism check).
- **R3 — search** (1–2 weeks). Step-level search (beam/MCTS-lite) with the jury as the process
  reward. **Gate:** beats frontier-model pass@1 (published Claude/GPT numbers) on ≥1 checkable slice
  at ≤10× token budget.
- **R4 — SHIP:** decode-time jury into pierre's server on `bet/jury-decode`; league-score
  (the league suite must report tokens/answer so search pays its rent transparently).

## Honest risk
Jury decorrelation may not survive a shared frozen base (correlated failure modes). R2 measures
error-overlap directly instead of assuming it.
