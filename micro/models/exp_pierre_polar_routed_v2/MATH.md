# MATH.md — Per-prompt routed K=2 composition (false-killed re-run)

## Hypothesis

Original `exp_pierre_polar_composition_v2_routed` was a false-kill per
Finding #831 audit. The original tested Gumbel top-2 hidden-state routing
and saw the exact bug-fingerprint collapse (humaneval=20%).

This experiment re-runs **per-prompt routing** with the corrected
infrastructure, using a simpler routing scheme (TF-IDF + Ridge — already
validated at 100% accuracy / 0.07ms in `exp_pierre_per_task_routing_math`)
to pick top-2 adapters per prompt.

> **Does per-prompt K=2 routing (1 strategy + 1 domain selected by prompt
> content) preserve specialization while gaining cross-task flexibility?**

## Method

For each eval prompt:
1. Classify prompt domain via TF-IDF + Ridge (math / code / medical / other).
2. Pick K=2 adapters: `(strategy_full, domain_X)` where X matches classification.
3. Compose K=2 via Fisher-Rao.
4. Install via PoLARLinear (B-only path) and evaluate that single prompt.
5. Repeat for every prompt — different composed B per prompt.

Different from existing experiments:
- `exp_pierre_per_task_routing_math` — only binary (math vs not), only routes math to single adapter.
- `exp_pierre_dare_vs_m2p_gated_postfix` — global K=7 with gate weights (not per-prompt selection).
- This: 4-way classifier (math/code/medical/other), K=2 per-prompt selection.

## Pre-registered Kill Criteria

- **K1** Per-prompt routed K=2 avg ≥ Fisher-Rao K=7 avg + 2pp. PASS = per-prompt routing beats fixed composition.
- **K2** Per-domain win: routed K=2 wins on each of the 3 specialized benchmarks (the routing should help on the matched domain).
- **K3** Routing classifier accuracy ≥ 85% on held-out validation.
- **K4** Original killed numbers (53/20/7 collapse) replaced — every benchmark ≥ 40%.

## Why this matters

Per-prompt routing is the simplest dynamic-composition primitive. If even
this trivial version doesn't beat fixed Fisher-Rao K=7, dynamic routing
isn't paying its cost on Pierre's adapter set. If it works, it's a
zero-extra-training shipping path.

## References

- False-killed original: `exp_pierre_polar_composition_v2_routed`
- Validated routing primitive: `exp_pierre_per_task_routing_math` (TF-IDF + Ridge, 100% acc, 0.07ms)
- Audit: research agent `afdac038d5737221a`
