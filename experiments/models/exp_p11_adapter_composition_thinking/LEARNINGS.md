# LEARNINGS.md — P11.J0: Adapter Composition via Exclusive Routing

## Core Finding (KILLED 2026-04-18)

All 3 KCs FAIL via three independent drivers — and J0 is the **first measurement-based
kill** in the P11.X chain (8th overall: F0/H0/B0/C0/D0/H1/I0/J0).

- **K1528 direct measurement FAIL**: router accuracy 187/280 = 0.668 vs ≥0.85 pre-reg
  (−18.2pp). Mean-pool of `embed_tokens` + cosine to centroids is too lossy for
  binary reasoning/knowledge classification.
- **K1526/K1527 untestable**: 4-of-4 referenced domain-adapter dirs (`adapters/{math,
  medical,legal,finance}/`) hold only `adapter_config.json` — no `adapters.safetensors`.
  Phase 4 crashed at `mlx.load` with no recoverable data.
- **MATH.md Theorem 1 premise inverted**: Phase 3 measured `acc_t(reasoning)=0.375 <
  acc_t(knowledge)=0.725` (35pp inversion). Premise `acc_t(P_r) ≥ acc_t(P_k)` broken;
  the loaded thinking adapter is H0's regressed artifact (H0 killed same day, 47.6%
  MMLU-Pro, humanities collapse).

## Why

Three orthogonal failure modes compound: (1) embedding-mean router is information-poor
(needs later-layer hidden states or learned head); (2) domain-adapter training scripts
in `exp_p1_t2_*` never persisted weights or had them cleaned — citing them as
dependencies without verifying `*.safetensors` is the consumer side of antipattern-017;
(3) inheriting H0's regressed thinking adapter propagates upstream cascade — antipattern-018's
shared-harness blast radius reaches composition experiments, not just training ones.

## Implications for Next Experiment

1. **Distinguish kill type for backlog planning**: J0 is *measurement-based*, prior 7
   were *preemptive on antipattern-018 recurrence*. Research lessons differ — J0
   teaches "weak router + dependency on regressed adapter"; the chain teaches
   "shared training harness needs P11.HARNESS unblock". Both unblocks are needed.
2. **P11.HARNESS blocks J0-v2, M0, K1527 retest** — atomic shared fix.
3. **Adapter-infra audit (antipattern-017 second instance)**: before any composition
   experiment, grep `micro/models/**/adapter_config.json` and assert sibling
   `adapters.safetensors` exists with size > 0. Two-instance recurrence (P11.B
   baseline_eval, P11.J0) makes this a *systemic* failure mode for the P11 chain.
4. **Router redesign as standalone work**: 66.8% on token-mean is a meaningful
   negative result regardless of cascade. Try later-layer hidden states (forward
   to layer ~10) or a learned linear head over centroids.
5. **F#560 baseline reconciliation** (62.1% cited vs 40.7% measured) still open;
   blocks any "≥ X% absolute" KC across P11. KCs that are relative deltas (J0's
   K1526/K1527) survive the drift; absolute thresholds (K1525, K1545) do not.
6. **L0 still motivated** but cannot use J0's router as-is — needs router redesign
   first OR exclusive single-adapter eval framing.
