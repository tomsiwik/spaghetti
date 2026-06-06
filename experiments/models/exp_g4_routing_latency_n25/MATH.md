# MATH.md — exp_g4_routing_latency_n25

## Claim (pre-registered)
KC #1597: per-sample ridge-routed Gemma 4 at N=25 adds ≤ 1.20× latency overhead
vs base generation.

## Pre-registered tripwire (UNMEASURABLE → KILLED)

This experiment is structurally gated on three preconditions. If any of
P1/P2/P3 FAILS, KC #1597 is UNMEASURABLE and the experiment KILLS per the
cohort's established tripwire pattern (Findings #605/#606/#608/#610/#611/
#612/#613/#615/#616/#617/#618/#619/#620/#621).

### P1 — N=25 Gemma 4 v_proj+o_proj r=6 LoRA safetensors
Latency for per-sample ridge routing requires ≥ 25 attachable adapters on
disk. Without them, there is nothing to route among.

### P2 — Upstream T2.1 (`exp_p1_t2_single_domain_training`) SUPPORTED
Finding #606 etc: upstream T2.1 verdict=KILLED, all_pass=false (K1030
metric-swap, K1028 format-artifact). Downstream routing-latency claims
cannot bind to a broken adapter manifest.

### P3 — Ridge router with Gemma 4 binding on disk
Prior-art routers (`exp_p1_c0_composition_port_gemma4`, `exp_g4_tfidf_ridge_*`)
either (a) lack Gemma 4 wiring or (b) depend on P1 themselves.

## Theorem (tripwire)
Let A = {required_adapters on disk}, T = {upstream T2.1 verdict}, R = {Gemma 4
ridge router on disk}. Measurement of KC #1597 requires |A| ≥ 25 ∧ T =
SUPPORTED ∧ R ≠ ∅.

If any conjunct is false, no ≤ 2h probe on M5 Pro 48 GB can produce a
trustworthy latency number — the ratio would be computed against an
empty adapter set or a broken upstream, yielding a meaningless
measurement (divide-by-zero in per-adapter overhead amortization, or
a latency-overhead figure that applies to non-existent weights).

QED — tripwire is structural, not hyperparameter-sensitive.

## Kill criterion outcome mapping
- P1 ∧ P2 ∧ P3 ALL TRUE → run experiment, measure KC #1597 empirically.
- Any FAIL → K1597 UNMEASURABLE → status=killed, evidence = probe
  bytes-on-disk, no MLX model load, no latency measurement.

## Assumptions (per Autonomy guardrail 1007)
- "Per-sample ridge routing" = the composition algorithm from Finding
  #310-class prior (TF-IDF + ridge regression over adapter registry).
  The exact ridge regulariser α is not a free parameter at the probe
  stage; P3 failure prevents it being exercised.
- "Base" = unadapted Gemma 4 E4B 4-bit `mlx-community/gemma-4-e4b-it-4bit`
  at matched decode budget. Matched-budget comparison only meaningful
  once P1/P2/P3 all PASS.
