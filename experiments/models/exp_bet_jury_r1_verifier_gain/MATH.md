# MATH — exp_bet_jury_r1_verifier_gain

**BET jury-decode R1.** Adapter-as-verifier best-of-8 vs self-consistency(8) on GSM8K, equal token budget.
Ladder: `.agents/bets/jury-decode.md` · Grounding: rStar-Math (arXiv:2501.04519) — small + verifier search
beats frontier on checkable tasks; bound: verifier quality (Stroebl, arXiv:2411.17501).

## Setup

Frozen base `mlx-community/gemma-4-e4b-it-4bit` + math LoRA (r=6, q_proj, scale 6 ≤ 8) as **generator**.
The SAME adapter, prompted as a judge, is the **verifier** (this is the asset claim: one existing adapter,
no new training, gives a usable reward signal at decode time).

For each of N ≥ 200 GSM8K test questions (seed 42, no-thinking harness — per harness-relative-EM memory):

- **Greedy** (floor reference, 1/8 budget): 1 greedy chain, predict its `####` answer.
- **Candidates**: 8 sampled chains, temperature 0.8 (untempered logprob recorded per chain).
- **SC(8)**: majority vote over the 8 candidates' parsed answers (tie → first sampled).
- **BoN(8)**: same 8 candidates, pick argmax verifier score; same generation tokens as SC(8)
  ⇒ generation budget is **identical by construction**. Verifier cost is prefill-only
  (no generated tokens) and is reported separately so search pays its rent transparently.

**Verifier score (primary, pre-registered):** single forward pass on
`Question + Proposed solution + "Is the final answer correct? Reply Yes or No."`,
score = logP(Yes) − logP(No) at the last prompt position, under base+math adapter.

**Diagnostic (not the gate):** BoN by mean untempered chain logprob (likelihood ranking) — the known-weak
baseline that a real verifier must beat to be more than sample likelihood.

## Theorem (prediction sketch)

Let p = per-chain correctness probability and let the verifier have ranking AUC = a on correct-vs-wrong
chains. SC(8) approximates the mode of the answer distribution: it fails whenever a single wrong answer
mode out-votes the correct one. BoN(8) succeeds whenever ≥1 of 8 chains is correct AND the verifier ranks
a correct chain on top. With math-solo p ≈ 0.85 per greedy chain (bet baseline), pass@8 ≈ 1−(1−p')⁸ ≫ p
for sampled p' ≈ 0.7–0.8, leaving ~8–15pp of headroom above SC that only a verifier can claim. A verifier
with a ≈ 0.70 converts roughly half of the SC-residual errors that have ≥1 correct candidate.

**Predicted numbers:** verifier AUC ≈ 0.65–0.75; acc(BoN8) − acc(SC8) ≈ +3 to +6pp.

## Pre-registered refutation thresholds (kill criteria, verdict gates)

- **K2315 (killed):** verifier AUC ≤ 0.55 on pooled correct-vs-wrong candidates — no better than random.
- **K2316 (killed):** acc(BoN8) ≤ acc(SC8) at equal generation budget.
- **Supported:** both kills clear AND acc(BoN8) − acc(SC8) ≥ +0.03 (the R1 gate).
- 0 < gain < 3pp with AUC > 0.55: `provisional` (signal exists but R1 gate unmet — not goalpost-moving;
  registered before the run).

Diagnostics (likelihood-BoN acc/AUC, pass@8 ceiling, token budgets per arm) are reported but gate nothing.
