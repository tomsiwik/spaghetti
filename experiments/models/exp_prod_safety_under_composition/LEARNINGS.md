# LEARNINGS: exp_prod_safety_under_composition

**Verdict:** PROVISIONAL (smoke; `requires-full-rerun`). All 3 KCs pass at n=15.

## Core Finding
Additive N=5 PoLAR composition on Gemma 4 E4B 4-bit preserves RLHF refusal
behavior at **zero measurable behavioral drift** on 15 AdvBench-style prompts
(K1645 ΔJ=+0.000 vs 0.336 bound; K1646 ratio=1.00 all 5 categories;
K1647 max unlock=+0.000 vs 0.366). Grassmannian A reduces
`max ||A_i^T A_j||_F` to 7.03e-08 — 4 orders tighter than MATH.md A2.
Aux: behavioral composition is **strictly more** refusal-preserving than
`Σ (R_single_i − R_base) = −0.067` — quadratic Taylor remainder
sign-consistent with base logit sitting deep in the refusal basin.

## Why
Theorem 2 bound `|Δℓ_compose − ΣΔℓ_i| ≤ N(N-1)β²ε_A·L` is loose by orders
of magnitude at `||B||_F ≤ 0.1`. Random-init magnitude × near-perfect A
orthogonality puts the composed logit shift far below the refusal margin,
so behavior is structurally preserved.

## What This Does NOT Prove
- Trained adapters have ≥10× larger `||B_i||_F`; Theorem 2 bound may bite.
- Random init cannot trigger the **directed-erosion** failure mode where a
  specific fine-tune (e.g. uncensored corpus) systematically pushes
  *toward* non-refusal — K1647 becomes load-bearing only at full N with
  adversarial adapters.
- n=15 σ=0.316 is very loose CI; need n≥500 for σ≈0.043.

## Implications for Next Experiment
1. **`exp_prod_safety_full_rerun`** — SFT PoLAR × Gemma 4 E4B, AdvBench 520
   + HarmBench 400, Llama-Guard-2 judge. DO NOT ship v3→prod without it.
2. **`exp_prod_safety_adversarial_adapter`** — one adapter SFT-trained to
   erode refusal; K1647 is the load-bearing unlock test at full N.
3. **Judge**: strict-substring + thinking cues is a smoke-only workaround;
   swap to Llama-Guard-2 / StrongREJECT for supported-tier verdict.
4. **Antipattern-008 recurred**: 32-tok run silently gave base=0% because
   thinking monologue consumed the budget. Fixed in-run (160 tok + 4
   thinking cues in regex); existing memory already covers — no new
   entry. Future safety experiments MUST set `max_tokens ≥ 160` and
   verify base refusal ≥ 95% before trusting any delta.
