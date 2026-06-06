# MATH.md — exp_g4_svd_truncate_adapter (PREEMPT-KILL, tautological-inter-variant-delta antipattern, 4th instance)

## Claim under review
K1611: "r=4 within 5% of r=6 on MMLU-Pro" — sole KC.

## Preempt-KILL conclusion (before any measurement)
KILLED on the **tautological-inter-variant-delta-ignores-base-baseline** §5-promoted antipattern (prior instances: K1552 / F#709 §5 promotion at 3rd). This is the **4th instance**, sub-variant **intra-adapter-rank-delta** (r=4 SVD-truncated vs r=6 baseline of SAME adapter). The §5 clause in reviewer.md applies byte-for-byte; direction-symmetric (|A − B| ≤ δ here, analogous to ≥ δ at K1552/K1577 and < δ at K1584).

## F#666 truth table (two-branch)
- **Branch 1: K1611 is a target-metric KC (MMLU-Pro accuracy).** Yes — MMLU-Pro is a real multi-domain MCQ benchmark, not a proxy. So this is **not F#666-pure-standalone**. Distinct from F#700/F#701/F#703/F#705/F#706/F#707/F#708/F#710/F#711. The tautological-inter-variant-delta antipattern (F#704/F#709) is the applicable clause.
- **Branch 2: K1611 as formulated is a _relative_ target KC without an absolute floor.** Even with a target metric on both sides, a comparison-only KC is satisfied by degenerate-equivalence regimes (both variants at base, both identical-below-base, both near-random) regardless of thesis progress.

## Five lemmas

### L1 — Degenerate-equivalence branch trivially satisfies K1611
Let `M_r ∈ [0, 1]` denote MMLU-Pro accuracy with the Gemma 4 adapter truncated to rank r. K1611 demands `|M_4 − M_6| ≤ 0.05`. Parent F#477 measured r=6 q_proj Gemma 4 target-FAILED on MCQ: K1226 FAIL with `adapted_acc = 0.480 < 0.50` (below random chance floor for 4-opt MCQ; base-comparable). In this regime `M_6 ≈ M_base` within CI width. SVD truncation to r ≤ 4 reduces adapter contribution further, so plausibly `M_4 ≈ M_base` as well. Therefore `|M_4 − M_6| ≈ 0 ≤ 0.05` **with zero mechanism** — both variants collapse to the same base-model accuracy, "PASS" signals no SVD-truncation quality preservation.

### L2 — Prerequisite gate unmet (F#166 base-beat requirement)
F#166 prerequisite gate (repo canon): single-adapter comparison KCs require per-variant base-anchor before inter-variant delta is testable. K1611 omits any per-variant KC of the form `M_r ≥ M_base + γ`. Without `M_6 > M_base` established, comparisons `M_4 ≈ M_6` are vacuous (you're comparing two failing configurations). F#477 K1226 FAIL is the explicit demonstration that `M_6 ≤ M_base + 0` on Gemma 4 MCQ.

### L3 — Parent-target-FAIL inheritance (F#477)
Parent F#477 (killed 2026-04-11): rank-6 q_proj LoRA on Gemma 4 E4B, K1226 FAIL (adapted_acc 0.480 < 0.50). The only r=6 q_proj adapters materialized on disk are `exp_p1_t2_single_domain_training/adapters/{code, math, medical}` (F#627, rebuilt 2026-04-19), trained for domain PPL, not multi-domain MCQ. MMLU-Pro is a 14-category benchmark (Biology, Business, Chemistry, Computer Science, Economics, Engineering, Health, History, Law, Math, Other, Philosophy, Physics, Psychology); only 3 of these plausibly match the trained adapters (CS ↔ code, Math ↔ math, Health ↔ medical). For the remaining 11 categories the adapter is distribution-out-of-sample — parent target state is "fails MCQ transfer" per F#477 impossibility-structure analysis (strong base priors, δ_d ≈ 0 when H(V_d|θ) is low).

### L4 — Intra-adapter-rank-delta is a sub-variant of the §5-promoted antipattern
§5 clause (promoted at F#709) fires direction-symmetrically on any KC of form `op(f(variant_i), f(variant_j)) op_2 δ` lacking a paired per-variant base-anchor KC. Sub-variants to date:
1. K1552 — inter-architecture (output-space QA vs NTP): `≥ 5pp`
2. K1577 / F#704 — inter-training (QA-format + cache-aware top-2 vs NTP swap-per-token): `≥ 5pp`
3. K1584 / F#709 — inter-routing (SOLE / CAT / LoRA-Flow / X-LoRA): `< 2pp`
4. **K1611 / this — intra-adapter-rank (r=4 SVD-truncated vs r=6 baseline): `≤ 5%`**

All four share the same structural defect; the variant axis differs. Resolution is identical: pair with per-variant base-anchor KC (e.g., `M_4 ≥ M_base + γ` AND `M_6 ≥ M_base + γ` AND `|M_4 − M_6| ≤ 5%`), or reject.

### L5 — Hygiene independence (2 defects below 3+ threshold)
DB record shows `success_criteria=[]`, `references=[]` (2 defects). Below F#703-canonical 3+ threshold for hygiene-multi-defect promotion. Hygiene is a secondary finding, not the preempt trigger — the tautological KC structure is.

## Standalone / parent topology
`depends_on=[]` (standalone). Not F#669-family (inter-experiment parent-dependency), not F#666-pure (has target metric), not hygiene-multi-defect (2 < 3 threshold). Cleanly inside tautological-inter-variant-delta §5 family.

## Prediction table (what would measurement have shown if we ran it)
| # | Prediction | Grounding | If measured |
|---|------------|-----------|-------------|
| P1 | M_6 ≈ M_base on MMLU-Pro (within ±5pp) | F#477 K1226 FAIL (q_proj r=6 Gemma 4 MCQ) | Confirms L1/L3 |
| P2 | M_4 ≈ M_6 on MMLU-Pro | Eckart-Young + degenerate-equivalence regime | K1611 passes trivially |
| P3 | M_2 ≈ M_3 ≈ M_4 ≈ M_6 ≈ M_base | Shared-failure regime per L1 | All SVD truncations indistinguishable |
| P4 | Adding per-variant base-anchor `M_r ≥ M_base + 0.05` would convert PASS to FAIL | L2/L4 remedy | Exposes vacuity |
| P5 | Domain-PPL reformulation (code/math/medical only) would allow non-degenerate measurement | F#325 SUPPORTED on Qwen3-4B domain PPL | v2 path |

## Unblock path (for v2 re-claim)
A valid follow-up `exp_g4_svd_truncate_adapter_v2_domain_ppl` would:
1. Restrict evaluation to the 3 trained domains (code, math, medical) held-out val sets — not MMLU-Pro.
2. Use F#325's PPL ratio metric: `PPL_r / PPL_base`, with both per-variant and inter-variant KCs.
3. Add absolute floor: `PPL_r < PPL_base` per rank AND `PPL_4 / PPL_6 ≤ 1.05` per pair.
4. Cite F#325 (SVD Eckart-Young on LoRA deltas) and F#627 (r=6 q_proj SUPPORTED on domain tasks) as grounding.

## References
- F#477 (killed, 2026-04-11, r=6 q_proj MCQ FAIL on Gemma 4)
- F#627 (supported, 2026-04-19, r=6 q_proj domain tasks SUPPORTED)
- F#325 (supported, 2026-04-06, SVD-rank=4 domain PPL 0.77× on Qwen3-4B-4bit)
- F#666 (conclusive, 2026-04-19, target-pair guardrail)
- F#704 (killed, 2026-04-24, 2nd instance, antipattern watchlist)
- F#709 (killed, 2026-04-24, 3rd instance, §5 promotion)
- F#166 (prerequisite gate — base-beat before composition)
- Eckart-Young theorem (optimal low-rank approximation)
- Davis-Kahan sin-theta theorem (F#324 impossibility structure)
