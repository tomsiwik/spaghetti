# Peer Review: Warm-Start Scale Validation (Third Pass)

## Prior Review Context

The first review identified a critical freeze bug causing 33x parameter overcount in adapter training. The second review verified the bug fix, confirmed K1/K2 PASS, identified K3 as vacuous, and requested three documentation fixes:

1. Relabel K3 as VACUOUS in the kill criteria table and verdict
2. Note the base_val_ppl discrepancy (165.15 vs 166.61)
3. Document the greedy decoding artifact in text samples

## Verification of Requested Fixes

**Fix 1 (K3 VACUOUS):** Applied. PAPER.md line 134 reads "K3 VACUOUS" in the narrative. The kill criteria table (line 167) shows verdict "VACUOUS" rather than "PASS." The overall verdict (line 213) reads "PARTIALLY SUPPORTED" with explicit justification that K3 was never meaningfully tested.

**Fix 2 (base_val_ppl discrepancy):** Applied. PAPER.md Limitation #6 (lines 186-191) documents the 165.15 vs 166.61 discrepancy and correctly attributes it to numerical path differences after LoRA module attachment with stop_gradient.

**Fix 3 (greedy decoding artifact):** Applied. PAPER.md K1 section (lines 84-87) explicitly states all text samples use greedy decoding (temperature=0.0), identifies repetition as a decoding artifact, and distinguishes it from model failure.

## Mathematical Soundness

No change from prior review. The math in MATH.md is correct: parameter counts, LoRA formulation, composition formula, data budget analysis. The K2 reformulation (>5% PPL improvement between steps 4000-8000 instead of the self-contradicting "still decreasing -> KILL") is well-justified.

## Consistency Check: Paper vs Results

Numbers in PAPER.md match results_fixed.json:
- PPL ratio 1.037x matches 1.0367... (rounded correctly)
- Adapter improvements (7.4%, 7.1%, 7.5%) match the JSON
- Composition ratio 1.0001 matches 1.000120...
- K2 improvement 20.1% matches 20.06...
- base_val_ppl values (165.15 adapter phase, 166.61 composition phase) match JSON

## Remaining Concerns (non-blocking)

**results_fixed.json still labels K3 as boolean true.** The JSON field `K3_composition_ratio: true` is technically correct (1.0001 < 2.0 threshold) but could confuse downstream tooling that reads this as "K3 passed." This is a minor metadata inconsistency -- the paper correctly labels it VACUOUS, and the JSON preserves the raw threshold check. Not worth a rerun to fix.

**Adapter loss trajectories remain flat.** As noted in the prior review, the 1000-step adapter losses do not show clear downward trends despite producing 7% domain PPL improvements. The prior review raised the possibility of train/eval overlap. This remains an open question but is not blocking -- the gradient norms confirm the adapters are learning something, and the mechanism (LoRA on ternary base) is the point being validated, not the magnitude of improvement.

## Verdict

**PROCEED**

All three requested documentation fixes are properly applied. The paper now accurately represents the experimental evidence:
- K1 PASS with appropriate caveats (architectural mismatch, greedy decoding)
- K2 PASS with reformulated criterion and clear justification
- K3 VACUOUS with honest acknowledgment that composition was never meaningfully tested
- Overall verdict PARTIALLY SUPPORTED

The experiment provides valid micro-scale evidence that (a) warm-start ternary QAT scales from d=512 to d=1024 with comparable PPL ratio, (b) LoRA adapters train stably on self-trained ternary bases when properly frozen, and (c) composition testing requires adapters with larger deltas -- a clear next-step for future work.
