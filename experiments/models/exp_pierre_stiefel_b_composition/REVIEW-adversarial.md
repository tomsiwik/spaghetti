# REVIEW — exp_pierre_stiefel_b_composition

**Verdict: KILL** (dependency unmet)

## Adversarial checklist

| # | Check | Result |
|---|-------|--------|
| a | results.json matches DB status | N/A — no run, no results.json |
| b | all_pass matches claim | N/A |
| c | PAPER.md verdict matches DB | PASS — both say KILLED |
| d | smoke → provisional | N/A |
| e | KC not modified after run | N/A — no run |
| f | No tautological KC | PASS — KCs are well-designed (moot) |
| g | Code measures MATH.md quantity | N/A — no run |
| h | Composition math correct | N/A |
| i | LORA_SCALE safe | N/A |
| j | Routing not tautological | N/A |
| k | No shutil.copy faking | N/A |
| l | No hardcoded pass | N/A |
| m | Model matches MATH.md | N/A |
| n | Base accuracy > 0% | N/A |
| o | n ≥ 15 | N/A |
| p | Target-metric KC exists | PASS — K1/K2/K4 are accuracy-based |

## Assessment

Correct kill. Upstream `exp_pierre_joint_stiefel_b_train` was killed (B-only
Stiefel insufficient for composition: A-matrix coupling). No weights saved.
Even if weights existed, upstream K3 failure (cross-contribution 3.4%–501k%)
proves composition operators on B alone cannot fix A-coupling. PAPER.md
correctly documents both the missing-weights and the theoretical dead-end.
