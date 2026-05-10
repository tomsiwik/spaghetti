# REVIEW-adversarial.md — exp_pierre_polar_routed_v2

## Verdict: KILL CONFIRMED

## Checklist

| # | Check | Result |
|---|-------|--------|
| a | results.json verdict matches DB | PASS — both "KILLED" |
| b | all_pass matches claim | PASS — 3/4 KC fail |
| c | PAPER.md verdict matches | PASS — "KILLED" |
| d | smoke → provisional | N/A |
| e | KC not modified post-run | PASS — consistent between MATH.md and code |
| f | No tautological KC | FLAG — K3 is always True (no val split). Non-blocking: experiment killed on K1/K2/K4 |
| g | Code measures what MATH.md says | PASS — per-prompt K=2 Fisher-Rao compose + eval |
| h | No independent A/B summation | PASS — B-only compose via compose_fisher_rao |
| i | LORA_SCALE < 12 | PASS — uses shared SCALE constant |
| j | Routing per-sample not global | PASS — composition recomputed per prompt |
| k | No shutil.copy fake | PASS |
| l | No hardcoded pass | PASS (K3=True flagged, not result fabrication) |
| m | Model consistent | PASS — gemma-4-e4b-it-4bit throughout |
| n | Base accuracy | N/A — no base run |
| o | n ≥ 15 | PASS — N=50 per bench |
| p | Target-metric KC | PASS — K1/K2/K4 are task accuracy |

## Notes

- K3 (classifier accuracy) is tautological — trained on synthetic data, evaluated on same distribution with no held-out val. But irrelevant: even with perfect routing, K=2 composition quality is the bottleneck (medqa 30%).
- Code is clean: per-prompt recomposition loop correctly resets model state between prompts.
- Finding #847 correctly synthesizes K=2 routed + K=3 strategy + K=3 domain results into unified conclusion: K=7 is minimum viable.

## Finding

**#847:** Per-prompt K=2 routing (strategy_full + domain_X) avg 58.0% vs K=7 64.7%. medqa collapse (30%) confirms composition quality, not routing accuracy, is the binding constraint. Sub-K=7 investigation line closed.
