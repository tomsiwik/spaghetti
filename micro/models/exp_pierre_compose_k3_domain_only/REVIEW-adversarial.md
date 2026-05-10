# Adversarial Review — exp_pierre_compose_k3_domain_only

**Verdict: KILL CONFIRMED** (with two non-blocking flags)

## Checklist

| # | Check | Result |
|---|-------|--------|
| a | results.json verdict matches DB | FLAG: results.json says "SUPPORTED", DB says killed. Script bug — verdict logic only checks K1 (which is tautological). Kill decision is correct per K2. |
| b | all_pass matches claim | N/A (field absent; K2.pass=false confirms kill) |
| c | PAPER.md verdict matches DB | PASS: "KILLED" |
| d | is_smoke → provisional | N/A |
| e | KC not modified post-run | PASS: single run, no git history of KC changes |
| f | No tautological KC | FLAG: K1 compares fisher_rao to itself (delta=0, threshold=0). PAPER.md acknowledges this. Kill rests on K2 which is real. |
| g | Code measures what MATH.md describes | PASS |
| h | Composition math correct | PASS: uses shared `compose_fisher_rao` via eval_runner |
| i | LORA_SCALE < 12 | PASS: scale=6.0 |
| j | Routing not tautological | N/A (no routing) |
| k | No shutil.copy fakes | PASS |
| l | No hardcoded pass | PASS |
| m | Model matches MATH.md | PASS: gemma-4-e4b-it-4bit |
| n | Base accuracy check | PASS: single_best 62% confirms model works |
| o | n ≥ 15 | PASS: n=50 |
| p | Target-metric KC present | PASS: gsm8k/humaneval/medqa are task accuracy |

## Non-blocking flags

1. **results.json verdict bug**: The eval_runner sets verdict="SUPPORTED" based only on K1 passing. Should check all KC. Non-blocking because the kill decision is externally validated by K2 data.
2. **K1 is tautological**: Comparing a Fisher-Rao subset to itself always yields delta=0. Future experiments should compare K=3 FR to K=7 FR reference (64.7%). Non-blocking because K2 carries the kill.

## Finding

K=3 domain-only composition avg 46.0% — catastrophic humaneval collapse (34%, -44pp). Domain axis is MORE interference-prone than strategy axis (46% vs 56%). K=7 is minimum viable adapter set for Pierre v3. Closes "ship fewer adapters" line.
