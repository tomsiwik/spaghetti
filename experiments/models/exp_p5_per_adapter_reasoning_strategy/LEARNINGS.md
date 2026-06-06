# LEARNINGS: P5.C1 — Per-Adapter Reasoning Strategy

## Core Finding
Direct prompting dominates all five domains (100% accuracy) at QA complexity — no per-domain strategy differentiation exists. CoT/PAL/Structured strategies either match or hurt accuracy while generating 38% more tokens.

## Why
Strategy differentiation requires tasks where direct recall fails but specialized reasoning succeeds. At QA complexity, Gemma 4 E4B-IT can answer correctly via direct recall across all domains — all strategies converge. The improvement (+12pp, +38% token savings) comes from avoiding CoT's harm (code: 60%, finance: 80%), not from positive strategy matching.

## Impossibility Structure
When a model can answer via direct recall, all reasoning strategies are equivalent in correctness and Direct wins on tokens. Per-domain strategy routing is impossible to distinguish from uniform Direct selection at this task complexity. Differentiation requires multi-step tasks where direct answering fails (e.g., multi-step proofs, complex debugging).

## Implications for Next Experiment
- Default to Direct prompting in the serving pipeline — no strategy routing needed for QA-level tasks
- Strategy routing should only be tested on harder tasks (multi-step math proofs, complex code debugging) where direct answering structurally fails
- TF-IDF routing on QA questions is degenerate (9/25 queries matched 0 keywords) — if routing is tested again, use embedding-based routing on richer domain signals
- The Room Model finding is reinforced: routing should be structural (adapter/weight selection), not strategic (prompting format)

## Status: SUPPORTED (Finding #497)
- K1279 PASS: +12pp accuracy (88%→100%) — mechanism: uniform Direct, not per-domain routing
- K1280 PASS: 37.9% token reduction (502→311 tokens)
- K1281 FAIL: TF-IDF routing 64% (threshold 80%) — degenerate fallback on QA queries
