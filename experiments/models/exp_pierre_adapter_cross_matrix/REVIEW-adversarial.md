# REVIEW-adversarial.md — exp_pierre_adapter_cross_matrix

**Verdict: KILL CONFIRMED**

## Adversarial Checklist

| Check | Result |
|-------|--------|
| (a) results.json verdict matches DB status | PASS — both KILLED |
| (b) all_pass matches claim | PASS — K2 fails, verdict=KILLED |
| (c) PAPER.md verdict matches | PASS — "KILLED (K2 fail)" |
| (d) is_smoke → provisional | N/A — not smoke |
| (e) KC not modified post-run | PASS — K1-K4 pre-registered in MATH.md |
| (f) No tautological KC | PASS — cross-domain matrix is real measurement |
| (g) Code measures same as MATH.md describes | PASS — single adapter on 3 benchmarks, delta vs raw |
| (h) Independent lora_A/B summation | N/A — single-adapter loading, no composition |
| (i) LORA_SCALE ≥ 12 | PASS — uses shared SCALE constant (≤8) |
| (j) Routing on single sample | N/A — no routing, exhaustive matrix |
| (k) shutil.copy fake adapter | PASS — loads from shared adapter store |
| (l) Hardcoded pass | PASS — real eval_fn calls |
| (m) Model mismatch | PASS — MATH.md says Gemma 4, code uses MODEL_NAME from shared config |
| (n) Base accuracy 0% | PASS — raw baseline 56/22/6, non-zero |
| (o) N < 15 | PASS — N=50 |
| (p) Target-metric KC | PASS — GSM8K/HumanEval/MedQA are task accuracy |

## Kill Rationale

K2 fail is clean: strategy_full medqa=0% (−6pp vs raw 6%). The code correctly detects this. The experiment produced highly valuable cross-transfer data despite the kill — 17/21 cells show positive transfer, informing future K=2 routing decisions.

## Notes

- strategy_full's pathological behavior (0% medqa) is a genuine finding, not a code bug
- Domain adapters as general boosters is actionable for routing design
- No code issues, no fabrication, no tautologies
