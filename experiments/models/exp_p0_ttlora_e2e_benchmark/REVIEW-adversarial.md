# Adversarial Review: exp_p0_ttlora_e2e_benchmark

## Verdict: PROCEED (SUPPORTED)

## Strengths

1. **Honest reporting**: All prediction misses clearly documented with deltas. The catastrophic MedMCQA failure is front-and-center, not buried.
2. **Theorem 2 confirmed exactly**: Routing independence is both mathematically trivial and empirically confirmed (98.3% match).
3. **Genuine behavioral finding**: The MedMCQA collapse (21% vs 50% baseline) while training loss was lowest (0.179) is a textbook case of metrics-as-proxies. This validates our guardrail that training loss is not behavioral quality.
4. **Compression economics validated**: 65x compression with 93% GSM8K retention makes the "$2 per domain" vision credible for reasoning tasks.

## Issues (non-blocking)

### 1. MedMCQA Prediction Directionally Wrong
MATH.md predicted "MCQ may retain better (less CoT)" — actual retention was 42% vs 93% for reasoning tasks. The prediction framework (uniform 84% retention from F#516) failed specifically where it was most confident. The post-hoc explanation (rank truncation destroys discriminative capacity) is plausible but not theorem-backed. This is an observation awaiting formalization, not a proven mechanism.

**Impact**: The finding is appropriately labeled SUPPORTED, not conclusive. No action needed.

### 2. Size Prediction 2.2x Off
Predicted ~0.15 MB/adapter, measured 0.33 MB. Parameter count is 135,492 per adapter (3,226/layer), not the ~1,500/layer estimated in MATH.md. The 2x discrepancy comes from v_proj+o_proj having more target dimensions than estimated. Doesn't affect conclusions but reveals the parameter count model needs refinement.

### 3. results.json Timing Bug
`total_time_s: 1.3` is clearly wrong — individual domain training times sum to ~1,894s (~31 min). Likely only captures the evaluation/routing phase. Should be noted for future result parsing.

### 4. K1428 "Marginal" Framing
1,000,152 bytes is >= 1 MB. It's correctly reported as FAIL. The argument that "raw binary would pass" is engineering-valid but the kill criterion doesn't specify format. Correctly handled — just noting the marginal nature.

## Key Finding for LEARNINGS.md

**Retention is task-type dependent, not uniform.** The F#516 model (84% uniform retention) is superseded by a two-tier model:
- Generative reasoning tasks (CoT): ~90% retention at TT-rank 6
- Discriminative MCQ tasks: ~42% retention at TT-rank 6

This has architectural implications: the 25-domain vision may need mixed adapter formats — TT-LoRA for reasoning domains, higher-rank for classification domains.

## Recommended Finding Status
SUPPORTED — 3/4 kill criteria pass, major unexpected finding (task-type sensitivity) with behavioral evidence, post-hoc explanation awaiting formalization.
