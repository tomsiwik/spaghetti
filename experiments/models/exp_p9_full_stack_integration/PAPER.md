# PAPER.md — P9.G0: Full Stack Integration — Current System Capability Assessment

**Type**: Guided Exploration  
**Date**: 2026-04-14  
**Status**: Pending full run (pueue task 8)

---

## Prediction vs. Measurement Table

| Kill ID | Prediction (MATH.md) | Measured | Pass? |
|---------|---------------------|----------|-------|
| K1387 | Math adapter ≥ base + 15pp on GSM8K (theoretical: 81.4% if base ≈ 55%) | TBD (task 8) | TBD |
| K1388 | Oracle-routed system ≥ base + 10pp on mixed-domain MMLU-Pro | TBD (task 8) | TBD |
| K1389 | 5 adapters < 5 MB (EXPECTED FAIL — documenting TT-LoRA gap) | 61.98 MB total | FAIL (expected) |

---

## Smoke Test Findings (from results.json, smoke=true)

### Phase 1: GSM8K
- HTTP 422 error from datasets-server API: all GSM8K results null
- **Fix applied**: replaced URL-based download with `datasets.load_dataset("openai/gsm8k")`
- K1387 result pending full run

### Phase 2: Oracle Routing (MMLU-Pro subset)
- Smoke test (n=2/cat): oracle=25%, base=16.7%, delta=+8.3pp (below 10pp threshold)
- **Not conclusive at n=12** — smoke sizes too small for reliable signal
- K1388 result pending full run (n=20/cat)

### Phase 3: Composition (α=0.5, math + medical)
- Smoke test ran successfully — composition loading/cleanup worked cleanly
- Adapter merge via `mx.save_safetensors` confirmed functional
- **Note on approximation**: code uses `α*B1 + (1-α)*B2` with shared `A1`, which differs from
  the ideal Theorem 2 formula `α*(B1@A1) + (1-α)*(B2@A2)`. This is a parameter-space average,
  not a weight-space exact sum. Interference metrics in full run should be interpreted accordingly.

### Phase 4: Footprint Audit
- math: 14.30 MB, code: 14.30 MB, medical: 14.30 MB, legal: 9.54 MB, finance: 9.54 MB
- **Total: 61.98 MB** (vs. MATH.md's "25MB" estimate — corrected here)
- MATH.md stated "5MB each = 25MB total" — actual files are 14.3MB (q+k+v+o+gate r=6) and 9.54MB
- K1389 FAIL confirmed as expected (TT-LoRA compression at 180KB was supposed to close this gap)

---

## Key Non-Blocking Issues

**N1 — Oracle math shows 0% in smoke MCQ (vs. base 25%)**  
Math adapter degrades MCQ performance. Consistent with Finding #515 (TT-LoRA experts too weak
for MCQ knowledge steering). q_proj adapters trained on generation tasks do not improve MCQ.
Expected behavior, not a bug. K1388 tests multiple domains — law/finance adapters may differ.

**N2 — Adapter footprint discrepancy in MATH.md**  
MATH.md states "measured 5MB each = 25MB total". Actual: 14.3MB (math/code/medical),
9.54MB (legal/finance) = 61.98MB total. MATH.md figure was incorrect. Corrected here.
K1389 FAIL conclusion (footprint >> 5MB target) unchanged.

---

## What This Establishes

1. **Floor measurement**: Base Gemma 4 E4B GSM8K and MMLU-Pro accuracy (prior to adapter)
2. **Routing value-add**: Delta between base and correct-domain-adapter on domain tasks
3. **Composition cost**: Interference from α=0.5 blending of math + medical adapters
4. **System state audit**: Documents current P9 capability before P11 reasoning improvements

---

## Connection to Theorem Predictions

**Theorem 1** (Routing-Gated Lower Bound): With routing accuracy α=0.977, the routed math system
should achieve ≥ 0.977×82% + 0.023×base_acc. If base≈55%, predicted floor = 81.4%. K1387
asks for ≥ base+15pp, so if base=55%, threshold = 70% (well below theoretical 81.4%).

**Theorem 2** (Composition Interference): At α=0.5 with ||Δ_1||≈||Δ_2|| (same r=6 q_proj),
expects 5–15pp degradation vs single-adapter on the primary domain.

**Theorem 3** (Footprint): 61.98 MB total is a design-time constant for the current adapter
architecture. Achieving <5MB requires either TT-LoRA compression (failed, MCQ) or architecture
change to smaller rank or fewer target modules.

---

## References

- Finding #421: math adapter 82% GSM8K (exp_p1_t2_single_domain_training)
- Finding #225: near-lossless composition (exp_persistence_diagram_diff)
- exp_p9_ttlora_moe_router PAPER.md: routing 97.7%, TT-LoRA MCQ failure
- arXiv:2202.05262 (ROME): FFN layers store factual knowledge — q_proj adapters miss this
