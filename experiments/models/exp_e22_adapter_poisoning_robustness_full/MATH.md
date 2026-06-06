# E22-full: Adapter Poisoning Robustness — Full Run

## Type
Verification — full-scale replication of E22 smoke (PROVISIONAL) at 35 layers, 5 clean adapters, 100 QA questions.

## Prior work
- **E22 smoke (PROVISIONAL)**: K2055 PASS (25pp < 30pp), K2056 PASS (55pp > 2pp). 3 layers, 3 adapters, 20 QA.
- **F#821**: Grassmannian provides 55pp poisoning protection via input-space feature isolation.
- **F#822 (E14-full)**: Grassmannian activation decorrelation is noise at scale (0.0018 vs 0.01 threshold) — confirms protection mechanism is NOT metric-based.
- **F#815**: B₁ᵀB₂ coupling dominates activation cosine but NOT behavioral degradation.
- **arxiv:2406.14563** (Model Merging Safety): one bad adapter spoils naïve task arithmetic.

## Theorem (Orthogonal Subspace Containment)
Inherited from E22 MATH.md — no changes.

**Claim**: Grassmannian A matrices provide poisoning containment via input-space feature isolation, independent of B-matrix correlation.

**Proof**: See E22 MATH.md. Key insight: A_poison ⊥ A_i forces poison to read from different input features. Output-space B-matrix interference adds noise but doesn't amplify through correlated input selection. Protection is behavioral (clean adapters preserve function) not metric (activation cosine unchanged).

## Kill Criteria (pre-registered)

**K2059** (target, behavioral): Grassmannian drop < 30pp at worst multiplier (35 layers, 100 QA).
- Inherits K2055 structure. Smoke measured 25pp at 20× — expect similar or better at scale (more layers → more averaging).
- PASS if < 30pp. FAIL if ≥ 30pp.

**K2060** (comparative, behavioral): Protection margin > 2pp at best multiplier (35 layers, 100 QA).
- Inherits K2056 structure. Smoke measured 55pp — expect ≥ 10pp at scale.
- PASS if > 2pp. FAIL if ≤ 2pp.

Both KCs behavioral (knowledge QA accuracy). Target-gated per F#666.

## Predictions
1. Grassmannian worst drop: 15-25pp (more layers → more averaging → less per-layer impact)
2. Random worst drop: 60-80pp (correlated input features amplify across 35 layers)
3. Protection margin: 30-55pp (scale should amplify the phase transition seen in smoke)
4. Phase transition: between 5× and 10× for random A (consistent with smoke)

## Experiment design
- 35 target layers (42 total - 7 global attention)
- 5 clean adapters + 1 poison per condition (Grassmannian vs random A)
- 100 knowledge QA questions
- Poison multipliers: [1, 3, 5, 10, 15, 20]
- v_proj only (F#627 compliant target)
