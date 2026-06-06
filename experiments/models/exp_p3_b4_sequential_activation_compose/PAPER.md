# PAPER.md — P3.B4: Pure Additive Composition (Interference-Free Baseline)

## Prediction-vs-Measurement Table

| Kill Criterion | Prediction | Measured | Pass? | Notes |
|---|---|---|---|---|
| K1189: max_cos_b (diagnostic) | ~0.16 (same as P3.B1) | 0.1607 | PASS | Unchanged adapters, exact match |
| K1187: style compliance, pure additive | ≥ 66% (better than B-GS 60%) | 24.0% | **FAIL** | Δ=+52pp; WORSE than B-GS by 36pp |
| K1188: math MCQ accuracy | ≥ 5% | 15.0% | PASS | Math actually improved (+5pp) |

**Verdict: KILLED** (K1187 fails hard: 24% vs 66% threshold)

---

## Experimental Setup

- Math adapter: rank=6, scale=6.0, all 42 q_proj layers (exp_p1_t2_single_domain_training)
- Personal adapter: rank=4, scale=4.0, layers 26-41 q_proj (exp_p1_t5_user_local_training)
- Composition: rank-10 concatenation [A_D|A_P] and [s_D×B_D; s_P×B_P], NO projection
- Power ratio: 1.077 (domain/personal) — nearly balanced (corrected PERS_SCALE=4.0)
- n_overlap_layers=16 (layers 26-41 have both adapters)
- N_style=25, N_math=20, elapsed=238.7s

---

## Comparison Across P3 Series

| Method | Style | Math | Delta (style) | vs Personal-Only | Notes |
|---|---|---|---|---|---|
| Personal-only baseline | 76.0% | — | — | 0pp | Reference |
| B-GS (P3.B1, Finding #462) | 60.0% | 10% | -16pp | -16pp | B-rows orthogonalized |
| Full-ΔW GS (P3.B2, Finding #463) | 40.0% | 20% | -36pp | -36pp | Over-amplification confound |
| Full-ΔW α=1.0 (P3.B3, Finding #464) | 0.0% | 40% | -76pp | -76pp | All style in col(ΔW_D) |
| **Pure additive (P3.B4)** | **24.0%** | 15% | **-52pp** | **-52pp** | WORSE than B-GS |

---

## Surprising Finding: Pure Additive < B-GS

MATH.md predicted pure additive ≥ 66% (better than B-GS 60%) because B-GS "removes style via projection." The reality: pure additive gives 24% — 36pp WORSE than B-GS.

**Explanation**: B-GS serendipitously HELPS by orthogonalizing personal adapter B-rows against domain B-rows. In pure additive, shared output directions (n_overlap_layers=16) create a different failure:

1. Domain adapter contributes to shared output directions with strength s_D × B_D rows
2. Personal adapter contributes the same directions with s_P × B_P rows (overlapping)
3. Domain signal in shared directions is STRONGER overall (total_norm_domain=78.356 vs pers=72.728)
4. The model receives domain+personal in shared directions → domain overwhelms the personal style signal

B-GS removes personal adapter's overlap with domain directions, forcing personal to only activate in "fresh" (non-domain) directions. This isolation accidentally preserves style better.

**BUT**: Even B-GS only achieves 60% (16pp loss from baseline). The remaining loss is from context shift: domain adapter changes hidden state distribution in layers 0-25 before personal adapter engages in layers 26-41.

---

## Impossibility Structure: Weight-Space Additive Composition

**Theorem (informal)**: For two LoRA adapters with overlapping layer ranges trained independently on the same base model:

1. Their B-matrices share output directions (n_overlap_layers > 0)
2. Pure additive double-counts shared directions → domain signal overwhelms personal
3. B-GS projection fixes double-counting but destroys style-encoding rows that overlap with domain
4. Full-ΔW projection fixes everything but removes ALL style (col(ΔW_P) ⊆ col(ΔW_D))

All weight-space composition strategies (additive, B-GS, full-ΔW GS) fail because:
- They operate on weights trained on DIFFERENT hidden state distributions
- Personal adapter was trained on base model's hidden states (no domain adapter)
- Domain adapter shifts the hidden states it sees
- Personal adapter in layers 26-41 now receives domain-shifted states ≠ what it was trained on

**The fix requires training-time alignment**: P3.B5 — retrain personal adapter ON TOP of domain-adapted model. This eliminates the distribution mismatch at training time.

---

## Next Experiment: P3.B5 (Adapted Personal Training)

**Design**: Load domain-adapted model (base + math adapter) → train personal adapter on top.
The personal adapter now learns: "given domain-shifted hidden states, how to add personal style?"

**Mathematical guarantee**: Training on domain-shifted distribution → personal adapter parameterized
to adapt FROM that distribution. Weight-space interference is irrelevant because the personal adapter
already accounts for domain adapter's effect in its learned parameters.

**Prediction**: style compliance ≥ 76% (matching personal-only baseline) because the trained model
already accounts for the domain context.

**Key risk**: if math accuracy degrades with personal style override, composition fails despite better style.

---

## Evidence

- results.json: is_smoke=false, n_style=25, n_math=20, elapsed=238.7s
- K1187: additive_rate=24.0% (FAIL, threshold=66%)
- K1188: additive_math=15.0% (PASS, threshold=5%)
- K1189: max_cos_b=0.1607 (PASS, diagnostic)
- all_pass=false
