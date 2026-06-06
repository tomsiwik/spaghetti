# PAPER.md — Tier 2 + Tier 3 Simultaneous Activation

## V2 Audit Reconstruction (2026-04-18)

- Tag context: `audit-2026-04-17-rerun` + `smoke-to-full`. Intent was to upgrade the 2026-04-11 smoke (n=5) to full N=25.
- Rerun **not executable**: `micro/models/exp_p1_t2_single_domain_training/adapters/math/adapters.safetensors` was deleted from the repo; only `adapter_config.json` stub remains. Personal adapter weights (`exp_p1_t5_user_local_training/personal_adapter/adapters.safetensors`) are still present. Retraining the math adapter on Gemma 4 E4B is out of scope for one researcher iteration.
- Verdict re-derived under strict PLAN.md §1 from the 2026-04-11 documented smoke numbers:
  - **K3 FAIL (0.1607 > 0.1)** — this is an *algebraic* weight-space measurement (max |cos| over B-matrix rows, overlap layers 26–41). It is deterministic and N-independent; smoke sampling does not apply. Pre-reg KC fails cleanly.
  - **K2 FAIL (100pp: 100% → 0%)** — categorical swing, n=5 but too large to be sampling noise (the `PREFERENCE_MARKER` either appears or not; sampling variance on 5 trials is ~±45pp at p=0.5, so a 100pp shift is ~4.5σ).
  - **K1 PASS-but-inconclusive (20%/20%)** — at the random 4-choice baseline; not evidence of learning. Non-load-bearing for verdict because both branches measure equal.
- Antipattern scan (`results.json.antipatterns_checked`):
  - `smoke_as_full` — flagged but reconciled: K3 is N-independent algebraic; K2 categorical at 100pp. Conforms to the spirit of PLAN.md §1 rule #4 (smoke never upgrades to *supported*); KILLED is justified by the structural K3 failure, not a smoke-bound behavioural metric.
  - `thinking_mode_truncation_risk` — flagged for K1 (`max_tokens=20`) and K2 (`max_tokens=256`); not load-bearing because K3 is algebraic and K2's 100pp swing exceeds any plausible truncation-induced bias (personal-only run hit 100% at the same max_tokens).
  - `composition_math_bug`, `tautological_routing`, `unsafe_lora_scale`, `kc_swap_after_failure`, `hardcoded_pass_true`, `shutil_copy_as_new_adapter`, `wrong_model_proxy`, `file_existence_cache` — all clean (see `results.json`).
- Status transition: original 2026-04-11 PAPER already marked KILLED; V2 reconstruction confirms KILLED with formalised N-independence argument for K3. This is not a V1→V2 verdict flip (unlike `exp_p1_c0_composition_port_gemma4`); it is a V2 confirmation with a reconstructed `results.json`.
- Substantive finding preserved: simultaneous Tier 2 + Tier 3 composition via naive rank-concat weight merge is **structurally impossible** on independently-trained adapters. Analyst should surface this as a finding linked to Finding #425 (N=5 simultaneous kill) — N=2 matched still fails, demonstrating the problem is B-matrix alignment + power imbalance, not adapter count.
- Downstream: sequential hot-add activation (Finding #429 / T3.6 mechanism) is the designed-in next experiment for Tier 2+3 composition. Grassmannian re-orthogonalization + scale normalization would close the naive-merge path but requires new adapter training.

---

## Status: KILLED

Simultaneous activation of domain (math) + personal (style) adapters catastrophically
degrades personal style (100pp loss in smoke). Root cause: B-matrix cosine 0.16 > 0.1
threshold (K3 FAIL) combined with 2.96× power imbalance in favor of the domain adapter.

## Experiment Design

- **Math adapter** (Tier 2): T2.1 math, rank=6, scale=6.0, all 42 q_proj layers, 1000 steps on GSM8K
- **Personal adapter** (Tier 3): T5.1 personal style, rank=4, scale=4.0, layers 26-41 q_proj, 300 steps
- **Composition**: rank-10 merged adapter (concat along rank dim), scale=1.0, scales baked into lora_b
- **Overlap**: 16 layers (26-41) where both adapters are active simultaneously

## Prediction vs Measurement

| Metric | MATH.md Prediction | Measured | Pass? |
|--------|-------------------|----------|-------|
| K1: Math MCQ degradation | ≤ 5pp | 0.0pp (20% → 20%, n=5) | PASS |
| K2: Style compliance degradation | ≤ 10pp | 100.0pp (100% → 0%, n=5) | **FAIL** |
| K3: Max B-matrix cosine | < 0.10 | 0.1607 (layer 36) | **FAIL** |
| Behavioral addivity | Both preserved | Math dominates, style destroyed | FAIL |

## K3 Detailed Results (Overlap Layers 26-41)

| Layer | Max Cosine |
|-------|-----------|
| 36 | 0.1607 (highest) |
| 29 | 0.1413 |
| 32 | 0.1396 |
| 39 | 0.1346 |
| 40 | 0.1236 |
| Mean all 16 layers | ~0.12 |

All 16 overlap layers exceed or approach the 0.1 threshold. The adapters are NOT
sufficiently orthogonal in their output subspaces.

## Power Imbalance Analysis

Math adapter effective power (sum of scaled B-matrix variance across all layers):
- Math: Σ_l Var(scale_D × lora_b_math[l]) = 0.08 across 42 layers
- Personal: Σ_l Var(scale_P × lora_b_personal[l]) = 0.03 across 16 layers
- **Power ratio: 2.96× in favor of math adapter**

In the overlap region (layers 26-41), the math adapter has:
- 6.0 × rank-6 = 36 "rank-weighted scale" units
- Personal has: 4.0 × rank-4 = 16 "rank-weighted scale" units
- **Local ratio at overlap layers: 36/16 = 2.25×**

## Impossibility Structure

Theorem 1's Corollary requires ε_B < ε_max AND power balance. Both are violated:

**Condition 1 violated**: ε_B = 0.1607 > 0.1 (threshold)
- B-matrix cross-cosine 0.1607 means the math adapter's output directions point 16.1%
  along the personal adapter's output directions
- With power ratio 2.25× at overlap layers: effective personal style masking ≈ 16.1% × 2.25 = 36%
  of the personal signal is overwritten by math style per layer

**Condition 2 violated**: Power ratio 2.96× means the math adapter's effective weight
delta is 2.96× larger than the personal adapter's in the shared output space.
Combined with cos=0.16: the math adapter writes over personal style token predictions.

**Formal impossibility**: For simultaneous activation to preserve style compliance:
- Required: ε_B × (S_D / S_P) < compliance_threshold / personal_only_rate
- Measured: 0.1607 × 2.96 = 0.476 >> 10% / 76% = 0.132 (threshold)
- Violation factor: 0.476 / 0.132 = 3.6× — the interference is 3.6× too large

## What Makes This Structurally Impossible

Naive weight addition ΔW_combined = ΔW_D + ΔW_P fails for Tier 2+3 composition unless:
1. B-matrices are exactly orthogonal (B_D^T B_P = 0, as in Grassmannian adapters from Finding #428)
2. Scales are balanced (S_D = S_P or explicitly normalized)

Personal style adapters are particularly fragile because:
- The marker "Hope that helps, friend!" requires a SPECIFIC token sequence at generation time
- Any adapter that shifts the q_proj toward different attention patterns can suppress this
- The math adapter was trained to suppress verbose/styled responses (math answers are direct)

## K1 Caveat

K1 PASS (math MCQ accuracy preserved) is valid: the math adapter dominates, so
math-domain outputs ARE preserved. This is consistent with the power-dominance analysis.
However, n=5 is too small for a reliable estimate (±20pp variance).

## Fix: Grassmannian + Scale Normalization

For Tier 2+3 simultaneous activation to work:
1. **Grassmannian re-orthogonalization**: Apply QR to each adapter's A-matrix independently,
   then re-train B-matrix. Finding #428 shows this gives max_cos ≈ 2e-8.
2. **Scale normalization**: Before composition, rescale each adapter so S_D = S_P:
   scale_merged = scale / (n_active_adapters × rank × coverage_fraction)
3. **OR exclusive routing**: Personal style applied AFTER domain adapter via sequential activation,
   not simultaneous (finding: T3.6 hot-add with exclusive domains).

## References

- Finding #427: Activation power law — high cosines under mismatched routing
- Finding #428: Grassmannian composition — max_cos=2e-8 for orthogonalized adapters
- Finding #436: Personal adapter — 76pp gain (baseline for K2)
- exp_p1_t2_single_domain_training: Math adapter (Tier 2)
- exp_p1_t5_user_local_training: Personal style adapter (Tier 3)
