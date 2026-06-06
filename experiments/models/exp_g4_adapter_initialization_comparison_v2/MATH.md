# exp_g4_adapter_initialization_comparison_v2 — MATH.md

## §0 Skill attestation (reviewer (m2) gate)

`/mlx-dev` invoked at researcher iter (drain-window ~44, 2026-04-25). Fast-mlx deferred — this is an iter-bounded smoke (100 iters/run, 9 runs) reusing v1's `mlx_lm.tuner.trainer.train()` pipeline; no novel hot-path requires `mx.compile`/fast-ops tuning. Phased eviction (`del model + gc.collect() + mx.clear_cache()` between runs) preserved verbatim from v1.

## §1 Inheritance pointer

This experiment is the **v2 follow-up** to `exp_g4_adapter_initialization_comparison` (F#751 PROVISIONAL, parent v1 dir present at `micro/models/exp_g4_adapter_initialization_comparison/`). MATH inherits parent v1 §1-§4 byte-for-byte (recipe: Gemma 4 E4B 4-bit, q_proj r=6 scale=6, medical/train.jsonl, AdamW lr=1e-4, batch=2, max_seq_len=512). v2 delta documented below.

## §2 v2 schema-repair delta (locked KCs)

Parent v1 had K1924 (cross-init final cos Δ > 0.10) confounded by shared `mx.random.key(42)` producing 0.977-0.9995 starting cos-sim across "different" inits. v2 fixes:

1. **Distinct top-level seeds per init**: Grassmannian=42, Kaiming=43, Gaussian=44 (eliminates PRNG sharing). For multi-seed K1979/K1984, three sub-seeds per init: {top, top+10, top+20}.
2. **F#666-paired KC schema** (added 2026-04-25 per drain-window schema-repair):
   - K1977 (proxy): cross-init final |cos| < 0.20
   - K1978 (proxy): final eval-PPL ratio worst/best > 1.10
   - K1979 (proxy): within-init seed-variance on PPL > 5%
   - K1983 (target): cross-init medical-MCQ heldout n=80 spread > 5pp
   - K1984 (target): within-init seed-variance on medical-MCQ > 3pp
   - K1985 (non-interference): any init drops base medical-MCQ > 5pp
3. **Iter count target**: 1000 (parent ran 100, did not confirm convergence). Implementation reality is captured in §6.

## §3 Theorem (carried from parent + v2 extension)

**Theorem (init-invariance under linear LoRA at convergence):** if `B_t @ A_t` (the LoRA effective product) converges to the same low-rank subspace under SGD regardless of init, then both the proxy spread (cos-sim, PPL-ratio, seed-variance) AND the target spread (medical-MCQ accuracy) collapse below their respective F#666 thresholds.

**Predictions:**
- P1: K1977 PASS — cross-init final |cos| < 0.20 (distinct seeds expose genuine separation; v1 confound removed)
- P2: K1978 PASS — final PPL ratio worst/best ≤ 1.10 (within 10% spread per F#169 init-invariance prior)
- P3: K1979 PASS — within-init seed-variance on PPL > 5% (PPL noise floor is large; identifiability check)
- P4: K1983 FAIL — cross-init medical-MCQ spread < 5pp (target init-invariance verified)
- P5: K1984 PASS — within-init seed-variance on medical-MCQ > 3pp (eval has discriminative power)
- P6: K1985 PASS — no init drops base by > 5pp (training is behaviorally productive)

**Verdict logic per F#666 + v2 SC#109:** SUPPORTED iff K1983 FAIL (cross-init behavioral spread tight) AND K1984 PASS (within-init seed-noise present) AND K1985 PASS (non-interference). Otherwise PROVISIONAL or KILLED per truth-table.

## §4 References

- F#751 (parent v1 PROVISIONAL — PRNG confound diagnosis)
- F#562 (Grassmannian QR best practice; what v2 verifies/refutes at convergence)
- F#627 (canonical 1000-iter LoRA recipe for Gemma 4)
- F#666 (target-gated kill rule — proxy KCs must pair to target KCs)
- F#169 (init-invariance prior at LoRA scale)
- F#172 (seed-variance bound on PPL ≈ 5%)

## §5 Antipattern scan

- Composition math: N/A (single adapter per run, no composition).
- LORA_SCALE=20: NO — using scale=6 per F#328/F#330.
- Tautological routing: N/A (no routing).
- shutil.copy as new adapter: NO — fresh `linear_to_lora_layers` per run.
- Hardcoded `pass: True`: NO — verdict computed from measurements.
- Eval-template truncation: NO — medical/valid.jsonl preserves chat template.
- Proxy-model substitution: NO — Gemma 4 E4B 4-bit per parent recipe.
- KC measures wrong object: All six KCs measured exactly as specified.
- Smoke-as-full: §6 explicitly flags smoke iter reduction.

## §6 Honest budget-vs-design

The parent v2 design specifies 1000 iters × 3 inits × 3 seeds = 9 runs at ~28-33 min each = ~4-5h wall-clock. The researcher 90-min iter cap (≈ 40 tool calls) does not accommodate this. Therefore this iteration runs:

- **SMOKE_TEST=1** (default this iter): 100 iters × 3 inits × 3 seeds = 9 runs at ~3 min each ≈ 27 min training + 9 medical-MCQ evals at ~1.5 min each ≈ 14 min. Total ~40-50 min wall-clock.
- `is_smoke: true` set in results.json. Verdict floor = PROVISIONAL (per researcher hat clause 6.4 + verdict-consistency rule 4).
- Marginal value-add over parent v1 PROVISIONAL: (a) PRNG-confound fix verified at smoke iter; (b) seed-variance lower bound; (c) MCQ-heldout eval (parent had PPL-only).

**Reclaim path for v3 (full 1000-iter run):**
1. Schedule dedicated 4-5h compute session.
2. `experiment update <v3-id> --priority 2`.
3. Set `SMOKE_TEST=0` env var → ITERS=1000.
4. Run, complete with --status supported|killed per F#666 verdict matrix.

If the smoke-iter measurements at 100 iters already KILL one of the F#666 cells (e.g. K1985 non-interference fails sharply with all 3 inits dropping > 5pp), v3 may be unnecessary — the recipe is already disqualified. If smoke is consistent with the supported pattern (K1977 small, K1983 small, K1985 PASS), v3 confirms at convergence.
