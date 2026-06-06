# LEARNINGS: P7.A1 — Null-Space Adapter Quality

## Audit (2026-04-17): Verdict revised SUPPORTED → KILLED (metric-swap)
DB KC #1297 pre-registers GSM8K accuracy and #1298 pre-registers MMLU; code+MATH.md measure training-loss ratio on 20 memorized math texts and PPL on 5 hand-curated general-knowledge prose snippets. Pre-flight antipattern #6 ("KC measures wrong object") applies. The "98.7% quality preservation" claim is NOT credited — training loss at memorization scale (PPL=1.03 both adapters) is mechanically ~1.0 regardless of null-space effect. K1299 orthogonality is the only KC that passed on its registered metric. V2 must pre-register GSM8K+MMLU at MATH.md time and train on non-trivial data (GSM8K train split, 1000+ steps).

## Core Finding (NOT CREDITED AS KC PASS — proxy metric, see audit above)
Null-space LoRA (A = A_null @ Q^T, arXiv:2512.15233) preserves 98.7% of unrestricted adapter quality while guaranteeing exact orthogonality to W_v (max violation 1.33e-5, 100x below threshold). The null-space restriction is essentially free: same final loss (0.037), same convergence rate, 20% fewer parameters. **Caveat: this measurement is at memorization scale (20 texts, 500 iters, both PPL=1.03 on train); whether null-space restriction preserves behavioral quality (GSM8K, MMLU) is UNKNOWN and is the purpose of the V2 experiment.**

## Reusable Behavioral Findings (credited — independent of metric-swap)
- **K1299 PASS (only credited KC):** Null-space reparameterization achieves exact orthogonality to W_v across 8 non-shared layers (max violation 1.33e-5, ~100x below 1e-4 threshold). Proved by Theorem 1 (Q from SVD null basis → W_v @ Q = 0 by construction); verified empirically.
- **Gemma 4 E4B KV-sharing (mandatory architectural check):** Layers 24-41 receive pre-computed KV from layers 22/23 via `shared_kv`. On those layers, v_proj is dead code — any LoRA has zero effect (zero gradient, zero logit delta). Detected via `previous_kvs` mapping + logit-delta probe. **All future Gemma 4 adapters on k_proj/v_proj MUST target layers 16-23 (non-shared) only.** First run of this experiment targeted 34-41 and produced vacuously identical loss curves before this was diagnosed.
- **Null-space parameter efficiency:** At rank 16, null-space adapter trains 327K params vs unrestricted 409.6K (20% reduction) without slowing convergence at memorization scale.

## V2 Design Requirements (exp_p7_null_space_adapter_quality_v2)
- Pre-register K1297 as lm-eval-harness GSM8K accuracy ratio, N>=100, at MATH.md time. No post-hoc proxy.
- Pre-register K1298 as MMLU-Pro accuracy delta < 1pp, N>=200 disjoint held-out questions, at MATH.md time.
- Train on GSM8K train split (or equivalent non-trivial domain), 1000+ iters, so loss is informative beyond memorization.
- Keep K1299 orthogonality check (currently valid).
- Target layers 16-23 only (non-shared) — mandatory per KV-sharing discovery.

## Why It Works (theorems still hold)
SVD of W_v yields a 2048-dim null space where gradients are unrestricted (Theorem 2: gradient retention ratio d_null/d_in = 0.80 lower bound). The reparameterization A = A_null @ Q^T ensures W_v @ A_eff^T = 0 by construction — no learned penalty, no regularization needed. Theorems are unaffected by the metric-swap; only the behavioral claims (K1297/K1298) are unsupported pending V2.

## Why It Works
SVD of W_v yields a 2048-dim null space where gradients are unrestricted (Theorem 2: gradient retention ratio d_null/d_in = 0.80 lower bound). The reparameterization A = A_null @ Q^T ensures W_v @ A_eff^T = 0 by construction — no learned penalty, no regularization needed.

## Critical Discovery: Gemma 4 KV-Sharing
Layers 24-41 of Gemma 4 E4B receive pre-computed KV from layers 22/23 via `shared_kv`. v_proj is **dead code** on those layers — adapters have zero effect. **Any future experiment targeting k_proj/v_proj on Gemma 4 must verify layers 16-23 (non-shared) only.**

## Implications for Next Experiment
P7.A2 (two null-space adapters on the same layer) is now unblocked. Capacity is ample: 2048 null dims / r=16 = 128 non-overlapping slots per layer, covering 25+ domains with 5x headroom. The remaining question is whether orthogonality in weight space translates to functional independence in activation space.

## Caveats
- Memorization scale (20 texts, PPL=1.03) — ratio may differ on harder tasks at larger data scale
- K1298 vacuous (base PPL=8154); null-space general PPL (362) is 44.7% worse than unrestricted (250) — relevant for composition
- P2 (post-hoc projection) untested; separate experiment if needed
