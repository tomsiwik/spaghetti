# REVIEW-adversarial.md — exp_composition_residual_analysis

**Reviewer hat, 2026-04-25. Verdict: PROCEED (SUPPORTED).**

## Adversarial checklist

| Item | Check | Result |
|---|---|---|
| (a) results.json verdict vs DB status | SUPPORTED vs `supported` | ✓ |
| (b) all_pass vs claim | `true` vs SUPPORTED | ✓ |
| (c) PAPER.md verdict line | "**Verdict: SUPPORTED.**" — no downgrade lexemes | ✓ |
| (d) is_smoke consistency | `false`, full-run claim | ✓ |
| (e) KC post-hoc modification | MATH.md untracked at runtime; thresholds 0.10/0.10 locked in DB pre-claim | ✓ |
| (f) Tautology sniff | K1926 measures hidden-state R; K1927 compares composed PPL vs adapter-i-alone PPL on domain-i held-out. Neither is `x==x` or single-adapter "composition" | ✓ |
| (g) K-ID ↔ MATH.md mapping | K1926 → `tau_final_layer`; K1927 → `behavioral_delta_max` — both match MATH.md §5 operationalization | ✓ |
| (h) Composition-math bug (sum(lora_A), add_weighted_adapter linear, indep safetensor sum) | NONE. r=18 stacked LoRALinear; `assert_concat_equiv` runtime check rel-diff = **6.28e-08** (FP noise) | ✓ |
| (i) LORA_SCALE ≥ 12 | `SCALE=6.0` (F#627 safe) | ✓ |
| (j) Single-sample routing | N/A — no routing | ✓ |
| (k) `shutil.copy` as new adapter | NONE — `train()` writes 3 distinct adapter files | ✓ |
| (l) Hardcoded `"pass": True` | NONE — results derived from `tau_final > 0.10` and `delta_max > 0.10` | ✓ |
| (m) Proxy-model substitution | `BASE_MODEL = "mlx-community/gemma-4-e4b-it-4bit"` matches MATH.md §0; no fallback | ✓ |
| (m2) Skill invocation evidence | MATH.md pre-flight cites `/mlx-dev`, `/fast-mlx`; code uses `mx.eval`, `mx.clear_cache`, `nn.losses.cross_entropy`, `mlx_lm.train` idiomatically | ✓ |
| (n) Base=0% / `avg_thinking_chars=0` truncation | Base PPL medical=2213.7 / code=9.93 / math=8.53 — non-degenerate | ✓ |
| (o) Headline n ≥ 15 | 5471 non-pad tokens across 45 batches × 3 domains | ✓ |
| (p) Synthetic padding | NONE — 3 distinct domains, distinct seeds (42/1337/2718) | ✓ |
| (q) Baseline drift | Base measured in-run, not cited | ✓ |
| (r) Prediction-vs-measurement table | Present, 4 rows covering P1–P4 (P4 flagged DEFERRED, non-blocking) | ✓ |
| (s) Math / unsupported claims | Theorem §3 gives LayerNorm/softmax/SiLU cross-term expansion; conclusions conservative (K1927 clears threshold by 2× even after removing floor-amplified medical) | ✓ |
| (t) Target-gated kill (F#666) | K1926 (proxy, structural) + K1927 (target, behavioral) properly paired; both PASS → safe SUPPORTED | ✓ |
| (u) Scope-changing fixes | NONE — ran as pre-registered, no silent scope reduction | ✓ |

## Adversarial re-interpretation

- **Could this be KILLED?** τ=0.48 and Δ_max≥0.21 across all domains — both KCs exceed thresholds by ≥2×. KILLED requires both FAIL. Excluded.
- **Could this be PROVISIONAL?** Both KCs pass; `adequately_trained=true` (min lift 79%); `is_smoke=false`. None of the PROVISIONAL triggers apply.
- **Medical Δ=2.19 floor-amplification**: well-documented in PAPER.md L3. Code (0.21) and math (0.37) alone clear K1927 by ≥2×, so the verdict survives even under the most hostile reading that discards medical.

## Non-blocking notes (carried into LEARNINGS.md)

1. **L1 — single-layer probe.** τ measured at final hidden state only; P4 depth-monotonicity deferred. Clean follow-up (`exp_composition_residual_layerwise`): compute τ per-layer to verify O(L) compounding prediction from MATH.md §3.
2. **L2 — single seed per adapter.** No variance bound on τ across seed-triplets. Within-experiment consistency (τ ∈ [0.474, 0.557] across 3 domains) is reassuring but not a variance estimate. Clean 3-seed-triplet replication is a cheap follow-up.
3. **L5 — q_proj-only target.** F#627 also supports v_proj+o_proj; generalization to other target sets not measured here.

## Assumptions

- F#752 already registered by researcher (verified via `experiment finding-get 752`). Reviewer does not add a duplicate finding.
- DB status already `supported` (completed by researcher with K1926/K1927 both pass). Reviewer PROCEED routes to Analyst for LEARNINGS.md, not back through `experiment complete`.

## Verdict

**PROCEED.** Both pre-registered KCs pass with large margins; antipattern matrix clean; target-gated kill discipline honored; math sound; limitations transparently logged with concrete follow-ups. F#302/F#334 are quantitatively replicated at Gemma 4 E4B 4-bit — first numeric τ at current target platform.
