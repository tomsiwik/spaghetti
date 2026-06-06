# REVIEW-adversarial.md — exp_g4_adapter_initialization_comparison

**Reviewer verdict: PROVISIONAL**
**Route: review.proceed (PROVISIONAL prefix)**

## Checklist (adversarial)

| item | result |
| --- | --- |
| (a) results.json["verdict"]=="PROVISIONAL" vs DB status "provisional" | CONSISTENT |
| (b) all_pass=false with status provisional (not upgraded to supported) | CONSISTENT |
| (c) PAPER.md verdict line says PROVISIONAL | CONSISTENT |
| (d) is_smoke=false; claim is 100-iter run pre-registered in MATH.md §6 | OK |
| (e) git-diff of MATH.md after first run | untracked dir, no mutation window → PASS |
| (f) tautology on KCs | K1924 = maxΔ cross-pair cos; K1925 = PPL ratio + best-init identity. Neither passes by identity. PASS |
| (g) K-ID vs code | run_experiment.py §kill_criteria computes both as MATH.md §5 specifies. PASS |
| (h) composition math in code | no composition; single-adapter runs only. PASS |
| (i) LORA_SCALE | 6.0 (F#627-canonical). PASS |
| (j) per-sample routing | no routing code. N/A |
| (k) shutil.copy | none. PASS |
| (l) hardcoded `pass` KC | none; both KCs computed from tensors. PASS |
| (m) target-model match | `mlx-community/gemma-4-e4b-it-4bit` in both MATH.md §0 and run_experiment.py BASE_MODEL. PASS |
| (m2) platform skill invocation evidence | MATH.md §5 pre-flight checklist explicitly cites `/mlx-dev, /fast-mlx`. Code uses `mx.eval`, `mx.clear_cache()`, `gc.collect()` between inits, `linear_to_lora_layers`, canonical `train()` API, `grad_checkpoint=True`. Idiomatic MLX. PASS |
| (n) base eval truncated | base PPL = 2070.79 from actual `default_loss` eval on the same val_set. PASS |
| (o) headline n | n=30 val rows, n=1 seed, n=100 iters — small, but PROVISIONAL self-acknowledges; not a STATS kill on a provisional filing |
| (p) synthetic padding | none |
| (q) cited baseline drift | baseline measured this run |
| (r) prediction-vs-measurement table in PAPER.md | present (§"Prediction vs Measurement", 4 rows P1-P4). PASS |
| (s) math/sci errors | see §Errors below |
| (t) target-gated kill (F#666) | K1925 target PASS (Δppl=3.5% < 5%); K1924 proxy FAIL (Δcos=0.068 < 0.10). Proxy-FAIL + target-PASS → finding about the proxy, not a kill. PROVISIONAL is correct. |
| (u) scope-changing fix | iters=100 was pre-registered in MATH.md §6 "iters reduced from 1000 for budget" — not a silent mid-run scope cut. PASS |

## Errors / caveats

1. **Hidden-size typo in MATH.md §0 / §3**: MATH.md writes `hidden_size=2048` and `d_in=d_out=2048`. Gemma 4 E4B is d=2560 (PAPER.md §0 uses 2560 correctly; PAPER.md §Clean-metric formula `1/sqrt(2560/6) ≈ 0.05` matches the measured Gaussian intra-|cos|=0.015–0.037). Code reads `in_dims` dynamically from `layer.lora_a.shape`, so actual runs are unaffected. Non-blocking; should be corrected in v2 MATH.md.

2. **P3 FAIL (train-loss ratio 1.17 vs 1.10 threshold)** is honestly reported in PAPER.md §"Why P3 fails" — Grassmannian's uniform singular values delay B-concentration. Not a verdict-changing issue (verdict tracks K1925 target, not P3 which was a supplementary prediction).

3. **PRNG confound (K1924)** is the core reason for PROVISIONAL. Researcher self-surfaces it in PAPER.md §3; intra-init metric (confound-free) shows Grassmannian intra-|cos|=0.032 vs Kaiming 0.090 — 2.8× cleaner. Under a distinct-per-init-seed design K1924 would be a real test of cross-init structural divergence.

## What would flip the verdict

- If v2 with distinct per-init seeds shows cross-init final cos < 0.10 across all pairs → K1924 PASS under the clean design → SUPPORTED.
- If v2 multi-seed variance on PPL > 3.5% → K1925 claim widens caveat ("within noise") but PROVISIONAL stands.

## Assumptions (reviewer-logged)

- Gemma 4 E4B is d=2560 per MLX_GEMMA4_GUIDE memory and PAPER.md; MATH.md's 2048 is treated as a typo, not a proxy-substitution per (m).
- Intra-init column-|cos| metric is accepted as a confound-free structural probe supplementing the K1924 pre-reg (K1924 itself was mis-designed but the researcher disclosed this rather than rewriting the KC post-claim).
- Single seed is acceptable for a PROVISIONAL; SUPPORTED would require multi-seed.

## Blocking fixes for v2 follow-up (not this experiment)

1. Use distinct per-init top-level seeds (42 / 43 / 44) so cross-init cos at init is near √(r/d)≈0.054 rather than 0.98–0.9995.
2. Register each init's seed as a CONFIG field with a per-init key split.
3. Add ≥3 seeds per init to bound seed-noise on PPL.

## Provenance

- MATH.md present with theorem (§3), predictions (§4), KCs (§5), skill pre-flight (§5).
- PAPER.md present with TL;DR, prediction-vs-measurement table, confound discussion, verdict reasoning.
- run_experiment.py: idiomatic MLX; KC logic in main() follows MATH.md §5 truth table.
- results.json has verdict, all_pass, is_smoke, both KCs with proxy/target type tagging.
- LEARNINGS.md lists three novel findings (FNa behavioral / FNb Grassmannian-persist / FNc antipattern PRNG-share).
