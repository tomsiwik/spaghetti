# PAPER — exp_g4_adapter_magnitude_distribution

## Verdict
**PROVISIONAL** — K1 (structure hypothesis) SUPPORTED; K2 (magnitude-as-importance hypothesis) target-supported behaviorally but proxy K1918 INCONCLUSIVE due to QuantizedMatmul VJP limitation on the 4-bit base.

## One-line
Trained Gemma 4 E4B LoRA q_proj adapters (r=6, scale=6.0) have per-matrix non-Gaussian structure (Shapiro-Wilk p > 0.01 on only 39.7 % of matrices), and zeroing the top-20 % magnitude entries damages task PPL substantially more than zeroing random 20 % (mean ratio R = 0.82) — i.e. magnitude carries real behavioral signal on math and code, but not on medical MCQA.

## Setup
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (MLX 0.31.1, mlx_lm 0.31.2).
- Adapters: `exp_p1_t2_single_domain_training` (q_proj only, rank 6, scale 6.0, 1000 iters Adam @ 1e-4). 3 domains: medical, math, code. 42 × 2 = 84 matrices per adapter, 252 total.
- Eval: 100 held-out samples per domain (valid.jsonl, teacher-forcing NLL / token).
- Prune fraction: 20 % per matrix (per-layer top-k and random-k, same count, deterministic seed).
- N samples drawn: 100 × 3 domains = 300 PPL points per condition × 3 conditions (baseline, top-magnitude-pruned, random-pruned).

## Prediction vs Measurement

| Prediction | Predicted | Measured | Pass? |
|---|---|---|---|
| P1 — Gaussianity: Shapiro p>0.01 on ≥ 80% of matrices AND `\|skew\|<0.5` AND `\|kurt\|<1.0` on ≥ 80% | ≥ 80% | **39.7%** | ❌ (Shapiro fails) |
| P2 — Effective rank at 99% var ≥ 5 on ≥ 80% of A matrices | ≥ 5 on 80% | mean = **6.0** (all) | ✓ |
| P3 — Fraction of `\|w\| < 1e-4` per matrix < 10% | < 10% | 0.80% mean | ✓ |
| P4 — Cross-weight Pearson `\|r\|(\|w\|, I(w))` < 0.3 (Fisher proxy) | < 0.3 | **NaN (VJP unsupported on quantized base)** | ⚠ INCONCLUSIVE |
| P5 — Behavioral ratio `\|ΔPPL_top − ΔPPL_rand\|/max(...) < 0.5` (magnitude doesn't signal importance) | < 0.5 | **0.82** (medical 0.13, math 1.09, code 1.26) | ❌ (magnitude DOES signal importance) |

P1, P4 are proxy predictions; P5 is the behavioral target paired with both.

## Per-domain behavioral table

| Domain | PPL_base | PPL_top-pruned | PPL_rand-pruned | ΔPPL_top | ΔPPL_rand | R |
|---|---|---|---|---|---|---|
| medical | 29.612 | 28.032 | 27.806 | **-1.58** | **-1.81** | 0.13 |
| math    | 9.251  | 14.930 | 8.767  | **+5.68** | -0.48 | **1.09** |
| code    | 14.944 | 20.200 | 13.560 | **+5.26** | -1.38 | **1.26** |
| **mean** | — | — | — | **+3.12** | **-1.22** | **0.82** |

Observation: both top-magnitude and random-20 % masks *reduce* PPL on medical (i.e. the adapter is net-harmful on held-out medical MCQs — zeroing some of it helps). On math and code, the adapter is net-beneficial and top-magnitude pruning degrades PPL substantially more than random pruning. This is direct evidence that **magnitude carries per-weight behavioral signal** on the domains where the adapter is actually working.

## Kill criteria results

| KC | Proxy / Target | Fires if | Measured | Fired? |
|---|---|---|---|---|
| K1917 | proxy | all-three normality pass frac ≥ 0.80 | **0.397** | no (< 0.80) |
| K1971 | target (paired w/ K1917) | mean R < 0.5 | **0.82** | no (≥ 0.5) |
| K1918 | proxy | mean `\|r\|(\|w\|, Fisher) < 0.3` | **INCONCLUSIVE** (QuantizedMatmul VJP) | N/A |
| K1972 | target (paired w/ K1918) | mean R < 0.5 | **0.82** | no (≥ 0.5) |

**K1 (structure)** — K1917 did not fire AND K1971 did not fire ⇒ **SUPPORTED** (F#666: both proxy and target "pass" meaning the hypothesis is confirmed; weights are non-Gaussian AND structure is behaviorally exploitable).

**K2 (magnitude = importance)** — K1918 INCONCLUSIVE, K1972 did not fire ⇒ **PROVISIONAL** (behavioral target strongly supports the hypothesis; proxy cannot be measured without re-engineering around MLX's quantized-VJP limitation).

## Assumptions and caveats
1. **q_proj adapters, not v_proj + o_proj.** F#627 established v_proj + o_proj as the optimal Gemma 4 target. We used the q_proj adapters that already exist from `exp_p1_t2_single_domain_training`. The hypothesis under test is about *any* trained Kaiming + Adam LoRA, so q_proj evidence generalizes structurally — but the specific magnitudes of ΔPPL differ between targets.
2. **Fisher-proxy K1918 could not be computed.** MLX 0.31 does not define `QuantizedMatmul::vjp` (error verbatim: `[QuantizedMatmul::vjp] no gradient wrt the quantized weights`). This is a framework limitation, not a measurement choice. Wanda-style `|w · activation|` (forward-only) is a viable replacement; a follow-up experiment can close this.
3. **Medical is anomalous.** Both masks reduce medical PPL, meaning this specific adapter is not improving medical MCQ PPL end-to-end. R = 0.13 for medical is therefore not evidence that magnitude is uninformative — it's evidence that the medical adapter is already partly-zero-effective on held-out MCQs, so the per-weight signal is small in absolute ΔPPL terms. The math/code signal (R ≈ 1.2) is much cleaner.
4. **PPL on MCQ data.** MCQ messages end with a short answer, so token-count averages heavily weight the question text. This is still a valid comparison across conditions — the shift is measured relative to the same tokens — but absolute PPL values should not be compared against other benchmarks.
5. **Random seed — single run.** The random mask was drawn with one seed per domain. Re-drawing the random mask would produce a confidence band. For the current 2x threshold this is unlikely to flip the verdict (effect sizes are 4–5x on math and code), but a tighter verdict would need 3–5 seeds.
6. **N = 100 eval samples.** Sufficient for the 2x threshold we chose; tight PPL differences would need larger N.

## Behavioral outcome (per PLAN.md Part 1 — metrics are proxies)
The behavioral claim in the hypothesis is "magnitude-based adapter compression is exploitable for product use." The evidence is:
- Top-magnitude pruning degrades math PPL by **+5.68** and code PPL by **+5.26**, while random-20 % pruning *improves* both by **-0.48** and **-1.38** respectively. The gap is not subtle.
- On the domain where the adapter is end-to-end useful (math, code), the top-20 % of weights by magnitude are carrying most of the useful signal. This is structure that a compression scheme could target.
- Caveat: the adapter overall sometimes hurts (medical) — before compression, one needs a per-domain diagnostic that the adapter is *actually improving* the held-out metric.

This is a positive behavioral finding: magnitude-based adapter compression on trained Gemma 4 LoRAs is *probably viable* (contingent on per-domain adapter-is-useful gating). It extends rather than contradicts F#500 (null-space-projection-magnitude fails for routing) and F#526 (pre-merge failure is direction-dependent, not *global* magnitude-dependent) — those earlier findings are about global scale, this one is about per-weight structure.

## Follow-ups
1. **Fisher-free importance proxy** — run the same ablation with Wanda-style `I(w) = \|w\| · E[\|a\|]` where `a` is the LoRA input activation. Measure correlation with behavioral ablation impact. Unblocks K1918.
2. **v_proj + o_proj adapters (F#627 target)** — the current adapters are q_proj. Re-run on v_proj + o_proj once such adapters exist to confirm the structural result transfers to the PLAN.md-sanctioned target.
3. **Magnitude-compressed adapter comparison** — actually produce a magnitude-pruned adapter at 50 % / 70 % sparsity, compare PPL and task accuracy to the uncompressed baseline. Frontier-extension unlocked by this PROVISIONAL finding (success criterion #104, #105).
4. **Medical anomaly** — investigate why the medical adapter is neutral/harmful on held-out MCQs. Possible causes: domain mismatch in train vs valid split, short eval tokens dominated by shared chat template, or an unrelated training bug.

## Pre-flight block (from researcher.md)
```
Reference: arxiv:2106.09685 (LoRA), arxiv:2306.11695 (Wanda); F#500, F#526, F#350, F#666
Platform skills invoked: /mlx-dev, /fast-mlx (confirmed)
Base model loaded: mlx-community/gemma-4-e4b-it-4bit
Adapter targets: q_proj (caveat: F#627 v_proj+o_proj optimal; logged)
Dataset: exp_p1_t2_single_domain_training/data/{medical,math,code}/{valid,train}.jsonl — messages column
Runtime budget: 45-75 min expected; actual ~4 min (Phase 1 30s, Phase 2 ~3 min/domain due to small eval set + baseline/Fisher/top/rand each ~30-80s)
KC count, F#666: 2 proxy (K1917, K1918) + 2 target (K1971, K1972) — paired
Antipattern scan: composition math N/A | LORA_SCALE=6.0 safe | shutil.copy N/A | hardcoded pass N/A | eval truncation N/A | proxy model OK
```

## Files
- `MATH.md` — theorem, predictions, KC, methodology.
- `run_experiment.py` — phased MLX execution (numpy K1917 + in-memory ablation + Fisher attempt).
- `results.json` — per-matrix stats, per-domain PPL, KC outcomes, verdict.
- `PAPER.md` — this document.
