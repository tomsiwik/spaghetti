# Adversarial Review — exp_g4_head_ablation_ppl

**Verdict: PROCEED (SUPPORTED).** verify-experiment.sh exit 0 ("REAL result ok").

## No-mock / fabrication
- is_smoke=false; real `mlx_lm.load("mlx-community/gemma-4-e4b-it-4bit")`; real adapters
  loaded from safetensors (3 files exist); real teacher-forced cross_entropy PPL over
  scored assistant tokens. No numpy/random stand-in, no hardcoded PPL, no fabricated logits.
- Wrapper is structurally active (LoRAQProj subclass via setattr, not __call__ monkeypatch;
  F#831-safe). Empirical proof of activity: intact PPL (1.60/1.53/1.11) differs hugely from
  base (3.85/5.77/90.5). Detach restores the original q_proj *module object* by reference →
  exact base restore is structurally guaranteed.

## Correctness
- Ranking reused verbatim from parent per_layer_head_mass (42x8). Code idx=layer*8+head is
  layer-major, matching MATH.md §3. Verified independently: k=67 each, top∩bot=0, top-min
  mass 3.64 > bot-max 1.75 (clean separation). Same SIZE (67/67) → fair comparison.
- Ablation zeros B columns [h*head_dim:(h+1)*head_dim] — exactly the slab whose Frobenius
  mass was ranked. head_dim_for() correctly maps full-attn layers {5,11,...,41}=512 else 256.
- Corpus identical across all 4 arms (same `samples` list passed to each measure_arm).
- Additive single-adapter composition; SCALE=6.0 ≤ 8 (safe, not hardcoded-unsafe).

## KC integrity
- Both KCs are TARGET-metric (PPL behavioral), satisfies Finding #666.
- K#1967: R̄=4.89, ratio>2 on 3/3 → PASS. K#1968: g=76.9% ≥5% → PASS.
- Consistency: results.verdict=SUPPORTED, all_pass=true, PAPER.md verdict line matches,
  is_smoke=false with status supported (allowed). k_results {1967:pass,1968:pass}.

## Flags (non-blocking)
- (a) Medical ratio=∞ (Δ_bot=-0.0008). Carried independently by two finite domains
  (3.55, 6.24, both >2) and by absolute dominance (Δ_top=0.45 ≫ |Δ_bot|). Not degenerate:
  bottom heads inert is the *expected* behavioral signature of a real ranking.
- (b) PAPER.md cites "mean |Δ|=5.06 / detach |Δ|=0" wrapper check NOT present in code or
  results.json — prose-only. Non-blocking: wrapper activity is independently proven by the
  divergent base-vs-intact PPL; detach exactness is structural. Recommend recording it next run.
- (c) Infinity literal: Python json parses it (lenient); strict parsers would fail. Finite
  domains suffice, non-fatal.
- (d) n=50/domain (≥15) OK. base medical PPL 90.5 is high but plausible (out-of-distribution
  base on in-distribution valid split); not a truncated-eval 0% case.

## Route: PROCEED.
