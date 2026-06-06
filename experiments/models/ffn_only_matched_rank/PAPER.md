# FFN-only Matched Rank Validation: Research Digest

## Hypothesis

Independently trained FFN-only LoRA adapters (gate_proj, up_proj, down_proj)
match all-modules quality at rank-16 and preserve the orthogonality properties
observed in retroactive subset analysis.

**Falsifiable**: Kill if (1) FFN-only PPL >5% higher than all-modules at rank-16,
or (2) independently trained FFN-only orthogonality differs >50% from retroactive
subset.

## What This Experiment Is

A macro-scale validation of the architecture pivot to FFN-only experts.
The prior experiment (ffn_only_vs_all_modules) showed FFN-only adapters are
more orthogonal (mean |cos| 0.0605 vs 0.0711), but used a confounded methodology:
FFN parameters were retroactively extracted from jointly-trained all-modules
adapters. This experiment eliminates the confound by training FFN-only adapters
from scratch.

Three components:
1. **Training** (GPU, RunPod): 5 FFN-only rank-16 adapters on Qwen2.5-7B
2. **Quality comparison** (CPU): per-domain PPL vs existing all-modules adapters
3. **Orthogonality comparison** (CPU): independent vs retroactive cosine matrices

## Lineage in the Arena

```
lora_gpt (MLP-only LoRA, rank 8, micro)
 |-- ffn_only_vs_all_modules (retroactive subset analysis, macro adapters)
      \-- ffn_only_matched_rank (THIS: independent training validation)
```

## Key References

- Geva et al. 2021, "Transformer Feed-Forward Layers Are Key-Value Memories"
- Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large Language Models"
- Prior experiment: micro/models/ffn_only_vs_all_modules/PAPER.md

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-7B (4-bit QLoRA) |
| Rank | 16 |
| Alpha | 16 |
| Steps | 300 |
| Effective batch | 8 (micro=1, grad_accum=8) |
| LR | 2e-4 |
| Optimizer | AdamW 8-bit |
| Seed | 42 |
| Domains | bash, math, medical, python, sql |
| Train data | 1000 examples/domain (distillation data) |
| Eval data | 50 examples/domain |

FFN-only target modules: gate_proj, up_proj, down_proj (30.3M params)
All-modules target modules: + q_proj, k_proj, v_proj, o_proj (40.4M params)

## Baseline: Existing All-Modules Adapter Orthogonality

Confirmed by re-analysis of existing adapters (matches prior experiment):

| Configuration | Mean |cos| | Max |cos| | Source |
|---------------|-----------|-----------|--------|
| All-modules (full) | 0.0711 | 0.7027 | adapters/ |
| FFN subset (retroactive) | 0.0605 | 0.5898 | adapters/ FFN keys |
| Attn subset (retroactive) | 0.0853 | 0.8501 | adapters/ attn keys |

The math-medical pair dominates: cos=0.59 (FFN), 0.85 (attn), 0.70 (full).
All other pairs have |cos| < 0.003.

## Empirical Results

### Quality Comparison (PPL)

**STATUS: AWAITING GPU TRAINING**

Training not yet completed. The training script is ready at
`micro/models/ffn_only_matched_rank/train_ffn_only.py`. Run on RunPod A5000
via `bash micro/models/ffn_only_matched_rank/run_on_runpod.sh`.

Expected results table (to be filled):

| Domain | FFN-only PPL | All-modules PPL | Gap % | Kill? |
|--------|-------------|----------------|-------|-------|
| bash | -- | -- | -- | -- |
| math | -- | -- | -- | -- |
| medical | -- | -- | -- | -- |
| python | -- | -- | -- | -- |
| sql | -- | -- | -- | -- |
| **Mean** | -- | -- | -- | -- |

Kill threshold: any domain with gap >5%.

### Orthogonality Comparison

**STATUS: AWAITING GPU TRAINING (for independent FFN-only adapters)**

Baseline retroactive FFN-only mean |cos| = 0.0605.
Kill threshold: independent differs >50% (i.e., outside [0.0303, 0.0908]).

## Cost Analysis

| Item | Time | Cost |
|------|------|------|
| 5 FFN-only adapters | ~75 min | ~$0.20 |
| 5 all-modules adapters (optional) | ~75 min | ~$0.20 |
| Total | 75-150 min | $0.20-0.40 |

(RunPod A5000 at $0.16/hr)

Well within $5 budget constraint.

## Micro-Scale Limitations

1. **Single seed.** Results may not generalize. 3 seeds recommended for
   definitive conclusion. Budget permits replication at ~$0.60 for 3 seeds.

2. **Eval PPL, not task accuracy.** PPL killed as task accuracy proxy
   (exp_ppl_vs_task_performance), but relative PPL comparison between
   matched-rank configurations is more reliable than PPL-to-accuracy mapping.

3. **Same training data.** Both configurations use the same 1000 distillation
   examples per domain. Different data could yield different results.

4. **No composition quality test.** We measure individual adapter quality
   and pairwise orthogonality, not the quality of the composed multi-expert
   model. Composition testing is a separate experiment.

5. **Existing all-modules adapters may have different software versions.**
   For maximum fairness, retrain all-modules with the same script using
   `--also-train-all`. This is optional but recommended.

## What Would Kill This

### Kill Criteria (both must pass)

1. **Quality**: FFN-only PPL must be within 5% of all-modules at rank-16
   for all 5 domains. Rationale: 25% fewer parameters should not cost
   more than 5% quality.

2. **Orthogonality**: Independent FFN-only mean |cos| must be within 50%
   of retroactive value (0.0605). Rationale: if independent training
   fundamentally changes orthogonality, the retroactive analysis was
   misleading and the FFN-only architecture pivot is not validated.

### If Killed

- If K1 triggers: FFN-only is insufficient. Use all-modules adapters
  (25% more expensive per expert but quality-proven).
- If K2 triggers: Retroactive analysis was confounded. Need to re-examine
  whether FFN-only is actually more orthogonal when independently trained.
- If both trigger: FFN-only is both lower quality AND differently structured.
  Strong evidence against the architecture pivot.

### If Proven

- Confirms FFN-only as the default adapter configuration for the composable
  architecture. All future experts (50-expert pilot and beyond) use
  gate_proj, up_proj, down_proj only.
- 25% parameter savings at scale: 5000 experts save 48 GB.
- Retroactive analysis validated as a legitimate proxy.

## Artifacts

- `micro/models/ffn_only_matched_rank/train_ffn_only.py` -- GPU training script
- `micro/models/ffn_only_matched_rank/analyze.py` -- CPU analysis script
- `micro/models/ffn_only_matched_rank/run_on_runpod.sh` -- RunPod orchestrator
- `micro/models/ffn_only_matched_rank/test_ffn_only_matched_rank.py` -- tests (8/8 pass)
- `micro/models/ffn_only_matched_rank/results.json` -- analysis output
- `micro/models/ffn_only_matched_rank/MATH.md` -- mathematical foundations
