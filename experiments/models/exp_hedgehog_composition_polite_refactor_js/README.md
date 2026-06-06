# exp_hedgehog_composition_polite_refactor_js

## Headline experiment
Compose three Hedgehog-distilled adapters (politeness + refactor + JS) at **runtime** (not pre-merge; F#571 killed pre-merge N>1) and verify the composition produces a polite JavaScript refactorer without cross-leakage.

**Parents:**
- `exp_hedgehog_behavior_adapter_politeness` → ΔW_polite
- `exp_hedgehog_procedural_adapter_refactor` → ΔW_refactor
- `exp_hedgehog_domain_adapter_js` → ΔW_js

## Paper
[arXiv:2604.14191](https://arxiv.org/abs/2604.14191) for the per-adapter recipe. Composition is our contribution — the paper does not test composition of multiple distilled adapters.

## Composition mechanism
Runtime addition on the forward pass (per Pierre F#627 — supported at N=24):

```
h_out = W_base @ h + sum_i(alpha_i * ΔW_i @ h)
```

No pre-merging weights. Adapters stay as delta matrices in memory; gating `alpha_i` can be fixed (all 1.0) or routed per-prompt.

## MLX translation
`mlx_lm.tuner.lora.LoRALinear.fuse()` merges into base — do NOT use for composition. Instead, keep LoRALinear layers active and load multiple sets of LoRA weights as separate parameter groups, summed at forward time. See `pierre/core/compose.py` in existing repo for the established runtime-compose primitive.

## Quick start
```bash
# Wait for all 3 parents to reach status=supported
experiment claim <worker-id> --id exp_hedgehog_composition_polite_refactor_js
experiment run exp_hedgehog_composition_polite_refactor_js
```
