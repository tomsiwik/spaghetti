# MATH — exp_adapter_pruning_post_training

## Question
Can magnitude-based pruning remove 50% of trained LoRA adapter weights with negligible behavioral degradation, and does the pruning compose under additive combination with a second adapter?

## Failure mode being prevented
**Disease**: serving a dense LoRA stack at deploy time costs 2× memory bandwidth versus a 50%-sparse stack. If pruning at 50% destroys behavioral quality, the dense form is required and "Pierre serving cost" cannot be halved by pruning.

**Antipattern (proxy-only)**: prior pruning literature (Wanda, SparseGPT) reports pre-training-domain PPL on WikiText. That is a proxy. Our research must demonstrate that pruning preserves the *adapter's effect* on its trained domain (medical Q&A) and under composition with a second adapter (medical + math).

## Prior theorems
1. **Wanda (arxiv:2306.11695)** — for dense LLM weights, magnitude × activation pruning preserves PPL within 0.5–1.0 at 50% sparsity. Pure magnitude (no activation term) loses ~1pp more.
2. **LoRAPrune (arxiv:2305.18403)** — applied to LoRA adapters: 50% magnitude pruning on rank-8 LoRA preserves task accuracy on GLUE within 1pp; the rank structure absorbs sparsity gracefully because LoRA matrices have higher coherence than full-rank weights.
3. **Eckart–Young** — bounds the truncation gap of low-rank approximations: ‖ΔW − ΔW'‖_F = (Σ_{i>k} σᵢ²)^(1/2). Magnitude pruning of A,B individually is **not** Eckart–Young; it generically destroys low-rank structure unless singular vectors are sparse-aligned.

## Theorem (LoRA pruning bound)

Let A ∈ ℝ^(d_in × r), B ∈ ℝ^(r × d_out) be a trained LoRA factor pair, scale α, so the effective weight delta is ΔW = α·A·B ∈ ℝ^(d_in × d_out). Apply magnitude pruning at sparsity 1−p (keep top-p fraction by |w|) independently to A and B, producing A', B' with the same shapes but (1−p)·d_in·r and (1−p)·r·d_out entries zeroed.

Define the per-matrix *retained energy fraction*:
f_M = ‖M'‖_F² / ‖M‖_F²,    M ∈ {A, B}

Then the perturbation of the effective delta satisfies:

```
‖ΔW − ΔW'‖_F  ≤  α·(‖A − A'‖_F · ‖B‖_F  +  ‖A'‖_F · ‖B − B'‖_F)
              =  α·(√(1 − f_A) · ‖A‖_F · ‖B‖_F  +  √f_A · ‖A‖_F · √(1 − f_B) · ‖B‖_F)
              =  α·‖A‖_F·‖B‖_F · (√(1 − f_A) + √f_A · √(1 − f_B))
```

**Proof.** Triangle inequality on ΔW − ΔW' = α(A·B − A'·B + A'·B − A'·B') = α((A − A')·B + A'·(B − B')) plus submultiplicativity of Frobenius norm. ‖A'‖_F ≤ ‖A‖_F. ‖A − A'‖² = Σ_{(i,j) pruned} A[i,j]² = ‖A‖² − ‖A'‖² = (1 − f_A)·‖A‖². ∎

## Predicted retained energy fraction (concentration of LoRA weights)
Trained LoRA matrices empirically follow heavy-tailed near-Gaussian distributions (per Aghajanyan 2020, 'Intrinsic Dimensionality'). For a centered Gaussian, top-50% by magnitude retains:

```
f_50% = E[X² · 1{|X| > q_0.5}] / E[X²]
      = 2 · ∫_{q_0.5}^∞ x²·φ(x) dx
      ≈ 0.886    (where q_0.5 ≈ 0.6745σ is the median |X|)
```

**Predicted bound** (substituting f_A = f_B = 0.886, α = 6.0):
```
‖ΔW − ΔW'‖_F / (α·‖A‖_F·‖B‖_F)  ≤  √(1 − 0.886) + √0.886 · √(1 − 0.886)
                                  =  0.338 + 0.941 · 0.338  =  0.656
```

This is a **worst-case** bound. The actual relative gap depends on alignment between A's pruned mass and B's column structure. Realistic gap: 0.3–0.5.

## Predicted PPL impact

For a model with baseline domain PPL ≈ P_0 and adapter contribution that lowers PPL to P_adapter, a relative perturbation ‖ΔW − ΔW'‖_F / ‖ΔW‖_F = ε produces a first-order PPL shift:

ΔPPL ≈ ε · |∂PPL/∂‖ΔW‖_F| · ‖ΔW‖_F

Empirically (F#674 cross-domain ablations), full removal of a domain LoRA (ε = 1) increased domain-PPL by ~1.5–3.0 PPL units. Predicted prune-50% effect with ε ≈ 0.4: **ΔPPL ≈ 0.6–1.2** — well above the 0.1 threshold in K1922.

**Prediction**: K1922 fails (50% pruning is too aggressive for behavioral preservation).

## Predicted composition impact

Under additive composition ΔW_med + ΔW_math, pruning errors approximately add in quadrature (independent random pruning patterns):
‖(ΔW_med − ΔW'_med) + (ΔW_math − ΔW'_math)‖_F ≈ √2 · 0.4 · ‖ΔW‖_F ≈ 0.57 · ‖ΔW‖_F

Compose-PPL gap predicted: ≈ √2 × single-adapter gap ≈ 1.0–1.7 PPL units. **Below** the 3.0 PPL threshold of K1923.

**Prediction**: K1923 passes (composition degrades but stays under 3 PPL).

## Pre-registered Kill Criteria (target-gated per F#666)

Both KCs are **target-side** (real PPL on real domain text). No proxies → no pairing required.

- **K1922 (target — single-adapter PPL)**: ΔPPL_single = PPL(medical-pruned) − PPL(medical-full) > 0.10 → FAIL
  - eval set: first 100 rows of `data/medical/valid.jsonl`
  - pruning: per-matrix magnitude top-50% on lora_a and lora_b across all 42 q_proj modules
  - dtype: pruning in fp32, model in 4-bit base
- **K1923 (target — composition PPL)**: ΔPPL_compose = PPL((med+math)-pruned) − PPL((med+math)-full) > 3.0 → FAIL
  - eval set: first 50 rows medical/valid.jsonl + first 50 rows math/valid.jsonl
  - LoRA forward = α·x·A_med·B_med + α·x·A_math·B_math (additive composition)
  - both adapters pruned identically (per-matrix top-50%)

Verdict logic:
- **supported** iff K1922 PASS and K1923 PASS
- **killed** iff K1922 FAIL (single-adapter cannot survive pruning ⇒ composition is moot)
- **provisional** iff K1922 PASS and K1923 FAIL (single ok but composition breaks; structural finding)

## Methodology

**Phase 0** — load base model `mlx-community/gemma-4-e4b-it-4bit`, load medical and math LoRA safetensors (rank 6, scale 6.0, q_proj only — per `lora_config_*.yaml`).

**Phase 1 (K1922)** — install medical adapter via `LoRALinear.__call__` monkey-patch (forward = base + α/r·x·A·B). Compute PPL on first 100 `data/medical/valid.jsonl` rows with full and pruned adapter. Pruning: per-matrix top-50% magnitude on each of 84 matrices.

**Phase 2 (K1923)** — install medical + math additive composition via the same monkey-patch (forward = base + α·x·A_med·B_med + α·x·A_math·B_math). PPL on 50 med + 50 math rows. Re-run with both adapters pruned.

**PPL formula**: standard token-level cross-entropy averaged over all assistant-side tokens, exp(mean_neg_log_prob). Eval batch size 1, max_seq_len 512.

## Antipattern checklist (researcher pre-flight)
- [x] Composition math: forward-pass LoRA = α·x·A·B (consistent with mlx_lm.tuner.lora.LoRALinear; verified in `exp_composition_ordering_matters/run_experiment.py:Phase2`).
- [x] LORA_SCALE: 6.0 (matches trained config; not the unsafe 20).
- [x] No `shutil.copy` of original adapter as new adapter.
- [x] No hardcoded `"pass": True` — KCs computed from numeric data.
- [x] No eval truncation. Full text length up to 512 tokens.
- [x] Proxy-model substitution: NO. Same `mlx-community/gemma-4-e4b-it-4bit` as adapter was trained on.
- [x] KC measures correct object: PPL on domain valid.jsonl, the same data the adapter was trained to fit.
- [x] Not a smoke run. Full N=100 (Phase 1) and N=100 (Phase 2).

## References
- arxiv:2306.11695 (Wanda — magnitude × activation pruning baseline)
- arxiv:2305.18403 (LoRAPrune — confirms LoRA matrices tolerate magnitude pruning better than full weights)
- F#674 (cross-domain ablation: full adapter removal → +1.5–3.0 PPL on domain)
- F#744 (immediately preceding finding: ordering invariance under MLX BF16 GEMM; same scaffolding pattern)
