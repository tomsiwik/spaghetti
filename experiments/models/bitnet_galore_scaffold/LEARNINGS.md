# Learnings: exp_bitnet_galore_scaffold

## Core Finding

GaLore-trained weights degrade 2.0–2.9x more than standard Adam weights under post-hoc ternary quantization, despite producing *better* FP32 models (0.81x PPL). The bottleneck for the base-free scaffold path is quantization, not training quality.

## Why This Happened (Literature-Grounded)

Three independent mechanisms converge to explain the degradation:

1. **GaLore's low-rank gradient projection leaves non-subspace weight dimensions uncalibrated.** Training dynamics research (Catalan-Tatjer et al., arXiv:2510.06213) shows that PTQ robustness depends on *all* weight dimensions being well-conditioned. GaLore restricts gradient updates to a rank-r subspace; the remaining d−r dimensions receive no optimization signal and accumulate frozen noise. These noise dimensions severely degrade ternary quantization.

2. **Undertrained/heterogeneous weight distributions quantize worse.** "Low-Bit Quantization Favors Undertrained LLMs" (arXiv:2411.17691, ACL 2025) demonstrates that fully-converged checkpoints have narrow weight fluctuations — quantization rounding pushes weights outside that narrow band. GaLore creates a paradox: its projected subspace is well-converged while the non-projected subspace is essentially random, producing a bimodal conditioning spectrum that is worst-case for PTQ.

3. **Ternary quantization specifically punishes non-bimodal weight distributions.** Ternary quantization literature (arXiv:2303.01505, arXiv:2306.17442) shows that weights must be near-bimodally distributed (peaks at ±threshold, minimal zero-mass) for efficient {-1,0,1} assignment. GaLore's frozen non-projected dimensions inflate zero-bin mass, directly degrading ternary assignment quality. The 21% CV in GaLore quant degradation (vs 1.7% for standard) confirms seed-sensitive distribution structure.

**Q-GaLore** (arXiv:2407.08296, ICML 2025) independently confirms the problem: they had to build an entire INT4/INT8 in-training quantization scheme specifically because naive quantization of GaLore weights loses too much information.

## Confirming Evidence

| Paper | arXiv | Key Finding | Relation |
|-------|-------|-------------|----------|
| Q-GaLore (Zhang et al., 2024) | 2407.08296 | GaLore weights need special quantization schemes; naive PTQ degrades quality | CONFIRMS: quantization-hostility is a known GaLore property |
| Training Dynamics & PTQ (Catalan-Tatjer et al., 2025) | 2510.06213 | Validation loss and quantization error diverge post-LR-decay; subspace-restricted training amplifies this | CONFIRMS: mechanistic explanation for GaLore's PTQ fragility |
| Low-Bit Quantization Favors Undertrained LLMs (ACL 2025) | 2411.17691 | Well-converged models are MORE fragile to PTQ; scaling laws show undertrained models quantize better | CONFIRMS: GaLore's well-converged subspace + untrained complement is worst-case |
| MagR (NeurIPS 2024) | N/A | High effective rank weights accumulate quantization error across more dimensions | CONFIRMS: GaLore's effective weight structure maps to the hard-to-quantize regime |
| LR-QAT (Qualcomm, 2024) | 2406.06385 | Low-rank adapters without QAT don't produce quantization-friendly weights; QAT is required as separate intervention | CONFIRMS: low-rank gradient methods need explicit QAT |
| PT-BitNet (2025) | ScienceDirect | Even standard Adam weights suffer under post-hoc ternary PTQ; GaLore makes it substantially worse | CONFIRMS: establishes PTQ-ternary baseline degradation |

## Contradicting Evidence

No paper directly contradicts our finding. The closest challenges are:

| Paper | arXiv | Key Finding | Nuance |
|-------|-------|-------------|--------|
| Q-GaLore | 2407.08296 | GaLore *projection matrices* are "quantization-friendly" when quantized in-training (INT4) | Does NOT apply to post-hoc PTQ of final weights — in-training quantization avoids the problem |
| LoQT (NeurIPS 2024) | 2405.16528 | Periodic merge of low-rank factors into quantized weights works well | Suggests the degradation is an artifact of *end-only* PTQ, not of low-rank training per se |
| GaLore 2 | 2504.20437 | Approximate projections are tolerable for gradient quality | Double-edged: tolerance of approximation ≠ tolerance of post-hoc ternary quantization |

**Key insight from contradictions**: The degradation is NOT fundamental to low-rank gradient training. It is specific to the combination of (GaLore FP32 training) + (post-hoc ternary PTQ). Every paper that solves the degradation does so by integrating quantization *during* training.

## Alternative Approaches (What We Could Try Instead)

### Tier 1: Most Actionable (directly addresses the bottleneck)

1. **Continual QAT Transition** (arXiv:2502.11895, ACL 2025)
   - Short FP16 warmup (10–20% of tokens) → transition to 1.58-bit QAT with optimizer state retention
   - Empirically beats both cold ternary start AND post-hoc quantization
   - **Hybrid variant**: GaLore warmup (memory-efficient) → QAT transition could give best of both worlds
   - This is the single most actionable finding for the base-free path

2. **Native BitNet QAT from random init** (arXiv:2402.17764)
   - STE + absmean quantization throughout training — no post-hoc step
   - Proven at 2B-4T scale by Microsoft
   - The "ground truth" method; GaLore is strictly a memory optimization, not a replacement for QAT

3. **LoQT-style periodic ternary merge** (arXiv:2405.16528, NeurIPS 2024)
   - GaLore gradient projection + periodic merge of low-rank factors into ternary weights
   - Prevents accumulation of quantization-hostile weight structure

### Tier 2: Worth Investigating

4. **BitDistill — Distillation into ternary** (arXiv:2510.13998, Microsoft)
   - 10x memory reduction, near-FP16 quality via dual-distillation (logits + attention)
   - Requires FP16 teacher (relaxes "base-free" constraint slightly)
   - Most practical path if we can accept using an existing FP16 model as teacher

5. **TernaryLLM DLT — Dual Learnable Ternarization** (arXiv:2406.07177)
   - Learns both scale AND shift (standard ternary only learns scale)
   - Could improve adapter expressiveness without breaking integer-grid composition

6. **RaBiT — Residual-aware multi-path binarization** (arXiv:2602.05367)
   - Train adapter N+1 to correct residual error of adapters 1..N
   - Could dramatically improve composition quality at large N (beyond naive 1/N averaging)

### Tier 3: Background Reference

7. **Spectra/TriLM** (arXiv:2407.12327, 2506.23025): Ternary models 99M–3.9B from scratch. Key finding: ternary benefits more from data scaling than parameter scaling.
8. **TernaryLM** (arXiv:2602.07374): 132M ternary from scratch using STE + adaptive per-layer scaling, RoPE + RMSNorm for stability.
9. **MoTE** (arXiv:2506.14435): Ternary routed experts on dense checkpoint. Validates ternary expert composition but requires pretrained shared expert.

## Implications for Next Experiments

1. **The base-free path MUST integrate QAT during scaffold training.** Post-hoc ternary PTQ of GaLore weights is a dead end. The next experiment (exp_bitnet_scaffold_fresh_adapters or a new STE-GaLore node) should use STE in the training loop.

2. **Consider the Continual QAT hybrid**: GaLore for memory-efficient FP16 warmup (~10–20% of steps) → transition to STE-aware ternary QAT for remaining steps. This combines GaLore's memory efficiency with QAT's quantization-friendly weight shaping.

3. **exp_bitnet_meta_scaffold should account for the QAT requirement**: MAML meta-learning on a scaffold must include the ternary quantization constraint in the inner loop (STE), or the meta-learned scaffold will suffer the same post-hoc degradation.

4. **The quant degradation variance asymmetry (CV 21% vs 1.7%) deserves a dedicated investigation** at macro scale. If GaLore weight distributions are fundamentally unstable w.r.t. quantization across seeds, even STE-aware training may not fully close the gap.

5. **Adapter initialization matters**: QuAILoRA (arXiv:2410.14713) and CLoQ (arXiv:2501.18475) show that quantization-aware LoRA initialization significantly reduces fine-tuning tokens needed. Worth integrating into adapter training pipeline.

## New References to Add

| Paper | arXiv | Relevance | Nodes |
|-------|-------|-----------|-------|
| Q-GaLore | 2407.08296 | Direct evidence for GaLore quantization challenges; in-training quantization solution | exp_bitnet_galore_scaffold, exp_bitnet_scaffold_fresh_adapters |
| Training Dynamics & PTQ Robustness | 2510.06213 | Mechanistic explanation for why certain training methods produce less quantizable weights | exp_bitnet_galore_scaffold |
| Low-Bit Quantization Favors Undertrained LLMs | 2411.17691 | Scaling laws for PTQ robustness; explains GaLore paradox | exp_bitnet_galore_scaffold |
| Continual QAT Pre-Training (16→1.58 bit) | 2502.11895 | Direct alternative to GaLore-from-scratch for base-free path; hybrid warmup strategy | exp_bitnet_galore_scaffold, exp_bitnet_scaffold_fresh_adapters, exp_bitnet_meta_scaffold |
| LoQT (periodic merge) | 2405.16528 | Periodic low-rank merge into quantized weights prevents PTQ degradation | exp_bitnet_galore_scaffold |
| LR-QAT (Qualcomm) | 2406.06385 | Integer-grid-aligned low-rank adapters for lossless quantized merge | exp_bitnet_scaffold_fresh_adapters, exp_bitnet_lori_sparse_b |
| BitDistill | 2510.13998 | Distillation into ternary as alternative to from-scratch training | exp_bitnet_scaffold_fresh_adapters |
| Spectra/TriLM | 2407.12327 | Ternary scaling laws: data scaling > parameter scaling for ternary models | exp_bitnet_galore_scaffold |
| TernaryLM | 2602.07374 | Small-scale ternary from scratch with STE + adaptive scaling | exp_bitnet_scaffold_fresh_adapters |
| RaBiT (residual binarization) | 2602.05367 | Residual-aware multi-path composition as alternative to 1/N averaging | exp_bitnet_per_token_routing |
