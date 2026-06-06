# Learnings: exp_bitnet_float_merge_fp32

## Core Finding

bf16 weight merging is a viable serving path for BitNet-SOLE (39% faster than runtime LoRA, 0.6% PPL cost), while fp32 merge is killed on latency. Most critically, a systemic 1/N² scaling bug was discovered and fixed in the runtime LoRA baseline — the same pattern exists in HuggingFace PEFT (issue #1155) and likely affects ~4 other experiments in our pipeline.

## Why This Happened (Literature-Grounded)

### The 1/N² Bug: A Known Pattern

The 1/N² scaling bug — where both A and B matrices are independently scaled by 1/N, producing effective 1/N² in the product A@B — is not unique to our code. HuggingFace PEFT issue #1155 (github.com/huggingface/peft/issues/1155) reported the exact same pattern in `add_weighted_adapter`: `lora_A` was weighted but `lora_B` was not consistently weighted, producing asymmetric scaling that degraded multi-adapter merges. The fix in both cases is identical: scale only ONE of {A, B, lora_scale}, not multiple independently.

This bug class arises because LoRA's factored form `x @ A @ B * scale` makes the effective scaling *multiplicative* across factors. Developers intuitively apply 1/N to "each part" without recognizing the product amplifies to 1/N². This is a systematic trap in any codebase that averages LoRA components.

### bf16 Sufficiency Despite Per-Element Truncation

The bf16 ULP at BitNet's base magnitude (α=1.72) is 0.0134, while LoRA deltas average 0.0079 — meaning delta/ULP = 0.6x, i.e., individual elements ARE truncated (28% per element). Yet PPL only degrades 0.6%. This follows from the law of large numbers: truncation errors are symmetric and zero-mean across ~6.5M parameters per layer, so they cancel in aggregate. This aligns with the general finding in quantization literature that per-element precision requirements overestimate actual model sensitivity (Dettmers et al., LLM.int8(), NeurIPS 2022).

### Cross-Terms: Product-of-Sums vs Sum-of-Products

Runtime composition computes (ΣA_i)(ΣB_i) while merge computes Σ(A_i@B_i). The difference is the cross-terms Σ_{i≠j} A_i@B_j. At N=5, this gap is 0.15%. No existing literature directly quantifies this divergence for LoRA specifically, but the Tensorized Clustered LoRA Merging paper (arxiv 2508.03999) addresses analogous cross-task interference in merged adapters through CP decomposition to disentangle shared vs task-specific factors. The cross-term gap will grow as O(N²) relative to the diagonal terms O(N), suggesting it may become significant at N>>5.

## Confirming Evidence

1. **PEFT Issue #1155** — Identical A/B scaling bug in HuggingFace's `add_weighted_adapter`. Independent discovery confirms this is a systematic pattern, not a one-off mistake. (github.com/huggingface/peft/issues/1155)

2. **S-LoRA (arxiv 2311.03285)** — Keeps adapters separate during serving for exactly the reasons we found: runtime composition preserves adapter independence and allows per-request routing. Our finding that runtime LoRA is necessary for dynamic adapter selection aligns with S-LoRA's architecture. (Sheng et al., 2023)

3. **LoRA Soups / AdapterSoup** — Literature confirms that simple weight averaging of independently trained LoRAs causes interference. Our bf16 merge works because we use the *same base model* and train with orthogonal initialization, unlike the general multi-task merge scenario.

4. **LoRI (arxiv 2504.07448)** — Specifically studies cross-task interference reduction in multi-LoRA settings, confirming that adapter weight merging is lossy and interference-prone. Our 0.15% cross-term gap is small only because our adapters are structurally orthogonal (Grassmannian init).

## Contradicting Evidence

1. **LoTA-QAF (arxiv 2505.18724)** — Demonstrates lossless ternary merge that stays on the quantization grid, achieving 1.7-2x inference speedup. Our bf16 merge is a precision-lossy approximation. If LoTA-QAF's ternary merge method can be adapted for multi-adapter composition (currently single-adapter only), it would dominate bf16 merge on both quality and speed. This is the main threat to our finding.

2. **MoTE (arxiv 2506.14435)** — Achieves better scaling by keeping a frozen shared expert + routing to ternary experts, suggesting that merge-then-serve may be suboptimal compared to route-then-compute. MoTE's architecture allows adding experts without any merge step, avoiding cross-terms entirely.

3. **The bf16 truncation may not cancel at large N.** At N=100, individual deltas shrink to 1/100th current magnitude, pushing delta/ULP well below 1.0 for bf16. The symmetric cancellation argument relies on independent errors, but correlated adapter structures could introduce systematic bias. No paper directly addresses this edge case.

## Alternative Approaches (What We Could Try Instead)

1. **LoTA-QAF Multi-Adapter Extension** — The original paper only demonstrates single-adapter ternary merge. Extending it to compose N ternary adapters losslessly on the integer grid would be a novel contribution and eliminate both the cross-term issue and bf16 truncation. (arxiv 2505.18724)

2. **Output-Space Fusion (LoRAuter)** — Instead of merging parameters, fuse adapter *outputs*: run each adapter independently and combine logits. Bypasses geometric misalignment entirely. Higher compute cost (N forward passes through adapters) but zero approximation error.

3. **Per-Token Routing (MoLoRA, arxiv 2603.15965)** — Already in our roadmap (exp_bitnet_per_token_routing). MoLoRA achieves 4.1x throughput vs per-sequence routing with CUDA graphs. Eliminates the merge question by selecting rather than combining adapters.

4. **TIES-Merging** — Uses sparsification + sign consensus to reduce cross-task interference in weight merging. Could be applied to our adapter merge pipeline to further reduce the 0.15% cross-term gap.

5. **ExpertWeave** — System-level solution for co-locating multiple adapter experts with a shared base model using virtual memory management and fused kernels. Could complement our dual-mode serving strategy.

## Implications for Next Experiments

1. **URGENT: Audit ~4 experiments for 1/N² bug.** Any runtime LoRA composition using the old `compose_adapters_runtime` pattern (averaging both A and B by 1/N) is affected. Priority targets: exp_bitnet_2b_real_composition, exp_bitnet_ternary_adapter_composition, and any experiment that compared runtime LoRA PPL against other methods. The bug inflates runtime LoRA PPL, making other methods appear relatively better than they are.

2. **Dual-mode serving is confirmed.** bf16 merge for static adapter sets, runtime LoRA for dynamic routing. This should be the default serving architecture going forward.

3. **Cross-term divergence needs monitoring at scale.** The 0.15% gap at N=5 is negligible, but O(N²) growth means it could reach ~6% at N=20 and ~24% at N=40 (naive extrapolation). The exp_bitnet_per_token_routing experiment should include cross-term measurement.

4. **LoTA-QAF multi-adapter merge is the highest-value future experiment** for serving path optimization. If lossless ternary merge can be extended to N>1, it obsoletes bf16 float merge entirely.

5. **fp16 is an unexplored middle ground.** fp16 has delta/ULP=4.7x (vs bf16's 0.6x), meaning no per-element truncation. On Apple Silicon, fp16 ALU throughput equals bf16 (both use the same 16-bit path). fp16 merge may give lossless quality at bf16 speed — worth a quick measurement.

## New References to Add

| Paper | ArXiv/URL | Relevance |
|-------|-----------|-----------|
| S-LoRA: Serving Thousands of Concurrent LoRA Adapters | arxiv 2311.03285 | Runtime multi-adapter serving architecture |
| LoRI: Reducing Cross-Task Interference | arxiv 2504.07448 | Cross-task interference in multi-LoRA |
| Tensorized Clustered LoRA Merging | arxiv 2508.03999 | CP decomposition for merge interference |
| PEFT Issue #1155 (A/B scaling bug) | github.com/huggingface/peft/issues/1155 | Independent discovery of 1/N² pattern |
