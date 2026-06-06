# Learnings: exp_pointer_routing_no_merge

## Core Finding

Per-layer adapter routing is actively harmful for jointly-trained LoRA adapters. The only routing method that beats uniform 1/N merge (learned gate, +11.9%) collapses to per-sequence oracle routing (same adapter at all 30 layers). Methods that actually vary adapter assignment across layers (hash: -1.1%, MLP: -0.5%) perform worse than uniform. The correct routing granularity for pre-trained LoRA adapters is per-SEQUENCE, not per-LAYER.

## Why This Happened (Literature-Grounded)

Three mechanisms explain why per-layer mixing hurts:

1. **Residual stream calibration mismatch.** Each adapter's B-matrices at layer L are calibrated for the residual stream produced by that same adapter at layers 0 through L-1. Mixing adapters across layers breaks this joint training distribution. This is a direct consequence of the LoRA training process: gradients flow through the full residual stream with one adapter active, so B_L implicitly depends on the cumulative effect of all preceding layers' adapter contributions. Switch Transformer (Fedus et al., arXiv 2101.03961) avoids this because each MoE layer's experts are trained *for* per-layer selection from scratch -- the experts never assume cross-layer consistency.

2. **1/N dilution vs full-strength scale confound.** The 11.9% improvement over uniform cannot be cleanly attributed to "output-space composition preserving nonlinearity" as PAPER.md claims. The review correctly identified a missing control: uniform applies scale/N=4.0 per adapter while oracle applies scale=20.0 for one adapter. Without comparing single adapter at scale=20.0 vs single adapter at scale=4.0, we cannot separate the 5x scale factor from the nonlinearity-preservation argument. Our own exp_output_averaging_vs_param_merge found that 1/k dilution (a linear scaling effect) is the *dominant* bottleneck, not cross-term interference or nonlinear mismatch.

3. **Hash routing has oracle access yet fails.** The hash implementation uses `domain_idx` (ground truth label) to assign adapters, meaning it has perfect knowledge of the input domain. Despite this oracle access, varying assignments across layers (-1.1%) is worse than uniform merge. This is a stronger negative result than acknowledged: even with perfect domain knowledge, per-layer mixing is harmful.

## Confirming Evidence

- **exp_output_averaging_vs_param_merge (our own):** Found that 1/k dilution is the dominant failure mode of parameter-space merge, not cross-term interference. At k=5, pre-merge wins (+3.0%), confirming that at SOLE's operating point (top-k=2-5) the dilution effect is manageable. Consistent with our finding that single-adapter at full strength beats diluted multi-adapter merge.

- **exp_tiny_routing_heads (our own):** Per-adapter binary routing heads achieve 100% accuracy and +19.9% over uniform at N=5. This experiment's 11.9% via oracle routing is *weaker* than the existing result, confirming that the routing mechanism matters less than selecting the correct adapter at the sequence level.

- **exp_molora_per_token_routing (our own):** Per-token routing provides no benefit over per-sequence routing on cleanly separated domains (-0.46%). "Routing granularity should match supervision granularity" -- our adapters were trained with per-domain labels (sequence-level), so per-layer routing is an architecture-data mismatch.

- **Mod-Squad (Chen et al., NeurIPS 2023):** Task-level routing outperforms token-level routing when domain boundaries are clear. Mutual information loss drives expert specialization at the task granularity, not finer granularities.

- **LoRA Soups (arXiv 2410.13025):** Uniform pre-merge scales linearly with N. The merge cost is in accumulation (memory bandwidth), not computation. Consistent with our latency sweep finding that pre-merge is viable at production scale.

## Contradicting Evidence

- **Switch Transformer (arXiv 2101.03961):** Achieves strong results with per-layer expert selection, but critically, experts are trained *for* per-layer routing from scratch. The training procedure creates experts that function in single-layer isolation. This is a fundamentally different paradigm from applying jointly-trained LoRA adapters per-layer. Our negative result is about training/inference mismatch, not about per-layer routing in general.

- **DeepSeek-V3:** Uses 256 experts with top-8 routing per token per layer, achieving state-of-the-art quality. Again, experts are trained from scratch with per-layer routing as an architectural invariant. Not comparable to post-hoc routing of pre-trained adapters.

- **MoLoRA (arXiv 2603.15965):** Qwen3-1.7B + 4 adapters beats 8B using per-token routing. Key difference: MoLoRA trains adapters *with* the routing mechanism in the loop (end-to-end), so adapters learn to function under per-token selection. Our adapters were trained independently and applied under a routing regime they never saw during training.

## Alternative Approaches (Paper-Referenced Only)

1. **End-to-end routed training (MoLoRA, arXiv 2603.15965):** Train adapters jointly with the routing mechanism so B-matrices are calibrated for per-layer isolation. Would require retraining all adapters -- not compatible with SOLE's plug-and-play design.

2. **LoRA-LEGO rank-wise clustering:** Decompose adapters into Minimal Semantic Units (per-rank), cluster, reassemble. Gets ensembling quality at merging speed by preserving fine-grained knowledge while avoiding destructive averaging.

3. **Top-k per-sequence routing with k>1 (proven in our system):** Our own top-2 Gumbel routing achieves +13.9% over uniform. Stays at per-sequence granularity (the correct level) while accessing multiple adapters. Already proven, should be the default.

4. **CLONE (arXiv 2506.02847):** MoE-style router for dynamic LoRA selection at edge. Per-query routing with lightweight gate, designed for the plug-and-play adapter scenario where adapters are pre-trained independently.

## Implications for Next Experiments

1. **Per-layer routing design branch is pruned.** Do not pursue per-layer adapter mixing for independently-trained adapters. The residual stream calibration argument is clean and the negative evidence is strong (three methods, all fail or collapse to per-sequence).

2. **Per-sequence routing is the correct granularity for SOLE.** This aligns with: SOLE hash-ring (5.3% displacement at N=20), routing heads (99.9% accuracy), Gumbel-sigmoid routing (0.58% overhead), and now the learned gate's convergence to oracle per-sequence routing.

3. **The nonlinearity-preservation claim remains unproven.** The missing scale control (scale=20 vs scale=4 for single adapter) means we cannot attribute the 11.9% gap to output-space vs parameter-space composition. The simpler explanation (1/N dilution elimination) is sufficient. A controlled experiment would be needed to isolate the nonlinearity effect.

4. **MLP routing needs hidden state normalization.** The float32 overflow and the hidden state distribution shift (trained on base model states, applied to adapter-modified states) both point to normalization as a prerequisite for any learned routing on BitNet-2B-4T hidden states.

## Recommended Follow-Up

**None required.** The per-layer routing branch is cleanly killed. The convergent recommendation from FIVE experiments (competitive_benchmark, batched_premerge, continual_learning, falcon_e3b_composition, and this one) is to pursue per-sequence top-k routing -- which is already implemented and proven in our system. The deployment track (P0) experiments (generation_quality_test, task_accuracy_real_benchmarks) are the correct next steps, not further routing architecture experiments.

## Numbers to Remember

| Metric | Value |
|--------|-------|
| Oracle single-adapter vs uniform 1/N | +11.9% mean PPL improvement |
| Hash per-layer routing vs uniform 1/N | -1.1% (worse, despite oracle domain access) |
| MLP per-layer routing vs uniform 1/N | -0.5% (worse, broken training) |
| Learned gate specialization (cross_layer_variation) | 0.0 (collapsed to oracle) |
| Learned gate same_adapter_fraction | 1.0 (all layers pick same adapter) |
| Best domain improvement (math, oracle) | +18.0% |
| Prior routing heads improvement | +19.9% (stronger than this result) |
| Experiment runtime | 136s |
| Peak memory | 5.35 GB |
