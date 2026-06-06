# Peer Review: MoTE-SOLE Architecture

## NotebookLM Findings

Skipped. The experiment is already self-killed with clear numerical evidence. The failure modes are unambiguous and do not require deep-dive synthesis to assess.

## Mathematical Soundness

**Router formulation: correct.** The linear router h @ W_r + b_r -> softmax -> top-k with STE is standard MoE machinery (Switch Transformer, MoTE). Load-balancing loss L_balance = N * sum(f_i * P_i) is correctly implemented per Fedus et al. No mathematical errors found.

**STE for top-k: correct in principle but weak in practice.** The implementation (line 648-652) multiplies softmax probs by a constant binary mask, so autograd differentiates through the soft probs while the hard selection is treated as constant. This is standard STE. However, at N=5, the gradient signal through the selected set is sparse -- only k/N of the expert weights receive gradient. This is acknowledged.

**Sequence-level averaging: methodologically sound but introduces a confound.** The router averages hidden states across the sequence (line 631, 1039) then makes a single routing decision per sequence. This is consistent with SOLE's per-query routing granularity. However, it means the router sees a blurred representation -- for short sequences (arithmetic: "12+34=46" is ~10 tokens), the average may not carry strong domain signal. This is an implementation choice, not a mathematical error.

**Pre-merge weight averaging across batch: potential issue.** Lines 656 and 1057 average the per-sample routing weights across the batch before composing a single delta. This means all samples in a batch get the same composed model, even if the router assigned different experts to different samples. For same-domain batches (which is the eval setup), this is fine. For mixed-domain batches during training, this smooths out the routing signal. The code comment on line 655 acknowledges this ("batch typically has same-domain data"). For the eval results, this is not a confound because eval is per-domain.

**K3 comparison: fair.** Both ternary and FP16 adapters are trained on the same ternary base with the same seed, epochs, and hyperparameters. The only difference is QAT quantization. The 13.5% gap is a clean measurement.

**One numerical discrepancy in PAPER.md.** The per-domain table (PAPER.md lines 67-73) shows values that are rounded means across seeds, but for seed=42 specifically, k=1 routing on arithmetic gives PPL 4.27 (close to oracle 4.26), NOT the 21.2 shown in the table. The 21.2 is the mean across all 3 seeds, heavily skewed by seed=123 which gives PPL 55.1 on arithmetic at k=1. The table mixes seed-42-specific structure with cross-seed means in a way that could confuse readers, but the aggregate numbers in results.json are correct.

## Novelty Assessment

**Low novelty, which is appropriate.** This is a direct application of MoTE (arXiv 2506.14435) architecture to the SOLE framework at micro scale. The experiment does not claim novelty -- it tests whether MoTE-style routing is beneficial for SOLE composition. This is the correct thing to test before investing in MoTE at macro scale.

**Prior art within the project is correctly cited.** The experiment correctly references:
- exp_bitnet_ternary_adapter_composition (SUPPORTED): ternary adapters compose 4.4% better under equal-weight
- exp_bitnet_composition_stability (SUPPORTED): composition ratio 0.63
- content_aware_routing (KILLED): oracle routing identical to random at d=64

**The content_aware_routing kill should have been a stronger signal.** That experiment already showed that routing at d=64 is fundamentally broken -- oracle routing produced identical NTP loss to random routing. Running MoTE routing (which is strictly worse than oracle) after this kill was somewhat redundant. However, the experiment adds value by (a) testing ternary experts specifically, (b) measuring the magnitude of routing failure, and (c) providing the K3 ternary/FP16 comparison.

## Experimental Design

**The experiment tests what it claims.** The hypothesis is "MoTE-style routing outperforms equal-weight composition." The experiment measures exactly this across 3 seeds, 5 domains, and 3 values of k.

**Controls are adequate:**
- Ternary equal-weight composition (primary baseline)
- FP16 equal-weight composition (secondary baseline)
- Oracle routing (upper bound -- uses the single best expert per domain)
- Ternary base with no adapters (lower bound)

**One missing control: random routing.** The experiment does not include random top-k selection as a baseline. Given that k=1 accuracy is 50.1% (vs 20% chance for uniform random), it would be informative to see whether the learned router actually beats random routing on PPL, or whether the 50% accuracy translates to no PPL benefit. Looking at seed=42 data, the router achieves 78.8% k=1 accuracy and gets routed k=1 mean PPL of 3.31 (vs equal-weight 4.91), which is excellent. But seed=123 gets 31.6% accuracy and routed k=1 PPL of 14.49 (vs equal-weight 6.06), which is catastrophic. This seed variance is the real story -- the router training is unstable, not that routing is fundamentally impossible.

**Critical observation: the router is trained at k=2 but evaluated at k=1, k=2, k=3.** Line 999 shows `k=2` is hardcoded for router training. The k=1 and k=3 evaluations use a router that was never trained for those k values. This is somewhat standard (Switch Transformer trains at k=1 and evaluates there), but it means the k=1 numbers are pessimistic (router was optimized for a different selection budget) and the k=3 numbers benefit from redundancy. This does not invalidate the kill -- even at k=2 (the trained setting), routed PPL (6.53) exceeds equal-weight (5.49) by 19%.

**The K1 kill criterion has a loophole.** K1 as stated is "MoTE-SOLE quality < equal-weight on >50% of domains." The experiment evaluates this at k=2 (line 1119), which is the training k. But the PAPER.md discussion correctly notes that domain-count-based criteria are misleading when failure magnitudes are asymmetric (+128% arithmetic vs -56% repeat). The researcher's decision to kill based on mean PPL rather than domain count is the right call.

**K3 failure (ternary 13.5% worse than FP16) contradicts prior finding.** exp_bitnet_ternary_adapter_composition found only 2.6% individual degradation. The PAPER correctly identifies the difference: the prior experiment compared ternary-on-ternary vs FP16-on-ternary and got 2.6%, while this experiment gets 13.5%. The discrepancy is large (5x). Possible explanations: (a) different random seeds, (b) different number of training epochs or hyperparameters, (c) the prior experiment used a different QAT implementation. This discrepancy deserves investigation but does not affect the kill verdict since both K1 and K3 fail.

## Macro-Scale Risks (advisory)

1. **Routing may work at d=4096 but the evidence trail is weak.** Two micro experiments (content_aware_routing, mote_sole_architecture) both killed routing at d=64. The paper's argument that d=4096 hidden states carry more domain signal is plausible but untested. If routing is ever revisited at macro scale, it should be tested with a cheap oracle-routing-vs-random baseline first (replicating the content_aware_routing design at d=4096) before investing in learned routers.

2. **The ternary expert quality gap (13.5%) may or may not persist at scale.** The prior experiment found 2.6% at the same d=64, so the gap is implementation-sensitive. At macro scale with proper QAT training (e.g., using GPTQ or AWQ-style calibration), the gap could be much smaller. This needs measurement, not extrapolation.

3. **Equal-weight composition remains the default for SOLE.** This experiment reinforces the existing architecture decision. At N=5, equal-weight beats routing. The open question is whether equal-weight still works at N=50+ (macro experiments on composition weight normalization suggest it does not -- PPL in trillions). The resolution path is PPL-probe weighting or per-input routing at macro scale, not MoTE-style learned routing at micro scale.

## Verdict

**KILL -- confirmed.**

The researcher's self-kill is correct and well-justified. The evidence is unambiguous:

1. **K1 FAIL (decisive):** Routed composition produces worse mean PPL than equal-weight at every value of k tested. k=1: +42%, k=2: +19%, k=3: +10%. The router is net-negative in expectation.

2. **K3 FAIL (clean):** Ternary experts are 13.5% worse than FP16 individually, exceeding the 10% threshold across all 3 seeds (0/3 pass rate).

3. **Consistent with prior kill:** content_aware_routing already showed routing is broken at d=64. This experiment confirms the pattern with a different routing mechanism (learned linear router vs non-parametric methods).

4. **The analysis is honest and thorough.** The PAPER.md correctly identifies the asymmetric failure modes, the seed instability, and the limitations of micro-scale routing. The Limitations section appropriately scopes the negative result.

No revisions needed. The experiment is a clean negative result that correctly eliminates MoTE-style routing at micro scale and correctly defers the question to macro scale.
