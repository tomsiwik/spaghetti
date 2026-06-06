# Learnings: exp_ternary_adapter_n50_composition

## Core Finding

Ternary LoRA adapter composition scales to N=50 (49 effective) on BitNet-2B-4T without composition catastrophe, and Gumbel-sigmoid routing captures 99.6% more composition benefit than uniform averaging (gamma_routed=0.632 vs gamma_uniform=0.996), with all 49/49 domains improving over base under routing.

## Why This Happened (Literature-Grounded)

Three mechanisms converge to make this work:

**1. High d/r ratio guarantees structural orthogonality.** With d=2560 and r=16, the capacity bound is N_max = d^2/r^2 = 25,600. At N=50, we use 0.2% of theoretical capacity. The Johnson-Lindenstrauss lemma and FlyLoRA (arXiv 2310.11034) predict that random low-rank projections in high-dimensional spaces are near-orthogonal with high probability. Our observed max cosine of 0.010 (5x below threshold) confirms this -- the adapters simply cannot interfere geometrically at this ratio.

**2. Routing eliminates the 1/N dilution catastrophe.** Our own prior work (macro/composition_dropout_robustness) killed equal-weight composition at macro scale (PPL in trillions, CV=112%). The N=50 result shows the same dilution trend: gamma_uniform degrades from 0.92 (N=5) to 0.996 (N=50), approaching 1.0 where adapters become neutral. Gumbel-sigmoid top-2 routing sidesteps this entirely by selecting 2 of 49 adapters, giving each effective scale s/k=10 instead of s/49=0.41. The L2R (Learning to Route, 2024) framework explains why sigmoid (not softmax) is correct: adapter gates are independent Bernoulli variables with no zero-sum competition, allowing multiple relevant adapters to activate simultaneously.

**3. QAT noise acts as implicit regularization.** NotebookLM confirms (via survey of ternary quantization literature) that quantization-aware training noise "behaves similarly to data augmentation or Dropout, enhancing robustness and adaptability." While no prior work directly links this to adapter composition, our micro-scale result (bitnet_ternary_adapter_composition) showed 19.3% lower pairwise |cos| for ternary vs FP16 adapters. The mechanism is plausible but unproven at scale.

## Confirming Evidence

**Scaling orthogonality:** Our own scaling trajectory (N=5 -> 15 -> 25 -> 50) shows consistent sub-threshold cosine throughout, confirming the JL-lemma prediction holds empirically. Mean cosine stays O(1/sqrt(d)) at all scales tested.

**Routing necessity:** The "When Are 1.58 Bits Enough?" paper (arXiv 2411.04965) confirms ternary works for decoder-only LMs but struggles in encoder-decoder architectures. Our decoder-only BitNet-2B-4T falls in the favorable regime. Separately, MoTE (arXiv 2506.14435) demonstrated ternary expert composition at real scale using top-k routing -- our Gumbel-sigmoid approach is architecturally similar.

**1/N failure is universal:** Our macro experiments (composition_dropout_robustness, composition_weight_normalization) conclusively killed equal-weight composition. The N=50 gamma_uniform=0.996 (only 0.4% improvement over base) confirms the trend extends to higher N. This is not a ternary-specific phenomenon -- it's a fundamental limitation of uniform parameter averaging at scale.

**QAT regularization:** Literature confirms QAT noise has regularizing effects analogous to Dropout (NotebookLM source [1]). The TNT (Target None-retraining Ternary) cosine objective from Zhang et al. (2019) shows the ternary community uses directional alignment, not divergence, as the optimization target -- making our observation of divergence between adapters a side effect of domain-specific training, not an intentional mechanism.

## Contradicting Evidence

**Cosine reduction may be a small-d artifact.** The adversarial review flags that at d=4096, random FP16 vectors are already near-orthogonal (cos ~ 1/sqrt(d) = 0.016). At our d=2560, cos ~ 0.020, and we observe 0.0019 -- meaning adapters are 10x more orthogonal than random. The ternary "decorrelation advantage" observed at d=64 (micro) may vanish at production scale where FP16 adapters are already geometrically near-orthogonal. NotebookLM confirms no papers have tested this specific question at d=4096+.

**lora_a is trained, not frozen.** The REVIEW-adversarial.md identifies that A matrices are trainable in the code despite MATH.md claiming they are frozen. The JL-lemma orthogonality guarantee applies to random (init) matrices, not trained ones. At N=200+, trained A matrices could converge toward similar subspaces for related domains. Empirically this hasn't happened at N=50, but the theoretical guarantee is weaker than claimed.

**Expert collapse in MoE routing at scale.** NotebookLM warns that dynamic learned routing suffers from severe instability at scale: routers frequently experience expert collapse (overly confident gates routing most tokens to a few dominant experts). Our Gumbel-sigmoid router achieves 86% accuracy at N=49, but 4 domains (chemistry, wikitext, dialogue, debate) show 0% accuracy -- early signs of selective collapse that could worsen at N=200+.

**Encoder-decoder ternary failure.** "When Are 1.58 Bits Enough?" found ternary training consistently degrades T5-style encoder-decoder models, even with increased hidden capacity. While not directly applicable (we use decoder-only), this constrains generalization claims about ternary composition.

**Parameter-averaging introduces cross-terms.** The review notes that averaging adapter parameters (not outputs) introduces cross-terms (A_1@B_2, A_2@B_1) and halves effective scale. This means gamma_routed=0.632 is conservative, but also means the composition mechanism is not mathematically clean. At lower d/r ratios or with more correlated adapters, cross-terms could become non-negligible.

## Alternative Approaches (What We Could Try Instead)

**1. Output-averaging instead of parameter-averaging.** Apply each selected adapter in a separate forward pass, average outputs. Eliminates cross-terms, gives exact composition. Costs k forward passes but is mathematically correct. Should be implemented for N=100 experiment.

**2. MoLoRA per-token routing (arXiv 2603.15965).** Route at token level, not sequence level. Our current sequence-level routing (mean-pooled hidden state) makes one selection per input. Per-token routing allows different experts within a single sequence. Our prior experiment (molora_per_token_routing) validated this on MLX; the top-2 approach won 8/15 domains.

**3. Arrow zero-shot routing (from "Towards Modular LLMs by Building and Reusing a Library of LoRAs").** Dynamically selects relevant adapters for new inputs without retraining. Could replace our supervised Gumbel router that requires labeled domain data.

**4. S-LoRA adapter serving (arXiv 2311.03285).** Stores thousands of adapters in system memory with dynamic fetching. Uses unified paging for heterogeneous adapter ranks. However, relies on CUDA kernels -- would need MLX port for our target platform.

**5. Test-time training (TTT).** From parameter-golf #1 entry: adapt weights per-document at inference. Sidesteps learned routing entirely. Reference: TTT Done Right (arXiv 2505.23884). Could be combined with our adapter pool: use TTT to identify which adapters to activate.

**6. Freeze A matrices explicitly.** If we actually freeze lora_a (add `module.freeze(keys=["lora_a"])` after unfreezing lora_b), we restore the JL-lemma guarantee and can make stronger theoretical claims. Trade-off: trained A may give slightly better individual adapter quality.

## Implications for Next Experiments

1. **Routing is the load-bearing mechanism, not orthogonality.** Orthogonality is necessary (prevents catastrophe) but not sufficient (gamma_uniform=0.996 is nearly useless). The router is what makes composition work. Future experiments should focus on router quality, not adapter geometry.

2. **Output-averaging is mandatory for N=100+.** Parameter-averaging's cross-terms are tolerable at high d/r=160 but will degrade. The N=100 experiment should implement proper output-averaging.

3. **The 4 zero-accuracy domains need investigation.** Chemistry, wikitext, dialogue, and debate have 0% routing accuracy. These may be fundamentally hard to distinguish from base hidden states, or may need more router training data (currently 20 samples per domain).

4. **Freeze A for cleaner theory.** The lora_a training vs frozen discrepancy should be resolved before N=100. Either freeze it (stronger theory) or update MATH.md (honest about empirical-only guarantee).

5. **Gamma scaling model remains undetermined.** The trajectory {0.92, 0.938, 0.982, 0.996} for uniform composition looks like it approaches 1.0 but the functional form (1-c/sqrt(N) vs 1-c/N) is not determined. This is moot if we commit to routing, but matters for understanding the failure mode.

6. **VISION.md Grassmannian references are stale.** Must reconcile with random-uniform A init actually used. This is a project-wide documentation debt.
