# Learnings: exp_bitnet_per_token_routing

## Core Finding

Top-2 per-sequence routing (from per-token router predictions) beats uniform 1/N composition by 13.9% average PPL on BitNet-2B-4T with 15 ternary LoRA adapters. A 659K-param router achieves 91.7% domain classification accuracy. Top-1 routing fails (-10.9% vs uniform) due to adapter overshoot. Importantly, this is per-sequence routing, not true per-token routing as originally named.

## Why This Happened (Literature-Grounded)

The top-2 advantage is explained by three converging mechanisms from the MoE and LoRA composition literature:

**1. Convex combination regularization.** Top-2 routing creates a convex combination of two adapters (weights summing to 1.0 after renormalization). This prevents any single adapter from applying its full delta to the base model. In our experiment, 4/15 adapters overshoot at full strength (individual PPL worse than base). Top-2 naturally clips this by giving each adapter 50-80% weight. This is the same mechanism that makes Mixtral-8x7B's top-2 routing outperform hypothetical top-1 variants — Jiang et al. (arXiv:2401.04088) chose k=2 specifically for this stability-quality tradeoff.

**2. Beneficial cross-domain transfer.** The secondary adapter adds complementary information. Physics (top-2: 22.83 vs individual: 26.47) benefits because the router's second choice (chemistry or math) provides related but non-identical signal. This aligns with our own earlier finding: "Composed PPL better than naive 1/N prediction (beneficial cross-domain transfer)" from pilot50_composition_quality.

**3. Signal dilution at 1/N.** With N=15, each adapter contributes only 6.7% under uniform weighting. Our own FINDINGS.md records that "Uniform static scaling cannot make composition competitive" (macro exp), with composed PPL approximately equaling base model — adapters cancel out. Top-2 routing concentrates signal on the 2 most relevant adapters, giving each ~50% contribution instead of 6.7%.

## Confirming Evidence

1. **Mixtral-8x7B** (arXiv:2401.04088) — Production top-2 MoE. Validates k=2 as the standard routing configuration for quality-efficiency tradeoff. Our result extends this to LoRA adapters on ternary base.

2. **MoLoRA** (arXiv:2603.15965) — Per-token routing of 4 LoRA adapters enables Qwen3-1.7B to exceed Qwen3-8B across four reasoning benchmarks. Key difference: MoLoRA does TRUE per-token routing (different adapters per token within one sequence), while our experiment does per-sequence routing. MoLoRA proves per-token is strictly better than per-sequence, suggesting our 13.9% improvement is a lower bound.

3. **LD-MoLE** (arXiv:2509.25684, ICLR 2026) — Replaces non-differentiable TopK with differentiable routing and adaptive expert count per token per layer. Directly addresses our W1 weakness (no load balancing) with an analytical sparsity control objective. Their approach lets the model learn k dynamically rather than fixing k=2.

4. **X-LoRA** (arXiv:2402.07148) — Dynamic layer-wise token-level LoRA mixing. Uses scaling vectors per layer, which is finer-grained than our single-layer routing. Confirms that input-dependent adapter weighting beats static composition.

5. **Our own FINDINGS.md** — Multiple entries confirm routing > uniform:
   - "Equal-weight composition is fragile" (CV=112.2%, one adapter poisons all)
   - "Uniform static scaling cannot make composition competitive" (best/single=2.57x)
   - "Validates per-input routing over static composition"

6. **Arrow routing** (Ostapenko et al., "Towards Modular LLMs by Building and Reusing a Library of LoRAs") — Zero-shot routing over LoRA libraries matches or outperforms joint training on held-out tasks. Confirms that even simple routing mechanisms provide substantial benefits.

## Contradicting Evidence

1. **Switch Transformers** (arXiv:2101.03961) — Showed k=1 routing works at scale with large experts. Our top-1 failure may be capacity-bound (small ternary adapters with only 400 training steps) rather than mechanism-bound. With better-trained adapters, top-1 might work. Our own REFERENCES.yml notes: "k=1 routing works at scale with large experts — our micro k=1 kill is capacity-bound, not mechanism-bound."

2. **LoRA Soups** (arXiv:2410.13025) — Simple weight averaging of LoRA adapters can compose skills effectively for practical tasks. Challenges the need for learned routing when domain boundaries are soft. However, LoRA Soups operates at merge time (static), while our routing is dynamic — these are complementary, not contradictory.

3. **Task Arithmetic / TIES / DARE** — Static merging methods achieve zero inference overhead by fusing adapter weights permanently. TIES (arXiv:2306.01708) resolves sign conflicts; DARE (arXiv:2311.03099) drops redundant params. These avoid the router entirely but lose the ability to activate different experts for different inputs. Our review noted the "7/15 domains where uniform wins" — for those domains, static merging might be equally good.

4. **Our own 1/N composition success** — "1/N scaling resolves composition catastrophe" (PPL from trillions to 2.36). At N=5, 1/N was already a massive win over unit-weight. The routing improvement (13.9% over 1/N at N=15) is significant but much smaller than the 1/N-vs-unit improvement. This suggests 1/N is the critical baseline; routing is an optimization on top.

## Alternative Approaches (What We Could Try Instead)

1. **LD-MoLE adaptive k** (arXiv:2509.25684) — Instead of fixing k=2, learn the optimal number of experts per token per layer. Would address our finding that some domains prefer uniform (low confidence → activate more experts) while others prefer concentrated routing (high confidence → top-2). Implementation: replace our fixed top-2 with Sparsegen-based differentiable routing.

2. **MoRAM self-routing** (arXiv:2501.13662) — Uses adapter weights as key-value pairs in associative memory, eliminating the external router entirely. Would solve the router retraining problem (adding/removing adapters currently invalidates the router).

3. **Union-of-Experts** (arXiv:2410.02285) — Expert weights contain intrinsic routing signal (via SVD analysis). Router-free approach that could scale better than our explicit classifier.

4. **Soft-MoE** (arXiv:2308.00951) — Soft token-to-expert assignment eliminates routing collapse and load balancing issues entirely. No top-k needed. Could be the cleanest path to scaling beyond N=15.

5. **FlyLoRA** (arXiv:2510.08396) — Frozen sparse A matrices serve as implicit routers. Zero router overhead, though less adaptive than learned routing.

6. **Confidence-gated hybrid** (our own recommendation from PAPER.md) — Use router confidence to switch between top-2 (high confidence) and uniform (low confidence). This is the simplest next step and doesn't require new infrastructure.

7. **TIES+routing hybrid** — Use TIES/DARE to pre-merge adapters with overlapping domains (e.g., merge science+chemistry+physics into a "STEM" adapter), then route among the merged groups. Reduces N for the router while keeping benefits of both approaches.

## Implications for Next Experiments

1. **exp_bitnet_per_token_routing should be renamed** to exp_bitnet_per_sequence_routing throughout HYPOTHESES.yml and PAPER.md. The naming overstates what was tested.

2. **True per-token routing is the obvious next step.** MoLoRA proves per-token > per-sequence. But it requires N forward passes per token (or fused kernels), which conflicts with our $0 Apple Silicon constraint. Parking this for when bitnet.cpp supports multi-adapter inference.

3. **Load balancing is critical before scaling to N=25+.** LD-MoLE's analytical sparsity control is the most principled approach. Without it, routing collapse is a known failure mode at high N.

4. **The confidence-gated hybrid (top-2 for high confidence, uniform for low confidence) should be tested.** This is the cheapest experiment that could capture most of the remaining gain — addressing the 7/15 domains where uniform still wins.

5. **Router retraining on adapter pool changes is a production concern.** For the evolve track (clone-and-compete), adapters change frequently. MoRAM or Union-of-Experts approaches that avoid explicit routers deserve evaluation.

6. **PPL-only evaluation remains a weakness.** exp_bitnet_task_eval showed PPL doesn't always predict task accuracy. The routing advantage should be validated with task-based evaluation (exp_bitnet_kr_test_evaluation).

## New References to Add

- LD-MoLE (arXiv:2509.25684): Adaptive expert count via differentiable routing. ICLR 2026. Relevant to exp_bitnet_per_token_routing scaling.
- MoLoRA (arXiv:2603.15965): True per-token LoRA routing. Qwen3-1.7B > Qwen3-8B. Relevant to exp_bitnet_per_token_routing and VISION.md router design.
- LoRA Soups (arXiv:2410.13025): Simple averaging for skill composition. Relevant to composition strategy selection.
- Arrow/Modular LLMs (Ostapenko et al.): Zero-shot routing over LoRA libraries. Relevant to router-free alternatives.
