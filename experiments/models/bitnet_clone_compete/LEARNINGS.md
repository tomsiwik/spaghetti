# Learnings: bitnet_clone_compete

## Core Finding

Clone-compete evolution (clone worst adapter, continue training on fresh data, tournament-select winner) monotonically improves legal-domain PPL from 15.82 -> 14.50 -> 13.04 with negligible regression (0.06% max) across other domains in the 1/N composition — but the per-sample tournament is underpowered (p=0.265 at N=38), making the selection step statistically inconclusive even though the aggregate signal is strong.

## Why This Happened (Literature-Grounded)

The monotonic PPL improvement is expected from **continued training on fresh data** — this is the standard finding in LoRA continual fine-tuning literature. The warm-start advantage (clone inheriting original weights) means the clone starts from a good initialization, and additional domain data provides gradient signal that hasn't been seen. This is precisely analogous to **Population-Based Training** (Jaderberg et al., 2017), where models are cloned, perturbed, and selected via tournament. The perturbation here is continued training rather than hyperparameter mutation, but the mechanism is identical: exploit + explore.

The negligible regression (0.06%) is explained by the **regression bound O(epsilon/N)** derived in MATH.md Section 3. Under 1/N scaling, replacing one of N=5 adapters perturbs each non-target domain by at most (1/5) * ||theta_clone - theta_original||. Since continued training from a warm start produces small parameter-space deltas (the clone stays close to the original), regression is suppressed. This is consistent with **LoRA Soups** (Prabhakar et al., 2410.13025), which found that independently-trained LoRA adapters compose without catastrophic interference when weights are properly scaled.

The divergence between per-sample win rate (62.1%) and aggregate PPL improvement (-8.3%) in round 1, and the even split (50/50 wins) vs strong aggregate improvement (-10.1%) in round 2, suggests the clone primarily improves on **hard samples** (high-PPL legal texts). This pattern is known in the adaptive merging literature — "The Appeal and Reality of Recycling LoRAs" (2602.12323) found that continued training on new data often yields improvements concentrated on underrepresented examples rather than uniform improvement.

## Confirming Evidence

1. **Sakana AI, "Evolutionary Optimization of Model Merging Recipes" (2403.13187)**: Evolutionary search over model merging recipes finds unintuitive but effective combinations. Our clone-compete is a simplified single-adapter version of this — mutation via continued training instead of crossover in weight space. Published in Nature Machine Intelligence 2024, validating the evolutionary approach to model improvement.

2. **Population-Based Training (Jaderberg et al., 2017)**: The canonical framework for clone-perturb-select training. Our protocol is PBT applied to LoRA adapters with training as the perturbation operator. PBT shows monotonic improvement given sufficient population size and selection pressure — consistent with our results.

3. **LoRA Soups (Prabhakar et al., 2410.13025)**: Demonstrates that independently-trained LoRA adapters can be composed. Their CAT method (per-layer trainable weights) outperforms data mixing by 12%. Key finding relevant to us: composition works, and interference is manageable. Our 1/N uniform scaling is simpler than CAT but shows similar non-interference.

4. **MoLoRA (Shah et al., 2603.15965)**: Recent work on composable specialization via per-token routing. Shows "specialization beats scale" — a 1.7B model with specialized LoRAs exceeds 8B. Validates the fundamental premise that composing specialized adapters is viable at scale.

5. **O-LoRA / CL-LoRA (CVPR 2025)**: Continual learning with LoRA, constraining new task subspaces to be orthogonal to previously-learned ones. Confirms that orthogonal adapter composition prevents interference — our Grassmannian initialization achieves this by construction.

## Contradicting Evidence

1. **"The Appeal and Reality of Recycling LoRAs" (2602.12323)**: This paper is a critical counterpoint. When adaptively merging from a pool of ~1000 LoRAs, the specific choice of LoRAs to merge had **little importance** — randomly initialized LoRA parameters yielded similar performance. This suggests adaptive merging works via a "regularization effect" rather than genuine knowledge transfer. **Implication for us**: The warm-start advantage of cloning (inheriting original weights) may be less important than we assume. The missing cold-start control (Limitation 7 in PAPER.md) is essential — if a fresh adapter trained on the same data performs equally well, the "evolutionary inheritance" claim is undermined.

2. **"Pause Recycling LoRAs" (2506.13479)**: Position paper arguing that the community should stop recycling LoRAs and instead study **mechanisms** of why merging works or doesn't. Key finding: orthogonality does NOT lead to semantic disentanglement — inter-LoRA orthogonality alone is insufficient for true compositionality. This challenges our reliance on Grassmannian packing for composition quality.

3. **Long training hurts merging**: Research on LoRA upcycling found that **long training of LoRA experts hurts merging performance** and aggressive early stopping is needed. Our clone v2 trains 200 additional steps on an already-trained adapter — if this pattern holds, further evolution rounds may eventually degrade composition quality even if the individual adapter improves. The v2 round already shows 50/50 per-sample win rate despite better aggregate PPL, which could be an early warning.

## Alternative Approaches (What We Could Try Instead)

1. **CAT from LoRA Soups**: Instead of uniform 1/N scaling, use per-layer trainable scalar weights to combine adapters. LoRA Soups shows this outperforms uniform merging by 43%. Could be layered onto SOLE as an optional optimization, especially when clone-compete produces adapters that are no longer orthogonal to the originals.

2. **MoLoRA-style per-token routing**: Instead of tournament-select a single winner, keep both original and clone and let a learned router select per-token. Avoids the statistical power problem entirely — no need for N=200 tournament samples if both adapters are used where they excel.

3. **DARE merging (2311.03099)**: Random drop + rescale of delta parameters before composition. Could reduce interference from evolved adapters without relying solely on 1/N scaling.

4. **Evolutionary Model Merging (Sakana 2403.13187)**: Full evolutionary search over merging recipes (weight interpolation + data flow). Our clone-compete is adapter-level PBT; Sakana's approach searches the composition space itself. Could apply CMA-ES or similar to optimize the 1/N weights.

5. **Cold-start baseline**: The simplest alternative — train a fresh adapter from scratch on all available data (original + fresh) instead of warm-starting from clone. If this matches the clone-compete result, evolution adds no value over simply having more data.

## Implications for Next Experiments

1. **exp_bitnet_clone_compete_powered is correctly prioritized**: The underpowered tournament (N=38, p=0.265) is the primary weakness. The powered replication with N=200 samples should also include a **cold-start control** arm to disambiguate warm-start advantage from additional-data benefit.

2. **Watch for composition degradation**: The literature warning about long training hurting merging means we should track composition quality (not just individual adapter PPL) across multiple evolution rounds. If composition degrades while individual PPL improves, the Evolve phase needs a composition-aware fitness function.

3. **Consider keeping both original and clone**: MoLoRA-style per-token routing could make tournament selection unnecessary — both adapters serve different samples. This is more parameter-efficient than discarding the loser.

4. **The cold-start control is critical**: The Recycling LoRAs paper (2602.12323) strongly suggests that the warm-start advantage may be illusory. This must be tested before claiming the "evolutionary inheritance" mechanism works.

## New References to Add

| Paper | ArXiv ID | Relevance |
|-------|----------|-----------|
| The Appeal and Reality of Recycling LoRAs with Adaptive Merging | 2602.12323 | Critical counterpoint: random LoRAs merge equally well, warm-start may be illusory |
| Pause Recycling LoRAs | 2506.13479 | Position paper: orthogonality ≠ semantic disentanglement |
| MoLoRA: Composable Specialization via Per-Token Routing | 2603.15965 | Per-token routing alternative to tournament selection |
| CL-LoRA: Continual Low-Rank Adaptation | CVPR 2025 | Orthogonal continual learning confirms interference mitigation |
| Population-based Model Merging via Quality Diversity (CycleQD) | Sakana AI 2024 | Quality-diversity extension of evolutionary model merging |
