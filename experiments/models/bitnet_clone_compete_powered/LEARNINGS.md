# Learnings: bitnet_clone_compete_powered

## Core Finding

Warm-start inheritance in LoRA adapter evolution provides no generalizable advantage over cold-start retraining — a fresh adapter trained from scratch on the same data matches or beats the warm-started clone on every metric. The prior N=38 "supported" result was a small-sample artifact (62.1% win rate collapsed to 28.9% at N=500).

## Why This Happened (Literature-Grounded)

The null warm-start result is predicted by two independent lines of research:

1. **"The Appeal and Reality of Recycling LoRAs" (2602.12323)** found that when adaptively merging from ~1,000 LoRAs, randomly initialized LoRA parameters yielded similar performance to carefully selected ones. The specific choice of LoRA to recycle had "little importance." This directly predicts our finding: warm-starting from the original adapter's weights is no better than random initialization because the inherited weights are effectively overwritten during continued training. The "recycling" mechanism works via regularization (constraining the loss landscape), not via genuine knowledge transfer.

2. **"Less is More: Undertraining Experts Improves Model Upcycling" (2506.14126)** found that long fine-tuning degrades merging performance because later training steps memorize a small set of hard examples that are subsequently forgotten during merging. Our clone's concentrated improvement on hard samples (winning few samples by large margins) is exactly this memorization pattern — the warm-start creates a "specialization bias" toward the hardest examples from the original's distribution, which doesn't generalize. The cold-start, lacking this bias, distributes improvement more evenly and produces a better per-sample profile.

The win-rate-vs-aggregate paradox (clone loses 71% of samples but matches aggregate PPL) is a variant of **Simpson's paradox** — the clone's large wins on a minority of hard samples compensate for its many small losses on easy samples. As the adversarial review correctly noted, this may also be partly a domain-mismatch artifact: the original was trained on law-stack-exchange while the tournament included lex_glue ECHR data, giving the original distributional advantage on one portion of the tournament.

## Confirming Evidence

1. **"The Appeal and Reality of Recycling LoRAs" (2602.12323)**: Random LoRAs merge equally well as carefully selected ones. Adaptive merging provides limited benefit over training a new LoRA on the same data. Directly confirms that warm-start inheritance adds no value. *CONFIRMS* our null result.

2. **"Position: Pause Recycling LoRAs" (2506.13479)**: Position paper arguing the community should stop recycling LoRAs and study mechanisms instead. Key finding: orthogonality does NOT lead to semantic disentanglement — inter-LoRA orthogonality alone is insufficient for true compositionality. Raises broader concern about assuming LoRA weight inheritance carries semantic meaning. *CONFIRMS* skepticism about inheritance mechanisms.

3. **"Less is More: Undertraining Experts Improves Model Upcycling" (2506.14126)**: Long training of experts that optimizes individual performance leads to degraded merging. Degradation traced to memorization of hard examples that dominate late training steps and are forgotten during merging. Easy examples survive merging; hard examples don't. *CONFIRMS* our observation that the clone's concentrated hard-sample improvement is fragile.

4. **"Subspace Geometry Governs Catastrophic Forgetting in Low-Rank Adaptation" (2603.02224)**: Forgetting follows geometric law F = alpha(1 - cos^2(theta_min)) + beta, where theta_min is the minimum principal angle between task gradient subspaces. At high subspace angles (as in our BitNet adapters with |cos|=0.001), forgetting is approximately rank-invariant. *CONFIRMS* that our near-random orthogonality means warm-start inheritance has minimal geometric advantage — the subspaces barely interact regardless of initialization.

## Contradicting Evidence

1. **Population-Based Training (Jaderberg et al., 2017)**: PBT demonstrates that clone-perturb-select can outperform random search by inheriting good hyperparameters and weights. However, the PBT advantage comes primarily from **hyperparameter scheduling** (learning rate, augmentation), not from weight inheritance per se. Our experiment isolated the weight inheritance component (same hyperparameters for clone and cold-start), and it provides no benefit. PBT's success is not contradicted — we simply showed the mechanism isn't weight inheritance.

2. **CA-LoRA and LoRA transfer methods**: Several methods (LoRASuite, CA-LoRA) demonstrate successful LoRA weight transfer across model versions or compressions. However, these involve transferring adapters to a **different** base model (e.g., compressed version), where the transfer maps provide structural advantages. Our setting — inheriting within the same model — is fundamentally different. The inherited weights are in the same parameter space and can be recovered from scratch.

3. **OPLoRA / O-LoRA (Orthogonal Projection LoRA)**: Continual learning methods that warm-start from previous task LoRAs while constraining orthogonality. These show benefits from warm-starting in sequential task learning. However, these methods explicitly **constrain** the new adapter to an orthogonal subspace, which provides structural guarantees that naive warm-starting (our clone) does not. The difference is: constrained warm-start (beneficial) vs. unconstrained warm-start (no benefit).

## Alternative Approaches (What We Could Try Instead)

1. **Retrain-from-scratch with combined data**: The simplest alternative and our experiment's clear winner. When fresh domain data becomes available, train a new adapter from scratch on all available data. This is simpler, cheaper, and produces better per-sample profiles than clone-compete. This becomes the default Evolve mechanism.

2. **Constrained warm-start (O-LoRA / OPLoRA style)**: If warm-starting is desired (e.g., for composition stability), constrain the new adapter to an orthogonal subspace of the original. This provides the structural guarantees that naive warm-starting lacks. Reference: O-LoRA (2310.14152), OPLoRA (2510.13003).

3. **Aggressive early stopping for composition**: Per "Less is More" (2506.14126), shorter training yields better merging/upcycling. If composition quality is the primary concern, train adapters for fewer steps than would optimize individual PPL. Our current 200-400 steps may already be in the right regime.

4. **Merge-before-forget continual merging**: "Merge before Forget" (2512.23017) proposes continual merging of LoRA adapters to prevent forgetting. Instead of discrete clone-compete rounds, continuously merge adapter updates into the base. Could be an alternative Evolve protocol.

5. **MoLoRA per-token routing**: Instead of tournament-selecting a single winner, keep both adapters and route per-token (MoLoRA, 2603.15965). Avoids the selection problem entirely — each adapter serves the samples where it excels.

## Implications for Next Experiments

1. **Evolve phase redesigned**: Clone-compete is dead. The Evolve mechanism becomes "retrain on more data" — when fresh domain data is available, train a new adapter from scratch on combined (original + fresh) data. This is simpler, equally effective, and avoids the complexity of tournament infrastructure.

2. **The per-sample PPL tournament methodology is validated**: Even though clone-compete is killed, the powered tournament methodology (N=500, binomial + Wilcoxon + t-test + Cohen's d) is proven useful for comparing adapters. Preserve it for future adapter comparison tasks (e.g., comparing retrained adapters, evaluating composition quality).

3. **Composition stability may benefit from initialization proximity**: The one surviving signal is that the warm-started clone showed marginally better composition behavior (all 5 domains improved vs original, compared to cold-start where 4/5 non-legal domains marginally regressed). This is within noise (0.26% max) but directionally consistent with the subspace geometry theory — adapters closer in parameter space to the original cause less composition perturbation. Worth tracking in future experiments but not worth optimizing for.

4. **Consecutive kills (2) suggest re-examining Evolve assumptions**: Both clone-compete and clone-compete-powered are killed. The Evolve phase should focus on data accumulation (proven mechanism) rather than evolutionary selection (disproven mechanism). Consider whether the Evolve phase needs tournament selection at all, or whether monotonic data accumulation with periodic retraining is sufficient.

5. **Watch for hard-sample memorization**: Per "Less is More" (2506.14126), any adapter trained for many steps risks memorizing hard examples that don't compose well. Monitor per-sample PPL distributions (not just aggregates) when evaluating adapter quality for composition.

## New References to Add

| Paper | ArXiv ID | Relevance |
|-------|----------|-----------|
| Less is More: Undertraining Experts Improves Model Upcycling | 2506.14126 | Long training hurts merging via hard-example memorization — explains clone's concentrated improvement pattern |
| Subspace Geometry Governs Catastrophic Forgetting in Low-Rank Adaptation | 2603.02224 | Geometric theory: forgetting = f(principal angle). At high angles (our |cos|=0.001), warm-start provides no geometric advantage |
| Merge before Forget: Continual LoRA Merging | 2512.23017 | Alternative Evolve protocol: continuous merging instead of discrete tournament rounds |
