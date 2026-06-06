# LEARNINGS: Energy-Gated Composition

## Core Finding

Binary energy gating of LoRA adapters is **structurally impossible**: LoRA adds low-rank
capacity that universally reduces NLL on all inputs (Delta_E <= 0 for all 25 adapter-domain
pairs), making any threshold at tau <= 0 vacuous. The gate never fires, producing output
identical to uniform composition. This is a mathematical necessity of overparameterization,
not an empirical surprise — it was provable a priori without running any experiment.

## Why This Happened

### The Overparameterization Trap

A LoRA adapter adds rank-r matrices (A, B) to each weight matrix W. The adapted model
W + B@A contains the base model as a special case (A=B=0). Any well-trained optimizer
therefore achieves NLL_adapted <= NLL_base on *any* input distribution — the adapted model
has a strictly larger hypothesis class. This is the standard bias-variance tradeoff:
more parameters = lower training loss on all data, including out-of-domain.

Our energy gap matrix confirmed this universally: all 25 adapter-domain pairs had
negative Delta_E (range: -0.49 to -2.18 nats). The Neyman-Pearson boundary at Delta_E=0
doesn't exist in LoRA-space.

OP-LoRA (arxiv 2412.10362) explicitly documents this "blessing of dimensionality" —
overparameterized LoRA converges faster and achieves lower loss precisely because the
enlarged parameter space makes optimization easier. Our gating failure is the downstream
consequence: if every adapter reduces loss everywhere, loss-based gating has zero
discriminative power.

### The Ranking-vs-Gating Distinction

This experiment conflated two different capabilities of the energy gap:
- **Ranking** (Finding #182): Which adapter reduces NLL *most*? AUC=0.942 on math. Works.
- **Gating** (this experiment): Which adapters to *include/exclude*? Requires a boundary
  between positive and negative gaps. Fails because all gaps are negative.

The fundamental error was treating a ranking signal as a classification signal. Finding #182's
AUC was computed on the ranking of negative gaps, not on a positive/negative boundary. The
finding should have been interpreted as "energy gap ranks adapter quality" (routing signal),
not "energy gap identifies harmful adapters" (gating signal).

### The Missing Proof Would Have Prevented This

The adversarial reviewer correctly identified that MATH.md contained no Theorem/Proof/QED.
The Neyman-Pearson lemma was cited but its preconditions (separable H0/H1 distributions)
were never verified. A one-paragraph proof:

> **Claim:** For any LoRA adapter (A, B) of rank r > 0 applied to model theta_0, if training
> converges (loss decreases), then NLL(x|theta_0 + BA) <= NLL(x|theta_0) for all x in the
> training support. By continuity, this extends to nearby distributions.
>
> **Corollary:** Delta_E <= 0 almost everywhere, so binary gating at tau=0 has zero rejection rate.

This would have killed the experiment in the proof stage, saving 115 minutes of compute.

## Confirming Evidence

- **OP-LoRA (arxiv 2412.10362)** — Explicitly shows overparameterized LoRA achieves lower
  loss via enlarged parameter space. The "blessing of dimensionality" is the mechanism
  that makes all adapters reduce NLL universally.

- **LoRA Learns Less and Forgets Less (arxiv 2405.09673)** — Shows LoRA preserves base
  model performance better than full fine-tuning, consistent with small but universal
  NLL reductions rather than large domain-specific shifts.

- **Our Finding #178** — PPL correlates at r=0.08 with task quality. The energy gap
  discriminator (Finding #182) partially fixed this by using relative NLL. But relative
  ranking ≠ absolute gating. This experiment proves the remaining gap.

- **Our generation_quality_test LEARNINGS** — The "two-world pattern" (math +142.1%,
  prose -5% to -13.5%) persists identically under energy gating, confirming the problem
  is in the adapters/composition, not in the selection mechanism.

## Contradicting Evidence

- **NotebookLM sources found no paper claiming LoRA universally reduces NLL.** The
  literature generally assumes adapters are domain-specific. Our finding that ALL adapters
  reduce NLL on ALL domains may be specific to our setup (scale=20 SFT adapters on
  BitNet-2B-4T). Adapters trained with different objectives (DPO, RLHF) or lower scales
  might not show universal reduction.

- **X-LoRA (arxiv 2402.07148)** demonstrates successful gating using hidden-state
  features, suggesting that a different signal (not NLL) can achieve what energy gating
  cannot. The approach uses learned layer-wise mixing weights, bypassing the NLL boundary
  problem entirely.

## Alternative Approaches

### 1. Energy Gap as Routing Signal (Top-k Selection)
Instead of binary gating (include/exclude), use energy gap magnitude to SELECT the best
adapter per query. This preserves the proven ranking signal (AUC=0.942) while avoiding
the gating failure. Equivalent to oracle top-1 routing with a learned signal.
- **Evidence:** Finding #182 (ranking works), generation_quality_test (oracle routing
  works for code +14.4%, math +142.1%).
- **Feasibility:** Zero additional training, same energy gap computation.

### 2. Hidden-State Routing (X-LoRA / L-MoE)
Learn a gating network that reads hidden states to produce per-token mixing weights.
Avoids NLL entirely — uses representational similarity instead.
- **Evidence:** X-LoRA (arxiv 2402.07148) demonstrated on biology+math+reasoning.
  L-MoE (arxiv 2510.17898) end-to-end trains lightweight gates. MoLoRA (arxiv 2603.15965)
  achieves Qwen3-1.7B+4adapters > 8B model via per-token routing.
- **Cost:** Requires training a router. Our Gumbel-sigmoid routing heads (2.32% overhead,
  100% accuracy) are already in this category.

### 3. Task-Representation Routing (LoRAuter)
Use small validation sets to compute task embeddings for routing. Training-free.
- **Evidence:** LoRAuter (arxiv 2601.21795) achieves 101.2% of oracle on StoryCloze.
- **Limitation:** Requires a few examples per task at inference time.

### 4. TIES Merging for Interference Resolution
Resolve parameter conflicts through sign-majority voting + magnitude pruning before merging.
- **Evidence:** LoRAuter found TIES merging recovers 70.09% on StoryCloze (vs oracle 72.00%).
  Our generation_quality_test LEARNINGS recommended this.
- **Feasibility:** No retraining needed. Weight-space operation compatible with pre-merge.

## Implications for Next Experiments

1. **The proof-first discipline would have caught this.** The impossibility was derivable
   from first principles (overparameterization → universal NLL reduction → vacuous gating).
   Future experiments MUST prove the test statistic has power before computing it. This is
   not a new rule (CLAUDE.md already requires it) but a concrete failure case to remember.

2. **Energy gap is reframed from gating signal to routing signal.** Finding #182 stands —
   the ranking capability is validated (AUC=0.942). But any future use must be top-k
   selection, not binary include/exclude.

3. **The NLL landscape under LoRA is flat-negative.** All adapters reduce NLL everywhere.
   This means NLL-based methods can only RANK, never GATE. Any future quality gate must
   use a non-NLL signal (hidden states, task representations, generation-based scoring).

4. **The two-world pattern is now confirmed across 3 experiments:** generation_quality_test,
   self_embedding_quality_discriminator, and energy_gated_composition all show the same
   split — adapters help structured tasks, hurt prose. The problem is upstream (training
   objective produces mode collapse on prose), not in composition or routing.

## Recommended Follow-Up

**Energy gap top-k routing** — Use energy gap magnitude to select the single best adapter
per query (top-1 routing by Delta_E magnitude). This tests the RANKING signal in a
GENERATION context, which Finding #182 validated for task accuracy but not for text quality.
- **Motivation:** Finding #182 (AUC=0.942 ranking), Finding #184 (gating fails but ranking
  signal survives), generation_quality_test (oracle routing works on code/math).
- **Literature:** MoLoRA (arxiv 2603.15965) validates per-token adapter selection.
- **Why this fixes the failure:** Top-k routing doesn't need a positive/negative boundary.
  It only needs the ranking to be correct, which is the proven capability.
