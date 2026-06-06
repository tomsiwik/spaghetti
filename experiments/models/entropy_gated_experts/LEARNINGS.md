# Learnings: exp_entropy_gated_expert_selection

## Core Finding

63% of tokens can skip LoRA adapter composition with only 1.13% PPL degradation,
using Otsu-thresholded base model output entropy as a binary gate. The mechanism
works consistently across 5 domains (0.5-1.9% degradation each) but the two-pass
implementation is 2.1x slower -- the value is as a pre-filter for routing, not
standalone speedup.

## Why This Happened (Literature-Grounded)

The core mechanism -- most tokens don't need expert computation -- is well-established
in the MoE and early-exit literature. Three forces explain it:

1. **Sensitivity concentration.** Critical task knowledge is not uniformly distributed
   across tokens. MoBiLE (Mixture of Big-Little Experts) showed that token importance
   varies dramatically, with most tokens being "unimportant" and safely handled by a
   lightweight branch (1.6-1.7x speedup, negligible accuracy loss). pQuant (Decoupled
   MoE) demonstrated the same principle: a 1-bit shared branch handles all standard
   tokens, with a top-1 router activating an 8-bit expert only for sensitive tokens.

2. **Micro-expert redundancy.** CAMERA's analysis of MoE layers showed that performance
   gains do not scale proportionally with the number of experts activated per token.
   Forcing "simple" tokens through multiple heavy experts provides no meaningful
   refinement. Our 63% skip rate is consistent with this -- most tokens are adequately
   served by the base model alone.

3. **Entropy as confidence proxy.** CALM (Schuster et al., 2022) and DeeBERT (Xin et al.,
   2020) established entropy-based early exit for Transformers. Our application is
   structurally similar but operates at the adapter composition level rather than
   layer-exit level. The mechanism is the same: Shannon entropy of the output distribution
   identifies tokens where the model is already confident.

## Confirming Evidence

- **MoBiLE** (Mixture of Big-Little Experts): Token importance routing achieves 1.6-1.7x
  speedup by reducing experts for unimportant tokens. Directly analogous to our approach
  but within MoE layers rather than adapter composition.

- **pQuant** (Decoupled MoE): Forces most parameters into a 1-bit backbone, routes only
  sensitive tokens to 8-bit experts. Confirms the "most tokens need only the base" principle
  that our entropy gating exploits.

- **CALM** (Schuster et al., 2022, arxiv 2207.07061): Softmax confidence for early exit.
  The foundational work showing that entropy-based computation skipping works in
  Transformers.

- **DeeBERT** (Xin et al., 2020, arxiv 2004.12993): Entropy-based early exit for BERT.
  Showed that entropy thresholds can identify "easy" tokens across diverse NLP tasks.

- **EdgeNav-QE**: Dynamic early exit achieving 82.7% latency reduction on edge devices.
  Confirms entropy-based gating is practical at edge scale (relevant to our M5 Pro target).

- **Diffusion LMs**: Entropy sum-based decoding that adapts to text complexity by unmasking
  tokens in parallel until cumulative entropy exceeds a threshold. Different domain, same
  principle: entropy reveals intrinsic complexity variation.

## Contradicting Evidence

- **Softmax calibration problem** (Guo et al., 2017, "On Calibration of Modern Neural
  Networks"): Deep networks produce poorly calibrated softmax scores -- overconfident on
  OOD inputs. Our raw softmax entropy gate will systematically MISS tokens where the base
  model is confidently wrong (low entropy but incorrect). This is the single biggest threat
  to entropy gating at scale. Temperature scaling or Platt scaling would improve threshold
  reliability.

- **Epistemic vs aleatoric blindness**: Information-theoretic uncertainty decomposition
  shows softmax entropy (SME) only captures aleatoric uncertainty (inherent data noise),
  not epistemic uncertainty (model's lack of knowledge). A token can have low entropy
  because the input looks unambiguous, even when the model has no domain knowledge. This
  means entropy gating will preferentially skip experts in EXACTLY the cases where domain
  expertise is most needed -- unfamiliar inputs that happen to pattern-match familiar
  structure.

- **SFT entropy regularization**: Supervised fine-tuning with entropy regularization
  flattens token distributions toward uniformity, artificially increasing entropy without
  improving reasoning. If adapters were trained with entropy regularization, the entropy
  distribution would shift, invalidating any fixed threshold. Our Otsu recalibration per
  deployment mitigates this, but the underlying signal becomes less informative.

- **Domain-level confound** (noted in adversarial review): Our Otsu eta=0.68 on the global
  distribution partially reflects domain separability (python mean H=0.79 vs legal H=2.61)
  rather than per-token confidence variation. With more similar domains, the clean Otsu
  split would degrade. No paper directly contradicts this, but the MoE scaling literature
  (Apple ICML 2025) shows that routing effectiveness degrades with more overlapping experts.

## Alternative Approaches (What We Could Try Instead)

### 1. Learned Routers (replace entropy with trained signal)
- **L2R (Learning to Route)**: Gumbel-sigmoid non-competing routing. Instead of competitive
  softmax gating, each adapter gets an independent binary activation score. This yielded
  up to +19.2 point improvement over softmax routing in continual learning tasks. Critical
  insight: our entropy gate is a binary all-or-nothing decision, but L2R shows that
  non-competing activation of multiple adapters simultaneously is strictly better.

- **pQuant token-level router**: Lightweight fixed top-1 router operating at token level.
  Avoids the calibration problems of raw entropy because the router is trained to identify
  sensitive tokens directly, not via a proxy signal.

### 2. Speculative Proxy Verification (TriSpec)
TriSpec introduces a lightweight proxy that evaluates draft tokens, approving easy sequences
locally and bypassing the heavy model for uncertain tokens. Achieves 35% speedup over
standard speculative decoding with 50% fewer target model invocations. This could replace
entropy gating entirely: instead of computing entropy then deciding, use a proxy model to
speculatively generate and verify.

### 3. Micro-Expert Granularity (CAMERA)
Instead of a monolithic on/off gate per adapter, CAMERA analyzes sub-matrix contribution
variance within experts. This suggests our binary "compose all or compose none" gate is too
coarse. A finer-grained approach would selectively apply specific adapter layers or
sub-matrices based on token sensitivity.

### 4. Task-Level Squads (Mod-Squad)
Rather than per-token routing (which causes memory thrashing on edge devices), Mod-Squad
activates sparse "squads" of experts at the task level, guided by mutual information loss
that encourages expert specialization while allowing cooperation. This might be more
practical for M5 Pro serving than per-token entropy gating.

### 5. Temperature-Scaled Entropy
Guo et al. (2017) showed that simple temperature scaling dramatically improves confidence
calibration. Before abandoning entropy gating, we should test whether calibrated entropy
(with temperature learned on a holdout set) produces a better threshold than raw entropy.
This is the cheapest improvement to try.

## Implications for Next Experiments

### 1. Entropy gating should be tested as pre-filter for tiny_routing_heads (confirmed)
The experiment correctly identifies the integration path. Entropy pre-filters 63% of tokens
cheaply; the routing heads handle the remaining 37%. The question is whether entropy +
routing heads beats routing heads alone on compute (not quality).

### 2. Calibration is the critical gap
Raw softmax entropy is the weakest possible confidence signal. Before scaling, test
temperature-scaled entropy on a holdout calibration set. If calibrated entropy significantly
improves the quality-skip tradeoff, entropy gating has legs. If not, switch to a learned
router (L2R Gumbel-sigmoid or pQuant-style).

### 3. Watch for epistemic failure mode
When testing entropy gating on new domains not seen during adapter training, explicitly
check for the epistemic blindness failure: low-entropy tokens where the base model is
confidently wrong and the adapter would have corrected it. This is the kill criterion
for entropy gating at scale.

### 4. Consider non-competing adapter activation
L2R's Gumbel-sigmoid routing is directly applicable to our architecture. Instead of
entropy -> {base, compose-all}, train a lightweight router that outputs independent
activation scores for each adapter. This subsumes entropy gating and handles the
multi-adapter case more naturally.

### 5. The two-pass problem is architectural, not implementational
S3 FAIL (2.1x slower) is not a bug to fix -- it reveals that entropy gating only makes
sense when it avoids additional computation (pre-filter for routing) rather than adding
a second pass. Any implementation must compute entropy as a byproduct of an existing
forward pass, not as a separate evaluation.
