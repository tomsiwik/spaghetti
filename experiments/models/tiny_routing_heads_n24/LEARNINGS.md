# LEARNINGS: Tiny Routing Heads at N=24

## Core Finding

Independent per-adapter binary routing heads collapse from 100% accuracy (N=5) to 39.6%
(N=24) due to a false positive cascade: with 24 heads at ~13% FPR each, ~3 false positives
compete against the true positive per query. Detection accuracy (87.2% per head) does not
imply ranking accuracy — this is the fourth routing mechanism killed at N=24, confirming
that decentralized routing is structurally incompatible with overlapping-domain scaling.

## Why This Happened

### False Positive Cascade (Detection ≠ Ranking)

Each binary head solves "is this my domain?" independently, achieving 87.2% average accuracy.
But the routing decision requires argmax over 24 uncalibrated sigmoid outputs — a ranking
problem, not 24 independent detection problems. With N=24 heads at FPR ≈ 0.13:

- Expected false positives per query: (N-1) × FPR = 23 × 0.13 ≈ 3.0
- The correct head fires with ~87% probability, but must outscore ~3 false positive competitors
- False positive scores from semantically close domains (economics/finance, medical/health) are HIGH
- Result: correct head loses the argmax competition ~60% of the time

This could have been predicted a priori from a simple combinatorial argument: P(correct argmax)
decreases with N when per-head FPR is constant. The MATH.md used Cover's theorem (point
separability) when the failure mode was in cross-head score ranking (distribution overlap).

### Class Imbalance Causes Head Degeneration

7 of 24 heads learned to reject everything (own-domain accuracy <50%). The 1-vs-23 BCE loss
with diverse negatives spanning 23 domains converges to "always predict negative" for domains
whose hidden states overlap with many others. This is a known failure mode of one-vs-rest
classification at high class counts (Rifkin & Klautau, 2004).

### N=5 Success Was Accidental

At N=5, domains were trivially separable (python code vs medical text vs legal language).
Only ~0.5 false positives per input. The mechanism appeared robust but was never stress-tested
against overlapping domains. Finding #54's 100% accuracy was an artifact of domain separation,
not routing mechanism quality.

## Confirming Evidence

- **MoE expert collapse literature**: NotebookLM confirms that routing instability and
  dominant experts absorbing routing mass is the primary failure mode of MoE systems. Our
  false positive cascade is the binary-head analog of this phenomenon.
- **DeepSeekMoE (arxiv 2401.06066)**: Uses normalized sigmoid gating — crucially, normalized
  across all experts, not independent binary decisions. The normalization is what makes
  independent-style scoring work at 256 experts. Our heads lack this cross-head normalization.
- **Expert Threshold (ET) routing**: Uses per-expert EMA thresholds calibrated against global
  token distribution. Succeeds at scale because thresholds are calibrated, unlike our
  uncalibrated sigmoid outputs. Key difference: ET routing calibrates independently but
  against a SHARED reference distribution.
- **Finding #189 (energy gap N=24 kill)**: Energy gap argmin routing collapsed to 8.3% at N=24
  due to adapter magnitude disparity (different failure mode, same structural conclusion:
  uncalibrated independent scoring fails at scale).
- **Finding #184 (energy gating kill)**: Energy gating impossible — non-negative energy gap.
- **Finding #27 (binary routing at N=24)**: Earlier kill at N=24 showed cooking regressed 26%,
  random routing beat trained routing — same false positive cascade mechanism.
- **Rifkin & Klautau (2004)**: "In Defense of One-Vs-All Classification" showed OVR can match
  multiclass methods when classifiers are well-calibrated, but acknowledged calibration
  failure at high class counts as the primary weakness.

## Contradicting Evidence

- **DeepSeekMoE normalized sigmoid**: Independent per-expert scoring DOES work at N=256,
  but only with normalization across experts. This suggests our failure is not intrinsic
  to independent evaluation, but to UNCALIBRATED independent evaluation.
- **DSelect-k (arxiv 2106.03760)**: Binary encoding formulation for MoE gating, successful
  at up to 128 tasks. But this uses a differentiable binary encoding with explicit
  sparsity control — a centralized mechanism in disguise.
- **Expert Threshold routing**: Independent per-expert thresholds succeed at scale, but
  thresholds are calibrated against global statistics. Not truly decentralized.

The pattern: every "independent" method that succeeds at scale includes a CALIBRATION
or NORMALIZATION step that introduces cross-expert information. Truly decentralized
(no cross-expert info) fails.

## Alternative Approaches

All proven approaches for routing to many (>10) adapters use centralized or calibrated scoring:

1. **LoRAuter (arxiv 2601.21795)**: Embedding-based routing via task representations.
   Scales to 1500+ adapters. Computes cosine similarity between query embedding and
   offline task embeddings. Centralized by design — all task representations compete
   in a shared embedding space. Training-free, O(T) complexity.

2. **MoLoRA (arxiv 2603.15965)**: Per-token learned routing with shared router network.
   Qwen3-1.7B + 4 adapters beats 8B model. Single centralized router sees all expert
   representations simultaneously. Requires training but achieves fine-grained routing.

3. **CLONE (arxiv 2506.02847)**: MoE router for dynamic LoRA selection on edge devices.
   Centralized router with hardware-accelerated LoRA Processing Unit for hot-swapping.
   Designed for exactly our deployment scenario (edge, dynamic, multi-adapter).

4. **CoMoL (arxiv 2603.00573)**: Dynamic Core Space Merging for MoE-LoRA. Merges
   adapters in shared core space, reducing interference. Centralized merge decision.

5. **MoE-Sieve (arxiv 2603.24044)**: Routing-guided LoRA for MoE fine-tuning. Profiles
   routing counts, applies LoRA only to top-25% routed experts. Confirms that routing
   is highly skewed — most tokens go to few experts. Reduces LoRA params by 70%.

6. **LoraRetriever (arxiv 2402.09997)**: Retrieve-then-compose framework using
   instruction-guided sentence embeddings to select top-k experts. Pushes routing
   outside the forward pass entirely.

7. **Our own tiny routing heads at N=5 (Finding #54)**: Still valid for small N.
   100% accuracy, 2.32% overhead. The mechanism works when domains are well-separated.

## Implications for Next Experiments

### Four routing kills confirm the structural conclusion

| Mechanism | Finding | N=24 Accuracy | Failure Mode |
|-----------|---------|---------------|--------------|
| Energy gap argmin | #189 | 8.3% | Adapter magnitude disparity |
| Energy gating | #184 | Impossible | Non-negative energy gap |
| Binary routing heads | #191 | 39.6% | False positive cascade |
| Softmax router | #28 | Earlier kill | Balance loss → uniform routing |

All four share one property: **no cross-expert calibration**. Each method scores
experts independently without reference to the other scores. This is the structural
disease, not the individual failure modes (which are symptoms).

### The fix is centralized ranking, not better detection

The project needs to pivot from "how to build better independent routers" to
"how to build a centralized router that fits within our constraints." The constraint
is: low overhead, no training on routing-specific data, scales to N=24+.

**LoRAuter-style embedding routing** is the strongest candidate:
- Training-free (uses existing sentence embeddings)
- O(T) complexity (one embedding per task, not per adapter)
- Proven at 1500+ adapters
- Fits our edge deployment constraints
- Our existing 24 adapters already have validation sets for building task representations

### What NOT to do next

- Do NOT try calibrating binary heads (temperature scaling, Platt scaling) — this adds
  a centralized component, defeating the decentralized advantage that was the whole point
- Do NOT try larger binary heads (h=64, h=128) — the false positive cascade is structural,
  not a capacity issue
- Do NOT try more training data per head — the 1-vs-23 imbalance is the problem, not
  data quantity

## Recommended Follow-Up

**Experiment: LoRAuter-style embedding routing at N=24**
- **Motivation**: Four routing kills (#184, #189, #191, #28) all share the same structural
  flaw — no cross-expert calibration. LoRAuter (arxiv 2601.21795) solves this by routing
  in a shared embedding space where all tasks compete on the same scale.
- **Literature**: LoRAuter proven at 1500+ adapters. Our 24 adapters with validation sets
  provide the required task representations.
- **Why it fixes the failure**: Embedding-based cosine similarity is inherently calibrated
  (all scores are in [-1, 1]) and centralized (all task embeddings live in the same space).
  No false positive cascade because there are no independent binary decisions.
- **Risk**: Requires a sentence embedding model (e.g., MiniLM, ~120MB). Must verify this
  fits within M5 Pro memory budget alongside BitNet-2B-4T + 24 adapters.

**Alternative: Use our existing tiny routing heads at N=5 granularity**
- Group 24 domains into ~5 clusters, route to cluster first (100% accuracy proven),
  then uniform-average within cluster. Hierarchical routing avoids the N=24 failure
  while leveraging proven N=5 mechanism.
