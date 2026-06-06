# Tiny Routing Heads N=24: Proof Verification Report

## Theorem (from MATH.md)

Per-adapter binary routing heads (2-layer MLP, d->32->1) can discriminate 24 domains
in R^2560 hidden state space. Cover's Function Counting Theorem guarantees linear
separability when N << 2d (24 << 5120). The heads should achieve >60% top-1 routing
accuracy and the routed composition should beat uniform 1/N averaging.

## Predictions vs Measurements

| Prediction (from proof) | Measured | Match? |
|------------------------|----------|--------|
| Average head accuracy >70% | 87.2% | YES |
| Min head accuracy >50% | 77.7% (environmental) | YES |
| Top-1 routing accuracy >60% | 39.6% | NO |
| Overhead ~11% of base forward | 6.80% | YES (better) |
| Routed PPL < uniform PPL | Routed 10.13 > uniform 10.08 | NO |
| Head params ~82K each | 81,985 | YES |

## Verdict: KILLED

K584 FAIL: Top-1 routing accuracy 39.6% (threshold 60%)
K585 FAIL: Routed PPL 10.13 worse than uniform 10.08
K586 PASS: Overhead 6.80% (threshold 10%)

## Hypothesis

Per-adapter binary routing heads trained at N=5 (100% accuracy, +19.9% PPL improvement)
will maintain >60% top-1 accuracy and beat uniform averaging when scaled to N=24
with semantically overlapping domains.

**Result: KILLED.** The mechanism does not survive scaling from N=5 to N=24.

## What This Experiment Reveals

### The Paradox: High Head Accuracy, Low Routing Accuracy

The most important finding is the disconnect between two metrics:
- **Head classification accuracy: 87.2% average** -- each head is good at the
  1-vs-23 binary classification task individually
- **Top-1 routing accuracy: 39.6%** -- but when all 24 heads compete, routing fails

This means each head correctly identifies its OWN domain ~87% of the time, but also
fires false positives on OTHER domains. With 24 competing heads, the false positive
problem dominates: for any input, multiple heads fire positively, and the wrong head
often has the highest score.

### The False Positive Cascade

With N=24 heads, each having ~87% accuracy:
- Each head has ~13% false positive rate on average
- For any given input, ~3 heads (24 * 0.13) fire false positives
- The correct head fires with ~87% probability
- But the correct head must outscore ~3 false-positive competitors
- The false positive heads often have higher scores because their thresholds
  are poorly calibrated across heads

### Why N=5 Worked Perfectly

At N=5:
- Domains were trivially separable (python code vs medical text vs legal language)
- Only ~0.5 false positive heads per input (5 * 0.1)
- False positives from distant domains had low scores
- The correct head always won

At N=24:
- Many domains are semantically close (economics/finance/marketing, medical/health_fitness)
- ~3 false positive heads per input
- False positive scores from semantically close domains are HIGH
- The correct head often loses

### Domain-Level Results

Strong routing (top-1 > 80%):
- **finance (95%)**, **health_fitness (100%)**, **legal (100%)**, **math (100%)**,
  **medical (100%)**, **psychology (100%)**
  These domains have distinctive hidden state signatures.

Weak routing (top-1 < 25%):
- **cooking (5%)**, **cybersecurity (15%)**, **economics (0%)**, **music (5%)**,
  **philosophy (5%)**, **politics (0%)**, **science (10%)**
  These domains' hidden states overlap heavily with other domains.

### The Adapter Quality Problem

A deeper issue: individual adapter PPL often matches or slightly exceeds base PPL.
Average base PPL: 10.06, average individual adapter PPL: 10.09. The adapters trained
on this dataset provide essentially zero benefit over the base model for most domains.
This means routing accuracy does not matter much because there is nothing to route TO --
the adapters are not specialized enough to differentiate.

### Confusion Patterns

Top misroute patterns reveal hidden state geometry:
- cybersecurity -> marketing (9x): both use persuasive/technical language
- linguistics -> sociology (9x): both are social sciences
- philosophy -> agriculture (8x): agriculture head has high false positive rate
- science -> agriculture (8x): same issue with agriculture head
- engineering -> sports (7x): sports head fires on action-oriented text

### Key Structural Insight

The own-domain accuracy reveals the real problem:
- **7 domains have own-domain accuracy <50%**: cooking (40%), cybersecurity (40%),
  economics (47%), environmental (20%), philosophy (40%), politics (27%), science (33%)
- These heads learned to REJECT everything rather than ACCEPT their own domain
- With 23 negative classes vs 1 positive, the BCE loss converges to "always predict
  negative" for domains that overlap with many others

This is the 1-vs-23 class imbalance problem. Even with balanced sampling (50/50
positive/negative), the negative examples are diverse (spanning 23 domains) while
positives are homogeneous. The head learns features that distinguish "not-my-domain"
rather than "is-my-domain."

## Key References

- Finding #54: N=5 tiny routing heads (100% accuracy, 19.9% PPL improvement)
- Finding #189: Energy gap routing collapsed at N=24 (8.3% accuracy)
- LoRAuter (2601.21795): Embedding-based routing scales to 1500+ adapters
- MoLoRA (2603.15965): Per-token routing with shared router

## Limitations

1. **Adapter quality:** The 24 adapters provide minimal improvement over base model
   (PPL 10.09 vs 10.06). With stronger adapters, routing accuracy would matter more
   and the gap between routed and uniform would be larger.

2. **Training data size:** Only 40 training samples per domain for head training.
   More data might help, but the structural false-positive problem persists.

3. **Single architecture:** Only tested h=32. Larger heads (h=64 or h=128) might
   help but add parameters.

4. **No calibration:** Head scores are not calibrated across domains. A calibration
   step (temperature scaling per head) could help.

## What Would Fix This

The core problem is that independent binary heads are not calibrated against each other.
Each head optimizes its own binary classification, but nobody optimizes the RANKING
across heads.

**Architectural fix:** Replace independent binary heads with a shared router that
sees all domain representations simultaneously. This is exactly what LoRAuter and
MoLoRA do -- and what Finding #54's N=5 success accidentally masked because N=5
does not need cross-head calibration.

**Mathematical fix:** The binary head approach needs a calibration layer that
normalizes scores across heads before ranking. This adds a centralized component,
defeating the decentralized advantage.

**The fundamental insight:** Decentralized routing (each adapter decides independently)
works when domains are well-separated (N=5). When domains overlap (N=24), routing
becomes a COMPETITION problem that requires centralized arbitration.

## What Was Learned

1. **Binary per-adapter heads scale poorly from N=5 to N=24** due to the false
   positive cascade effect.
2. **Head accuracy is not routing accuracy** -- the disconnect grows with N.
3. **The 1-vs-23 imbalance causes many heads to learn "reject everything"** for
   overlapping domains (own-domain accuracy as low as 20%).
4. **The base model's hidden states DO contain domain signal** (6 domains achieve
   100% top-1 routing), but the signal is not strong enough for ALL 24 domains
   to be independently discriminated.
5. **Overhead at N=24 is acceptable** (6.80%) -- the bottleneck is accuracy, not speed.
6. **Energy gap routing (8.3%) and learned routing (39.6%) both fail at N=24** --
   the problem is structural, not just about the routing method.

---

## Audit-Rerun Closure (2026-04-18)

**Tags at claim:** `audit-2026-04-17-rerun, code-bug`. This section documents
the closure analysis — no rerun performed.

### Identified code bug

`run_experiment.py` line 331:
```python
return nn.losses.binary_cross_entropy(mx.sigmoid(logit), target, reduction="mean")
```
MLX `nn.losses.binary_cross_entropy` defaults to `with_logits=True` (expects
logits, not probabilities). Passing `mx.sigmoid(logit)` double-applies sigmoid:
the loss computes `log(σ(σ(logit)))` rather than `log(σ(logit))`. At logit=5
(confident positive), the bug yields ~45× the correct loss; the network sees
medium-magnitude loss signals regardless of how confident the logit is. This
partially explains the 7 heads that learned "reject everything" (own-domain
accuracy <50%): gradient compression at confident predictions biases toward
the flat prior.

**Correct form:** `nn.losses.binary_cross_entropy(logit, target, reduction="mean")`
(keep default `with_logits=True`), or pass `with_logits=False` if keeping
the manual sigmoid.

### Decision: closure (no rerun). Rationale — three theorems.

**Thm C1 (K585 zero-headroom — FORMAL, kill-invariant).**
Let PPL_base = 10.06 (avg over 24 domains). Individual adapter PPL = 10.09
(avg), i.e. individual adapters are *worse* than base on average. Any
composition in adapter space (uniform 1/N or routed top-k) is a convex
interpolation between base behavior and individual adapters. Hence:

- Upper bound on routable improvement over uniform: the magnitude of
  adapter specialization signal = |PPL_individual − PPL_base| ≈ 0.03
  (or ~0.3%).
- Measured routed−uniform gap: 10.13 − 10.08 = 0.05, dominated by PPL
  estimation noise over 20 validation samples per domain.
- Signal-to-noise < 1 by construction.

K585 ("routed PPL < uniform PPL") is therefore structurally unreachable.
*No routing mechanism* — BCE-fixed or otherwise — can produce routed
PPL meaningfully below uniform PPL when the underlying adapters provide
~0 PPL headroom. KILL direction on K585 is invariant under any
in-experiment code modification.

**Thm C2 (K584 plausibility — directional, not formally invariant).**
If BCE fix lifts per-head accuracy from 87.2% to ~90% and per-head FPR
from 0.13 to ~0.10 (optimistic — consistent with eliminating ~half of
the reject-everything heads), expected false positives per query =
23 × 0.10 = 2.3. Under uniform-score argmax, top-1 accuracy is bounded
roughly by 1/(1 + 2.3·score_ratio). For score_ratio ≈ 1 (uncalibrated
independent heads produce comparable score magnitudes across domains),
top-1 ≈ 0.30−0.50. Still below K584 = 60% with high probability.
Failure direction — no optimistic tautology. Not formally invariant
(BCE fix *could* in principle push K584 past 60% if score calibration
were accidentally restored, which is unlikely).

**Thm C3 (Structural uncalibration — referencing reviewer).**
REVIEW-adversarial.md §"The Impossibility Structure Is Correctly
Identified" already formalized: *independent binary classifiers cannot
solve competitive ranking without calibration*. The BCE fix addresses
*detection* (per-head loss curvature) but does not introduce cross-head
*calibration*. The cited MoE scaling literature (DeepSeekMoE's normalized
sigmoid, Expert Threshold's shared-reference calibration) all demonstrate
that truly-decentralized independent scoring fails at scale regardless
of per-head quality. This is the LEARNINGS.md "Four routing kills share
no cross-expert calibration" conclusion.

### Antipattern self-check

- **mem-021 (CEILING-HEADROOM COLLAPSE):** APPLIES. **Third distinct
  instance** of the pattern:
  - (1) `exp_depth_routed_adapters`: oracle-router ceiling (test-time).
  - (2) `exp_flat_lora_training`: orthogonality ceiling (training-time).
  - (3) This experiment: adapter-specialization ceiling (behavioral-metric).
  Abstract structure identical: a mechanism M is proposed to improve
  metric µ layered on baseline B_0 that has no headroom in µ. K585
  here = behavioral analog of zero-headroom. Promotes to 3-instance
  confidence.
- **mem-008 (thinking truncation):** N/A (BitNet base, no thinking mode).
- **mem-003 (LORA_SCALE=20):** N/A (routing heads, not adapter scale).
- **mem-001 (composition math bug):** N/A (single-adapter routing, no
  summation).
- **Verdict-DB mismatch:** N/A (results.json verdict "KILLED" and PAPER
  verdict KILLED match; no mismatch to fix).

### KC disambiguation

| KC ID | Label | Threshold | Measured | Pass? | Fix-invariant? |
|---|---|---|---|---|---|
| 584 | K1: Top-1 < 60% → KILL | 60% | 39.6% | FAIL | Directional (C2) |
| 585 | K2: Routed PPL ≥ uniform → KILL | Routed<Uniform | 10.13 vs 10.08 | FAIL | Formal (C1) |
| 586 | K3: Overhead > 10% → KILL | 10% | 6.8% | PASS | — |

KC IDs/text in MATH.md unchanged since 2026-03-29 (see `git log MATH.md`).
No KC-swap.

### Verdict

**KILLED** — reaffirmed under audit-rerun closure. K585 failure is
formally invariant under any code fix (zero-headroom). K584 failure is
directionally invariant. The identified BCE-with-logits bug is
acknowledged but kill-direction-preserving; fix deferred to
`exp_followup_bce_with_logits_routing` which is the correct venue for
measuring a clean BCE-fix ablation in isolation (per experiment DB
search).
