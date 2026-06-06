# LEARNINGS: capability_benchmark_full_system (KILLED)

## Core Finding

SFT adapters at per-domain optimal scales (s=20) degrade ALL out-of-distribution
benchmarks (GSM8K -15pp, code -10pp, NER -7.4pp, MMLU -5pp). Prior in-distribution
gains (+700% math, +36% code) do not transfer. The format-capability equivalence
hypothesis is refuted: adapter FORMAT helps only when evaluation distribution matches
training distribution.

## Why This Happened

### 1. SFT Adapters Are Distribution-Specific Format Copiers

SFT (supervised fine-tuning with response-only loss) trains the adapter to reproduce
the exact format of its training data. When applied to prompts from a different
distribution (e.g., GSM8K format vs our math training data format), the learned format
transformation becomes noise that *interferes* with the base model's general capability.

This is consistent with the LIMA hypothesis (Zhou et al., arXiv:2305.11206) — but with
a crucial refinement: LIMA says fine-tuning teaches format, not knowledge. What we
discovered is that the format learned is **distribution-specific**, not
**task-general**. A math SFT adapter learns "format of our math training data," not
"format of mathematical reasoning."

### 2. High Scale Amplifies Distribution Mismatch

At s=20, the LoRA perturbation is large enough that distribution-specific formatting
overwhelms the base model's general capability. The perturbation-to-base ratio
ρ=0.144 at s=20 (Finding from lora_scale_ablation) is modest in absolute terms, but
the FORMAT direction of the perturbation is misaligned with OOD task requirements.

MMLU finance at s=1 showed 0pp delta (base preserved), while MMLU math at s=20
showed -10pp. This confirms the scale-interference relationship.

### 3. NTP vs SFT Confound

Finding #237 showed +10pp GSM8K with NTP adapters. This experiment used SFT adapters
and got -15pp. The difference is structural:

- **NTP (next-token prediction):** Trains on ALL tokens (prompt + response). Learns
  broader distributional patterns. Energy gap routing works at 88% accuracy (Finding #205).
- **SFT (response-only loss):** Trains only on response tokens. Learns narrow
  instruction-following format. Energy gap routing collapses to 36% (Finding #205).

The OOD degradation may be SFT-specific, not fundamental to adapter composition.
This is the single most important confound in the experiment.

### 4. Statistical Power Was Insufficient

The adversarial review correctly identified that no individual benchmark result is
statistically significant (GSM8K: Fisher p=0.45, code: p=1.0, MMLU: p=0.56). The
kill was based on directional consistency across all benchmarks (all negative), which
provides weak collective evidence but not confident falsification.

## Confirming Evidence

- **LIMA (Zhou et al., arXiv:2305.11206):** Fine-tuning teaches format, not knowledge.
  Our result refines this: the format learned is distribution-specific.

- **TIES-Merging (Yadav et al., arXiv:2306.01708, NeurIPS 2023):** When merging
  multiple task-specific adapters, parameter interference causes OOD degradation.
  Their solution (trim, elect sign, merge) addresses multi-adapter conflicts but not
  single-adapter OOD transfer. Confirms that adapter weight perturbations degrade
  OOD performance.

- **LoRA Land (Zhao et al., arXiv:2405.00732):** Comprehensive study of 25+ LoRA
  adapters showing that fine-tuned LoRAs are highly task-specific and transfer poorly
  across tasks. Directly supports our in-distribution vs OOD finding.

- **Finding #180 (this project):** SFT composed adapters improve GSM8K from 0.36 (NTP)
  to 0.52 (SFT), exceeding base 0.44 — but this was measured in-distribution. The
  improvement doesn't survive OOD evaluation.

- **Finding #258 (this project):** CPT is a no-op at 200 iter / 80K tokens / rank-16.
  Adapters encode format, not knowledge. Consistent with the observed failure.

## Contradicting Evidence

- **Finding #237 (this project):** GSM8K +10pp was "consistent across 3 experiments"
  — but those used NTP adapters, not SFT. The consistency of NTP adapter gains on
  GSM8K has not been retested in this experiment, so it may still hold. The contradiction
  is between adapter types, not between experiments.

- **LoRAHub (Huang et al., arXiv:2307.13269):** Composing multiple LoRA adapters via
  learned coefficients can improve OOD task performance. However, LoRAHub uses
  gradient-free optimization on a few OOD examples to find composition weights — a
  fundamentally different approach from our fixed-scale oracle routing.

## Alternative Approaches (With Paper References)

1. **NTP Adapters for OOD Robustness:** Finding #237 showed +10pp GSM8K with NTP
   adapters. NTP trains on all tokens, producing broader distributional shift that
   may transfer better to OOD. The simplest fix: repeat this experiment with NTP
   adapters from real_data_domain_experts/adapters/.

2. **DARE (Yu et al., arXiv:2311.03099):** Drop And REscale — randomly drops a
   fraction of LoRA delta parameters before merging. Sparsifying the perturbation
   reduces interference while preserving task-relevant directions. Could reduce OOD
   degradation at high scales.

3. **Distribution-Aware Scale Selection:** Instead of fixed per-domain scales, detect
   OOD queries and reduce scale. The entropy gating pre-filter (63% tokens skip at
   1.13% cost, from our prior work) is a prototype: skip/reduce adapter application
   when the base model is already confident.

4. **Task-Arithmetic (Ilharco et al., arXiv:2212.04089):** Treating adapters as task
   vectors and using arithmetic operations (addition, negation) for composition.
   Combined with TIES or DARE, this provides principled multi-adapter OOD composition.

5. **Inverted-U Scale Optimization:** Finding from lora_scale_ablation showed
   GSM8K peaks at scale=8 (59.3%), not scale=20 (52.3%). Lower scales may avoid
   OOD degradation while capturing most of the in-distribution benefit.

## Implications for Next Experiments

### The In-Distribution Value Proposition Is Valid

For a DEPLOYMENT system where routing matches queries to domain-appropriate adapters,
in-distribution-only improvement is fine. The user sends a math question → routed to
math adapter → adapter helps because the query is in-distribution. The benchmark
failure is a deployment-scenario mismatch, not a fundamental architecture failure.

### The Benchmark Question Requires NTP Adapters

The NTP vs SFT confound (Finding #237 vs this experiment) is the most important
unresolved question. Before concluding that "composition doesn't help on benchmarks,"
we must test NTP adapters on the same benchmarks. This is a clean, falsifiable test.

### Scale=4-8 May Be the Sweet Spot

The inverted-U pattern (lora_scale_ablation) and the s=1 preservation (MMLU finance)
suggest that moderate scales preserve base capability while still providing format
benefit. The two-regime model (FORMAT at s≤4, CAPABILITY at s≥20) should be revised:
moderate scales (4-8) may combine format benefit with OOD robustness.

### Statistical Power Must Increase

Future benchmark experiments need N≥100 per benchmark to achieve meaningful statistical
power. At N=20, Fisher exact test cannot detect even 15pp differences reliably (p=0.45).

## Recommended Follow-Up

1. **Repeat with NTP adapters on same benchmarks** — Directly resolves the NTP/SFT
   confound. Motivated by Finding #237 (+10pp GSM8K with NTP) and Finding #205
   (NTP preserves routing distinguishability). Literature: NTP preserves broader
   distributional properties (all-token training signal) that may transfer OOD.

2. **Scale sweep on OOD benchmarks** (s=1,2,4,8,20) — Find the scale where OOD
   degradation begins. Motivated by inverted-U pattern from lora_scale_ablation
   (peak at s=8 for GSM8K). Would test whether moderate scales avoid interference.

Neither follow-up is blocking for the P0 deployment track. The deployment use case
(in-distribution routing) is not affected by OOD benchmark degradation.
