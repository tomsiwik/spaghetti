# LEARNINGS: Hierarchical Two-Stage Routing at N=24

## Core Finding

Hierarchical routing perfectly solves cluster selection (97.3%) but within-cluster routing (41.5%) is identical to flat routing (39.6%) — the representation bottleneck is hierarchy-invariant. The deeper discovery (Finding #198): oracle PPL = base PPL (0.04% delta), meaning all 24 adapters provide near-zero specialization. Seven routing kills at N=24 are explained by one root cause: there is nothing meaningful to route.

## Why This Happened

The seven routing kills follow a diagnostic sequence that progressively peeled back layers of symptoms to reach the disease:

1. **Kills 1-3 (energy gap, energy gating, binary heads)**: Blamed routing mechanism failures (magnitude disparity, non-negative energy, false positive cascade).
2. **Kill 4 (centralized softmax)**: Revealed architecture-independence — same 6/24 domains succeed regardless of mechanism. Diagnosed as "weak domain signal in mean-pooled hidden states."
3. **Kill 5 (embedding routing)**: Centroid collapse (cos 0.986). Proved transformer layers ADD signal, not destroy it.
4. **Kill 6 (hierarchical)**: Stage-1 clustering works perfectly (97.3%), but stage-2 = flat. The N=5 success exploited domain distinctness, not N-smallness.
5. **Kill 7 (Finding #198)**: Oracle PPL (10.05) = base PPL (10.06). The adapters themselves provide 0.04% benefit. Routing is vacuous.

The progressive diagnosis matches the SIGReg method: kills 1-5 treated symptoms, kill 6 exposed the wrong assumption (N<=5 sufficiency), and kill 7 found the disease (adapter quality, not routing quality).

**Why adapters don't specialize**: 400 samples at 200 steps on a 2.4B ternary model is insufficient for domain-specific knowledge acquisition. The SFT data composition research (referenced in NotebookLM) shows general alignment plateaus at ~1000 samples, but complex specialized tasks (math, code) require significantly more data with continued scaling improvement. Our 400-sample adapters likely learned only surface-level formatting, not domain knowledge.

**The N=5 success was real but misleading**: Those 5 adapters (medical, code, math, legal, creative) were trained on maximally distinct domains with different data pipelines. The 26.5% PPL improvement proves adapters CAN specialize — under the right conditions. The N=24 setup (400 samples each, rank-16, 200 steps) simply doesn't produce that specialization.

## Confirming Evidence

- **QLoRA (arxiv 2305.14314)**: Achieves 99.3% ChatGPT performance on Vicuna benchmark with "small high-quality dataset" — but this is general alignment, not multi-domain specialization requiring inter-adapter discrimination.
- **SFT data composition research**: General alignment at ~1000 samples, but math/code continue scaling. Our 400 samples per domain is below even the alignment plateau.
- **MoE-Sieve (arxiv 2603.24044)**: Confirms routing is highly skewed — small subset of experts handles most tokens. When experts don't specialize, routing collapses to uniform.
- **Expert specialization collapse** (arxiv 2602.14159): Auxiliary load balancing loss causes expert overlap and uniform routing, hindering specialization. Our adapters show a similar lack of differentiation, but caused by insufficient training rather than load balancing interference.
- **MoCE (arxiv 2509.10513)**: Dual-stage routing (sequence-level cluster then token-level expert) achieves specialization through data partitioning — validates our hierarchical approach structurally, but requires genuinely specialized experts as input.

## Contradicting Evidence

- **LoRAuter (arxiv 2601.21795)**: Routes across 1500+ adapters successfully. Key difference: uses SupCon-trained encoder on task embeddings, not raw hidden states. More critically, those adapters were individually trained to high quality on diverse tasks with sufficient data.
- **Task-Aware LoRA Adapter Composition**: Dynamically merges adapters across 22 NLP datasets using frozen encoder + retrieval. Success depends on adapters that actually perform differently on their target tasks.
- **DeepSeekMoE**: 256 experts with finely segmented routing + isolated shared experts. Critical enabler: experts are trained jointly with the routing mechanism end-to-end, ensuring specialization emerges from the training process itself.
- **QVAC Fabric**: LoRA fine-tuning of 1-bit LLM on ~300 documents (18K tokens). Apparent contradiction to our "insufficient data" diagnosis — but QVAC targets general capability, not discriminable domain specialization among 24 competing adapters.

The pattern across all contradicting evidence: successful multi-adapter systems either (a) train adapters to genuine specialization with sufficient data/compute, or (b) use end-to-end training where routing and specialization co-evolve.

## Alternative Approaches

1. **Adapter quality audit + retraining** (motivated by Finding #198): Before any further routing experiments, retrain N=24 adapters with more data (>=2000 samples) and/or more steps until oracle PPL is >=5% below base. The N=5 success (26.5% PPL improvement) proves this is achievable.

2. **LoRA scaling laws (arxiv 2501.03152, MIUB)**: Use mutual information upper bound to predict minimum data/rank for specialization at N=24. Could provide a principled data requirement rather than trial-and-error.

3. **Distillation pipeline** (from NotebookLM): Query a large teacher model to generate specialized training data per domain. SOLE framework achieves quality adapters for ~$0.25 each. Could scale to 24 domains affordably.

4. **End-to-end routing + adapter co-training** (MoCE arxiv 2509.10513, Grouter arxiv 2603.06626): Train routing and experts jointly so specialization emerges from the training signal. Grouter decouples routing from representation for accelerated MoE training.

5. **Leave-One-Out PPL screening** (from NotebookLM): Pre-routing quality gate — any adapter that increases PPL when added gets pruned. Would have caught the vacuous adapters immediately.

6. **Multi-discipline fine-tuning scaling laws (arxiv 2602.11215)**: LoRA-MoE with asymmetric parameter sharing enables cross-discipline knowledge sharing then expert specialization. May provide a principled approach to training 24 domain experts.

## Implications for Next Experiments

1. **No more routing experiments at N=24 until Finding #198 is resolved.** The reviewer's recommendation is correct: oracle PPL must be >=5% below base before any routing mechanism can be meaningfully tested.

2. **The P0 deployment track must address adapter quality first.** The critical path item `exp_generation_quality_test` requires adapters that actually produce different (better) text per domain. Current adapters cannot do this.

3. **The N=5 setup is the existence proof.** Whatever was different about N=5 adapter training (data quality, domain distinctness, training duration) should be replicated at N=24.

4. **Seven kills provide a complete routing diagnostic framework:** energy gap (magnitude disparity), energy gating (impossible), binary heads (false positive cascade), centralized softmax (weak signal), embedding routing (centroid collapse), hierarchical (stage-2 = flat), oracle PPL (nothing to route). Future routing work at any N should run the oracle PPL check FIRST.

5. **The "Theorem 0" gap is a methodological lesson.** All seven experiments assumed adapters provided meaningful specialization without proving it. Future experiment designs must include a precondition check: "Theorem 0: Under training conditions X, adapters achieve oracle PPL at least Y% below base."

## Recommended Follow-Up

**Immediate (P0):** `exp_adapter_quality_audit` — Measure per-domain oracle PPL for all 24 adapters. Identify which adapters provide zero benefit vs which provide marginal benefit. Determine if the problem is uniform (all adapters weak) or bimodal (some work, some don't).

**Motivation:** Finding #198 showed average oracle = 10.05 vs base = 10.06, but some domains (agriculture: 14.14 > 14.06 base) actively hurt. An audit would reveal the distribution of adapter quality, not just the mean.

**Then:** Retrain failing adapters with more data (>=2000 samples, inspired by SFT scaling research) or with distillation from a teacher model. The N=5 adapters proved 26.5% PPL improvement is achievable on this same base model.
