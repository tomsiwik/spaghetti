# LEARNINGS: exp_bitnet_sft_generation_v3 (KILLED)

## Core Finding

Third generation quality kill confirms a structural tension: SFT training optimizes adapters for general instruction-following, making one adapter (code) universally dominant at NLL reduction. This collapses energy gap routing from 88% (NTP) to 36% (SFT), killing 3/5 domains. The TWO-WORLD pattern persists: structured tasks (code +37%, math +600% correctness) benefit enormously while prose tasks (medical -14%, legal -7%, finance -6%) degrade. The disease is not routing architecture but the assumption that NLL separability survives SFT training.

## Why This Happened

**Single-expert dominance is a well-known MoE failure mode.** The code SFT adapter absorbed 42/50 routing decisions because SFT on structured data (code) produces the largest NLL reduction across ALL domains. This matches the "expert collapse" phenomenon documented in MoE literature:

- **Representation collapse** (arxiv 2204.09179): Sparse MoE routers collapse to a subset of experts via self-reinforcing feedback — experts receiving more tokens train faster, attracting even more tokens. Our energy gap routing has no load-balancing mechanism to counteract this.
- **LoRAuter** (arxiv 2601.21795, 2602.21222): Found that instruction scaffolding IMPROVES task disambiguation, contradicting the hypothesis that SFT destroys domain signal. But LoRAuter uses a trained SupCon encoder, NOT raw NLL. The issue is not that SFT destroys signal — it's that NLL is the wrong routing signal for SFT adapters.
- **MoLoRA** (arxiv 2603.15965): Per-token routing over LoRA experts achieves 1.7B outperforming 8B, but uses learned router weights trained jointly with experts, not post-hoc energy gap.

**The structural tension:** SFT training makes adapters better at instruction-following generally. Better instruction-following = lower NLL on all inputs. Energy gap routing selects the adapter with lowest NLL. Therefore, the best general instruction-follower wins ALL routing decisions. These goals (good adapters vs. distinguishable adapters) are in fundamental tension under NLL-based routing.

## Confirming Evidence

1. **Expert collapse in MoE** (arxiv 2204.09179, arxiv 1701.06538): Without load balancing, routing collapses to a few dominant experts. Our energy gap routing has zero load-balancing mechanism.
2. **Standing Committee phenomenon** (MoE Illusion of Specialization): MoE models develop domain-invariant "standing committee" experts handling general tasks, with peripheral experts for domain-specific work. Our code adapter IS the standing committee.
3. **Auxiliary-Loss-Free Load Balancing** (arxiv 2408.15664): Shows that even standard load-balancing losses can introduce interference gradients. Dynamic bias injection or Expert Threshold routing are cleaner solutions.
4. **Our own Finding #203**: All adapters are general-purpose improvers (routing errors cost only ~13% PPL). DDR = 1.13 means specificity gap is small. SFT amplifies this by making the best adapter even more generally capable.

## Contradicting Evidence

1. **LoRAuter** (arxiv 2602.21222): Successfully routes 1500+ adapters using SupCon-trained sentence encoder. Instruction scaffolding HELPS distinguish tasks, contradicting "SFT destroys signal." The key: LoRAuter routes on task description embeddings, NOT on NLL profiles.
2. **Instruction Tuning Helps MoE** (Mixture-of-Experts Meets Instruction Tuning): MoE models benefit MORE from instruction tuning than dense models. SFT doesn't kill expert specialization when routing is learned jointly during training.
3. **OMoE** (arxiv 2501.10062): Orthogonal MoE prevents expert collapse via Gram-Schmidt process on expert representations. Collapse is caused by lack of orthogonal constraints, not by SFT.
4. **NeuroLoRA** (arxiv 2603.12378): Uses Contrastive Orthogonality Loss to enforce separation between expert subspaces. Expert similarity is addressable at the architectural level.

## Alternative Approaches (Paper-Backed)

1. **Contrastive Retriever Routing** (LoraRetriever, arxiv 2402.09997): Train a retriever on contrastive pairs (same-task vs different-task) to select adapters. Decouples routing signal from NLL entirely. Proven at multi-task scale.
2. **Task-Aware Vector DB Routing** (LoRAuter, arxiv 2602.21222): Route at task level using SupCon-trained encoder + vector database. Training-free at inference, scales to 1500+ adapters. Eliminates energy gap entirely.
3. **Expert-Router Coupling Loss** (ERC): Treats each expert's router embedding as proxy token, enforces bidirectional coupling. Prevents any single expert from dominating by design.
4. **Per-Token Learned Routing** (MoLoRA, arxiv 2603.15965): Train lightweight router jointly with LoRA experts. Per-token granularity, Qwen3-1.7B+4 adapters > 8B. Router learns to distinguish despite SFT similarity.
5. **Expert Threshold Routing** (arxiv 2603.11535): Maintains EMA threshold per expert. Token routed only if score exceeds expert-specific threshold. Auxiliary-loss-free, prevents single-expert absorption.
6. **Contrastive Orthogonality Loss** (NeuroLoRA, arxiv 2603.12378): Enforce separation between expert subspaces via contrastive loss during training. Addresses root cause (expert similarity) rather than routing mechanism.

## Implications for Next Experiments

1. **Energy gap routing is dead for SFT adapters.** Three kills (NTP+Gumbel, LLM-judge, SFT+energy gap) prove that NLL-based routing fails when one adapter dominates. No further energy-gap experiments should be attempted.

2. **The routing signal must be decoupled from NLL.** LoRAuter and LoraRetriever prove that task-level semantic routing works at scale. The next generation quality attempt must route on task embeddings, not energy gaps.

3. **The code adapter's universal improvement is the most actionable finding.** Math correctness: 10% → 70% (+600%) via wrong adapter. This suggests that for structured tasks, ANY good SFT adapter improves generation. The question is whether domain-correct routing adds incremental value beyond this universal improvement.

4. **Prose evaluation remains unsolved.** Three kills, and medical/legal/finance are still evaluated by keyword density (r=0.08 with quality). Without execution-based eval for prose domains, we cannot distinguish "adapter hurts quality" from "metric doesn't measure quality."

5. **The reviewer's prerequisite for v4 is correct:** Before any next generation quality attempt, MATH.md must prove that routing works when one adapter has lower NLL than all others on all domains.

## Recommended Follow-Up

1. **exp_contrastive_routing_n5** — Use LoraRetriever-style contrastive retriever (arxiv 2402.09997) to route SFT adapters. Motivation: decouples routing from NLL, proven at multi-task scale. Tests whether task-semantic routing beats energy gap on our 5 SFT adapters.

2. **exp_universal_adapter_ablation** — Test whether the code SFT adapter alone (no routing) matches or beats routed composition on all 5 domains. Motivation: Finding #206 (code adapter = universal improver) + Finding #203 (routing errors cost ~13%). If one adapter matches routed composition, routing is unnecessary and the architecture simplifies dramatically.

3. **exp_prose_execution_eval** — Build execution-based evaluation for prose domains (USMLE for medical, bar exam for legal, CFA for finance). Motivation: Three generation kills with keyword density, which Finding #179 showed correlates r=0.08 with quality. Cannot make progress without proper prose evaluation.
