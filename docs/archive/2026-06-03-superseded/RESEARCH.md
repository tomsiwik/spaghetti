# Pierre — Supporting Research

> Literature survey supporting the strategy-composition thesis. 100+ papers organized by thesis pillar. Each entry links to the Pierre component it validates.

---

## Thesis Pillars

| Pillar | Claim | Pierre Component | Papers |
|---|---|---|---|
| **Strategies transfer, knowledge doesn't** | Problem-solving approaches generalize cross-domain; domain facts don't | Strategy adapters | 1-25 |
| **Behaviors distill separately from knowledge** | Reasoning patterns, approaches, and styles can be extracted independently from factual content | Hedgehog distillation, M2P | 26-60 |
| **Orthogonal composition works** | Adaptation deltas in orthogonal subspaces compose without interference | NRE compositor, Grassmannian init | 61-75 |
| **Small models + right strategies beat large monoliths** | A 4B model with composed strategies outperforms a 400B model with one compromised approach | Full Pierre stack | 76-90 |
| **Personal patterns extract from sessions** | User-specific approaches, preferences, and domain expertise can be extracted from interaction history | M2P, MEMENTO | 91-100+ |

---

## 1. Strategies Transfer Across Domains

**Pierre link:** Strategy adapters are domain-general. A `systematic` adapter helps math AND code AND medical reasoning.

| # | Paper | Arxiv | Key Finding |
|---|---|---|---|
| 1 | ProtoReasoning: Prototypes as the Foundation for Generalizable Reasoning in LLMs | 2506.15211 | Reasoning prototypes learned in one domain transfer directly to planning, proof, and other domains |
| 2 | General Reasoning Requires Learning to Reason from the Get-go | 2502.19402 | Disentangling knowledge from reasoning is necessary for general intelligence; reasoning strategies generalize better when trained independently |
| 3 | Self-supervised Analogical Learning (SAL) | 2502.00996 | Solution strategies transfer across structurally similar problems in different domains |
| 4 | MetaLadder: Analogical-Problem Reasoning Transfer | 2503.14891 | Abstract reasoning patterns transfer across problem types via analogy |
| 5 | STaR: Bootstrapping Reasoning With Reasoning | 2203.14465 | Reasoning strategies are self-reinforcing and domain-general; small seed of rationales bootstraps across tasks |
| 6 | Can Models Learn Skill Composition from Examples? | 2409.19808 | Training on mixed skills improves compositional generalization to novel combinations |
| 7 | Learning Composable Chains-of-Thought | 2505.22635 | Reasoning decomposes into reusable primitives that combine for novel tasks |
| 8 | Skills-in-Context Prompting (SKiC) | 2308.00304 | Strategies compose through context; knowledge doesn't need to |
| 9 | Combining Modular Skills in Multitask Learning | 2202.13914 | Modular skills improve generalization by disentangling and recombining known skills |
| 10 | Discovering Modular Solutions that Generalize Computationally | 2312.15001 | Modular hypernetworks discover compositional structure and meta-learn transferable policies |
| 11 | Circuit Component Reuse Across Tasks | 2310.08744 | Mechanistic circuits discovered for one task are reused across different tasks |
| 12 | Function Vectors in Large Language Models | 2310.15213 | Attention heads encode abstract functions, not facts; function vectors compose to create new tasks |
| 13 | At Which Training Stage Does Code Data Help LLMs Reasoning? | 2309.16298 | Code training enhances reasoning across ALL domains, not just code |
| 14 | On Code-Induced Reasoning in LLMs | 2509.21499 | Code data induces transferable reasoning strategies |
| 15 | DeepSeek-R1: Distillation beats direct RL | — | Distilling reasoning from a strong model to a small model produces better results than training the small model with RL directly |
| 16 | VibeThinker-1.5B | — | A 1.5B model with distilled reasoning beats a 671B model |
| 17 | FutureMind: Thinking-pattern priors transfer | — | Thinking pattern priors from one model transfer to another |
| 18 | Self-Discover: Self-compose reasoning structures | — | Models can self-compose reasoning structures from atomic modules |
| 19 | Trace2Skill: Skills transfer across model scales | — | Skills extracted from one model scale transfer to others |
| 20 | Disentangling Recall and Reasoning in Transformers | 2510.03366 | Recall and reasoning are neuroanatomically separable mechanisms in transformers |
| 21 | Causal Head Gating | 2505.13737 | Attention heads have separable behavioral functions (instruction following vs in-context learning) |
| 22 | Route to Reason: Adaptive Routing for Strategy Selection | 2505.19435 | Adaptive routing between reasoning strategies improves over single-strategy approaches |
| 23 | LIMA: Less Is More for Alignment | 2305.11206 | 1000 high-quality examples match GPT-4, proving knowledge is in pretraining, not fine-tuning |
| 24 | LIMO: Less is More for Reasoning | 2502.03387 | 817 samples sufficient for strong reasoning, confirming reasoning is a skill not knowledge |
| 25 | s1: Simple test-time scaling | 2501.19393 | 1000 reasoning traces beat o1-preview; reasoning strategies transfer |

---

## 2. Behaviors Distill Separately from Knowledge

**Pierre link:** Hedgehog distillation extracts strategy behaviors from teacher prompts; M2P extracts personal patterns from sessions. Both separate behavior from content.

### Behavioral Distillation

| # | Paper | Arxiv | Key Finding |
|---|---|---|---|
| 26 | Command-V: Pasting LLM Behaviors via Activation Profiles | 2506.19140 | Behaviors are copy-pasteable between architecturally different LLMs without backprop or training data |
| 27 | Subliminal Learning: Hidden Signal Behavioral Transfer | 2507.14805 | Behavioral traits transmit through semantically unrelated data; traits persist after filtering |
| 28 | SkillFactory: Self-Distillation for Cognitive Behaviors | 2512.04072 | Cognitive skills (problem-solving strategies) are isolatable and reinforceable through self-distillation |
| 29 | Structured Agent Distillation | 2505.13820 | Agent behaviors decompose into reasoning + action spans, independently transferable |
| 30 | RL vs Distillation: Accuracy and Capability | 2505.14216 | Distillation transfers behavioral capability; RL only transfers accuracy |
| 31 | MiniLLM: Knowledge Distillation of LLMs | 2306.08543 | Reverse KL transfers behavioral distribution patterns (precision, quality, diversity) distinct from knowledge |
| 32 | Distilling Step-by-Step | 2305.02301 | Small models on LLM rationales outperform the original LLMs with less data |
| 33 | Symbolic Chain-of-Thought Distillation (SCoTD) | 2306.14050 | Step-by-step reasoning is a behavioral pattern transferable via distillation |
| 34 | Teaching Small Language Models to Reason | 2212.08410 | Reasoning is an independently distillable skill via CoT outputs |
| 35 | Orca: Progressive Learning from Explanation Traces | 2306.02707 | Imitating reasoning process (behavior) > imitating answer (knowledge) |
| 36 | Small Models Struggle to Learn from Strong Reasoners | 2502.12143 | Challenge is behavioral pattern matching, not knowledge transfer |
| 37 | Mentor-KD: Multi-step Reasoning Distillation | 2410.09037 | Graduated behavioral transfer outperforms direct knowledge transfer |
| 38 | Phi-4-Mini-Reasoning | 2504.21233 | Structured CoT training cultivates reasoning as behavioral skill independent of parameter count |
| 39 | Mixed Distillation for Better Reasoning | 2312.10730 | Combining PoT and CoT behavioral patterns transfers reasoning ability |
| 40 | TinyLLM: Learning from Multiple Teachers | 2402.04616 | Multiple LLM teachers improve reasoning diversity through behavioral transfer |
| 41 | LaMini-LM: Distilled from Instructions | 2304.14402 | Instruction-following behaviors compress well to small models |
| 42 | TAPIR: Task-aware Curriculum Distillation | 2405.13448 | Instruction-following is a behavioral skill distilled via curriculum planning |
| 43 | SPaR: Self-Play with Tree-Search Refinement | 2412.11605 | Instruction-following is an improvable behavioral capability |

### Knowledge vs Behavior Separation

| # | Paper | Arxiv | Key Finding |
|---|---|---|---|
| 44 | Representation Engineering (RepE) | 2310.01405 | Cognitive phenomena (honesty, harmlessness) are independently manipulable from factual knowledge |
| 45 | Activation Addition (ActAdd) | 2308.10248 | Behaviors are steerable at inference time without optimization or fine-tuning |
| 46 | Precise Attribute Intensity Control | 2510.12121 | Behavioral attributes (formality, sentiment) are continuous separable dimensions |
| 47 | AlphaEdit: Null-Space Knowledge Editing | 2410.02355 | Knowledge editable without affecting behavior; occupies separable subspaces |
| 48 | Knowledge Editing through CoT (EditCoT) | 2412.17727 | Reasoning strategies editable as separate layer from factual knowledge |
| 49 | Refusal Is Mediated by a Single Direction | 2406.11717 | Safety behavior is a single direction in activation space, independently controllable |
| 50 | Who's Harry Potter? Approximate Unlearning | 2310.02238 | Factual knowledge removable while preserving reasoning capabilities |
| 51 | LoRA Learns Less and Forgets Less | 2405.09673 | LoRA underperforms full FT in target but better preserves other capabilities — behavioral regularization |
| 52 | A Closer Look at Instruction Tuning Limitations | 2402.05119 | Instruction tuning fails to inject new knowledge; it shapes existing knowledge expression |
| 53 | From Language Modeling to Instruction Following | 2310.00492 | SFT rotates representations toward instruction-following format, doesn't add knowledge |
| 54 | SFT Data Composition Effects on Abilities | 2310.05492 | Different abilities (math, code, instruction) scale independently with data composition |
| 55 | Catastrophic Forgetting via Implicit Inference | 2309.10105 | Forgetting is an inference phenomenon, not a weight-overwrite problem — separable |
| 56 | How Abilities in LLMs Are Affected by SFT | 2310.05492 | Abilities respond independently to data composition, confirming separability |
| 57 | Attention Heads Survey | 2409.03752 | Heads serve specialized roles (induction, copying, retrieval) — behaviors are structurally localized |
| 58 | Inferring Head Functionality from Parameters (MAPS) | 2412.11965 | Behavioral roles encoded in weight structure, readable without inference |
| 59 | Interpretability in the Wild (IOI Circuit) | 2211.00593 | 26 attention heads form 7 functional groups for a single task — behaviors are distributed but compositional |
| 60 | Attention Filters and MLP Stores | 2508.00901 | Self-attention filters information; MLP stores it — structural separation of retrieval vs knowledge |

---

## 3. Orthogonal Composition Works

**Pierre link:** Grassmannian A-init ensures adapters occupy mathematically disjoint subspaces (cos=2e-8). NRE composes them without interference.

| # | Paper | Arxiv | Key Finding |
|---|---|---|---|
| 61 | Editing Models with Task Arithmetic | 2212.04089 | Task vectors are composable arithmetic objects in weight space |
| 62 | TIES-Merging: Resolving Interference | 2306.01708 | Parameter interference identifiable and resolvable through sign conflict resolution |
| 63 | AdaMerging: Adaptive Model Merging | 2310.02575 | Task-specific behaviors are independently tunable merging parameters |
| 64 | Task Singular Vectors: Reducing Interference | 2412.00081 | Task behaviors decompose into low-rank separable singular vectors |
| 65 | Composing Parameter-Efficient Modules with Arithmetic | 2306.14870 | PEFT modules are composable behavioral units via arithmetic |
| 66 | Composable Interventions for Language Models | 2407.06483 | Multiple behavioral interventions compose with measurable interaction effects |
| 67 | LoRAHub: Dynamic LoRA Composition | 2307.13269 | Cross-task generalization via dynamic LoRA composition |
| 68 | LoRA Soups: Merging LoRAs for Skill Composition | 2410.13025 | LoRA modules compose through concatenation for multi-skill tasks |
| 69 | Universal Weight Subspace Hypothesis | 2512.05117 | Fine-tuned weights share a universal subspace across tasks |
| 70 | Intrinsic Dimensionality Explains Fine-Tuning | 2012.13255 | Fine-tuning operates in low intrinsic dimension — composition is structurally simple |
| 71 | Rethinking Inter-LoRA Orthogonality | 2510.03262 | Orthogonal Monte Carlo analysis of LoRA orthogonality in merging |
| 72 | Merging by Matching in Task Subspaces | 2312.04339 | Task-specific subspaces can be aligned and merged with minimal interference |
| 73 | Localize-and-Stitch: Sparse Task Arithmetic | 2408.13656 | Localized parameter regions from different tasks compose without overlap |
| 74 | Standing Committee in MoE Models | 2601.03425 | Domain-invariant shared components exist alongside specialized experts |
| 75 | Configurable Foundation Models | 2409.02877 | LLMs decomposable into functional brick modules (reasoning, knowledge, generation) |

---

## 4. Small + Right Strategies Beat Large Monoliths

**Pierre link:** 4B model + composed strategy adapters outperforms monoliths. Evidence: F#204 (code adapter helps math 10%→70%).

| # | Paper | Arxiv | Key Finding |
|---|---|---|---|
| 76 | LIMA: Less Is More for Alignment | 2305.11206 | 65B with 1000 examples matches GPT-4 — knowledge is in pretraining |
| 77 | LIMO: Less is More for Reasoning | 2502.03387 | 817 samples achieve strong reasoning |
| 78 | s1: Simple test-time scaling | 2501.19393 | 1000 traces beat o1-preview |
| 79 | Distilling Step-by-Step | 2305.02301 | Small model on rationales outperforms original LLM |
| 80 | Orca: Progressive Distillation from GPT-4 | 2306.02707 | 13B model beats larger models by imitating reasoning process |
| 81 | RL vs Distillation: Accuracy and Capability | 2505.14216 | Distillation transfers both accuracy AND capability to small models |
| 82 | Phi-4-Mini-Reasoning | 2504.21233 | Structured CoT training makes small models competitive on reasoning |
| 83 | VibeThinker-1.5B | — | 1.5B beats 671B with distilled reasoning |
| 84 | DeepSeek-R1 distillation | — | Small distilled models beat larger RL-trained models |
| 85 | Activation Addition (ActAdd) | 2308.10248 | Behavior changes without any training — inference-time only |
| 86 | Representation Engineering (RepE) | 2310.01405 | High-level cognitive control without weight changes |
| 87 | Knowledge Distillation of LLMs (MiniLLM) | 2306.08543 | Small models capture behavioral distribution from large models |
| 88 | SCoTD: Symbolic Chain-of-Thought Distillation | 2306.14050 | Small models learn step-by-step reasoning from distilled traces |
| 89 | Mentor-KD: Graduated Reasoning Transfer | 2410.09037 | Intermediate behavioral models bridge the gap between large and small |
| 90 | Code Data Enhances Reasoning | 2309.16298 | Code training enhances all reasoning, not just code — cross-domain strategy transfer |

---

## 5. Personal Patterns Extract from Sessions

**Pierre link:** M2P extracts approach, domain, preferences, style, codebase from MEMENTO session buffer. One forward pass. No fine-tuning.

| # | Paper | Arxiv | Key Finding |
|---|---|---|---|
| 91 | Subliminal Learning: Hidden Signal Transfer | 2507.14805 | Behavioral patterns (preferences, tendencies) transmit through interaction data |
| 92 | Command-V: Behavior Pasting | 2506.19140 | Behaviors extractable and injectable without training data or backprop |
| 93 | Precise Attribute Intensity Control | 2510.12121 | Behavioral attributes continuously controllable from representations |
| 94 | SkillFactory: Self-Distillation for Cognitive Skills | 2512.04072 | Cognitive skills learnable through self-distillation from behavior |
| 95 | AlphaEdit: Null-Space Editing | 2410.02355 | New patterns injectable without disrupting existing ones |
| 96 | Refusal Direction Identification (COSMIC) | 2506.00085 | Behavioral directions identifiable and steerable automatically |
| 97 | Representation Engineering (RepE) | 2310.01405 | High-level cognitive patterns readable and writable from representations |
| 98 | Activation Addition (ActAdd) | 2308.10248 | Behaviors injectable at inference time, zero training cost |
| 99 | Personality Traits via Activation Engineering | 2412.10427 | Personality dimensions extractable and controllable in LLMs |
| 100 | Geometry of Refusal: Concept Cones | 2502.17420 | Behavioral concepts are geometric objects with representational independence |

---

## Cross-Reference: Pierre Findings to Supporting Literature

| Pierre Finding | What we proved | Supporting papers |
|---|---|---|
| F#203: Wrong adapter captures 87% benefit | Strategies transfer, not knowledge | 1, 3, 5, 11, 12, 13, 20, 26, 30 |
| F#204: Code adapter helps math 10%→70% | Cross-domain strategy transfer | 13, 14, 20, 25, 90 |
| F#262: NTP preserves reasoning, SFT destroys | Behavior ≠ knowledge in training | 44, 45, 49, 51, 52, 53, 55 |
| F#362: M2P one-shot at 99.6% SFT quality | Personal patterns extract without fine-tuning | 26, 91, 92, 94, 97, 98 |
| F#428: Orthogonal composition cos=2e-8 | Adaptation deltas compose in disjoint subspaces | 61, 62, 63, 64, 65, 67, 70, 71 |
| F#458: Routing at 98.8% | Strategy selection from query features | 22, 74 |
| F#508: E2E +19-56pp with adapters | Small + strategies beats base model alone | 76-90 |
| F#510: Pre-merge fails without orthogonality | Orthogonality is mandatory for composition | 62, 64, 68, 71 |
| F#766: Hot-swap <1ms | Serving infrastructure viable | 45, 85, 86 |

---

## Foundational Papers (Referenced in Architecture)

| Paper | Arxiv | Relevance |
|---|---|---|
| Transformer Feed-Forward Layers Are Key-Value Memories | 2012.14913 | Structural basis: attention routes, MLP stores |
| Toy Models of Superposition | 2209.10652 | Why orthogonality matters: polysemanticity causes interference |
| Gemma Scope: Open SAEs | 2408.05147 | Interpretability tools for verifying adapter behavior |
| Switch Transformers | 2101.03961 | Sparse MoE: routing as architectural primitive |
| Mixtral 8x7B | 2401.04088 | Expert composition at production scale |
| Finding Neurons in a Haystack | 2305.01610 | Sparse probing reveals localized behavioral features |
| JumpReLU SAEs | 2407.14435 | Better SAE reconstruction for feature identification |
| Scaling and Evaluating SAEs | 2406.04093 | SAE scaling laws for interpretability |
| Disperse-Then-Merge: Alignment Tax Reduction | 2405.13432 | Multiple abilities can coexist with proper merging |
| Token-Level LoRA Adaptation | 2311.10847 | Per-token routing of adapters validates dynamic composition |
| In-context Learning and Induction Heads | 2209.11895 | Mechanistic basis: induction heads implement behavioral strategies |
| ROME: Locating and Editing Factual Associations | 2202.05262 | Factual knowledge surgically editable without affecting behavior |
| MEMIT: Mass-Editing Memory | 2210.07229 | Batch factual editing preserves behavioral capabilities |

---

## Counter-Evidence and Important Nuances

| Paper | Arxiv | Warning |
|---|---|---|
| Rethinking Inter-LoRA Orthogonality | 2510.03262 | Weight-space orthogonality (99.6%) does NOT produce behavioral specialization (0.3-5.7% PPL differentiation); routing/activation-space composition is needed |
| Model Merging and Safety Alignment | 2406.14563 | One misaligned model degrades safety of entire merged model; behavioral alignment is fragile under naive composition |
| COSMIC: Refusal Direction Identification | 2506.00085 | Refusal directions are discoverable and exploitable; safety behaviors must be hardened |

**Implication for Pierre:** Naive weight-space orthogonality is necessary but insufficient. Our Grassmannian init + NRE composition + activation-space routing (F#458) addresses this — composition must happen in the right space, not just weight space.
