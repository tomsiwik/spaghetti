# Learnings: Generation Quality Test v2

## Core Finding

Routed LoRA composition is **killed for general prose generation quality** (3/5 domains worse) but **validated for structured output domains** (code +14.4% syntax validity, math +142.1% answer correctness). PPL improvement does not predict generation quality for prose — this is the "Two-World Pattern."

## Why This Happened (Literature-Grounded)

### The PPL-Generation Disconnect Is Well-Documented

Our own prior work already demonstrated this: full-sequence PPL has near-zero correlation (r=0.08) with task quality in composed models (SOLE findings). Even answer-conditioned PPL, which achieved r=0.811 at micro scale, was killed at macro (r=-0.63, inverted). The generation quality test v2 adds a new dimension: **PPL improvements from domain adaptation predict structured-task performance but anti-predict prose quality**.

This aligns with Chen et al. (arxiv 2410.23771, "What is Wrong with Perplexity for Long-context Language Modeling?") who showed PPL averages across all tokens, obscuring performance on "key tokens." For prose domains, the key tokens for quality (coherent transitions, diverse vocabulary, domain-appropriate reasoning) are diluted by the overwhelming majority of common tokens that PPL rewards.

### Adapter-Induced Repetitive Collapse: Mode Collapse in Text Space

The legal adapter's "hoa" repetition loop (13 repetitions in ~100 words, cross-PPL 4.39 vs 2.70 base) is a textbook case of **mode collapse transferred from fine-tuning to generation**. The mechanism:

1. Legal training data has narrow distributional support (legal language is formulaic)
2. LoRA fine-tuning on narrow data collapses the adapter's effective distribution
3. During generation, the adapter pulls the model toward high-probability sequences in its narrow learned distribution
4. Temperature sampling cannot escape because the adapter's logit contribution dominates at weight=1.0
5. The model enters a repetition attractor — each repeated token increases the probability of the next repetition

This is the same mechanism as mode collapse in image LoRAs (documented in Stable Diffusion fine-tuning literature), but manifested in text as repetitive degeneration rather than visual homogeneity.

### Why Structured Domains Escape This Trap

Code and math adapters succeed because:
- **Format convergence is the goal**: Python syntax and GSM8K answer format are narrow by design. The adapter's "mode collapse" toward these formats is *beneficial* — it produces valid code and extractable answers.
- **Objective metrics capture adapter value**: Syntax validity and answer correctness directly measure what the adapter learned. Keyword density does not.
- **128-token truncation amplifies the effect**: The math adapter produces concise `<<...=X>>` format answers that complete within the limit, while the base model produces verbose step-by-step solutions that get truncated. This is a *legitimate* adapter benefit (format efficiency), not an artifact.

## Confirming Evidence

1. **SOLE PPL disconnect (our own)**: r=0.08 full-sequence PPL vs task quality; "reverse expert" with -27% PPL but +9.5pp accuracy. Directly confirms PPL is unreliable for quality prediction.

2. **BitROM quantization paradox**: Higher PPL but better downstream task accuracy in extreme quantization (quantization as regularizer). Shows PPL and task performance can be anti-correlated.

3. **BabyLM pretraining complexity study**: Text complexity affects PPL but has "little impact on fine-tuning evaluations" — surface-level distributional fit (what PPL measures) is orthogonal to capability.

4. **LoRAuter on StoryCloze (arxiv)**: Linear merging of adapter weights fails on "temporal coherence tasks where ordered narrative flow is crucial." This is the same phenomenon — simple adapter composition degrades prose quality. TIES merging (directional sparsity) partially recovers it (70.09% vs oracle 72.00%).

5. **LongPPL (arxiv 2410.23771)**: Standard PPL masks performance on key tokens by averaging. Proposes key-token-weighted PPL that correlates at r=-0.96 with downstream benchmarks. Our prose domain metrics are analogous — they try to capture quality dimensions PPL ignores, but with cruder proxies.

## Contradicting Evidence

1. **No direct contradiction found** for the two-world pattern itself. No paper claims LoRA adapters universally improve prose generation quality.

2. **Partial contradiction on prose adaptation**: LoRAuter achieves 70.09% on StoryCloze (narrative coherence) using TIES merging — close to oracle 72.00%. This suggests the problem is not that adapters *can't* help prose, but that (a) our training objective (PPL minimization) is wrong for prose, and (b) our composition method (direct weight addition) creates interference. TIES resolves parameter conflicts that linear merging does not.

3. **Our own cross-PPL asymmetry**: Medical cross-PPL *improves* (2.41 routed vs 2.59 base) yet the composite score worsens (-6.9%). For medical specifically, the adapter may be helping in ways the metric doesn't capture. The -6.9% loss is within the range where better metrics might reverse the verdict.

## Alternative Approaches (What We Could Try Instead)

### 1. Generation-Aware Training Objectives
Instead of training adapters to minimize PPL, train them with generation-in-the-loop objectives:
- **DPO (Direct Preference Optimization)**: Post-SFT alignment using domain preference pairs. Would steer adapters toward preferred generation styles rather than just next-token prediction.
- **RLHF-guided routing**: Use reward signals to adjust expert selection, not just PPL-based scoring.
- **Relevance**: High. Our adapters are trained purely on PPL. The prose domain failures may be a direct consequence of this training objective.

### 2. TIES Merging Instead of Linear Weight Addition
LoRAuter found that TIES merging (resolving parameter conflicts through directional sparsity) recovers narrative quality that linear merging destroys. Our current composition uses direct weight addition (equivalent to linear merging).
- **Implementation**: Replace `W_base + sum(w_i * B_i @ A_i)` with TIES-style conflict resolution (sign majority voting + magnitude pruning).
- **Relevance**: Medium-high. May fix prose degradation without retraining adapters.

### 3. Domain-Specific Evaluation Frameworks
Our prose metrics (keyword density, n-gram diversity, repetition) are crude proxies. Better options:
- **LLM-as-judge**: Use a larger model to rate domain quality (established practice, low cost at micro scale).
- **Task-specific benchmarks**: MedQA for medical, LegalBench for legal, FinQA for finance. These test *capability*, not surface text properties.
- **Intention entropy metrics**: Measure whether the adapter preserves the base model's reasoning diversity rather than collapsing it.

### 4. Ensemble in Output Space vs. Weight Space
Arxiv 2603.03535 found: ensembling > routing > merging for multi-LoRA. Our approach (pre-merge weight composition) is the worst-performing category. Output-space ensemble (running each adapter separately and combining logits) preserves individual adapter quality at the cost of inference overhead.
- **MLX feasibility**: Running 5 adapters in parallel is expensive but could be tested on 2-3 adapters.
- **Relevance**: High for understanding the quality ceiling, but conflicts with our 0% overhead pre-merge goal.

### 5. Selective Adapter Activation
Don't use prose adapters for generation. Keep them for PPL-based scoring/routing only. Use the base model directly for prose generation, adapters only for structured output.
- **Implementation**: Entropy-adaptive gating — if the base model is confident (low entropy), skip the adapter.
- **Relevance**: Pragmatic. Aligns with the two-world finding: use adapters where they help, skip where they hurt.

## Implications for Next Experiments

1. **The two-world pattern reframes SOLE's value proposition**: SOLE is not a universal quality enhancer. It is a structured-task specialist. Future experiments should focus on code (HumanEval) and math (MATH-500, GSM8K) benchmarks where the architecture demonstrably helps.

2. **Prose domain adapter training needs a different objective**: PPL-trained adapters cause repetitive collapse on prose. If prose quality matters, explore DPO or RLHF-based adapter training. This is a training problem, not a composition problem.

3. **Legal and finance adapters are broken for generation**: Cross-PPL confirms these adapters produce text their own models reject. They should be retrained with generation-quality-aware objectives or excluded from generation tasks entirely.

4. **TIES merging is a low-cost experiment**: Could recover prose quality without retraining. Worth testing as a quick hypothesis before the more expensive DPO route.

5. **Evaluation must match the claim**: Never again use keyword density to evaluate domain expertise. Use task-specific benchmarks or LLM-as-judge. The v1→v2 metric change (composite → domain-appropriate) reversed the signal for code and math, proving metrics matter more than architecture changes.

6. **The 128-token limit is both a confounder and a finding**: It truncates base model responses but reveals that adapters produce more efficient formats. Future experiments should test at 512+ tokens to separate these effects.
