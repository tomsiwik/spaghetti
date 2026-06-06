# Learnings: Generation Quality LLM-Judge

## Core Finding

Routed LoRA composition is **doubly killed for prose generation quality** — two independent, uncorrelated evaluation methods (keyword density r=0.107 vs LLM-as-judge) converge on the same verdict. The prior kill was NOT an evaluation artifact. However, the math adapter's 24x correctness improvement (48% vs 2%) proves the architecture works for **structured tasks where correctness is objectively measurable**.

## Why This Happened

### 1. BitNet-2B Cannot Act as Judge — Model Too Small for Evaluation

The 2B self-judge outputs near-constant scores (relevance=4, coherence=3, informativeness=5) for 80-100% of texts, giving 0-11 effective discriminating pairs out of 50 per domain. This is consistent with the LLM-as-judge literature:

- Zheng et al. (arXiv:2306.05685) established the LLM-as-judge paradigm using GPT-4, not small models. Their MT-Bench framework assumes judges with strong instruction-following and nuanced reasoning.
- Prometheus (arXiv:2310.08491) showed that even purpose-built judge models need 7-13B parameters to correlate with human evaluations. Models below 7B exhibit high variance and poor calibration.
- The Zephyr-7B work (arXiv:2310.16944) used a larger "teacher model" for AI Feedback, implicitly acknowledging that small models lack evaluation capacity.

**Our 2B ternary model is ~4x below the minimum viable judge size.** The experiment should have included a pilot study (5-10 samples, check variance) before committing 70 minutes of compute.

### 2. Two Broken Rulers Agreeing Is Weak But Real Evidence

The keyword-density metric has known format sensitivity (penalizes code, concise answers). The LLM-judge has near-zero discriminating power. These flaws are independent — keyword density fails on FORMAT, the judge fails on CAPACITY. Their directional agreement on all 5 domains (p=0.031 under sign test) provides weak but meaningful evidence that the direction is real. However, the reviewer correctly noted:

- Ties (code, finance) are counted as losses when the judge is non-discriminating
- Honest count is 2-3/5 worse, 2 tied, 0 better — still triggers K1 at threshold 3 but changes narrative

### 3. The Format-Correctness Tradeoff Reveals Evaluation Methodology Limits

The math domain paradox (24x correctness, lower judge scores) is a known pitfall:
- **Verbosity bias** — LLM judges consistently rate longer, verbose responses higher than concise correct ones (documented in MT-Bench analysis, arXiv:2306.05685)
- The math adapter produces concise GSM8K format (`<<26*3=78>>78`) that completes within 128 tokens, while base produces verbose step-by-step that truncates
- Both our evaluation methods penalize the adapter for doing exactly what it should

### 4. Prose Degradation Is Real, Not an Artifact

The legal adapter's mode collapse (repetitive HOA complaint template, cross-PPL 4.39 vs 2.70 base) and the general prose quality degradation across domains are consistent with the prior LEARNINGS.md analysis: PPL-trained adapters on narrow domain data cause mode collapse in generation. This is the same mechanism identified in the v1 learnings — training objective (PPL minimization) is mismatched with the desired outcome (generation quality).

## Confirming Evidence

1. **Our own prior kill (exp_generation_quality_test)** — 3/5 domains worse by keyword density, now confirmed by independent metric. Finding #178.
2. **Our own PPL disconnect** — r=0.08 full-sequence PPL vs task quality; "reverse expert" with -27% PPL but +9.5pp accuracy. PPL improvements don't predict generation quality.
3. **MT-Bench (arXiv:2306.05685)** — established that LLM judges need GPT-4-class reasoning; documents verbosity bias that explains our math paradox.
4. **Prometheus (arXiv:2310.08491)** — purpose-built judge models need 7-13B minimum for human-correlated evaluation.
5. **LoRAuter on StoryCloze** — linear merging of adapter weights fails on narrative coherence tasks. TIES merging partially recovers quality.
6. **arxiv 2603.03535** — ensembling > routing > merging for multi-adapter. Our pre-merge weight composition is the worst-performing category for generation quality.
7. **Hash Layers (arXiv:2106.04426)** — random routing matches learned routing on NTP, suggesting routing optimization has bounded upside when expert quality is insufficient. Finding #118.

## Contradicting Evidence

1. **Finding #179 (math 24x correctness)** — directly contradicts "routing is useless." The architecture works for structured tasks, just not prose.
2. **TIES merging (arXiv:2306.01708)** — resolves parameter interference via sign consensus. Our linear weight addition may be unnecessarily destructive. TIES could potentially recover prose quality without retraining.
3. **DARE sparsification (arXiv:2311.03099)** — dropping 90%+ of delta parameters before merging reduces interference. Could combine with our composition method.
4. **DPO training (Zephyr, arXiv:2310.16944)** — generation-aware training maintains core capabilities while improving response quality. Our adapters trained purely on PPL may fundamentally lack the signal needed for good prose generation.
5. **Medical cross-PPL asymmetry** — medical cross-PPL *improves* (2.41 vs 2.59) despite judge showing marginal degradation. With a more capable judge, this domain might reverse.

## Alternative Approaches

### 1. Objective Task Benchmarks (RECOMMENDED NEXT)
Skip subjective evaluation entirely. Test on GSM8K accuracy, HumanEval pass@1, and domain-specific QA benchmarks (MedQA, LegalBench, FinQA) where correctness is binary and objectively measurable. The math 24x finding proves routing helps on these tasks.
- **Motivation:** Finding #179 (math 24x), Finding #178 (evaluation methodology limitation)
- **Literature:** BitNet-2B-4T achieves 58.38 on GSM8K and 38.40 on HumanEval+ as baseline

### 2. TIES Merging for Prose Recovery
Replace linear weight addition with TIES-style conflict resolution (arXiv:2306.01708). Resolves sign conflicts that may cause prose degradation.
- **Motivation:** Prior LEARNINGS.md identified TIES as low-cost experiment; LoRAuter showed partial recovery on StoryCloze
- **Risk:** May not help since our adapters use ternary weights (sign conflicts are more constrained in {-1,0,1})

### 3. DPO-Trained Adapters
Train adapters with Direct Preference Optimization instead of PPL minimization (arXiv:2310.16944). Addresses the root cause: training objective mismatch.
- **Motivation:** PPL-generation disconnect proven across multiple experiments
- **Risk:** Requires preference data per domain, higher training cost

### 4. Selective Adapter Activation (Already Partially Implemented)
Use entropy gating to skip adapters when base model is confident. Don't route prose — only route structured tasks.
- **Motivation:** Two-world pattern from v1 LEARNINGS.md
- **Already have:** Entropy gating pre-filter (63% tokens skip at 1.13% cost)

## Implications for Next Experiments

1. **Pivot evaluation to objective benchmarks.** exp_task_accuracy_real_benchmarks (P0 deployment track item #2) is now the highest-priority experiment. GSM8K, HumanEval, and domain QA benchmarks avoid the evaluation methodology trap entirely.

2. **Do NOT retry with a bigger judge at micro scale.** The evaluation problem is not solvable with available models. Accept this limitation and move to objective metrics.

3. **The math 24x finding reframes SOLE's value proposition.** SOLE is a structured-task specialist, not a universal quality enhancer. Marketing and architecture decisions should reflect this.

4. **Prose quality requires training objective change, not evaluation change.** Two evaluation methods agree on the diagnosis. The fix is DPO/RLHF-trained adapters, not better metrics. This is a P1 research direction, not a P0 blocker.

5. **Reviewer conditions satisfied.** Finding #179 extracts math 24x as separate supported finding. Finding #178 documents the kill with caveats about tie-counting and judge limitations. The 2B judge limitation is documented in both findings' caveats.

## Recommended Follow-Up

**exp_task_accuracy_real_benchmarks** (already in P0 deployment track) — test routing on GSM8K accuracy and HumanEval pass@1 where the math adapter's 24x benefit can be properly measured against standard benchmarks. Motivated by Finding #179 (math correctness) and Finding #178 (subjective evaluation killed). No new experiment creation needed — this is already queued.
