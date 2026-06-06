# Peer Review: Generation Quality Test v2

## NotebookLM Findings

Skipped for v2 -- the experiment is primarily empirical (generate text, score with automated metrics). The mathematical content in MATH.md is scoring function definitions and aggregation formulas, not derivations requiring deep verification. Analysis below covers what matters: whether the 6 fixes were properly applied, whether the two-world pattern is trustworthy, and what it means for the architecture.

## Fix Verification

All 6 required fixes from the v1 review were applied. Verification:

| Fix | Required | Applied | Verified In |
|-----|----------|---------|-------------|
| 1. Top-1 routing only | Single expert, weight=1.0 | `get_oracle_routing_weights_top1()` returns [0,0,...,1,...,0] | run_experiment.py:484-493, RoutedMultiAdapterLoRALinear skips w<1e-6 (line 303) |
| 2. XPPL diagnostic only | Not in primary scoring | `compute_domain_score()` has no XPPL component | run_experiment.py:694-728 |
| 3. Domain-appropriate metrics | Code: syntax+KD, Math: correctness+KD | Separate scoring branches per domain | run_experiment.py:705-728 |
| 4. 3+ seeds | Mean and std reported | Seeds [42, 137, 2024], per-seed details in results.json | results.json:232-313 |
| 5. Same K1 criterion | >= 3/5 domains worse -> KILL | K1 test unchanged, 3/5 triggers kill | results.json:157-162 |
| 6. Document XPPL asymmetry | In MATH.md | Full algebraic derivation of the one-sided bias | MATH.md:144-179 |

**Assessment: All 6 fixes properly applied.** The XPPL asymmetry documentation (Fix 6) is particularly well done -- the algebraic proof that base always gets 0 and routed can only benefit is clear and correct.

## Mathematical Soundness

### Scoring Functions Are Correctly Implemented

The domain-appropriate scoring functions match MATH.md:
- Code: `0.5 * syntax_ok + 0.5 * kw` (line 708). Range [0, 1]. Correct.
- Math: `0.5 * correct + 0.5 * kw` (line 719). Range [0, 1]. Correct.
- Prose: `0.45 * kw + 0.25 * div + 0.10 * coh + 0.20 * rep` (line 728). Weights sum to 1.0. Correct.

### Aggregation Across Seeds Is Sound

Mean across 3 seed-level means, std across 3 seed-level means. This is the correct approach -- it captures inter-seed variance (the stochastic component of temperature sampling) rather than conflating within-seed and between-seed variance. With only 3 seeds, the standard error is sigma/sqrt(3), which is honestly reported.

### The Cross-PPL Diagnostic Reveals a Real Problem

Cross-PPL for legal (4.39 routed vs 2.70 base) and finance (3.30 vs 2.36) shows that even with top-1 clean routing (no secondary interference), these adapters make text LESS probable under their own models. This is no longer explainable by secondary expert interference (the v1 confound). This is a genuine signal that the legal and finance adapters are steering generation away from their own learned distribution -- possibly toward training-data-like patterns (the legal sample's "hoa" repetition loop is characteristic of training data memorization/degeneration).

### Math Variance Deserves Scrutiny

The math domain has std=0.098 on a mean of 0.304. Looking at per-seed details:
- Seed 42: 0.369
- Seed 137: 0.377
- Seed 2024: 0.165

Seed 2024 is dramatically lower than the other two (0.165 vs ~0.37). With binary answer correctness weighted at 50%, this means seed 2024 got far fewer answers correct. This level of seed sensitivity (one seed at half the performance of the other two) means the math improvement, while directionally real, is fragile. The 142.1% improvement headline number is driven primarily by two seeds; the third seed shows a much more modest gain over base (0.165 vs 0.146 = +13%).

**This does not invalidate the finding** -- the routed model wins math on all 3 seeds (0.369 > 0.136, 0.377 > 0.095, 0.165 > 0.146). But the magnitude varies 10x across seeds, and the paper's "+142.1%" headline is the mean of a bimodal distribution.

## Experimental Design

### The Kill Is Clean and Correct

Unlike v1, there are no confounds muddying the K1 verdict:
- Top-1 routing eliminates secondary expert interference
- Domain-appropriate metrics eliminate cross-domain metric bias
- 3 seeds eliminate single-seed noise concerns
- XPPL removal eliminates the pro-routed scoring bias

The kill triggers on 3/5 domains (medical, legal, finance) with consistent negative deltas across all 3 seeds. The standard deviations are tight for all three losing domains (0.006, 0.010, 0.003), meaning this is not seed noise.

### The Two-World Pattern Is Real But Has a Simpler Explanation

The paper frames the two-world pattern as "structured vs prose" domains. A simpler explanation exists: **code and math have task-specific binary metrics (syntax validity, answer correctness) that capture adapter value; prose domains use surface text statistics that do not.**

Consider what the prose composite actually measures:
- **Keyword density (45%):** Whether the text contains domain words. The adapters were trained on domain text, so they should increase domain keywords. But the data shows they do not reliably do so (medical: 0.045 routed vs 0.044 base; finance: 0.043 vs 0.052 -- actually worse). This suggests the adapters are not producing more domain-relevant text by this crude measure.
- **N-gram diversity (25%):** Unique trigrams / total trigrams. The legal routed sample ("hoa...hoa...hoa") would score terribly here. This is measuring degeneration, not domain quality.
- **Repetition (20%):** Unique words / total words. Same problem as diversity -- captures degeneration.
- **Coherence (10%):** Sentence length proxy. Essentially noise at 10% weight.

The prose domains are not "losing because routing hurts prose." They are losing because (a) the legal and finance adapters produce degenerate/repetitive text (observable in the samples and confirmed by high cross-PPL), and (b) the medical adapter produces marginally less diverse text. The metric is correctly detecting real quality degradation in these domains.

### The Legal Sample Is Catastrophically Bad

The routed legal sample contains "hoa" repeated 13 times in ~100 words. The prompt is about speed bumps on a "non-hoa road" and the routed output spirals into an "hoa" repetition loop. This is not a metric artifact -- this is genuine adapter-induced degeneration. The legal adapter is causing repetitive collapse on this prompt. The base model produces a coherent, structured response to the same prompt.

Cross-PPL of 4.39 (vs 2.70 base) confirms: even the legal adapter's own model disagrees with this output. The adapter is broken for generation on this type of prompt, not just poorly measured.

### The Finance Sample Shows Vocabulary Narrowing Without Domain Value

The routed finance response ("Invest in a business... Start a business... Build a new home with $10,000") is less informative than base ("stocks, REITs, side ventures") and uniform ("stocks, bonds, real estate, crypto allocation"). The adapter constrains vocabulary toward generic advice rather than adding financial expertise. The keyword density drop (0.043 routed vs 0.052 base) is correctly detecting this.

### ast.parse Is Generous but Defensible

The code syntax checker tries 3 strategies: full text, extracted code blocks, and contiguous code-like lines. The third strategy (lines starting with `def`, `class`, `import`, etc.) is generous -- it could find a parseable subset of broken code by skipping error lines before the code starts. However, this generosity applies equally to all configurations (base, uniform, routed), so it does not bias the comparison. The 60% vs 53.3% routed win is directionally valid.

### Answer Extraction Has an Important Asymmetry

The math answer extractor (line 630-662) tries patterns in priority order:
1. "the answer is X"
2. "= X"
3. "$X"
4. Last number in text

The routed math sample uses GSM8K training format: `<<3*26+2*83+1*90=243>>243...#### 231`. The `<<...=X>>` markers and `####` format make answer extraction easy. The base model produces step-by-step LaTeX that gets truncated at 128 tokens before reaching an answer.

This is not an extraction artifact -- it is a real phenomenon. The adapter steers the model toward the concise answer format it was trained on. The 128-token limit then becomes an advantage for routed (complete answer) vs base (truncated steps). But this is a legitimate benefit of domain adaptation: producing more efficient outputs for the task.

**One concern:** The `extract_ground_truth_answer` function looks for `<<...=X>>` patterns in the training data response. If the training data has errors (wrong answers in the `<<>>` markers), ground truth would be wrong. This is a data quality assumption, not a code bug. Acceptable for micro-scale.

## Novelty Assessment

No novelty is claimed. This is a validation experiment testing whether PPL improvements from prior adapter training (exp_real_data_domain_experts: 26.5% mean PPL improvement) translate to generation quality. The finding that PPL improvements predict generation quality for structured tasks but not prose tasks is a genuinely useful architectural insight.

## The Two-World Pattern: What It Actually Means

The paper's interpretation is partially correct but needs refinement:

**What the data shows:**
1. Domain adapters improve task-specific correctness (code syntax +6.7pp, math answers +40pp)
2. Domain adapters degrade text quality on 3/5 prose domains (legal catastrophically, finance moderately, medical marginally)
3. The degradation correlates with cross-PPL increase (legal 4.39, finance 3.30 vs base 2.70, 2.36)
4. Uniform 1/N composition is uniformly worse than base on all prose domains and math/code

**What this means for the architecture:**
- The adapters work for their trained objective (PPL reduction on domain text) but this does not transfer cleanly to generation quality on all tasks
- The legal and finance adapters appear to be poorly suited for generation -- possibly overtrained on narrow training distributions that cause repetitive collapse on out-of-distribution prompts
- The code and math adapters benefit from having a clear target format (Python code, GSM8K answer format) that is both trained-on and measurable
- The prose domains lack both: the adapters have no clear target format, and the metrics cannot capture domain expertise

**What the paper should NOT claim:**
- "This is a measurement problem" (partially true for medical, not true for legal/finance where qualitative inspection confirms real degradation)
- "The architecture IS NOT killed for structured output domains" (this is correct, but overstates confidence from 10 prompts x 3 seeds)

## Macro-Scale Risks (advisory)

1. **Legal/finance adapter degeneration will persist at macro.** The repetitive collapse in legal and vocabulary narrowing in finance are training/data problems, not scale problems. Macro experiments should either retrain these adapters with generation-aware objectives or test with different legal/finance data.

2. **Task-specific benchmarks are mandatory.** HumanEval for code, GSM8K/MATH-500 for math. The micro-scale code syntax rate (60%) and math answer rate (56.7%) are promising but low in absolute terms. Macro must establish whether these improve with scale.

3. **The 128-token limit masked potential problems.** At longer generation lengths, adapter-induced repetitive collapse (seen in legal) may emerge in other domains. Macro should test at 512-1024 tokens.

4. **Prose domain evaluation requires human judgment or established benchmarks.** Keyword density and n-gram diversity are insufficient. At macro, use domain-specific benchmarks (MedQA, LegalBench) or LLM-as-judge evaluation.

## Verdict

**PROCEED** (accept the kill, with the two-world pattern as a documented finding)

### Justification

The v2 experiment is well-executed. All 6 fixes from v1 review were properly applied. The K1 kill is clean and unconfounded -- routed composition degrades text quality on 3/5 domains even under ideal conditions (oracle top-1 routing, domain-appropriate metrics, 3 seeds). This kill should stand.

The two-world pattern (structured domains win, prose domains lose) is a trustworthy finding supported by:
1. Consistent direction across all 3 seeds for all 5 domains
2. Tight standard deviations on the prose domain losses (0.003-0.010)
3. Qualitative sample inspection confirming real degradation (legal "hoa" loop)
4. Cross-PPL diagnostics corroborating the pattern (legal/finance adapters produce text their own models reject)
5. Task-specific metrics showing genuine improvement (code syntax +6.7pp, math answers +40pp)

### What Should Be Recorded

The following findings should be added to FINDINGS.md as a killed experiment:

- **Killed claim:** "Routed LoRA composition improves generation quality across all domains"
- **Surviving claim (micro-scale evidence only):** Domain adapters improve task-specific correctness for structured output domains (code: 60% vs 53% syntax validity; math: 57% vs 17% answer correctness). This needs macro validation with HumanEval and MATH-500.
- **Architectural insight:** PPL improvement does not predict generation quality for prose domains. Adapters trained for PPL reduction on prose can cause repetitive collapse during generation.
- **Implication for SOLE:** Routing to domain experts is beneficial for structured tasks, neutral-to-harmful for prose generation. Production systems should consider generation-aware training objectives or selective adapter activation.
