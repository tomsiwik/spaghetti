# LEARNINGS.md — P11.Z1: Injection Decoding

## Core Finding
Zero-training injection decoding ("Wait, keep thinking", arXiv:2503.10167) plus Plan-and-Solve prompting (arXiv:2205.01068) **does not** lift Gemma 4 E4B 4-bit past 65% MMLU-Pro. At N=100 across 4 STEM categories: base=75.0%, injection=76.0% (+1.0pp, within ±4.3pp noise), PS=57.0% (−18pp), PS+inj=56.0%. K1532 FAIL, K1533 nominal PASS but statistically fragile, K1534 PASS.

## Why (Mechanism)
1. **Gemma 4 E4B over-thinks, not under-thinks.** Mean thinking ≈ 3489 chars — more than double the injection trigger threshold. Injection decoding was designed for models that truncate reasoning prematurely; Gemma does the opposite. The ~36% of samples that dipped below 1500c are the few where injection even fires, and even there the gain is marginal.
2. **Plan-and-Solve conflicts with terse-answer instructions.** Question prompt ends "Answer with ONLY the letter... Do not explain." PS prefix says "Let's first plan what steps are needed." Gemma resolves the conflict by degrading output — especially on chemistry (60%→32%).
3. **Saturation ceiling**: Theorem 2 bounds the gain at $p_{sat}-p_{base}$. With base already 75% (STEM-heavy sample), there is minimal room left.

## Antipattern Candidates (for analyst review)
- **"Inference-time injection on extended-thinking models"**: budget forcing assumes under-thinking. Before applying arXiv:2501.12599 / 2503.10167 techniques, verify the target model exhibits premature termination. Gemma 4 does not.
- **"PS prefix over terse-answer prompt"**: combining Plan-and-Solve with "Do not explain / letter only" actively hurts. Do not compose prompt techniques whose output-style demands conflict.

## Implications for Next Experiments
- `exp_p11_full_pipeline_v2` (blocked by this experiment) cannot rely on injection decoding as an accuracy lever on Gemma 4. Redesign or drop that component.
- For Gemma 4 accuracy work, focus on: (a) training-based reasoning improvements (s1K SFT, ThinkPO, GRPO) which actually change weights; (b) prompt reformulation that is compatible with thinking mode; or (c) **inverse** problem — budget compression (shorten thinking while preserving accuracy) — since over-thinking is the real failure mode. Any follow-up on (c) must start with its own paper-grounded MATH.md.
- Honest negative result: Finding #536's 62.1% baseline should be cited as the authoritative base; today's 75% comes from STEM-only category selection and is not a general MMLU-Pro baseline.

## Status
KILLED on K1532. K1533 nominal PASS is noise (+1pp at N=100, σ≈4.3pp). Experiment complete; PAPER.md updated with full prediction-vs-measurement table.
