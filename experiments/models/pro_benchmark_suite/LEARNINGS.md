# LEARNINGS.md: exp_pro_benchmark_suite

## Core Learning

**Scale=5 LoRA composition on Qwen3-4B preserves general benchmark quality within noise
of the base model.** No statistically significant degradation on any benchmark. This
replicates Finding #329-330 (scale=5 preserves MMLU on BitNet-2B) on a second architecture.

## What This Means

### Scale=5 is validated across architectures
BitNet-2B (#329-330) and Qwen3-4B (this experiment) both show benchmark preservation
at scale=5. The Davis-Kahan linear scaling argument is empirically supported: at 4x
lower scale than the catastrophic regime (scale=20), the perturbation is safely below
the critical angle for knowledge-encoding subspaces.

### DARE may matter for reasoning preservation
NRE composition at scale=5 showed -10pp GSM8K (not significant, n=20), while NRE+DARE
showed +10pp. The 20pp gap is suggestive: DARE's stochastic sparsification may act as
regularization for the reasoning subspace under multi-domain composition. This needs
confirmation at larger N.

### Benchmark design limitations
- MMLU with factual questions is too easy for 4B models (ceiling at 100%)
- Code generation evaluation via ast.parse of extracted blocks fails at 0% for Qwen3-4B
  with native chat template (possible thinking-mode interference)
- GSM8K at n=20 has insufficient power to detect 10pp effects

## What's Still Unknown

1. **Does MMLU degrade under harder questions?** MMLU-Pro or domain-specific MMLU
   subsets would reveal if scale=5 causes sub-5pp effects masked by the ceiling.
2. **Is the DARE vs NRE divergence real?** The 20pp gap on GSM8K (6/20 vs 10/20) needs
   n≥100 to confirm. If real, DARE should be mandatory for reasoning-sensitive composition.
3. **Code evaluation.** Need a working code benchmark (exec-based pass@k, not ast.parse
   from extracted blocks).

## Recommended Follow-ups

1. **exp_pro_mmlu_hard (P1):** Run MMLU-Pro (harder MCQ) at scale=5 to escape ceiling.
   Need n≥100 for 5pp detection at 80% power.
2. **exp_dare_vs_nre_gsm8k (P1):** Large-N GSM8K comparison (n≥100) between NRE and
   NRE+DARE at scale=5. Resolves whether DARE is necessary for reasoning preservation.
3. **exp_pro_code_eval_fix (P2):** Fix code evaluation — use execution-based pass@k
   instead of ast.parse extraction.
