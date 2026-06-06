# LEARNINGS.md: exp_tiny_benchmark_suite

## Core Learning

**Domain-specific LoRA adapters on BitNet-2B-4T do NOT improve standardized benchmarks
(MMLU, GSM8K, HumanEval). The scale dilemma is fundamental: scale=20 enables domain
generation but destroys benchmarks (-6 to -17pp); scale=1 preserves benchmarks but
provides zero domain improvement.**

This is NOT an architectural failure. Finding #323 showed the integrated pipeline
achieves -2.8% vs oracle on domain text. The adapters work for their intended purpose
(domain generation), just not for general reasoning/knowledge benchmarks.

## The Scale Dilemma (Formalized)

The rank-16 LoRA perturbation cannot selectively target domain subspaces while
preserving benchmark subspaces, because:
1. At scale=1: ||alpha * x@A@B|| << ||base(x)||, so adapter is invisible to BOTH
   domain tasks AND benchmarks
2. At scale=20: perturbation is large enough for domain tasks but also disrupts
   general knowledge representations
3. No intermediate scale exists where adapters help domain tasks AND preserve benchmarks

This connects to Finding #320 (MMLU degradation at scale=20) and extends it:
the problem is universal across MMLU, GSM8K, and HumanEval.

## Finding #262 Contradiction

NTP math adapter: +10pp GSM8K in Finding #262, -17pp here. Three possible causes:
1. TernaryLoRALinear (STE quantization during forward) vs RuntimeLoRA (direct bfloat16)
2. Per-task scale optimization vs fixed scale=20
3. Different adapter attachment mechanism (unpacked vs native BitLinear)

This matters: if RuntimeLoRA is systematically worse for benchmark tasks, the serving
architecture choice (RuntimeLoRA for speed) has a quality cost.

## Statistical Note

At n=50/30/15, the 95% CI widths are 26/34/44pp. Most measured deltas (-2 to -6pp)
are within noise. Only the NTP math -17pp on GSM8K is borderline significant (p=0.19).
The headline result ("no improvement") is robust but the magnitude of degradation is
uncertain.

## Impossibility Structure

**Why general benchmark improvement from domain adapters is structurally unlikely:**
- Adapters are trained on domain-specific text (medical reports, legal filings, etc.)
- Benchmarks test domain-independent capabilities (knowledge recall, reasoning, coding)
- Finding #319 (Ghosh et al. 2402.05119): SFT teaches style, not knowledge
- LIMA (2305.11206): alignment is superficial, doesn't inject capabilities
- The adapter can make the model SOUND more like a doctor, but cannot make it KNOW more
  medicine (knowledge is in the base weights)

## Recommended Follow-ups

1. **exp_domain_specific_benchmarks (P0):** Create domain-specific evaluation tasks
   (medical QA, legal clause extraction, financial analysis) where adapters SHOULD
   improve over base. This is the correct evaluation for domain adapters.

2. **exp_task_adaptive_scaling (P1):** Route determines not just adapter but SCALE
   per-token. Domain tokens get scale=20, benchmark tokens get scale=0. Requires
   distinguishing domain vs general text at token level.

3. **exp_f262_reproduce (P1):** Reproduce Finding #262 conditions to determine which
   difference (TernaryLoRA vs RuntimeLoRA, scale optimization, attachment mechanism)
   causes the +10pp → -17pp GSM8K flip.
