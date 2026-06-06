# LEARNINGS.md — exp_followup_format_compat_peft_required

## Core Finding

SUPPORTED (K1576 all 3 gates, Finding #604). MLX→PEFT transpose bijection
(Thm 1, Finding #433) survives a **real runtime load**, not just a schema
check, once the parent kill's (#585) silent-ImportError-bypass and
subset-direction fallacy are structurally eliminated. Fused-QKV runtimes
require **rank-expansion to 3r** block-diagonal fusion — naive row-stack
is invalid whenever A_q, A_k, A_v are distinct (measured max|A_q−A_k|=0.589,
max|A_q−A_v|=0.640).

## Why

Two design changes made the parent failure mode impossible, not merely
unlikely:
1. **Hard-required top-level imports** (`peft, transformers, torch`): the
   module cannot load without them, so the silent-bypass path ceases to
   exist.
2. **Real `PeftModel.from_pretrained` + forward on tiny CPU Llama** plus a
   **wrap-count check** (12 = 4 layers × {q,k,v}) catches `target_modules=∅`
   silent-skips that a schema subset check would miss. Distinct-A probe
   refuses the "PEFT-compat ⇒ vLLM-compat" shortcut Thm 3 relied on.

## Implications for Next Experiment

- **Template** for future runtime-X-compat claims: top-level hard imports
  + real load + behaviourally-sensitive count check. Flag any proposal
  missing the triad at design time.
- **vLLM/Unsloth serving** remains correctly out-of-scope on Apple Silicon
  (handled by `exp_p1_t4_format_compat_v2` SKIP). Do not resurrect.
- **Fused-QKV downstream work** must use `rank=3r` block-diagonal fusion.
  Thm 2 is architecture-agnostic for separate-QKV HF bases (Llama/Gemma/
  Qwen); live-load against Gemma 4 / Qwen 3 is a low-priority follow-up
  (base-specific quirks only).
- **Training-drift** claims still require a separate interference
  experiment; format-compat proves nothing about drift.

No new antipattern: this experiment **retires** the silent-ImportError-
bypass antipattern for the format-compat lane by providing a structurally
immune template.
