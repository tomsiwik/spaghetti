# MATH.md — exp_g4_memory_footprint_n25

## Pre-registered hypothesis

**Theorem (claim):** Gemma 4 E4B (4-bit quantized) with N=25 LoRA adapters
(r=6, v_proj + o_proj, all 42 layers) simultaneously mounted fits within the
M5 Pro 48 GB unified-memory envelope with peak resident set size (RSS)
≤ 5 GB on Apple Silicon via MLX.

**Motivation (citation):** Finding #74 (memory-optimized serving; adapter
overhead is dominated by base weights, not deltas). Adapter memory per
layer is 2·r·(d_in+d_out) fp16 ≈ O(kilobytes); N=25·42 layers·2 matrices
is a budget question, not a novel-algorithm question.

## Kill criterion

**K1596:** peak process RSS ≤ 5 GB while the base model is loaded AND all
N=25 adapters are attached AND a single forward pass executes to end of
first generated token on a reference prompt. Failure → KILLED.

## Pre-registered precondition probe (tripwire)

Per the audit-2026-04-17 cohort-wide standing rule (Findings
#605/#606/#608/#610/#611/#612/#613/#615/#616/#617/#618/#619/#620), this
experiment MUST NOT launch the multi-adapter MLX load path before verifying
three structural preconditions exist on disk:

- **P1.** Gemma 4 E4B 4-bit base model resolvable through `mlx-lm` — this
  is a prereq for **any** memory measurement; without a loadable base, RSS
  measurements have no referent. Probe: at least one of the canonical
  `mlx-community/gemma-4-e4b-*4bit*` dirs is present or reachable.
- **P2.** N=25 v_proj+o_proj r=6 Gemma 4 adapter safetensors exist on
  disk. Adapter configs without weights cannot be attached to a base
  model; attempting to do so either errors or silently attaches zero
  deltas (which inflates the claim: "25 attached at no cost"). Probe:
  ≥25 `*.safetensors` under any canonical Gemma 4 N=25 adapter dir.
- **P3.** A Gemma 4 multi-adapter mount path exists as runnable code
  — not just a toy LoRA single-adapter swap, but a loader that actually
  attaches N>1 adapters **simultaneously** on Gemma 4 (dimensions,
  naming, tie-points, and LoRA algebra all validated). Without this,
  "peak RSS with 25 adapters attached" is uncomputable because the
  attachment doesn't happen.

If any of P1/P2/P3 fail: K1596 is UNMEASURABLE → status=killed, NO
multi-adapter load is attempted, follow-up experiment becomes the
upstream rebuild.

## Predicted verdict path

- 3/3 preconditions PASS → proceed to heavy MLX multi-adapter load,
  record peak RSS via `psutil.Process().memory_info().rss`, decide
  K1596 on observed peak.
- ANY precondition FAIL → status=killed, K1596 result=fail
  (UNMEASURABLE), evidence logs exact missing artifacts.

## Assumptions logged (guardrail 1007 autonomy)

- "Peak RSS" is measured at process level (`psutil` RSS), not VmData or
  Mach task footprint, because RSS is what the OOM-killer and the M5 Pro
  48 GB budget actually watch.
- "N=25 adapters attached" means each has non-zero delta loaded into
  MLX arrays and reachable from the forward-pass module graph — not
  merely listed on disk. The probe cannot verify attachment from files
  alone, but if P2 fails (no files), attachment is impossible and the
  verdict stands.
- "Single forward pass to first token" is the minimum evidence that the
  graph materializes; without it, RSS could be under-reported because
  lazy MLX graph construction defers allocation.
