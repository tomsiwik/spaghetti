# PAPER.md — exp_prod_llama_cpp_bridge

## Verdict

**KILLED_PREEMPTIVE** — ap-017(s) hardware-topology-unavailable,
**2nd instance** (1st = F#650). All 5 theorems block (defense-in-
depth; 5/5 all_block=true). Any one alone is sufficient.

## One-line

The llama.cpp runtime is structurally out-of-scope for the
Apple-only-MLX parent (`exp_prod_adapter_format_spec_v1`); no
Gemma 4 GGUF converter, no PoLAR→GGML-LoRA converter, no llama.cpp
binary on this host, no MMLU-Pro-with-thinking harness for
cross-runtime comparison. A claim of "runs with quality parity"
cannot be measured on the local platform.

## Target claim (DB)

| KC | Text |
|----|------|
| K1654 | GGUF export of Gemma 4 E4B + 1 adapter runs in llama.cpp; MMLU-Pro within 5pp of MLX reference |
| K1655 | Adapter hot-swap works in llama.cpp runtime |

## Prediction vs. measurement

The runner pre-registers the preempt structure (MATH.md) and then
measures whether each theorem's blocking condition holds on the
local platform.

| Theorem | Prediction | Measurement | Blocks? |
|---------|------------|-------------|---------|
| T1 artifact shortfall | ≥ 3 cross-runtime artefacts missing | shortfall = 6 (llama.cpp binary, Gemma 4 converter, PoLAR converter, GGML LoRA converter, MMLU-Pro-thinking harness, parent-loader-ok baseline) | PASS |
| T2 resource budget | 1/2 backends reachable | backends_reachable=1, required=2, coverage=0.5 | PASS |
| T3 schema completeness | `success_criteria: []` + ⚠ INCOMPLETE literal | confirmed from DB claim output; 7th F#502/F#646 cohort hit | PASS |
| T4 KC pin count | pin_ratio ≤ 0.20; K1655 non-falsifiable | 1/10 pins = 0.10; K1655 has 0 pins ("works" = any stdout) | PASS |
| T5 source literal breaches | ≥ 3 breaches of parent Apple-only-MLX scope | 5/5 breaches (A hardware, B llama.cpp stack, C weights-vs-MMLU, D PoLAR-in-GGML, E no-converter) | PASS |

`all_block = true`, `defense_in_depth = true`, `t_blocking = 5/5`.

## Kill criteria

| KC | Verdict | Reason |
|----|---------|--------|
| K1654 | **fail** | T1 ∧ T5(A,B,C,D): no Gemma 4 llama.cpp converter, no PoLAR converter, parent Apple-only scope breach, MMLU-Pro-with-thinking harness absent |
| K1655 | **fail** | T4 (non-discriminating "works") + T1 (no llama.cpp binary on host) |

## Precedent map

**Reusable (direct):** F#650 killed sister
`exp_prod_adapter_loader_portability` on ap-017(s) axis.
llama.cpp was explicitly named in F#650 T5(B) as one of the three
out-of-scope loader stacks (`safetensors-rs / torch / llama.cpp`).
This experiment is the llama.cpp-specialisation realisation — the
2nd ap-017(s) instance. A 3rd instance would promote the axis to
a top-level guardrail.

**NOT-transport (F#60):** llama.cpp serves **BitNet-2B-4T** with
multi-LoRA hot-swap. Does not transport to Gemma 4 + PoLAR:
- Different base arch (BitNet TQ2_0 vs Gemma 4 4-bit MQA).
- F#60 needed 3 llama.cpp convert-script patches; no equivalent for
  Gemma 4 in this repo.
- F#60 explicitly: "Metal NOT supported (TQ kernels missing)";
  inference was M1 Max **CPU-only**. PROD target is M5 Pro
  Metal-backed.
- F#60 LoRA rank-16 standard factorisation; Pierre adapters are
  **PoLAR r=6** with Grassmannian A — GGML LoRA format does not
  express this factorisation.
- F#60 did not evaluate **thinking mode**; Gemma 4's
  `<|channel>thought` template is not ratified by llama.cpp's chat
  templater.

**Reinforcing (F#61):** "MLX runtime LoRA KILLED on Apple Silicon —
Always pre-merge." Non-native adapter runtimes degrade on Apple
Silicon; supports "pre-merge, don't dynamically swap" conclusion.

## Operator unblock

The preempt does **not** say "llama.cpp is impossible"; it says
"the current repo + parent-scope + local platform cannot measure
it." To unblock, any of:

1. **Extend parent scope.** Widen
   `exp_prod_adapter_format_spec_v2` to commit to a cross-runtime
   bytes-to-inference contract (GGUF + PoLAR→GGML-LoRA converter
   spec). Currently spec_v1 Assumption 1 pins MLX; Assumption 3
   defers hash verification. Both must be lifted.
2. **Add llama.cpp Gemma 4 infra.** Land
   (a) `convert_hf_to_gguf.py` patches for Gemma 4 arch with
   thinking-mode template, (b) a PoLAR→GGML-LoRA converter (or
   accept PoLAR→standard-LoRA rank-expansion collapse as a
   documented lossy mapping), (c) a llama.cpp binary with Gemma 4
   Metal kernels in the sandbox.
3. **Add a MMLU-Pro-with-thinking harness** targetable at
   llama.cpp stdout, seed-deterministic, with the MLX reference
   number pre-registered as a baseline pin.
4. **Downgrade P3.** Until (1)–(3), tag `out-of-scope-local-apple`
   and move to P ≥ 4; this matches F#650's proposed mitigation.

## Runner

Pure-stdlib, zero MLX, zero model load, zero llama.cpp invocation.
Drains the 5-theorem stack via filesystem + `grep` + `shutil.which`
probes. Wall time < 5s.

## Assumptions (logged per guardrail 1008)

1. Local platform = Apple Silicon M5 Pro only. PLAN.md Part 2
   names no remote runner or pre-built llama.cpp-Gemma-4 sandbox.
2. Parent `exp_prod_adapter_format_spec_v1` Assumption 1 literally
   pins MLX; Apple-only-MLX scope is authoritative.
3. F#60's BitNet result does not transport to Gemma 4 + PoLAR for
   the 5 enumerated differences.
4. GGML LoRA format does not express PoLAR r=6
   orthogonal/Stiefel factorisation as of 2026-04 upstream.

## Related

- F#650 ap-017(s) hardware-topology-unavailable (1st instance,
  sister preempt, reusable).
- F#652 ap-017(s2) software-infrastructure-unbuilt (sub-axis;
  schema-incomplete sibling).
- F#60 llama.cpp + BitNet (SUPPORTED; not transportable here).
- F#61 MLX runtime LoRA KILLED (reinforcing; pre-merge on Apple
  Silicon).
- F#627 PoLAR targets `v_proj+o_proj` on Gemma 4 E4B.
- `exp_prod_adapter_format_spec_v1` (parent; SUPPORTED; Apple-only).
