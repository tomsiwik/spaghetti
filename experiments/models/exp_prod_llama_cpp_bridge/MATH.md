# MATH.md — exp_prod_llama_cpp_bridge

## Preemptive-Kill Via 5-Theorem Stack (ap-017(s) 2nd instance)

**Verdict: KILLED_PREEMPTIVE** (all five theorems block; any single sufficient).

## Target claim (per DB)

K1654: GGUF export of Gemma 4 E4B + 1 adapter runs in llama.cpp;
MMLU-Pro within 5pp of MLX reference (compound structural +
target-gated).

K1655: Adapter hot-swap works in llama.cpp runtime.

## Parent/source finding and precedents

**Direct parent (dep):** `exp_prod_adapter_format_spec_v1`
(SUPPORTED). Scope: **Apple-Silicon / MLX only**, per its MATH.md
Assumption 1 (`mlx==0.31.1` `save_safetensors` / `load` is the tensor
primitive); cross-version drift and hash verification explicitly
deferred. Writing into a non-MLX consumer (llama.cpp) literally
breaches the source scope.

**Sister preempt (REUSABLE):** F#650 killed
`exp_prod_adapter_loader_portability` on exactly this axis
(ap-017(s) hardware-topology-unavailable). llama.cpp is the named
realisation of T5(B) loader-stack breach in that finding
(`safetensors-rs / torch / llama.cpp`). This experiment is a
specialisation of the same axis — 2nd instance.

**NOT-TRANSPORT precedent:** F#60 (llama.cpp serves **BitNet-2B-4T**
with multi-LoRA hot-swap) is **not transportable** to Gemma 4 E4B:
- Different base architecture (BitNet ternary vs Gemma 4 4-bit
  GPTQ-like).
- F#60 required 3 llama.cpp convert-script patches (arch name,
  `sub_norm` mapping, vocab). No equivalent Gemma 4 patches exist
  on disk.
- F#60 explicitly notes **Metal NOT supported (TQ kernels missing)**;
  inference was M1 Max **CPU-only**. The PROD target is M5 Pro
  Metal-backed.
- BitNet rank-16 LoRA ≠ Pierre PoLAR r=6 with Grassmannian A;
  llama.cpp GGML LoRA format does not express the PoLAR
  factorisation.
- Thinking-mode preservation (`<|channel>thought`) is untested under
  llama.cpp's template engine; F#60 evaluated on non-thinking BitNet.

**Reinforcing precedent:** F#61 ("MLX runtime LoRA KILLED on Apple
Silicon — Always pre-merge on Apple Silicon") established that
runtime adapter composition outside MLX's native path is 4.4× worse
than llama.cpp-CPU; but F#61 was for pre-merged single adapters, not
cross-platform format-identity.

## 5-Theorem Stack

### T1 — Artifact shortfall (hardware-topology-unavailable)

Required artifacts for the claim:
- Apple-Silicon MLX reference (✓ exists via parent `adapter_format_v1`)
- llama.cpp GGUF converter for Gemma 4 E4B-it-4bit with thinking
  mode support (✗ absent — repo grep for `convert_hf_to_gguf` + Gemma
  4 specific patches returns 0 hits)
- llama.cpp adapter converter for PoLAR r=6 factorisation (✗ absent
  — GGML LoRA format expresses standard LoRA A/B only, not polar
  factorisation; no converter script in repo)
- llama.cpp binary on `$PATH` with Gemma 4 arch support (✗ not
  installed in this sandbox; even if installed, Metal-Gemma-4 support
  is not upstream as of Apr 2026)
- MMLU-Pro with-thinking eval harness against llama.cpp output (✗
  absent; Pierre's harness is MLX-native)

shortfall ≥ 4 missing artifact categories. Can run N = 0 of the
claim's required export→run→eval chain on the local platform.

### T2 — Resource budget (physical-topology ceiling)

Claim requires 2 backends (MLX reference, llama.cpp runtime) × full
MMLU-Pro (≈ 12k items) × ≥ 1 adapter swap round-trip.

On local M5 Pro Apple Silicon: only MLX backend is physically
reachable as a first-class runtime. llama.cpp would need (i) arch
patching (F#60 precedent: 3 patches for a simpler arch), (ii) PoLAR
converter write, (iii) Metal/CPU kernel audit for Gemma 4's MQA
grouped-query block (no Gemma 4 llama.cpp kernel path upstream).

Achievable coverage = 1/2 at most without kernel work that exceeds
any reasonable iter budget (F#60 was 8 min, $0 — but that was
*after* the convert-script patches existed for BitNet).

### T3 — DB schema completeness

Literal DB annotation at claim time:

```
success_criteria: [] # MISSING
# ⚠ INCOMPLETE: missing success_criteria
```

Matches F#502 / F#646 schema-completeness-vs-instance-fix axis; same
failure pattern as F#650 (5th F#502 schema hit) and F#652 (5th F#502
hit). Patching success-criteria on this instance alone does not
address the cohort pattern.

### T4 — KC pin count (under-pinned)

| KC | Text | Pins present |
|---|---|---|
| 1654 | GGUF export runs in llama.cpp; MMLU-Pro within 5pp of MLX reference | 1/5: ε-pin (5pp) — no baseline pin (no "MLX reference" number), no pool pin (prompt-set size), no rescale pin (temperature/sampling), no enum pin (thinking on/off) |
| 1655 | Adapter hot-swap works in llama.cpp runtime | 0/5: **non-falsifiable** ("works" has no threshold — any stdout produced could be claimed as "works") |

Pin audit across 5-pin template {baseline, pool, enum, rescale, ε}:
1/10 = 10% ≤ 20% threshold. KC1655 is definitionally unfalsifiable.
F#645 under-pinned KC axis reproduced in a new domain (cross-backend
runtime).

### T5 — Source-finding LITERAL breaches (5 independent)

**(A) Hardware/runtime-scope breach.** Source SUPPORTED scope =
Apple-Silicon MLX only. Target claim spans MLX + llama.cpp runtime.
llama.cpp runs on Metal/CPU/CUDA/BLAS backends — none ratified by
source. Cross-runtime transport is not merely a software
recompilation; GGML quantisation kernels (`Q4_K_M`, `Q5_K_M`)
produce different logit-space outputs than MLX's 4-bit GPTQ-like
path. Bitwise weight identity does not imply logit or task-score
identity.

**(B) Loader-stack breach (llama.cpp-specific realisation).** F#650
flagged generic "`safetensors-rs / torch / llama.cpp`" as the T5(B)
breach. This experiment IS the llama.cpp realisation — one of the
three backends F#650 used as counter-examples. Source ratifies
`mx.save_safetensors` / `mx.load` only.

**(C) Observable-scope breach.** Source K1637 verifies bitwise
weight bytes. Target K1654 verifies MMLU-Pro task score (within 5pp)
under **full inference with thinking**. MMLU-Pro delta depends on:
(i) GGML quantisation scheme choice, (ii) Gemma 4 thinking-template
rendering in llama.cpp's chat templater (not ratified for thinking
mode), (iii) sampler/seed determinism across backends. F#650 T5(C)
applies verbatim here.

**(D) Adapter-factorisation breach (PoLAR).** Pierre adapters use
PoLAR r=6 on `v_proj+o_proj` (F#627). GGML LoRA storage expresses
standard LoRA A/B only. PoLAR's orthogonal/Stiefel structure is not
in the GGML schema. A round-trip would either (i) collapse PoLAR to
standard LoRA (semantic breach) or (ii) require a GGML schema
extension (out-of-source). Source spec_v1 does not commit to a
lossless cross-format encoding.

**(E) No reference llama.cpp loader on disk.** Grep for
`convert_hf_to_gguf` or `llama_cpp` Gemma 4 adapter converter in
repo returns zero Gemma-4-specific hits. There is no artefact
against which an Apple-local test could be compared. T5(E)
reinforces T1.

## Defense-in-depth

T1 alone blocks (required cross-backend artefacts absent).
T3 alone blocks (schema-incomplete per DB).
T5 alone blocks (5 independent source-scope breaches; any single
breaks transport).
T2 and T4 are reinforcing.

## ap-017(s) super-family — 2nd instance

1st: F#650 `exp_prod_adapter_loader_portability` (cross-backend
generic).
**2nd (this exp):** `exp_prod_llama_cpp_bridge` (llama.cpp-specific
realisation of F#650 T5(B)).

If a 3rd ap-017(s) instance arrives (e.g. a CUDA-specific or
`safetensors-rs`-specific PROD follow-up), the axis should be
promoted to a top-level guardrail: "PROD experiments requiring
runtime outside the Apple-only MLX parent scope preempt-KILL
without a remote-hardware / alternate-runtime unblock in PLAN.md
Part 2."

## Kill criteria verdicts (pre-registered for this drain)

- **K1654** result = **fail** (no llama.cpp Gemma 4 converter + no
  Metal kernel + parent scope breach — T1 ∧ T5(A,B,C,D))
- **K1655** result = **fail** (non-discriminating KC; no threshold;
  T4 + T1 hardware absence)

## Assumptions (per guardrail 1008)

1. Local platform = Apple Silicon M5 Pro only; llama.cpp Gemma 4
   converter absent; no remote non-Apple runner named in PLAN.md
   Part 2.
2. Parent `exp_prod_adapter_format_spec_v1` scope is Apple-only MLX
   per its MATH.md Assumption 1. Source artefacts on disk are
   authoritative.
3. F#60 (BitNet llama.cpp) does not transport to Gemma 4 + PoLAR
   per 5 differences enumerated above.
4. llama.cpp GGML LoRA format does not natively express PoLAR r=6
   Stiefel/orthogonal factorisation.

## Success criterion for this drain

This drain SUCCEEDS iff ≥ 3 of the 5 theorems block with evidence
in `results.json`. Exits as KILLED_PREEMPTIVE; never silently
upgraded (guardrail 1010).
