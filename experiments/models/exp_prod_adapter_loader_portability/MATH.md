# MATH.md — exp_prod_adapter_loader_portability

## Preemptive-Kill Via 5-Theorem Stack

**Verdict: KILLED_PREEMPTIVE** (all five theorems block; any single sufficient).

## Target claim (per DB)

The `.pierre` adapter file written on one hardware backend (Apple
Silicon / MLX) produces bitwise-identical weights — and logit
cosine > 0.999 on 100 fixed prompts — when loaded on CUDA (KC1656)
and CPU (KC1657) backends, with documented endianness/dtype
handling (KC1658).

## Parent/source finding

`exp_prod_adapter_format_spec_v1` (SUPPORTED, PAPER.md lines 1–114).
The parent freezes the on-disk layout and verifies bitwise-lossless
round-trip for Apple-Silicon / MLX only (K1637, K1638). The PAPER
explicitly lists `exp_prod_adapter_loader_portability` as a
follow-up for cross-language/cross-backend loaders that do not yet
exist.

Source MATH.md Assumption 1: "`mlx==0.31.1` `save_safetensors` /
`load` is the tensor primitive." MLX only executes on Apple Silicon.

Source MATH.md Assumption 4: "No `mlx-lm` dependency in this spec
(pure MLX-core + stdlib)." Target requires full inference under a
real LM to compare logits.

## 5-Theorem Stack

### T1 — Artifact shortfall (hardware-topology-unavailable)

Required artifacts for the claim:
- Apple-Silicon reference loader *(✓ exists: adapter_format_v1.py)*
- CUDA reference loader *(✗ absent — grep of repo: 0 matches for a
  .pierre / adapter_format_v1-compatible loader binding CUDA tensors)*
- CPU-backend reference loader for Gemma 4 inference *(✗ absent —
  mlx-lm Gemma 4 path is Apple-only; no torch CPU inference harness
  for .pierre adapters on disk)*
- Physical CUDA GPU access *(✗ absent — `nvidia-smi` is not found;
  `uname -m = arm64`; `system_profiler` returns "Apple M5 Pro")*
- Physical non-Apple CPU backend *(◦ CPU exists but no loader stack)*

shortfall = 4 missing artifact categories. Can run N = 0 of the
claim's required cross-backend identity checks on the local
platform.

### T2 — Resource budget

Even granting the missing artifacts existed, the claim requires:
- 3 backends (Apple, CUDA, CPU) × 100 prompts × full-model inference
- On local machine: only one backend physically reachable.
- Achievable in-iter fraction: 1/3 at most (Apple baseline; CUDA
  and CPU runs are structurally impossible on arm64 Apple Silicon
  without remote hardware that is not referenced in PLAN.md Part 2).

need = 3 backends / feasible = 1 ⇒ coverage ≤ 33%, cannot satisfy
an *identity* claim that requires all three. Budget is not a time
ceiling here — it is a *physical-topology* ceiling.

### T3 — DB schema completeness

Literal DB annotation at claim time:

```
Success Criteria: NONE — add with: experiment success-add ...
⚠ INCOMPLETE: missing success_criteria
```

This matches F#502 / F#646 schema-completeness-vs-instance-fix
axis: the row is flagged incomplete before any scientific question
is asked. Patching success-criteria on this instance alone would
not address the cohort-wide pattern.

### T4 — KC pin count (non-discriminating)

Three KC:

| KC  | Text                                                              | Pins present |
|-----|-------------------------------------------------------------------|--------------|
| 1656 | Apple vs CUDA loaded adapter: logit cosine >0.999 on 100 fixed prompts | ε-pin only (>0.999) |
| 1657 | Apple vs CPU loaded adapter: logit cosine >0.999 on 100 fixed prompts  | ε-pin only (>0.999) |
| 1658 | Endianness + dtype handling documented and tested                   | **no pin** (non-falsifiable — "documented" has no threshold) |

Pin audit across the 5-pin template {baseline, pool, enum, rescale,
ε}: 1/5 pins present. KC1658 is definitionally unfalsifiable as
stated. F#645 subcategory axis: under-pinned KC reproduces the
N-scale / subcategory-aggregation failure mode in a new domain
(loader verification).

### T5 — Source-finding LITERAL breaches

Five independent scope/semantic gaps between source (spec_v1) and
target (loader_portability):

**(A) Hardware-scope breach (NEW AXIS).** Source SUPPORTED scope =
Apple-Silicon MLX only. Target claim = Apple ∪ CUDA ∪ CPU
identity. The source experiment's Assumption 1 names MLX as the
tensor primitive; MLX physically does not execute on CUDA. Source
contains no cross-hardware evidence; the SUPPORTED verdict cannot
be transported across a physical-hardware boundary that the source
never tested.

**(B) Loader-stack breach.** Source uses `mx.save_safetensors` /
`mx.load` as ground-truth encoder/decoder. A CUDA or CPU consumer
would use `safetensors-rs` (Rust), PyTorch's `safetensors` Python
binding, or `llama.cpp`'s GGUF-adjacent loader. None of these are
on disk in this repo; their bitwise-determinism with respect to
MLX's writer is un-audited.

**(C) Observable-scope breach.** Source K1637 verifies *weight
bytes* are bitwise-identical. Target KC1656/1657 verify *logit
cosine* under full-model inference. Logits are a downstream
observable that depends on the base model, tokenizer, dtype
promotion rules, and math-library numerics per backend. Bitwise
weight identity does not imply logit identity when the inference
kernel differs across backends. Source ratification does not
transport.

**(D) Untested invariant — signing.** Source Assumption 2 literally
defers signing to `exp_prod_adapter_signing` ("Signing is a
downstream experiment; slot is reserved, not exercised."). A
cross-backend *identity* claim that a shipped adapter loads
identically must also verify signature-slot zeroing is read
identically across endianness/dtype boundaries — the source never
exercised the slot, so the cross-backend invariant for signed files
is definitionally un-ratified.

**(E) No reference CUDA loader on disk.** Grep of the entire repo
for a CUDA binding of `adapter_format_v1` / `.pierre` returns zero
matches. There is no artefact against which an Apple-local test
could be *compared* even if we stubbed a CUDA backend. T5(E)
reinforces T1.

## Defense-in-depth

T1 alone blocks (zero required hardware artifacts on this
platform). T3 alone blocks (schema-incomplete per DB). T5 alone
blocks (five independent source-scope breaches; any single breaks
transport). T2 and T4 are reinforcing.

## Novel ap-017 axis candidate

**hardware-topology-unavailable**: target claim requires
observations on hardware physically absent from the local platform.
Distinct from prior ap-017 preempts (a)–(r), which are all
software/semantic scope gaps on a single platform. This is a
*physical-hardware-absence* breach. Proposed registration:
ap-017 preempt (s).

## Kill criteria (pre-registered for this drain)

- **K1656** result = **fail** (no CUDA hardware; structurally
  unmeasurable — T1(hardware) ∧ T5(A,B,C))
- **K1657** result = **fail** (no CPU inference stack for `.pierre`
  on Gemma 4 — T1(loader) ∧ T5(B))
- **K1658** result = **fail** (non-discriminating KC; no threshold
  stated; T4)

## Success criterion for this drain

This drain SUCCEEDS iff ≥ 3 of the 5 theorems block, each with
evidence in `results.json`. It exits as KILLED_PREEMPTIVE, not
SUPPORTED, per guardrail 1009 (never silently upgrade).

## Assumptions (per guardrail 1007)

1. Local platform = Apple Silicon only. CUDA/non-Apple hardware not
   reachable in this iter; PLAN.md Part 2 does not name a remote
   runner.
2. The parent `exp_prod_adapter_format_spec_v1` artefacts on disk
   are authoritative for source scope.
3. Repo grep for "cuda" / "nvidia" / cross-backend .pierre readers
   returns zero structural matches (confirmed at claim time).
