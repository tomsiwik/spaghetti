# PAPER: exp_prod_adapter_loader_portability

**Verdict: KILLED_PREEMPTIVE.** All five theorems block; defense-in-depth
holds (≥3 of 5 block independently). The claim is structurally
unmeasurable on the local Apple-only platform and is the 31st
preemptive-kill in the audit-2026-04-17 drain. New ap-017 preempt
axis candidate **(s) hardware-topology-unavailable**.

## Hypothesis (from DB / MATH.md)

A `.pierre` adapter file written on Apple Silicon loads to
bitwise-identical weights on CUDA and CPU backends, such that full-
model logit cosine > 0.999 on 100 fixed prompts (K1656, K1657), with
endianness/dtype handling documented and tested (K1658).

## Predictions vs. Measurements

| Theorem | Predicted to block? | Measured | Status |
|---------|---------------------|----------|--------|
| T1 artifact-shortfall | yes (missing ≥3 artefacts) | shortfall=3 (cuda_reference_loader, cpu_inference_stack_for_pierre, cuda_gpu_hardware); `nvidia-smi`=absent; `uname -m`=arm64 | ✅ blocks |
| T2 resource-budget | yes (backends_reachable < 3) | 1/3 backends reachable; coverage=0.333 | ✅ blocks |
| T3 schema-completeness | yes (DB literal incomplete) | success_criteria count=0; DB flag "⚠ INCOMPLETE" | ✅ blocks |
| T4 kc-pin-count | yes (≤1/5 template) | 2/15 pins present = 0.133 ratio; K1658 non-falsifiable ("documented") | ✅ blocks |
| T5 source-literal-breaches | yes (≥3 of 5) | 5/5 breaches confirmed: (A) hardware-scope, (B) loader-stack, (C) weights-vs-logits, (D) untested-signing, (E) no-CUDA-loader-on-disk | ✅ blocks |

`all_block = True`, `defense_in_depth = True`, `t_blocking = 5/5`.

## Kill criteria

| KC    | Result | Reason |
|-------|--------|--------|
| K1656 (Apple vs CUDA cos>0.999) | **fail** | no CUDA hardware; T1(hardware) + T5(A,B,C) |
| K1657 (Apple vs CPU cos>0.999)  | **fail** | no CPU inference stack for `.pierre` on Gemma 4; T1(loader) + T5(B) |
| K1658 (endianness+dtype doc+test) | **fail** | non-discriminating KC (no threshold); T4 |

## Novel ap-017 axis

**(s) hardware-topology-unavailable** — target claim requires
observations on hardware physically absent from the local platform
(CUDA, non-Apple CPU backend). Distinct from prior preempts (a)–(r)
which are all software/semantic gaps on a single hardware class.
This is the first *physical-hardware-absence* preempt in the audit-
2026-04-17 drain. Reusable on: `exp_model_peer_comparison_mistral_nemo`,
`exp_model_quantization_composition_stability`, and any remaining
PROD experiments whose KCs mention "CUDA", "CPU backend",
"endianness", or cross-platform determinism.

## Parent / source

`exp_prod_adapter_format_spec_v1` (SUPPORTED 2026-04-18). Source
scope explicitly Apple-Silicon / MLX only (Assumptions 1 & 4 of
source MATH.md). Source PAPER lines 106–110 list this experiment
as a *follow-up for cross-language readers that do not yet exist*
— the SUPPORTED verdict is not transportable across hardware
boundaries it never tested.

## Assumptions (per guardrail 1007)

1. Local platform is Apple Silicon only. PLAN.md Part 2 does not
   name a remote CUDA runner.
2. The parent's on-disk artefacts are authoritative for source
   scope.
3. `nvidia-smi` absence + `uname -m = arm64` + `system_profiler`
   reporting "Apple M5 Pro" is sufficient evidence that no CUDA
   hardware is reachable in this iter.

## Verdict-consistency pre-flight (PLAN.md §1)

1. `results.json["verdict"] = "KILLED"` ✅ (not silently upgraded).
2. `results.json["all_pass"] = false` ✅.
3. PAPER verdict line contains "KILLED_PREEMPTIVE" — explicit kill,
   not PROVISIONAL / PARTIAL. ✅
4. `is_smoke = false`; this is a structural preempt, not a
   smoke run. ✅
5. KC text was not altered between MATH.md and now (reproduced
   verbatim from `experiment get exp_prod_adapter_loader_portability`
   at claim time). ✅
6. Antipattern check (type=fix memories): no composition math here
   (T5(A–E) are scope breaches, not arithmetic errors); no LORA
   scale (no training); no tautological routing (no router); no
   `shutil.copy` producing empty adapters (no adapters written); no
   hardcoded `"pass": True` (KC results are `"fail"`); no
   smoke-as-full mislabel (`is_smoke=false`, no training loop). ✅

## Reproducibility

```bash
experiment run exp_prod_adapter_loader_portability
```

Pure stdlib. ~<0.1 s wall. No MLX, no model load, no inference, no
network. Grep + `shutil.which("nvidia-smi")` + `platform.machine()`
are the only system probes.

## Follow-ups

None from researcher hat. Operator unblock required:
- (A) Python 3.14 `datasets` fix (prior HALT).
- (B) T2.1 reopen (prior HALT).
- (C) Analyst cap raise (prior HALT addendum).
- **(D NEW)** Remote-CUDA runner declaration in PLAN.md Part 2
  (or explicit downgrade of all PROD hardware-portability
  experiments to priority > 2 as out-of-scope for local-apple
  drain).
