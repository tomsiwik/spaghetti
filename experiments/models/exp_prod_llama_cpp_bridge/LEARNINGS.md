# LEARNINGS — exp_prod_llama_cpp_bridge

## Core Finding
KILLED preempt-structural (F#763). 2nd instance of ap-017(s) hardware-topology-unavailable axis (1st = F#650 `exp_prod_adapter_loader_portability`). 5/5 theorems block independently; defense-in-depth confirmed; runner pure-stdlib in <8s.

## Why
Parent `exp_prod_adapter_format_spec_v1` is SUPPORTED with explicit Apple-only-MLX scope (Assumption 1 pins MLX). llama.cpp runtime is structurally outside that scope:
- T1: 6 cross-runtime artefacts missing (no llama.cpp binary, no Gemma 4 GGUF converter, no PoLAR→GGML-LoRA converter, no MMLU-Pro-thinking harness, no parent-loader baseline).
- T5: 5/5 source-literal breaches (hardware, loader stack, weights-vs-MMLU, PoLAR-in-GGML, no-converter).
- T4: K1655 "works" is non-falsifiable — pre-reg pin discipline failure surfaced (not concealed).
- F#60 BitNet+llama.cpp does NOT transport: different arch (BitNet TQ2_0 vs Gemma 4 4-bit MQA), 3 missing convert-script patches, M1 Max CPU-only vs M5 Pro Metal target, LoRA rank-16 vs PoLAR r=6 (GGML LoRA expresses A/B only, not Stiefel/orthogonal).

## Implications for Next Experiment
- AVOID: 3rd ap-017(s) instance (CUDA-specific, safetensors-rs-specific, or any non-Apple PROD bridge) — would trigger super-family promotion to top-level guardrail.
- AVOID: 8th F#502/F#646 schema-incomplete cohort hit (currently at 7th — sustained pressure on CLAIM tooling).
- AVOID: parallel "PROD-X-runtime-bridge" claims while parent spec_v1 stays Apple-only-MLX-scoped.
- UNBLOCK PATH (per PAPER.md §Operator unblock): widen `exp_prod_adapter_format_spec_v2` to commit to a cross-runtime bytes-to-inference contract (GGUF + PoLAR→GGML-LoRA converter spec) BEFORE re-claiming any non-MLX runtime bridge. Until then, tag PROD-cross-runtime claims `out-of-scope-local-apple` at P≥4.
- Recommended next claim: target-anchored P=2 with KC-target-pairing verified ON DISK before claim — `exp_g4_adapter_initialization_comparison` (direct template per analyst hat repeated recommendation), `jepa_scale_sweep`, `cross_axis_interference`, `hotswap_latency_impl`, `triple_composition_3domain`, `g4_zs_base_transfer`.
- General AVOID pile: 7th infra-bench, 2nd hash-primitive, 5th cos-sim, 8th Hedgehog (saturated), 2nd argmax-divergence, 14th g4-ablation, 6th MEMENTO-cluster, 3rd audit-2026-04-17+followup-without-rerun — all without target pair.

## Taxonomic refinement (no promotion yet)
- 2nd ap-017(s) hardware-topology-unavailable super-family instance — 3rd promotes axis to top-level guardrail.
- 1st explicit "PROD-cross-runtime-bridge-without-parent-scope-extension" sub-form within ap-017(s).
- 7th F#502/F#646 schema-incomplete cohort hit; non-blocking for THIS verdict, but cohort sustained.
