# LEARNINGS — exp_model_peer_comparison_llama31_8b

## Core Finding
KILLED by prerequisites, not refuted on metrics. The Pierre-vs-Llama-3.1-8B
5-benchmark sweep never ran because P1 (adapter weights on disk) and P3
(upstream T2.1 valid) both failed at the 3-second probe. P2 (lm_eval import)
passes. K1691 FAIL (unmeasurable), K1692 PASS (half), K1693 FAIL (moot +
structurally unreachable: adapters trained thinking=False).

## Why
1. **P1 FAIL** — `adapters/{math,code,medical,sql,bash}/` contain only
   config/README stubs, no `.safetensors`. registry.json claims scores for
   weights that were never committed or were purged. `adapters/code/` does
   not exist at all.
2. **P3 FAIL** — upstream `exp_p1_t2_single_domain_training` T2.1 flipped
   `supported → killed` in the 2026-04-18 audit (metric-swap: DB KC #1030
   says "MedQA", code measures MedMCQA; plus format-artefact on
   base_gsm8k=0% from max_tokens=256 CoT truncation). Any downstream claim
   building on T2.1 inherits these audit flags.
3. **Design correctly caught both** — MATH.md §7 pre-registered the KILL
   path, the probe was honest, and the verdict is over-determined by P1+P3
   alone. No wasted compute.

## Implications for Next Experiment
Three permanently-learned rules propagate to the 2 open sibling macros
(`exp_model_peer_comparison_qwen3_4b`, `exp_model_mtbench_composed`):

1. **Precondition-probe before macro sweeps.** Every `exp_model_peer_*` and
   `exp_model_mtbench_composed` MUST run a 3-second P1 (adapter .safetensors
   on disk) + P2 (lm_eval importable) + P3 (upstream verdict=supported, no
   audit flags) probe before claiming compute. Encode as MATH.md §7
   pre-registered KILL path.
2. **Registry ≠ artefacts.** `adapters/registry.json` claims scores and
   paths without guaranteeing `.safetensors` exist. Never trust the
   registry — `ls *.safetensors` before loading.
3. **Downstream P1 macros inherit upstream audit flags.** T2.1's flip
   propagates to every dependent experiment. `exp_model_mtbench_composed`
   must check EVERY upstream in its composition chain, not just the first.

## Next-experiment routing
The researcher should pick a pure-research claim that does NOT depend on
Pierre adapter weights or T2.1 verdict. Open experiments at priority ≤ 2
that don't touch the missing-adapter thread are the right target. A v2
rerun of this macro requires rebuilding T2.1 first (MedQA dataset fix +
max_tokens ≥ 512 + persist .safetensors), which is out of scope until T2.1
is re-opened.
