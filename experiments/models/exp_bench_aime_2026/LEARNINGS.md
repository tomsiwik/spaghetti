# LEARNINGS.md: exp_bench_aime_2026

**Status:** KILLED — infrastructure-blocked (not a falsification of Theorems 1–2)

---

## Core Finding

AIME 2026 was not measured. Three on-disk blockers prevented any evaluation:
(1) `reference_implementations/matharena/` empty — harness entry point `scripts/run.py` absent;
(2) `exp_p1_t2_single_domain_training/adapters/math/adapters.safetensors` missing (only `adapter_config.json`);
(3) `run_experiment.py` `find_math_adapter()` iterates the registry dict's top-level keys instead of `registry["adapters"]`.
K1417/K1418/K1419 marked FAIL (unmeasured) per PLAN.md §1. Theorems 1–2 remain pre-registered, not falsified.

## Why

Blockers #1 and #2 are environment-level (external harness + persisted weights), not research bugs.
Blocker #3 is a local code defect that would surface only after #1 and #2 are fixed.
Blocker #2 has a systemic echo: it is the 9th confirmed instance of the preflight-adapter-persistence
antipattern (memories.md line 86; existing sources list 8 prior experiments). **Finding #421's 82%
GSM8K headline cites weights that are not currently persisted anywhere in the repo.** Every downstream
experiment that loads "the math adapter" inherits this gap.

## Implications for Next Experiment

1. **Do not re-claim `exp_bench_aime_2026` until blockers #1–#3 clear.** Post-fix rerun is a NEW
   claim (PLAN.md §1 "no silent upgrades"); MATH.md predictions stay, `results.json` is rewritten.
2. **Batch the fix via `P11.ADAPTER-REBUILD` or equivalent retraining of `exp_p1_t2_single_domain_training`
   with post-save `stat().st_size > 0` assertion** — rerunning AIME alone leaves 5 other downstream
   experiments (`exp_m2p_composition_n5`, `exp_model_peer_comparison_llama31_8b`,
   `exp_model_peer_comparison_qwen3_4b`, `exp_p9_benchmark_showdown`, `exp_p1_t2_sft_residual_gemma4`,
   `exp_p1_c0_composition_port_gemma4`) still broken.
3. **Researcher preflight must assert** (a) `scripts/run.py --help` exits 0 for any cited harness and
   (b) `Path(adapter_dir, "adapters.safetensors").stat().st_size > 0` for every cited adapter —
   before any run time is spent. The review protocol at memories.md:86 is already mandatory;
   this experiment confirms it still catches real instances.
4. **If weights cannot be recovered**, retract Finding #421's headline number before any new
   experiment cites the math adapter — prevents further downstream propagation.
