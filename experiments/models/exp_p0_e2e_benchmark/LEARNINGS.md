# LEARNINGS: exp_p0_e2e_benchmark

## Core Finding
The full Pierre pipeline (v_proj+o_proj LoRA + TF-IDF routing on Gemma 4 E4B 4-bit) passes all 5 kill criteria: +56pp GSM8K, +45pp HumanEval, +19pp MedMCQA, 98.3% routing, 1.82s latency. The complete system works end-to-end on standard benchmarks.

## Why
Distribution-aligned training (train on task data, eval on matching benchmark) produces large accuracy gains because the adapter learns task-specific behavior (answer format, reasoning chains). Training cost is ~$2 / 20 min / domain for a 21.8 MB adapter — the $2/domain claim is verified.

## Key Observations
- **Base model is weaker than expected**: Gemma 4 E4B 4-bit scores 17-31% on these benchmarks (not 40-60% as estimated from #421). Measure base first before predicting deltas in future experiments.
- **Latency is adapter-reload dominated**: routing <1ms, adapter load ~1s, generation ~0.5s. Pre-merge serving (#503) eliminates the reload cost entirely.
- **Finding #506 resolved**: HF data degrades vocab density but improves benchmark accuracy when evaluation is distribution-aligned. These are orthogonal effects.
- **v_proj+o_proj confirmed**: competitive with q_proj on benchmarks, proven better for behavioral quality (#504). This is the canonical adapter target going forward.

## Implications for Next Experiment
The open gap is **composition under benchmark evaluation** — #505 proved composition preserves PPL at N=5, but accuracy under composition has not been tested. Next: merge adapters (room model: W_combined = Σ ΔW_i) and re-run GSM8K/HumanEval/MedMCQA to verify accuracy is preserved post-merge.
