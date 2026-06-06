# LEARNINGS — exp_memento_streaming_inference

## Core Finding
KILLED preempt-structural. 5th MEMENTO-cluster child preempt-KILL (F#669 ≥12 reuses, hybrid with F#666-pure-standalone). Both KCs target per F#666 (K1939 streaming-vs-batch task-acc parity = behavioral; K1940 per-block streaming latency = engineering) but unidentifiable: K1939 = NaN/NaN, K1940 = NaN. Parent `exp_memento_gemma4_replication` PROVISIONAL per F#685 — no Gemma-4-MEMENTO checkpoint, MEMENTO 2-stage SFT + block-mask attention not executable via `mlx_lm.lora`. F#758 registered.

## Why
- Streaming/batch task-acc parity ratio is unidentifiable when neither regime is computable (no trained checkpoint).
- Per-block streaming latency at 20ms granularity requires an INLINE streaming-mode forward path with decoder-loop integration — strict superset of F#739's offline-streaming surface (block-mask attn + per-block hook during inline generation + KV-cache mutation hooks + dual-regime eval harness).
- Substituting PPL parity (antipattern-i, weakly-correlated proxy) or base-Gemma-4 latency (antipattern-t silent objective swap) would each fail F#666 / preempt-structural carve-outs.

## Implications for Next Experiment
- **CANONICALIZATION at 3rd obs**: "target-only KC panel under preempt-KILL on PROVISIONAL parent" micro-pattern now fires across 3 distinct sub-forms — pure-behavioral (F#738), pure-engineering (F#739), mixed-behavioral-engineering (this). Future MEMENTO-cluster (and structurally analogous) target-only preempt-KILLs cite this finding rather than re-deriving.
- **AVOID 6th MEMENTO-cluster child** until parent SUPPORTED via `_impl` companion at full pooled OpenMementos training corpus AND parent K1799/K1800/K1801/K1802 SUPPORTED AND parent-side inline streaming-mode forward path landed.
- **AVOID 5th cos-sim-bucket form** (4th was F#757 cross-instance dual-tail; auto-preempt at 5th).
- **AVOID 8th Hedgehog-ablation sub-type** (super-family saturated at 7).
- **AVOID further routing-acc-only / infra-bench-only standalone** unless target-paired.
- **PREFER target-anchored P=2** candidates: `exp_g4_adapter_initialization_comparison_v2` (direct template — K1977 cos proxy + K1978 PPL-ratio target + K1979 within-init seed-PPL-variance target, runnable), `exp_jepa_scale_sweep_5m_15m_50m`, `exp_g4_lora_rank_importance_per_task`, `exp_cross_axis_interference`, `exp_fingerprint_uniqueness`, `exp_hotswap_latency_impl`, `exp_triple_composition_3domain`, `exp_g4_zs_base_transfer`.
- **No new antipattern memory** — canonical `mem-antipattern-f666-pure-standalone-preempt-kill` and `mem-antipattern-preempt-child-parent-target-unverified` cover this case; non-blocking `run_experiment.py` SystemExit(1) divergence already noted in REVIEW (canonical pattern at line 258 is "main() never raises").
- **No new lit ref** — preempt-structural; Kontonis et al. arxiv:2604.09852 already canonical anchor.
