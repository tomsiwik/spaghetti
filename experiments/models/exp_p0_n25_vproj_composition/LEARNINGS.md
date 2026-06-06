# LEARNINGS: exp_p0_n25_vproj_composition

## Core Finding
N=25 composition WORKS mechanically (PPL -8%, latency 3.2%, ensemble creates signal for zero-solo domains), but fails kill criteria because adapter quality — not composition interference — is the binding constraint.

## Why
With f=0.44 dead adapters (zero solo improvement), K1324 is mathematically impossible regardless of composition quality. SIR degradation from N=5→25 is 0.59x vs predicted 0.408x — ensemble compensates 1.45x better than random-B-matrix theory, meaning real interference is lower than worst-case bounds.

## Key Discoveries
1. **Composition mechanism is not the bottleneck** — PPL improves, 6/11 zero-solo domains gain constructive signal
2. **SIR scaling is 0.59x (N=5→25)** — better than 1/sqrt(5)=0.408x predicted; ensemble partially compensates
3. **v_proj.lora_b = 0 in ALL adapters** — the "dual-target" configuration is effectively o_proj-only; v_proj initialization needs investigation
4. **vocab_retention metric breaks when solo < base** — negative retention (-200%) signals composition outperformed solo, not failure; need standard benchmarks

## Implications for Next Experiment
Before scaling experiments resume: (a) train quality adapters (1000+ iters, curated vocabulary-dense data), OR (b) switch to standard benchmarks (GSM8K accuracy, MedMCQA) that don't break at the boundary. The composition architecture is validated — adapter training quality is the only remaining blocker.

## References
- Finding #505: N=5 composition, 113% mean retention (ensemble effect baseline)
- Finding #506: Distribution mismatch — task-completion data degrades vocab density metric
- Finding #503: Pre-merged serving, 1ms swap, 0% inference overhead
- LoRA (arXiv:2106.09685): rank-r adaptation math
