# E1: Function Vector Strategy Extraction

## Theorem

**Claim:** Problem-solving strategies (systematic decomposition, step-by-step verification, conservative safety) are encoded as linear directions in Gemma 4 E4B's attention activation space. These directions can be extracted via mean-difference probing and injected as LoRA adapter weights, producing behavioral change without any gradient-based training.

**Proof sketch:**

1. Function Vectors (2310.15213) proves attention heads encode abstract input-output functions as vectors transportable across contexts.

2. ActAdd (2308.10248) proves adding a single activation vector at layer L steers behavior without any weight change.

3. Refusal=Single Direction (2406.11717) proves behavioral directions are 1-dimensional in activation space.

4. F#480 (Pierre) proves v_proj+o_proj are the behavioral projection targets.

5. Therefore: the difference in attention output between a strategy-prompted forward pass and a baseline forward pass contains the strategy direction. This difference, projected into LoRA format, should produce the same behavioral steering as the strategy prompt — but baked into adapter weights, composable with other adapters via NRE.

**Prediction 1:** Mean activation difference (strategy prompt - neutral prompt) at layer L will have cos > 0.5 across different prompts using the same strategy (strategy-specific, not prompt-specific).

**Prediction 2:** Injecting this difference vector as a LoRA B-matrix (with Grassmannian A) will produce measurable behavioral change (>3pp) on out-of-domain tasks.

**Prediction 3:** The extracted vector will compose with domain adapters via NRE without interference (cos < 0.1 with domain adapter direction).

QED (conditional on experimental verification).

## Kill Criteria

- K#2017: Extracted activation vector has zero correlation with strategy condition (cos < 0.1 between systematic-prompt and baseline activations)
- K#2018: Injected LoRA from extracted vector produces no behavioral change on GSM8K (< 2pp improvement)
- K#2019: Strategy vector is prompt-specific, not strategy-specific (same strategy different prompts → cos < 0.3)

## Grounded by

- arXiv:2310.15213 (Function Vectors)
- arXiv:2308.10248 (ActAdd)
- arXiv:2406.11717 (Refusal = Single Direction)
- F#480 (v_proj+o_proj behavioral targets)
- F#428 (Grassmannian composition)
- F#203 (strategies transfer, not knowledge)
