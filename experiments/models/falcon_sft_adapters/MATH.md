# SFT vs NTP Loss for Adapter Training on Instruction-Tuned Base Models

## Type: Guided Exploration (Type 2)

The mathematical framework (LoRA composition, pre-merge weight addition) is proven
from prior work. The unknown is whether SFT loss resolves the empirically observed
degradation from NTP-trained adapters.

## A. Failure Mode Identification

**The disease:** NTP-trained adapters degrade instruction-tuned base model performance.
Three independent experiments confirm this:
1. exp_falcon_e3b_composition: composed GSM8K 0.44->0.36, medical MMLU 0.55->0.30
2. exp_generation_quality_llm_judge: 5/5 domains worse (Finding #166)
3. exp_top2_output_space_falcon: 1/5 domains beat single (Finding #166)

**Root cause analysis:** The instruction-tuned base model M has been trained to produce
distribution P_instruct(y|x) = P(response|instruction). NTP loss trains adapters on
the FULL sequence including instruction tokens. This means the adapter learns a
perturbation delta that tries to predict instruction tokens -- tokens the base model
has already been calibrated to process as conditioning, not as generation targets.

Formally: Let the training sequence be [x_1, ..., x_I, y_1, ..., y_R] where x are
instruction tokens and y are response tokens. NTP loss computes:

    L_NTP = -(1/T) sum_{t=1}^{T} log P(t_t | t_{<t})

where T = I + R covers ALL tokens. The gradient from instruction tokens pushes the
adapter to modify the model's conditioning behavior, conflicting with the base model's
instruction-following calibration.

## B. The Right Question

Not: "How do we prevent adapter degradation?"
But: "What loss function preserves the base model's instruction-following calibration
while specializing the response distribution?"

Answer: SFT (Supervised Fine-Tuning) with response-only masking. This is standard
practice in the RLHF pipeline (Ouyang et al., 2022, "Training language models to
follow instructions with human feedback").

## C. Prior Mathematical Foundations

**Property (Response-only gradient under SFT loss):** If the loss is computed
only on response tokens y given instruction tokens x:

    L_SFT = -(1/R) sum_{t=I+1}^{I+R} log P(y_t | x, y_{<t})

then the SFT loss produces zero gradient signal from the task of PREDICTING
instruction tokens. Specifically:

    dL_SFT / d(delta) has no contribution from positions t <= I

This follows directly from the chain rule: the loss has no terms at instruction
positions, so no gradient flows from instruction-token prediction.

**Important caveat:** This does NOT mean "instruction-processing pathways receive
zero gradient." In transformers, shared attention weights at positions t > I attend
to instruction positions t <= I. The response-token loss at position t > I DOES
produce gradients that flow through attention over instruction tokens. The correct
statement is narrower: SFT eliminates gradient signal from the task of predicting
instruction tokens themselves, but response-token gradients still flow through
shared parameters that process instructions as context.

**Prior art:**
- Ouyang et al. (2022): SFT is Step 1 of RLHF, applied with response-only loss
- Hu et al. (2022, LoRA): LoRA preserves pre-trained features by low-rank constraint
- The combination (SFT + LoRA) is standard practice in production fine-tuning

**LoRA composition (proven in prior work):**
Pre-merge composition: W_composed = W_base + sum_i (alpha_i * B_i @ A_i)
This is linear in adapter parameters and preserves each adapter's contribution
when alpha_i are set correctly.

## D. Predictions

### Behavioral Predictions
1. SFT-trained adapters will NOT degrade base model performance on instruction-following
   benchmarks (MMLU), because instruction-processing gradients are zero
2. SFT adapters will improve response quality for domain-specific queries, because
   the adapter specializes the response distribution P(y|x, domain)
3. Composed SFT adapters will preserve improvements, because pre-merge is linear

### Quantitative Predictions (from prior experiment baselines)
- Falcon-E-3B base: GSM8K=0.44, MMLU avg=0.54
- NTP adapters degraded to: GSM8K=0.36, MMLU avg=0.43 (avg across domains)
- **P1:** SFT single adapters: domain MMLU >= base (>= 0.54 per domain)
- **P2:** SFT composed: overall MMLU >= base (>= 0.54 avg)
- **P3:** SFT math adapter: GSM8K >= base (>= 0.44)
- **P4:** >= 3/5 SFT adapters improve over base on their domain benchmark

## E. Assumptions & Breaking Conditions

1. **Instruction boundary is identifiable**: We assume we can correctly identify
   which tokens are instruction vs response. If the boundary is misidentified,
   some instruction tokens receive gradient (partial NTP contamination).
   - Breaking: If >20% of masked tokens are actually response tokens, improvement
     will be reduced but degradation should still not occur.

2. **Low-rank perturbation is sufficient for SFT**: LoRA rank-16 has enough capacity
   to specialize the response distribution. Prior experiment showed NTP LoRA trains
   successfully (loss decreases), so capacity is sufficient.

3. **200 training iterations is sufficient**: The prior experiment used 200 iters with
   NTP. SFT has fewer loss terms per example (only response tokens), so effective
   gradient signal per example may be lower. We may need more iterations.
   - If 200 iters insufficient: loss will plateau high, adapter will be near-identity.

4. **Pre-merge composition preserves SFT benefits**: Linear weight addition is exact
   for single adapters but introduces interference for composed adapters. Prior work
   (TIES, DARE) addresses this for NTP adapters. SFT adapters may interfere less
   because they don't modify instruction-processing pathways.

## F. Worked Example (conceptual)

Sequence: "### Instruction:\nSolve 2+3\n### Response:\n5"
Tokenized: [I1, I2, I3, I4, I5, I6, R1, R2]  (8 tokens total)

NTP loss mask: [1, 1, 1, 1, 1, 1, 1, 1]  -> gradient on ALL 8 positions
SFT loss mask: [0, 0, 0, 0, 0, 0, 1, 1]  -> gradient on ONLY 2 positions

The NTP adapter receives 3x more gradient signal, but 75% of it pushes the adapter
to modify how the model processes instructions -- directly conflicting with the
instruction-tuned base. The SFT adapter receives less gradient but ALL of it
reinforces the desired response distribution.

## G. Complexity & Architecture Connection

Training cost: Identical to NTP LoRA. Only difference is the loss mask.
Inference cost: Identical (same adapter architecture, same pre-merge composition).
Memory: Identical (same rank-16 LoRA on q_proj, v_proj, o_proj).

The SFT approach adds zero hyperparameters beyond what NTP already requires.
The only implementation change is the loss mask.

## Self-Test

1. What is the ONE mathematical property that makes the failure mode impossible?
   SFT loss masks instruction tokens, so adapter parameters receive zero gradient
   from the task of predicting instruction tokens. Note: shared attention weights
   still receive gradient from response tokens attending to instruction context,
   so the preservation is partial, not total.

2. Which existing theorem(s) does the proof build on?
   Chain rule of calculus (zero loss -> zero gradient). Ouyang et al. 2022 (RLHF SFT).
   Hu et al. 2022 (LoRA preserves pre-trained features).

3. What specific numbers does the proof predict?
   P1-P4: SFT adapters >= base accuracy on domain benchmarks (>= 0.54 MMLU avg).
   NTP adapters gave 0.43 avg (20% degradation). SFT should give >= 0.54 (0% degradation).

4. What would FALSIFY the proof?
   If SFT adapters ALSO degrade base performance, the root cause is NOT instruction-token
   gradients but something else (e.g., LoRA rank interference, weight scale issues).

5. How many hyperparameters does this approach add?
   Count: 0. Same LoRA config, only the loss computation changes.

6. Hack check: Am I adding fix #N to an existing stack?
   No. This is fixing the root cause (wrong loss function), not adding a mitigation.
