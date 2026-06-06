# Peer Review: MTP Composition

## NotebookLM Findings

Skipped (negative result already well-characterized; deep review unnecessary for a clean KILL).

## Mathematical Soundness

### Derivations

The MATH.md formulations are correct. The MTP module architecture follows DeepSeek-V3 faithfully:

```
h_k = RMSNorm(W_k @ h_{k-1} + emb(token_{t+k}))
logits_k = lm_head(h_k)
```

Sequential chaining (h_k depends on h_{k-1}) is correctly implemented. The loss formulation averaging over valid positions and depths is standard.

### Implementation vs. Math Verification

**Correct**: The code at `/Users/tom/Code/tomsiwik/llm/micro/models/mtp_composition/mtp_composition.py` lines 130-153 implements the sequential chaining correctly. For MTP depth k (0-indexed in code):
- `h_slice = h_prev[:, :T-1-k, :]` -- positions that have enough future context
- `emb_slice = tok_emb[:, k+1:k+1+max_pos, :]` -- embeddings at offset positions
- Target in `mtp_loss()` is `targets[:, k+1:k+1+max_pos]`

Since `targets[t] = tokens[t+1]`, the MTP-k target at position t is `tokens[t+k+2]`, which matches the MATH.md specification of predicting `token_{t+k+1}` (adjusting for the paper's 1-indexed notation).

**Minor code smell (non-blocking)**: Lines 148-149 allocate a full `(B, T, d)` zero tensor just to place `h_k` into positions `[:max_pos]` for chaining. This is wasteful but functionally correct since the next iteration only reads `[:max_pos-1]` which is within the populated region.

### Parameter overhead

Claimed: `(D-1) * d^2`. Code test (`test_param_count`) verifies `params[2] - params[1] == d*d` and `params[3] - params[1] == 2*d*d`. However, MATH.md line 88 says "RMSNorm: 0 parameters (no learnable params)" while the docstring on line 74 says "+8,320 params (~4% overhead)" for D=3, which includes RMSNorm params. Looking at the test: it passes with `expected_per_module = d * d` (line 98), meaning `d*d = 1024` per module for d=32. The RMSNorm in the code uses the project's custom `RMSNorm` which has no learnable parameters (confirmed by test passing). The PAPER.md table shows 202,560 -> 206,656 (+4,096) for depth 2, which is `64*64 = 4,096`. This is consistent. The docstring's "+8,320" number is wrong (should be +8,192 for D=3 at d=64), but this is a comment error, not a code error.

### Hidden assumptions

1. **Gradient flow through shared lm_head**: The MTP loss backpropagates through the shared `lm_head`, meaning MTP gradients update the language model head during fine-tuning. Since only capsule groups are unfrozen during fine-tuning (lines 333-336), `lm_head` is frozen, and the MTP gradient through `lm_head` only reaches the capsule groups. This is correct and the assumption in MATH.md (Assumption 1) is explicitly stated.

2. **MTP uses teacher-forced token embeddings**: The embedding input to each MTP module uses `tok_emb` from the ground-truth sequence, not from model predictions. This is standard for MTP training (same as DeepSeek-V3) but means the model never learns to chain its own predictions. Not a flaw -- this is the established approach.

## Novelty Assessment

### Prior Art

MTP for training is from DeepSeek-V3 (2024). Using MTP specifically to improve expert composition quality is novel -- no prior work tests this. However, this is a negative result, so novelty is primarily in the experimental finding, not in a new mechanism.

### Delta Over Existing Work

The hypothesis (MTP forces richer capsule specialization, improving composition) is a reasonable extension of the project's composition line. The experiment is a clean A/B test of this hypothesis. The finding that MTP is neutral for composition but harmful for absolute quality at character-level micro scale is useful directional evidence.

### References Check

`references/deepseek-v3/` and `references/qwen3-coder-next/` are cited. No prior work on MTP-for-composition was found in the references. The experiment correctly builds on the existing `capsule_moe` composition protocol rather than reinventing it.

## Experimental Design

### Does this test what it claims?

**Yes.** The experiment isolates MTP's effect on capsule specialization by:
1. Using identical pretraining across all conditions (same base, same steps)
2. Varying only the fine-tuning objective (NTP vs. NTP+MTP)
3. Using NTP-only inference for all conditions (fair evaluation)
4. Running the same composition protocol (compose, calibrate, evaluate)

This is a clean causal test: the only independent variable is the MTP training signal during fine-tuning.

### Could a positive result be explained by a simpler mechanism?

Not applicable (result is negative). But the paper correctly identifies that a positive result could have been explained by regularization effects rather than "richer multi-step structure."

### Controls

**Adequate.** The depth=1 condition is effectively the NTP baseline (MTP modules exist but produce zero loss). Three seeds provide basic reproducibility. Per-seed gaps are reported (Table in PAPER.md lines 73-79), showing consistency.

### Kill criteria evaluation

**Kill 1** (MTP composes >5pp worse): Correctly evaluated. MTP-2 gap is -3.15% vs NTP -2.40%, difference -0.75pp. MTP-3 gap is -1.82%, difference +0.57pp. Both well within the 5pp threshold. PASS is correct.

**Kill 2** (MTP provides <2% quality improvement): Correctly evaluated. The "improvement" is negative (MTP composed losses are HIGHER than NTP composed losses). -0.84% and -2.12% are both below the +2% threshold. KILL is correct.

**Note on kill 2 formulation**: The kill criterion as formulated in MATH.md (line 185) is `(composed_loss(1) - composed_loss(D>1)) / composed_loss(1) * 100 < 2.0`. Since lower loss is better, positive improvement means MTP has lower loss. The actual values show MTP has higher loss (worse), so the improvement is negative, which is below 2.0. The kill triggers correctly. The criterion is well-designed -- it would also trigger if MTP improved by only 1.9%, which is appropriately strict.

### Hypothesis graph consistency

The experiment matches `exp_mtp_composition` in HYPOTHESES.yml. Kill criteria match. The status is correctly set to `disproven`. Evidence summary accurately reflects the findings.

### One minor methodological note

The joint model (step 3 in the protocol) is also trained with MTP during fine-tuning. This means the joint baseline for MTP conditions is slightly worse than the joint baseline for NTP. The composition gap comparison is fair (each depth's composed is compared to its own joint), but the absolute composed loss comparison (kill criterion 2) compares NTP-composed against MTP-composed directly, which is the right metric for "does MTP help the final composed product."

## Macro-Scale Risks (advisory)

The PAPER.md Limitations section (lines 159-179) is unusually thorough and honest. It correctly identifies five specific limitations, all of which are genuine:

1. **Token-level MTP vs. character-level**: At subword tokenization, predicting t+2 is qualitatively different (next word vs. next character). The negative micro result does NOT kill MTP for macro.

2. **MTP during pretraining vs. fine-tuning only**: DeepSeek-V3 and Qwen3-Coder-Next use MTP during pretraining. This experiment only tested MTP during fine-tuning. MTP may need to shape base representations (not just capsule specialization) to be effective.

3. **Sequence length**: T=32 characters severely limits the MTP horizon. At T=2048+ with subword tokens, MTP-3 could capture multi-sentence structure.

4. **Routing granularity**: G=4 groups may be too coarse for MTP to create discriminative routing patterns. At 512 experts, MTP could drive more fine-grained specialization.

5. **Speculative decoding**: MTP's inference-time value is untested and orthogonal to composition quality.

These are all legitimate open questions for macro. The micro result establishes that MTP is not a magic bullet for composition at any scale, but does not close the door.

## Verdict

**PROCEED** (with the KILL conclusion as stated)

The experiment is methodologically sound, the math is correct, the code implements the math faithfully, the experimental design properly isolates the MTP variable, and the kill criteria are appropriately formulated and correctly evaluated. The negative result is clean and informative.

Specific strengths:
- Counter-arguments stated upfront in MATH.md (lines 140-143) -- the paper predicted the negative result was possible
- Dose-response (depth 1 vs 2 vs 3) shows monotonic degradation, strengthening the conclusion
- MTP-2 variance reduction noted as a secondary finding worth tracking
- Limitations are honest and correctly scope the generalizability

No revisions needed. The experiment accomplished its purpose: testing whether MTP training improves capsule composition at micro scale. The answer is no. This is a useful negative result that saves compute on the "MTP for composition" hypothesis at micro scale while correctly leaving macro-scale MTP questions open.

The HYPOTHESES.yml status of `disproven` and the VISION.md entry under "SOTA-aligned novel directions" marking MTP as tested are appropriate updates.
