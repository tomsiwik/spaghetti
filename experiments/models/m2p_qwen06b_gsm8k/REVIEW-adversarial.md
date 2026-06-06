# Peer Review: m2p_qwen06b_gsm8k

## Experiment Type
frontier-extension (Type 3) -- MATH.md correctly identifies this as extending proven M2P results (synthetic domains, toy scale) to a real model on a real NLP task.

## Hack Detector
- Fix count: 0 (no mechanism hacks; this is a straight application of existing M2P recipe)
- Is MATH.md a proof or a description? **Description dressed in equations.** MATH.md contains no Theorem/Proof/QED block. It describes what will be measured and makes predictions, but there is no formal proof. For a Type 3 frontier-extension this is acceptable IF the proven framework is correctly cited, but the MATH.md should at minimum state the theorem from prior work being extended.
- Metric used as evidence: quality_ratio (M2P accuracy / SFT accuracy). Not applicable since both were 0.0.
- Kill criteria source: Partially from prior findings (70% threshold relaxed from 85% for frontier), but the 30% K_KILL threshold is an arbitrary judgment call, not derived from any proof.

## Self-Test Audit
**MATH.md has no Self-Test section.** This is a BLOCKING omission per the review protocol. All six self-test items are missing:
1. One-sentence impossibility property -- ABSENT
2. Cited theorems -- ABSENT (references #362, #365, #318, #330 but no formal theorem statements)
3. Predicted numbers -- PRESENT in the predictions table, but not derived from a proof
4. Falsification condition -- ABSENT as a self-test item (kill criteria exist but are not tied to a proof)
5. Hyperparameter count -- ABSENT (there are at least 7 hyperparameters: d_M2P, L_M2P, N_MEMORY, LORA_RANK, LORA_SCALE, TRAIN_STEPS, MAX_SEQ_LEN)
6. Hack check -- ABSENT

## Mathematical Soundness
There is no proof to verify. MATH.md is a well-structured experiment plan that cites prior findings (#362, #365, #318, #330) and makes reasonable predictions, but contains zero formal mathematical derivations. The "proven recipe" claim (d_M2P=64, L_m2p=2, N_memory=32) references toy-scale synthetic results that have no formal guarantee of transfer to real NLP.

The only mathematical claim is the invocation of Aghajanyan et al. (2012.13255) regarding intrinsic dimensionality. The prediction that d_M2P=64 might be tight for real NLP (d_int=100-1000 in their Table 1) is reasonable but is not a derived bound -- it is an empirical reference.

For a Type 3 frontier-extension, this is borderline acceptable: the proven framework exists (M2P quality scaling at toy scale), and the frontier question is clearly stated. However, the experiment cannot produce a finding stronger than "provisional" regardless of outcome.

## Prediction vs Measurement
PAPER.md contains a prediction-vs-measurement table. All measured values are 0.0 across all conditions (base, SFT, M2P), making the table uninformative. The quality_ratio is 0/0, defaulting to 0.0. This means:
- No prediction was tested. The experiment was structurally incapable of producing any signal.
- The kill criterion K908 triggered on a degenerate case, not on a genuine measurement of M2P quality.

## Bug Analysis

### Bug #1 (correctly identified): LoRA applied to wrong operation
PAPER.md correctly identifies this as the primary structural failure. The `forward_with_lora` function computes `h = h + attn_out + LORA_SCALE * (h_norm @ A) @ B` -- an additive residual correction to the hidden state, not a modification of the q/v projection weights. Standard LoRA computes `W_out = W + A @ B`, meaning the low-rank correction participates in the q/k/v computation. The experiment's "LoRA" is a free-standing linear transformation on the hidden state that bypasses the attention mechanism entirely.

Comparing with the correct implementation in `m2p_macro_quality/run_experiment.py` (lines 356-364), which does `base_out = linear_fn(x_in); return base_out + LORA_SCALE * (x_in @ A) @ B` for each of q, k, v projections -- the correction is applied to each projection's output individually, not as a bulk additive term.

**Severity: Fatal.** This means neither SFT nor M2P training was actually training a LoRA adapter. SFT was training a random linear layer added to the residual stream.

### Bug #2 (correctly identified): max_gen_tokens=128 too short
PAPER.md correctly identifies that GSM8K chain-of-thought solutions need 200-400 tokens. With max_gen_tokens=128, the "#### answer" terminator is rarely reached, causing answer extraction to return None for all examples. This explains base_accuracy=0.0 despite Qwen3-0.6B having non-zero baseline GSM8K performance.

**Severity: Fatal for evaluation.** Even if training were correct, evaluation would report 0% accuracy.

### Bug #3 (NOT identified in PAPER.md): Missing causal attention mask
Both `forward_with_lora` (line 262) and `get_layer_hidden_states` (line 296) call `layer.self_attn(h_norm)` without passing a mask argument. Qwen3's `Attention.__call__` signature is `(self, x, mask=None, cache=None)`. Without a mask, `mask=None` propagates to `mx.fast.scaled_dot_product_attention`, which performs **bidirectional** (non-causal) attention.

For autoregressive next-token prediction training, this is incorrect: every token can attend to future tokens, so the model sees information it should not have access to during training. The training loss is artificially lowered (explaining the deceptively low SFT loss of 0.0038 noted in PAPER.md), and the learned parameters do not transfer to causal generation.

For M2P's `get_layer_hidden_states`, the extracted hidden states are computed with bidirectional attention, giving them a different distribution than what the model produces during inference. M2P learns to generate B-matrices from the wrong distribution.

The standard Qwen3 forward path (`Qwen3Model.__call__`, line 153) creates the mask via `mask = create_attention_mask(h, cache[0])`, which returns `"causal"` for multi-token sequences. The custom forward functions bypass this entirely.

**Severity: Fatal for training.** This corrupts both SFT and M2P training signals. The low SFT loss (0.0038) is meaningless -- the model was trained with bidirectional attention on a next-token-prediction objective.

### Bug #4 (NOT identified in PAPER.md): GQA dimension mismatch (latent)
`MODULES_DIMS = [("q_proj", D_MODEL, D_MODEL), ("v_proj", D_MODEL, D_MODEL)]` treats both projections as having output dimension 1024. But Qwen3-0.6B has:
- q_proj: (1024, 16 * 128) = (1024, 2048) -- n_heads * head_dim
- v_proj: (1024, 8 * 128) = (1024, 1024) -- n_kv_heads * head_dim

The q_proj output dimension should be 2048, not 1024. In the current (broken) implementation this is latent because the LoRA is applied to hidden states (dimension 1024), not to the actual projection outputs. But when Bug #1 is fixed by hooking into q_proj/v_proj properly, this dimension mismatch will immediately cause a shape error for q_proj. The A matrix would be (1024, 4) which is correct for the input side, but the B matrix would be (4, 1024) when it needs to be (4, 2048) for q_proj.

**Severity: Latent, blocks the fix.** Not a cause of the current failure, but will cause a crash when Bug #1 is fixed.

## Assessment of Kill Decision

**The kill decision is CORRECT but for the wrong reason.** K908 triggered because quality_ratio = 0/0 = 0.0 < 30%. But this is a degenerate case where no condition produced any signal. The experiment's conclusion -- that M2P cannot capture real NLP task structure at d_M2P=64 -- is NOT supported by the evidence, because the experiment never actually tested this.

PAPER.md correctly identifies this in the "Impossibility Structure" section: "With base_accuracy=0.0, sft_accuracy=0.0, and m2p_accuracy=0.0 all equal, the experiment cannot distinguish between 'M2P matches SFT' and 'nothing works.'" This is honest and accurate.

The kill should be recorded as "killed due to implementation bugs" not "killed due to M2P failure on real NLP." The impossibility structure should note that at least 4 compounding bugs prevented any valid measurement.

## What Carries Forward

PAPER.md's "What Carries Forward" section is mostly correct. The minimum fixes listed (use standard LoRA, increase max_gen_tokens, increase train_steps, increase max_seq_len) address Bugs #1, #2, and #4 (training data truncation). But it misses:

**Missing fix #5:** Pass causal attention mask in both `forward_with_lora` and `get_layer_hidden_states`. Use `create_attention_mask(h, None)` from mlx_lm's base module.

**Missing fix #6:** Fix GQA dimensions in MODULES_DIMS. q_proj output should be 2048 (n_heads * head_dim), not 1024.

**Recommended approach for retry:** Rather than maintaining a custom `forward_with_lora`, use mlx_lm's built-in LoRA support (which handles all of these correctly) for the SFT baseline, and only use the custom forward for M2P-generated B-matrices where you must hook in dynamically. This reduces the bug surface dramatically.

## Finding #373 Assessment

The user mentioned Finding #373. I could not locate this finding in the codebase. If it records this experiment's result, it should state:
- Status: killed
- Failure mode: Implementation bugs (4 compounding), not M2P capacity limitation
- Impossibility structure: quality_ratio = 0/0 is degenerate; no valid measurement of M2P on real NLP was obtained
- The d_M2P=64 capacity question for real NLP remains completely open

## Novelty Assessment
The experiment concept (M2P on real Qwen model + GSM8K) is the correct next step per the research roadmap. There is no prior art that already answers whether M2P hypernetwork generation works on real NLP tasks. The frontier question is well-posed and important. The execution failed before any novel measurement could be obtained.

## Macro-Scale Risks (advisory)
Not applicable -- the experiment did not reach the point where macro-scale considerations matter. If retried:
- d_M2P=64 is likely insufficient for real NLP (Aghajanyan's d_int measurements suggest 100-1000)
- Memory: 15M M2P params + 0.6B base is manageable, but 28 layers x 2 modules x custom forward = slow
- Consider using mlx_lm's native LoRA machinery rather than custom forward passes

## Verdict

**KILL** -- confirmed.

The kill decision is correct. The experiment produced zero valid measurements due to 4 compounding implementation bugs, two of which were not identified in the post-mortem:

1. LoRA applied as residual addition instead of weight-space modification (identified)
2. max_gen_tokens=128 too short for GSM8K (identified)
3. **Missing causal attention mask in custom forward passes** (NOT identified)
4. **GQA dimension mismatch in MODULES_DIMS** (NOT identified, latent)

Additionally:
- MATH.md has **no Self-Test section** (BLOCKING omission)
- MATH.md has **no Theorem/Proof/QED block** (acceptable for Type 3, but means findings are capped at provisional)

**For the retry experiment, the following must be addressed:**

1. Use mlx_lm's native LoRA for SFT baseline (eliminates bugs #1, #3, #4 for the ceiling measurement)
2. For M2P-generated adapters, write a custom forward that (a) hooks into q_proj/v_proj output computation, (b) passes causal mask, (c) uses correct GQA dimensions
3. Set max_gen_tokens >= 384
4. Set max_seq_len >= 512 to avoid truncating answer suffixes
5. Increase train_steps to >= 1000 for SFT
6. Add a sanity check: verify base model accuracy > 0 before proceeding (fail-fast)
7. Add Self-Test section to MATH.md
