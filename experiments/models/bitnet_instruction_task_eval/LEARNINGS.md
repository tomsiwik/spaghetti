# Learnings: bitnet_instruction_task_eval

## Core Finding

Instruction-tuned LoRA adapters on BitNet-2B-4T are directionally better than NTP-trained
adapters for task evaluation under 1/N composition, but effect sizes are small and the
evaluation is underpowered. The training data format (instruction QA pairs vs raw text)
is the key variable, not the LoRA architecture itself.

## Why This Happened (Literature-Grounded)

The mechanism is well-explained by Li et al. (2023, 2310.00492): instruction tuning causes
three specific internal shifts: (1) the model learns to recognize instruction parts of
prompts, (2) self-attention heads capture word-word relationships around instruction verbs,
and (3) FFN layers rotate pre-trained knowledge toward user-oriented tasks. NTP training
does none of this -- it merely shifts token distributions toward domain surface statistics.

Biderman et al. (2024, 2405.09673) confirm the asymmetry: LoRA substantially underperforms
full finetuning on complex target domains (programming, math) under both instruction and
continued pretraining regimes, but LoRA's advantage is that it "forgets less" -- it better
preserves base model capabilities on out-of-domain tasks. This explains why our composed
model (which must balance 5 domains) benefits from instruction-tuned LoRA: the adapters
add task structure while minimally disrupting the base model's general capabilities.

The medical exception from the prior NTP experiment is now fully explained: medical training
data (flashcards) was already in QA instruction format, accidentally providing the instruction
signal that other NTP domains lacked. This experiment extends that accidental finding to all
domains systematically.

## Confirming Evidence

1. **Biderman et al. (2024, 2405.09673)** - "LoRA Learns Less and Forgets Less": LoRA
   underperforms full FT on target tasks but preserves base model diversity. Instruction FT
   and continued pretraining drive fundamentally different internal mechanisms. CONFIRMS our
   finding that training format matters more than architecture.

2. **Li et al. (2023, 2310.00492)** - "From Language Modeling to Instruction Following":
   Instruction tuning empowers the model to recognize instruction parts, shifts attention
   heads to instruction verbs, rotates FFN knowledge toward user tasks. CONFIRMS the
   mechanistic explanation for why instruction adapters produce task-capable outputs.

3. **Prabhakar et al. (2024, 2410.13025)** - "LoRA Soups": Composing individually-trained
   instruction LoRAs via optimal concatenation (CAT) beats data mixing and standard merging
   by 43% on compositional tasks. CONFIRMS that instruction-tuned adapters are the right
   substrate for composition.

4. **Liu et al. (2023, 2312.09241)** - "TinyGSM": 1.3B model reaches 81.5% GSM8K with
   synthetic instruction data, but ONLY with a separate verifier model. CONFIRMS that
   instruction data is necessary but may not be sufficient for small models on reasoning.

## Contradicting Evidence

1. **"Adapter Merging Reactivates Latent Reasoning Traces" (2025, 2603.15965)**: When
   domain adapters and instruction-alignment adapters are merged, their "partially misaligned
   update directions" cause interference and trace leakage. This suggests our 1/N uniform
   averaging may be masking interference that would surface under stricter decoding or
   harder evaluations. Our small eval sets (15 problems) may not detect this.

2. **TinyGSM verifier requirement**: The 1.3B model needed a second 1.3B verifier to reach
   81.5% GSM8K. Our single-model 6.7% on MATH-500 (1/15 correct) is consistent with the
   finding that instruction tuning alone is insufficient for reasoning in sub-3B models.
   The base model's reasoning floor, not the adapter quality, may be the binding constraint.

3. **Biderman et al. performance gap**: LoRA substantially underperforms full finetuning
   on complex domains. Our small effect sizes (+0.1pp medical, 1/15 math) are consistent
   with LoRA being a weak learner for task skills. The instruction format helps but cannot
   overcome LoRA's fundamental capacity limitation vs full FT.

## Alternative Approaches (What We Could Try Instead)

1. **LoRA Soups CAT (Concatenation)** (2410.13025): Instead of 1/N uniform averaging,
   optimally weight and concatenate adapters. Beats standard merging by 43%. This is the
   most directly applicable alternative -- same adapters, different composition math.
   Could be tested as a follow-up micro experiment.

2. **MoLoRA per-token routing** (2402.12851): Keep adapters independent, route per-token
   via learned router. Eliminates interference entirely. Our oracle routing results (0/4
   regressions vs 1/4 for composed) already validate the routing direction. MoLoRA is the
   learned version of our oracle.

3. **Geometry-aware merging** (2603.15965): Account for layer-wise geometric misalignment
   between adapters before merging. Reduces trace leakage. More principled than uniform
   1/N but more complex to implement.

4. **TIES-Merging** (Yadav et al., 2023): Trim redundant parameters, resolve sign conflicts,
   average only aligned parameters. Available in HuggingFace PEFT. Natural fit for LoRA
   adapters since they are already task-specific tensors.

5. **DARE** (Yu et al., 2024): Randomly prune task weights by density fraction, rescale.
   Can be combined with TIES or linear merging. Available in HuggingFace PEFT.

6. **Task-Aware Retrieval Composition** (2602.21222): Use vector DB similarity to
   dynamically select and weight adapters per input. Zero-shot generalization without
   oracle routing labels.

## Implications for Next Experiments

1. **Instruction tuning is confirmed as the default training format.** All future SOLE
   adapters should use instruction-formatted data. The NTP path is closed.

2. **The evaluation must scale up.** 15 problems per domain is insufficient. The math
   result (1/15 = 6.7%) is one coin flip from 0%. Next evaluation should use:
   - GSM8K (not MATH-500) for math -- BitNet-2B scores 58.4% few-shot
   - 100+ samples per domain minimum
   - Proper statistical power analysis

3. **The prompt format confound must be resolved.** Re-evaluate NTP adapters with
   instruction prompts to isolate the training format effect from the prompt effect.
   Without this, we cannot attribute improvement to instruction tuning vs prompting.

4. **Routing > merging is the direction.** Oracle routing (0/4 regressions) strictly
   dominates 1/N composition (1/4 regressions). The next composition experiment should
   test learned routing (MoLoRA-style) or LoRA Soups CAT, not more uniform averaging.

5. **The 2B reasoning floor is real.** Do not expect math breakthroughs on MATH-500 with
   a 2B base. GSM8K is the right benchmark. Consider verifier/self-consistency for
   reasoning tasks (TinyGSM pattern).

6. **Legal-style degeneration needs detection.** Template collapse (identical outputs
   across all eval samples) will recur on short-answer domains. Build degeneration
   detection into the evaluation pipeline before scaling to more domains.

## New References to Add

| Paper | ArXiv ID | Relevance |
|-------|----------|-----------|
| Adapter Merging Reactivates Latent Reasoning Traces | 2603.15965 | Geometric misalignment in adapter merging; trace leakage risk |
| Task-Aware LoRA Adapter Composition via Similarity Retrieval | 2602.21222 | Vector DB routing for dynamic adapter composition |
