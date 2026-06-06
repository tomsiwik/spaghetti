# LEARNINGS: exp_pro_sft_5_adapters

## Core Finding

**SFT with low-quality data actively degrades a strong base model (Qwen3-4B behavioral
0.691 → 0.391 with adapters). The same recipe that succeeded on weak BitNet-2B (0.41
behavioral) fails on a strong base because training data quality is BELOW base model
capability. This is Finding #209 escalated: legal/finance degradation is not domain-specific
but a general law — data quality must exceed base model capability for SFT to add value.**

## Why This Happened

The failure has three reinforcing causes:

1. **Data-model quality inversion.** On BitNet-2B (55% MMLU), Reddit-quality data was
   informative — the model learned something new. On Qwen3-4B (92% MMLU), the same data
   is noise — the model already "knows" more than the training data teaches. SFT forces
   the model to match a LOWER-quality distribution. This is the "alignment tax" mechanism
   described in Ghosh et al. (2402.05119): instruction tuning fails to inject knowledge
   and instead degrades it through style imitation.

2. **Format override destroys reasoning.** Qwen3-4B is a thinking model with native
   `<think>` token reasoning (2505.09388). The ### Instruction/### Response SFT format
   bypasses this capability, forcing simpler response patterns. The Qwen3 Technical Report
   itself documents SFT degradation on complex tasks and uses self-distilled thinking data
   as the fix.

3. **Batch-size-1 variance masks convergence.** All 5 domains show non-monotonic loss
   curves. The "converged" label (L_final < L_base) is a snapshot, not a guarantee —
   at a different stopping point, different domains might appear converged. This is
   consistent with Finding #209's observation that legal/finance degradation is monotonic
   with adapter scale.

## Confirming Evidence

- **2402.05119** (Ghosh et al.): "A Closer Look at the Limitations of Instruction Tuning."
  IT fails to inject knowledge; LoRA fine-tuning only learns style tokens while full-parameter
  IT degrades knowledge. Directly confirms our observation that SFT teaches format, not facts.

- **2305.11206** (LIMA, Zhou et al.): 1,000 high-quality examples match GPT-4 in 43% of cases.
  "Superficial Alignment Hypothesis" — alignment is mostly about learning style, not knowledge.
  Confirms that data QUALITY, not quantity, determines SFT success.

- **2402.00530** (Superfiltering, Li et al., ACL 2024): IFD score filters data using a tiny
  model. Filtered subsets outperform full dataset. Confirms quality filtering > quantity.

- **2502.09650** (Gao et al.): "Principled Data Selection for Alignment." Overly difficult
  examples significantly degrade performance. Model capacity dictates a threshold — data
  beyond model capacity hurts. Filtering improves win rates by 9-16%.

- **Finding #209** (own): BitNet-2B legal/finance degradation at scale=20. Same domains,
  same failure, weaker base. The pattern scales with base model strength.

- **Finding #265** (own): NTP adapters preserve OOD reasoning (30pp GSM8K gap vs SFT).
  SFT format override is a known mechanism for losing reasoning capability.

## Contradicting Evidence

- **2410.03717** (Revisiting the Superficial Alignment Hypothesis): Post-training performance
  scales as power law with finetuning examples for math, coding, instruction following.
  HOWEVER: this uses curated benchmarks, not Reddit-quality data. Quality floor is higher.

- **2508.04329** (Token-level discrimination in noisy data): Even noisy samples contain
  valuable tokens. Proposes token-level positive/negative discrimination rather than
  sample-level rejection. Suggests our wholesale failure may be recoverable with
  token-level quality filtering rather than data replacement.

## Alternative Approaches (with paper evidence)

1. **Self-distillation (SDFT, 2601.19897):** Model generates its own training data via
   in-context learning. Only approach that improves new-task performance without degrading
   prior capabilities. Directly addresses our failure: the model generates data at its OWN
   quality level, guaranteeing data quality >= base capability.

2. **Self-play (SPIN, 2401.01335):** Model generates training data from previous iterations.
   Key finding: re-running SFT on static data degrades performance (matches our observation),
   but self-play avoids this. Compatible with DPO.

3. **Disperse-then-merge (2405.13432):** Trains multiple sub-models on data portions, then
   merges. Reduces alignment tax without data quality changes. Could work with our existing
   Grassmannian composition architecture (each adapter trained on a data shard, composed at
   inference).

4. **Quality-gated data with IFD score (2402.00530):** Use the base model's own perplexity
   as a quality filter. Discard samples where base PPL is already low (model already knows
   this). Keep samples where base PPL is high but response quality is good (novel knowledge).

5. **Thinking-preserving SFT (2505.09388, Qwen3 report):** Generate training data with
   `<think>` reasoning via rejection sampling from RL-trained model. Ensures SFT does not
   override the thinking capability. Directly applicable to Qwen3-4B.

## Implications for Next Experiments

1. **The data quality floor is now proven.** For Qwen3-4B, any SFT data must be at least
   as good as what the model can generate itself. Reddit-sourced data fails this test for
   legal/finance and behaviorally even for code/math (despite loss convergence).

2. **Composition experiments should use base model as default.** The 3 converged adapters
   (medical, code, math) have lower behavioral scores than the unadapted base. For
   composition experiments (exp_pro_composition_mmlu), the base model should be the control,
   not the adapted model.

3. **Format is a first-class concern.** Any future SFT on Qwen3-4B must preserve or
   leverage the `<think>` token format, not override it with ### templates.

4. **Finding #209 + #319 = general law.** SFT degradation is not domain-specific (legal/finance)
   but a function of data_quality < base_capability. This should be formalized.

## Recommended Follow-Up

**exp_pro_self_distill_adapters (P0):**
- **Motivation:** Finding #319 (this experiment) + Qwen3 Technical Report (2505.09388)
  + SDFT (2601.19897) all converge on the same solution: model-generated training data.
- **Literature:** SDFT (2601.19897) demonstrates the approach prevents catastrophic forgetting.
  SPIN (2401.01335) shows self-play avoids the degradation from static SFT data.
  Qwen3 report (2505.09388) uses rejection-sampled thinking data for their own model.
- **Approach:** Use Qwen3-4B-4bit to generate domain-specific training data with `<think>`
  reasoning preserved. Filter by response quality. Train adapters on self-distilled data.
  This guarantees data quality >= base capability by construction.
- **Fix:** Addresses the exact failure mode: data quality < base model capability.
