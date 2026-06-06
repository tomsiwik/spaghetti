# Peer Review: Falcon-E-3B LoRA Composition

## NotebookLM Findings

Skipped. The experiment is a direct successor to the competitive_benchmark kill, with identical methodology. The artifacts are self-contained and the key questions are answerable from code and results inspection alone.

## Mathematical Soundness

### Pre-merge composition formula: CORRECT

The composition math is identical to the competitive_benchmark and correctly implemented. `W_new = W_base + sum_i (1/N) * scale * B_i^T @ A_i^T` where scale=20.0 and N=5, giving effective per-adapter contribution of 4.0x the learned delta. The code in `premerge_adapters()` (lines 373-444) matches this exactly. The inline derivation comments showing that LoRALinear forward computes `y = x @ W^T + scale * (x @ A) @ B` and therefore `dW = scale * B^T @ A^T` is correct and demonstrates genuine understanding of the transpose conventions.

### LoRA parameter count: CORRECT

q_proj: 2048x16 + 16x2048 = 65,536. v_proj: 2048x16 + 16x256 = 36,864. o_proj: 2048x16 + 16x2048 = 65,536. Per layer: 167,936. 32 layers: 5,373,952. The 0.18% figure checks out against 3B total params.

### GQA description: CORRECT

16 query heads, 2 KV heads, ratio 8:1, d_k=128. KV cache is O(n * 2 * 128) = O(n * 256). This is accurately described.

### Memory analysis: HONEST but the 6GB kill threshold is questionable

MATH.md correctly predicts the bf16 unpack will expand 999MB to ~6.1GB, and that peak will exceed 6GB. The actual measurements confirm: 6.74 GB base, 8.80 GB composed. The paper is honest that K3 will likely fail before running the experiment.

**Issue with the threshold itself:** The 6GB K3 threshold appears to be set knowing it would fail. On an M5 Pro with 48GB unified memory, 8.80 GB is 18% utilization -- completely viable for production use. The threshold seems calibrated to force a kill on any bf16-unpacked ternary model over ~2.5B params, which makes it a test of "does MLX have native ternary kernels" rather than "is this architecture viable." The paper acknowledges this (Section 3.1: "The K3 failure is an engineering limitation, not architectural"), which is fair, but it means K3 is not testing a research hypothesis -- it is testing an engineering prerequisite that was already known to fail.

### Ternary unpacking: CORRECT

The bit-extraction logic (lines 138-148) correctly unpacks 4 ternary values per uint8 byte using shifts and masks. The `[:out_features]` slice handles the padding from ceil division.

## Novelty Assessment

### This is not a novel mechanism experiment

This is a benchmark replication with a different base model. The experimental contribution is: "Falcon-E-3B is better than BitNet-2B as a ternary base." No new composition mechanism, no new routing strategy, no new training procedure. The LoRA pipeline, pre-merge composition, and evaluation methodology are all directly reused from competitive_benchmark.

### The interesting finding is accidental

The most valuable result -- that Falcon-E-3B BASE beats Qwen-3B 5/6 WITHOUT adapters -- is not actually testing the stated hypothesis (composition). The paper correctly identifies this reframing. However, this raises a question: if the base model already wins, what does composition add? The experiment answers: nothing positive. Uniform composition only hurts. This is a useful negative result but not novel; the competitive_benchmark already showed uniform composition hurts math/legal.

## Experimental Design

### 1. Are the benchmark numbers real or synthetic?

**Verdict: REAL, with high confidence.**

Evidence of genuine execution:
- `results.json` contains precise timing data: falcon_base 350.5s, falcon_composed 1533.2s, qwen_3b 112.4s. The 4.4x slowdown for composed Falcon (which must unpack ternary + merge adapters) vs Qwen is physically plausible.
- Peak memory values are precise to sub-MB (6.735532151 GB, 8.8030298 GB, 2.431559756 GB), consistent with MLX memory API output, not round fabricated numbers.
- GSM8K timing: Falcon-base 269.5s for 50 problems (~5.4s/problem), Falcon-composed 1165.9s (~23.3s/problem), Qwen 99.5s (~2.0s/problem). The composed model being 4.3x slower than base is suspicious -- pre-merge should have zero inference overhead. This could indicate the merge itself is being re-run per evaluation, OR the larger memory footprint causes more cache pressure. Either way, this timing anomaly actually increases confidence the experiment ran (a faker would not introduce this inconsistency).
- Training data comes from `real_data_25_domain_adapters/data/` (an existing experiment's artifacts).
- The code uses `load_dataset("openai/gsm8k")` and `load_dataset("cais/mmlu")` -- real HuggingFace datasets.
- Total time 2000.5s (~33 min) is plausible for 3 model evaluations on M5 Pro.

**One concern:** `training.note: "skipped - adapters pre-trained"` means the training phase was skipped in the recorded run. The training stats in PAPER.md and LEARNINGS.md come from a prior run. This is not fabrication, but means we cannot verify training convergence from this results.json alone. The PPL numbers in PAPER.md (medical 4.26->2.19, etc.) are not in results.json.

### 2. Is the Qwen comparison fair?

**Partially unfair, in Qwen's disfavor, and the paper acknowledges this.**

The paper states: "Qwen published GSM8K ~65-70%, we measured 36%." This 30+pp gap between published and measured performance is alarming. Possible causes:
- 4-bit quantization degrading Qwen more than expected
- Wrong prompt template (the ChatML template looks correct for Qwen2.5-Instruct)
- Answer extraction failures on Qwen's output format
- The 50-question sample happening to be harder than the test distribution

The Falcon prompt template uses a generic `### Instruction / ### Response` format rather than Falcon's actual chat template. This could also underperform. But since Falcon BASE scores 44% on GSM8K (reasonable for a 3B model), the prompt format seems adequate.

**Critical point:** Both models are likely underperforming their true capability due to the evaluation harness. The comparison is internally consistent (same eval code, same questions), but the absolute numbers should not be compared to published benchmarks. The paper correctly notes this in Limitation 6.

**The 4-bit vs ternary comparison is inherently apples-to-oranges.** 4-bit quantization preserves more information per parameter than 1.58-bit ternary. The fact that Falcon-E-3B (1.58-bit) beats Qwen (4-bit) on 5/6 benchmarks is actually remarkable and suggests Falcon's training procedure is very effective. But this is a comparison of two specific models, not of quantization methods. Qwen2.5-3B was not designed for 4-bit; it was quantized post-hoc. Falcon-E-3B was trained at 1.58-bit from scratch. These are fundamentally different approaches.

### 3. Is "uniform composition degrades all 6" properly measured?

**Yes, but the signal is mostly within noise for individual benchmarks.**

With 20 MMLU questions per domain, each question is worth 5pp. The 95% CI at p=0.5 is +/-22pp. So:
- Medical: 55% -> 30% (-25pp): likely real (>1 CI width)
- Finance: 60% -> 45% (-15pp): borderline (within 1 CI)
- Code: 60% -> 50% (-10pp): within noise (2 questions)
- Legal: 40% -> 35% (-5pp): within noise (1 question)
- Math: 55% -> 55% (0pp): no change

GSM8K with n=50 is better: CI +/-14pp at p=0.5.
- GSM8K: 44% -> 36% (-8pp): borderline

**The pattern of ALL 6 degrading (or staying flat) is more informative than any individual benchmark.** Under the null hypothesis (composition has no effect), the probability of 5/6 degrading and 1/6 flat is very low (~1/64 for direction, ignoring magnitude). So the aggregate conclusion "uniform composition hurts" is supported, even though individual benchmarks are noisy. The paper should have made this statistical argument explicitly.

**However, the claim "degrades all 6" is overstated.** Math is 55% -> 55% (exactly 0 change). That is "degrades 5/6, neutral on 1." The paper says this in one place and "all 6" in another. This inconsistency should be fixed.

### 4. Does MATH.md show genuine understanding?

**Yes -- above average for this project's experiments.**

Specific evidence of understanding rather than buzzword recitation:
- Correct derivation of LoRA forward pass transpose conventions with self-correction in the premerge function comments
- Accurate GQA tensor shapes and KV reduction factor
- Honest prediction that K3 would fail before running the experiment
- Correct identification of why wider MLP (13312 vs 6912) makes attention-only LoRA potentially insufficient, citing DeepSeek-V3 MLP routing as evidence
- The "Why LoRA works on ternary" section explains the differentiability through bf16 unpack, not just "it works"

**One gap:** Section 2.1 claims "ternary weight matrices have a constrained gradient landscape" and "adapter updates exist in a lower-dimensional effective subspace." This is asserted without derivation. The cosine similarity numbers (0.00125 ternary vs 0.142 fp16) are cited as evidence, but correlation is not mechanism. This remains an empirical observation, not a mathematical understanding.

### 5. K2 criterion evaluation

**K2 is evaluated on the COMPOSED model, which is the correct reading of the criterion.**

The analysis code (line 823) compares `falcon_composed` vs `qwen_3b`, not `falcon_base` vs `qwen_3b`. The composed model loses 3/6 (medical, legal, finance), which is <=4, so K2 PASS.

**However, the K2 evaluation is generous to Falcon.** Looking at the numbers:
- GSM8K: 36% vs 36% -- this is a TIE, counted as "not losing"
- Code: 50% vs 40% -- Falcon wins
- Math: 55% vs 45% -- Falcon wins
- Medical: 30% vs 70% -- Falcon loses badly
- Legal: 35% vs 40% -- Falcon loses
- Finance: 45% vs 55% -- Falcon loses

The code counts ties as "not worse" (line 823: `falcon.get(b, 0) < qwen.get(b, 0)`, strict less-than). This means GSM8K's exact tie at 36% counts as a pass. With n=50, a tie at 36% is within noise of either winning or losing. If GSM8K had gone the other way (17 vs 18 correct instead of 18 vs 18), K2 would show 4/6 losses -- exactly at the threshold. The pass is fragile.

The paper's framing of "K2 PASS" for the composed model, while technically correct by the criterion definition, obscures that the composed model barely passes and is significantly worse than the BASE model on the same comparison. The paper is honest about this in "The Honest Assessment" section, so this is a presentation issue, not a data integrity issue.

### 6. Memory K3 threshold justification

**The 6GB threshold is poorly justified and effectively pre-determined to fail.**

The threshold is stated as K3 (#534) but its origin is not documented in the experiment files. Given that:
- Falcon-E-3B unpacked = ~6.1GB (known before the experiment)
- Any KV cache or composition overhead pushes past 6GB
- The M5 Pro has 48GB (8.8GB = 18% utilization)

The 6GB threshold appears designed around a "smaller than Qwen-3B-4bit * 2.5x" mental model, but this is never stated. A more meaningful K3 would be either:
- "Fits in 16GB" (consumer MacBook Air) -- would PASS at 8.8GB
- "Smaller than Qwen-3B-4bit at 2.43GB" -- would fail but for a defensible reason
- "Fits in M5 Pro working memory alongside other processes" (~32GB) -- would PASS trivially

The current 6GB threshold kills an otherwise functional system for a somewhat arbitrary reason. The paper's own analysis identifies this as an engineering limitation (bf16 unpack) not an architectural one, which is the right framing.

## Macro-Scale Risks (advisory)

1. **Native ternary inference is the real blocker.** Every ternary experiment will hit this wall until MLX implements BitLinear kernels or the project builds custom Metal kernels. This is not a macro-scale risk -- it is a current-scale engineering debt.

2. **The "Falcon BASE beats Qwen" result needs validation at larger sample sizes.** 20 questions per MMLU domain is far too few for the +20pp Code claim. A macro validation should use 200+ questions per domain.

3. **Composition strategy is the open research question.** This experiment confirms (for the second time) that uniform composition is harmful on instruction-tuned models. The project has proven routing mechanisms (Gumbel-sigmoid, per-adapter heads) that should be tested on Falcon-E-3B. Until routed composition beats the base model, composition adds no value.

4. **The Qwen comparison may flip at proper scale.** If Qwen's GSM8K is actually 65-70% (published) and the 36% measured here is an artifact of the evaluation harness, then Falcon-E-3B at 44% GSM8K would lose that benchmark too. The head-to-head depends heavily on the evaluation methodology being equally fair (or unfair) to both models.

## Verdict

**PROCEED** (with caveats)

### Justification

The experiment ran on real hardware, produced genuine results, and the analysis is honest and self-aware. The code is correct. The most important finding -- Falcon-E-3B BASE beats Qwen 5/6 -- is directionally significant even with small samples. The K3 FAIL on memory is correctly identified as an engineering limitation.

### Caveats (not blocking, but should be addressed)

1. **Fix the "degrades all 6" inconsistency.** Math is 55%->55%, which is flat, not degraded. The paper says both "worse than Falcon base on ALL 6" and later shows the same-score result. Pick one framing and be consistent.

2. **Acknowledge K2 fragility.** The composed model passes K2 by one tie on GSM8K (18 vs 18 correct out of 50). State this explicitly so downstream experiments do not treat K2 PASS as a robust result.

3. **The training data provenance needs to be clearer.** The adapters were pre-trained and the training phase was skipped in the recorded run. The PPL numbers in PAPER.md are from a prior execution. Note this in the results section so readers do not expect to find training data in results.json.

4. **The K3 threshold should be revisited at the project level.** A 6GB limit that kills any bf16-unpacked 3B ternary model is testing "does native ternary inference exist in MLX" rather than "is the architecture viable." This is a project-level methodology issue, not specific to this experiment.

5. **The Qwen underperformance (36% vs published 65-70% GSM8K) should be investigated.** If the evaluation harness systematically underestimates both models, the relative comparison is still valid. If it disproportionately penalizes one model, the comparison is confounded. A quick sanity check: run Qwen with its tokenizer's built-in chat template via `tokenizer.apply_chat_template()` instead of manual ChatML formatting.
