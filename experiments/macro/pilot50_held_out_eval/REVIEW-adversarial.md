# Peer Review: pilot50_held_out_eval

## NotebookLM Findings

Skipped -- this is a macro-scale evaluation experiment with straightforward methodology. The review can be conducted entirely from the code and documents.

## Mathematical Soundness

### MATH.md

The formalization is correct and appropriately modest:

1. **Log-probability scoring** (argmax over log P(c|p; theta) for c in {A,B,C,D}) is standard MMLU methodology. No errors.

2. **Kill criteria formalization** is clean:
   - K1 (win rate > 80%) and K2 (avg delta > 2pp) are reasonable thresholds.
   - K3 (HumanEval python > base) is appropriately binary.

3. **Statistical considerations are honest.** The paper correctly notes that per-subset comparisons at n=100 have SE ~0.045, making 5pp deltas barely 1 SE. The aggregate across ~5000+ questions provides the real statistical power. This is the right way to think about it.

4. **One concern with K1 formulation.** MATH.md defines K1 as proportion of adapters with delta_i > 0, but the code (line 351) computes it as adapters whose *average* delta across mapped subsets is > 0. These are consistent in intent but the MATH.md notation delta_i is ambiguous -- it should explicitly state this is the adapter-level aggregate, not per-subset.

5. **Expected magnitudes are realistic.** The +1-5pp expectation for distilled adapters on MMLU is well-calibrated. The adapters were trained on synthetic instruction-response pairs, not MMLU-format QA. Expecting modest transfer is appropriate.

### Cost estimate

The ~2.5 hour / ~$0.10 estimate appears reasonable for sequential evaluation of 23 adapters on 2-4 MMLU subsets each plus HumanEval.

## Novelty Assessment

This is an evaluation experiment, not a novel method. No novelty claim is made. The experimental design follows standard practices (MMLU log-prob, HumanEval execution-based). The reference to Raschka (2025) for MMLU eval methodology is appropriate.

The domain-to-MMLU mapping is the main original contribution. It is a judgment call, not a derivable result.

## Experimental Design

### Strengths

1. **Addresses the right problem.** The contaminated eval (last 100 of 1000 training examples) is a genuine concern. Using completely independent benchmarks (MMLU, HumanEval) is the correct fix.

2. **Clean lineage.** This experiment has a single purpose: upgrade exp_distillation_pilot_50 from "supported" to "proven" (or kill it). The scope is tight.

3. **Code is well-structured.** Base caching, intermediate saves, per-adapter memory cleanup, clear logging. Production-quality eval scripts.

4. **Honest about limitations.** 27/50 adapters without MMLU mapping, Python-only for HumanEval, single-adapter evaluation only. All acknowledged.

### Issues Found (numbered for action)

**1. MMLU tokenization of answer choices -- potential silent failure.**

In `eval_mmlu.py` lines 89-97, the code tokenizes "A", "B", "C", "D" and also " A", " B", " C", " D", then takes the max log-prob between them. This is reasonable but has a subtle issue: for Qwen2.5 tokenizer, the single letter and space-prefixed letter may map to the same token ID (since many tokenizers merge single characters with preceding spaces). The code should verify that `choice_ids[letter] != choice_ids[f" {letter}"]` or handle the case where they are identical. Currently if they are the same, `max(log_probs[tid].item(), log_probs[tid_space].item())` just returns the same value twice, which is harmless. **Non-blocking** -- this is a no-op bug, not a correctness issue.

**2. Adapter loading/unloading without merge.**

In `eval_mmlu.py` line 283, `PeftModel.from_pretrained` loads the adapter on top of `base_model`. On line 327, `del adapted_model` deletes the PeftModel wrapper. However, the `base_model` object is reused across adapters. The assumption is that `PeftModel.from_pretrained` does not modify `base_model` in place. This is correct for PEFT's default behavior (adapters are stored separately), but if any adapter fails to load and leaves the base in a modified state, subsequent adapters would be contaminated. **Low risk** -- PEFT handles this correctly in normal operation.

**3. K1 kill criterion in HYPOTHESES.yml says "composed 50-expert model" but the experiment evaluates individual adapters.**

The HYPOTHESES.yml kill criterion reads: "composed 50-expert model does not beat base on >80% of expert domains." But this experiment tests each adapter *individually*, not a composed multi-expert model. The PAPER.md correctly describes this as single-adapter evaluation and notes composition quality is a separate hypothesis. **The HYPOTHESES.yml wording should be corrected** to match the actual experimental design: "individual adapters do not beat base on >80% of evaluated domains."

**4. Domain-to-MMLU mapping overlaps create non-independence.**

Several adapters share MMLU subsets:
- `math` and `abstract-math` both map to `abstract_algebra` and `college_mathematics`
- `python`, `cpp`, `java`, `javascript` all map to `high_school_computer_science`
- `biology` and `ecology` both map to `high_school_biology`

This means K1 (adapter win rate) is not computed over independent measurements. If the base model is particularly weak on `high_school_computer_science`, four adapters benefit simultaneously. The 80% threshold was presumably set assuming roughly independent adapter results.

**Mitigation:** The paper should note this non-independence. Consider reporting both the raw win rate and a "unique subset" win rate where shared subsets are counted once for the adapter with the strongest mapping.

**5. HumanEval execution safety.**

`eval_humaneval.py` line 108 runs generated code via `subprocess.run(["python3", tmp_path])` with a 10-second timeout but no sandboxing (no container, no seccomp, no resource limits beyond timeout). On a RunPod instance this is acceptable (disposable environment), but the code should document this assumption. A malicious or confused completion could write to disk, make network calls, or consume memory. **Non-blocking** for a research eval on a disposable VM.

**6. HumanEval stop sequences may truncate valid completions.**

`eval_humaneval.py` line 90 uses `"\nprint("` as a stop sequence. Some HumanEval solutions legitimately contain `print()` calls within the function body (for debugging or as part of the solution). This could truncate valid completions. More importantly, `"\ndef "` will stop at any new function definition, which is correct for most cases but would break solutions that define helper functions.

**Mitigation:** Consider using indentation-based stopping (stop when indentation returns to column 0) or matching HumanEval's canonical stop sequences from the original paper. This affects absolute pass@1 numbers but affects base and adapter equally, so the *delta* is still valid for the kill criterion.

**7. No few-shot prompting for MMLU.**

The standard MMLU evaluation protocol uses 5-shot prompting (Hendrycks et al., 2021). This experiment uses 0-shot (the `format_mmlu_prompt` function provides only the question and choices). Zero-shot will produce lower absolute scores but the delta between base and adapter should be approximately preserved.

However, there is a risk: if the adapters were trained on instruction-following data that implicitly teaches the model to answer multiple-choice questions directly, the adapter may show a larger improvement in 0-shot than in 5-shot (where the base model already "gets it" from the examples). This could inflate the measured delta.

**Recommendation:** Run at least one adapter in both 0-shot and 5-shot to check whether the delta is robust to prompting format. If deltas differ substantially, the 0-shot results may overestimate generalization.

**8. No confidence intervals or significance testing on kill criteria.**

The kill criteria are point estimates (win rate, average delta). Given the statistical noise acknowledged in MATH.md (SE ~0.045 per subset), a borderline result (e.g., 79% win rate, 1.8pp average delta) would be uninterpretable without confidence intervals. The experiment should report bootstrap CIs on K1 and K2.

## Hypothesis Graph Consistency

The experiment matches `exp_pilot50_held_out_eval` in HYPOTHESES.yml. The kill criteria align (with the caveat in issue #3 about "composed" vs "individual" wording). The dependency on `exp_distillation_pilot_50` (status: supported) is correct -- this experiment exists precisely to resolve that supported status.

The experiment does not block any other nodes, which is accurate -- composition quality is a separate hypothesis.

## Macro-Scale Risks (advisory)

1. **Qwen2.5-7B NF4 quantization** may affect MMLU absolute scores differently than literature reports (which use FP16 or BF16). The delta comparison is still fair since both base and adapter run in NF4.

2. **MMLU dataset version.** The code uses `cais/mmlu` from HuggingFace. There are multiple MMLU variants (original, MMLU-Pro, MMLU-Redux). Ensure this is the standard Hendrycks et al. test split, not a contaminated variant.

3. **Adapter path assumptions.** The scripts assume adapters are at `/workspace/llm/adapters/{domain_name}`. If the pilot 50 used different naming conventions, adapters will silently skip. The script handles this gracefully (prints SKIP), but verify naming matches before running.

## Verdict

**PROCEED** with 3 required fixes and 2 recommended improvements.

### Required Fixes

1. **Correct HYPOTHESES.yml wording** (issue #3): Change "composed 50-expert model does not beat base on >80% of expert domains" to "individual adapters do not beat base on >80% of evaluated domains on held-out benchmarks." The experiment tests individual adapters, not composition.

2. **Document the non-independence of adapter results** (issue #4): Add a note in PAPER.md's limitations that shared MMLU subsets across adapters (abstract_algebra shared by math and abstract-math, CS subsets shared by 4 programming adapters, biology shared by biology and ecology) inflate the effective sample size for K1. Report both raw and deduplicated win rates.

3. **Add bootstrap confidence intervals** (issue #8): After computing K1 and K2, bootstrap resample adapters (with replacement, 1000 iterations) to produce 95% CIs. If the CI for K1 straddles 80% or K2 straddles 2pp, the result is inconclusive rather than pass/fail.

### Recommended Improvements (non-blocking)

4. **Validate 0-shot vs 5-shot delta** (issue #7): Run at least 2-3 adapters in 5-shot to confirm the delta direction is preserved. If 0-shot inflates deltas, note this caveat.

5. **Review HumanEval stop sequences** (issue #6): Consider adding indentation-based stopping. The current stop sequences are aggressive and may undercount pass@1 for both base and adapter.

### Justification

The experimental design is sound in principle. It tests the right thing (held-out generalization of distilled knowledge), uses standard benchmarks, has clear kill criteria, and is honest about limitations. The code is well-written and handles edge cases (caching, memory management, error handling). The three required fixes are straightforward additions to documentation and post-hoc analysis -- none require re-running the experiment. The core question (do distilled LoRA experts generalize beyond their training distribution?) will be answered by this methodology.
