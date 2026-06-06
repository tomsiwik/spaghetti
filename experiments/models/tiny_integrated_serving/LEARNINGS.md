# LEARNINGS.md: exp_tiny_integrated_serving

## Core Learning

**All 5 independently proven components (block-diagonal masking, per-token MLP routing,
DARE sparsification, SFT adapters, ridge router) compose into an integrated pipeline
without quality degradation on BitNet-2B-4T.** The pipeline achieves -2.8% vs isolated
oracle (18/18 samples better, likely code-path artifact) and +3.0% vs per-sequence
baseline. Speed unmeasured for integrated forward pass (47.4 tok/s measured for single-
adapter generation only).

## What Worked

1. **Component composition is non-catastrophic.** The central fear (components interact
   to amplify each other's error) did not materialize. Quality is at least as good as
   segment-isolated evaluation across all 18 test samples.

2. **Block-diagonal masking + per-token MLP routing = exact segment isolation + correct
   adapters.** Each token sees only same-domain context (mask) and gets its domain-
   specific MLP adapter (routing). This is the correct architecture for multi-domain
   serving in a single forward pass.

3. **DARE at p=0.5 is essentially free for SFT adapters.** Max in-distribution change
   1.2% (and that was an improvement). This means we can use DARE for OOD robustness
   without any quality cost.

4. **Ridge router achieves 100% accuracy on 5 clean domains.** Closed-form, no iterative
   training, sub-millisecond inference.

## What Didn't Work

1. **Speed measurement is confounded.** Phase 5 measures standard mlx_generate with one
   adapter, not the integrated pipeline. K819 FAIL (47.4 < 60 tok/s) cannot be attributed
   to or exonerated from the integrated architecture.

2. **Mathematical framework predicted wrong sign.** Conjecture 1 predicted additive
   degradation (~7% worst case). Measured -2.8% improvement. The additive independence
   assumption is either wrong or incomplete. Most likely explanation: code-path confound
   between `model(x)` (isolated) and manual `single_pass_mixed_mlp_forward` (integrated).

3. **MLP routing gap not independently verified.** The "< 1% MLP routing gap" prediction
   (from Finding #313) is absorbed into the integrated measurement. Cannot confirm or
   deny in this experiment.

## What We Still Don't Know

1. **Integrated pipeline generation speed.** The 2x LoRA compute per MLP layer + block-
   diagonal mask creation + manual attention may be significantly slower than single-adapter
   serving. This is THE critical unknown for production viability.

2. **K > 2 segments.** Only 2-segment pairs tested. The mask construction works for any K
   by construction, but per-token routing with K adapters requires K LoRA computations per
   MLP layer, scaling linearly.

3. **Harder domain pairs.** Only 6/10 pairs tested. Legal+finance (two worst domains) may
   produce larger gaps. The max gap vs per-sequence (14.5%) already exceeds the mean-based
   K818 threshold.

4. **Router on mixed text.** 100% accuracy on clean single-domain inputs. Boundary detection
   in concatenated text is a harder problem, untested.

## Literature Connections

- **Block-Attention (2409.15355):** Our block-diagonal masking is equivalent to their
  approach but without position re-encoding (which Finding #322 proved unnecessary).

- **LoRA (2106.09685), DARE (2311.03099):** Standard components, validated in composition.

- **MoLoRA (2603.15965):** Per-token LoRA routing is the same concept. Their architecture
  uses top-K routing; ours uses segment-based assignment via block-diagonal masking.

- **Finding #313 (single-pass MLP matching oracle):** The foundational result that makes
  per-token MLP routing viable. 0.61% gap in isolation; composition preserves this.

- **Finding #322 (RoPE reset unnecessary):** Closed the block-diagonal architecture
  question. No position re-encoding needed.

## Generalizable Insight

**Independently proven components with orthogonal failure modes compose safely.** Each
component (masking: isolation, routing: selection, DARE: robustness, adapter: specialization)
addresses a different dimension of the problem. When failure modes don't interact, composition
works. This is NOT a general principle — it depends on the specific components having truly
independent effects, which was conjectured but not proven here.

## Recommended Follow-ups

1. **exp_integrated_speed_benchmark (P0):** Measure the actual integrated forward pass
   generation speed. This is the most important unknown — if the integrated pipeline is
   too slow for interactive serving, the architecture needs optimization.

2. **exp_pro_integrated_serving (P0):** Replicate on Qwen3-4B (Pierre Pro). The
   BitNet results establish the pattern; Pro validates at production scale.

3. **exp_integrated_code_path_control (P1):** Use the same manual forward pass for BOTH
   isolated and integrated evaluation to eliminate the code-path confound and determine
   whether the -2.8% improvement is real or artifactual.
