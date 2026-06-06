# Peer Review: Hidden-State MLP Probe for Per-Token Adapter Routing (Re-review)

## Experiment Type
Guided exploration (Type 2)

**Framework cited:** Hidden-state routing for LoRA adapter selection (X-LoRA arXiv 2402.07148, TT-LoRA MoE arXiv 2504.21190, Finding #276: ridge regression on mean-pooled hidden states at 96% accuracy). Theoretical grounding: Cover's theorem (1965), UAT (Hornik 1991).

**Unknown identified:** Per-token hidden-state classification accuracy with a post-hoc MLP probe on frozen ternary adapters (vs. the proven mean-pooled/sequence-level case).

**Unknown narrowed:** Yes. The experiment conclusively shows that (a) per-token hidden states are linearly separable at 98.3%, (b) MLP adds only 0.2pp over ridge, and (c) the SNR degradation predicted by isotropic-noise theory does not materialize because domain signal is structured per-token. Valid Type 2 execution.

## Hack Detector
- Fix count: 1 (MLP probe on hidden states). No stacking. CLEAN.
- Is MATH.md a proof or a description? MIXED -- Claim 1 is correctly labeled as a framework-grounded prediction (not a proof). Theorems 2 and 3 are formal proofs with QED. The document is honest about the distinction.
- Metric used as evidence: Token-level classification accuracy (valid proxy for routing correctness) and PPL gap (right metric but tautologically tested -- see below).
- Kill criteria source: K784 from heuristic SNR analysis (acknowledged as such). K785 from Theorem 3 (valid). K786 from Theorem 2 (valid).

## Revision Verification: All 6 Fixes

### Fix 1: K785 tautological -- APPLIED CORRECTLY
PAPER.md lines 130-137 explicitly labels K785 as tautological: "routing accuracy is 100% (200/200 segments correct), so the probe selects exactly the same adapter as the oracle for every segment. Matching oracle PPL is guaranteed by construction when routing is perfect." The kill criteria table (line 173) marks it as "PASS (tautological)". Results.json K785 note echoes this. No attempt to hide the tautology.

### Fix 2: Theorem 1 renamed to Claim 1 -- APPLIED CORRECTLY
MATH.md line 118 reads "Claim 1 (Probe Routing Accuracy Prediction)." Lines 138-144 add an explicit note: "This is a prediction grounded in the proven framework (Cover's theorem + Universal Approximation Theorem), not a formal proof. The UAT guarantees existence of a suitable weight configuration but provides no constructive bound on the required width for a given accuracy target." No QED on Claim 1. Only Theorems 2 and 3 retain QED.

### Fix 3: Failed SNR prediction acknowledged -- APPLIED CORRECTLY
MATH.md lines 208-223 contain a dedicated "Failed Prediction: SNR Degradation" section. It states the prediction (16x degradation), the actual result (0.2pp), and a correct diagnosis: "The isotropic noise assumption (Var = sigma^2 I) is wrong. Domain signal is structured per-token, not just per-sequence noise reduced by averaging." PAPER.md prediction table (line 27) also marks this as "Unpredicted" with the same explanation. Intellectually honest.

### Fix 4: Model config mismatch documented -- APPLIED CORRECTLY
PAPER.md Limitation 7 (lines 226-232) states: "Phase 1 extracts hidden states from the raw BitLinear model. Phase 3 evaluates probe routing on the unpacked-bf16 + zeroed-LoRA model configuration. These are different model configurations." Notes that 100% routing accuracy suggests it is benign at this scale but flags it as a potential issue at larger scale.

Verified in code: Phase 1 (line 247) calls `load_model_and_tokenizer(MODEL_ID)` and runs forward on the raw BitLinear model. Phase 3 (lines 694-696) calls `load_model_and_tokenizer(MODEL_ID)`, then `replace_bitlinear_with_linear(model)`, then `apply_lora_to_model(model)`. The `get_base_hidden` function (line 780) zeros the adapter but still uses the unpacked-bf16 model, not the raw BitLinear. So hidden states used for probe classification in Phase 3 come from a different model configuration than those used for probe training in Phase 1/2. The mismatch is real and correctly documented.

### Fix 5: Oracle-worse-than-base analyzed -- APPLIED CORRECTLY
PAPER.md lines 148-165 provide three substantive explanations:
(a) LORA_SCALE=20.0 may be too aggressive for 128-token segments lacking full context.
(b) Segment isolation forces adapter cold-start without prior context.
(c) Adapters trained on complete sequences (instruction+response) face distribution shift when applied to isolated 128-token segments.
Concludes: "adapters trained on full sequences may hurt when applied to isolated short segments. This is a real limitation of segment-isolated routing at this adapter configuration, not an anomaly to be dismissed." This is substantive and honest. No longer dismissive.

### Fix 6: Rahimi & Recht citation corrected -- APPLIED CORRECTLY
MATH.md lines 104-114 now state: "The Rahimi & Recht bound is specifically about kernel approximation error, not classification accuracy directly. The connection to classification is indirect... The choice of w=128 is a heuristic informed by the scale of the kernel approximation bound... not a rigorous derivation of the minimum width for 85% classification accuracy." The original incorrect claim of a direct classification accuracy bound has been removed.

## Self-Test Audit

1. **One-sentence impossibility property:** "Hidden states in d=2560 are linearly separable with 96% accuracy at sequence level (Finding #276); Cover's theorem guarantees near-certain separability at d/N >> 1, and a single hidden layer MLP can capture any nonlinear residual." This is still a compound statement (Cover + MLP capacity), but it articulates a coherent argument chain rather than a list of disconnected properties. The core property is "d >> N guarantees separability." MINOR -- acceptable for guided exploration.

2. **Cited theorems:** Cover (1965), UAT (Cybenko 1989, Hornik 1991), Concentration of Measure (Vershynin 2018 Thm 3.1.1), Rahimi & Recht (2007). All real. Conditions correctly noted -- Cover applies to random labelings (structured labels are easier), UAT is existence-only, Rahimi & Recht is about kernels not classification. The self-test explicitly notes only Theorems 2 and 3 are formal proofs. PASS.

3. **Predicted numbers:** Token-level accuracy >= 85%, latency < 0.01ms, PPL within 5% if accuracy > 95%, PPL within 15% if accuracy = 85%. Specific and falsifiable. PASS.

4. **Falsification condition:** "If token-level hidden states are NOT domain-informative (accuracy ~ 20% = random), the assumption that hidden states carry per-token domain signal is wrong." Targets the core assumption. PASS.

5. **Hyperparameter count:** 2 (hidden width, learning rate). Width heuristically derived, learning rate is Adam default. PASS.

6. **Hack check:** Single mechanism extending proven framework. No stacking. PASS.

## Mathematical Soundness

### Claim 1 (Probe Routing Accuracy Prediction) -- HONESTLY WEAK
Now correctly labeled as a prediction, not a theorem. The argument chain is: (1) domain centroids are separated empirically (Finding #276), (2) Cover's theorem says d=2560 >> N makes linear separability near-certain, (3) UAT says an MLP can capture any nonlinear residual. The 85% threshold comes from the SNR heuristic, which turned out to be wrong (predicted 16x degradation, measured 0.2pp). This is all disclosed. No formal bound, but that is appropriate for the "unknown" in a Type 2 experiment -- the experiment itself is meant to discover the actual number.

### Theorem 2 (Probe Inference Cost) -- SOUND
FLOPs = 2w(d+K) = 2*128*(2560+5) = 656,640. Verified arithmetic. Latency prediction (0.066 microseconds) is a theoretical lower bound; measured 170 microseconds dominated by MLX dispatch overhead. Paper correctly distinguishes raw compute from measured latency. Batched measurement (1.4 microseconds per token) is closer to the theoretical prediction. PASS.

### Theorem 3 (PPL Bound Under Routing Errors) -- SOUND WITH CAVEAT

The derivation:
- NLL_probe = NLL_oracle + epsilon * log(R)
- PPL_probe = PPL_oracle * R^epsilon
- <= PPL_oracle * (1 + epsilon*(R-1)) for small epsilon

Verified: For R=2, epsilon=0.15: R^epsilon = 2^0.15 = 1.1096, bound gives 1.15. Correct direction and numerically reasonable.

The linearization uses R-1 >= log(R) for R >= 1, which is standard. The bound is conservative but not vacuous.

Caveat (from first review, still valid): The decomposition assumes misrouted tokens incur at most log(R) additional NLL per token independently. In autoregressive models, error could propagate across tokens within a segment. However, this is mitigated by segment isolation (errors do not propagate across segment boundaries), and the paper evaluates segments independently. Acceptable.

### SNR Analysis -- CORRECTLY MARKED AS FAILED
The sqrt(T) = 16x degradation prediction assumed isotropic token-level noise. The experiment falsified this assumption. MATH.md correctly diagnoses why: domain signal is per-token, not noise reduced by averaging. This is a legitimate scientific finding -- the experiment discovered that the isotropic noise model is wrong for these hidden states.

## Prediction vs Measurement

PAPER.md contains a prediction table (lines 24-31). Assessment:

| Prediction | Predicted | Measured | Match? | Assessment |
|-----------|-----------|----------|--------|------------|
| Token-level MLP >= 85% | >= 85% | 98.5% | YES | Threshold was conservatively low (from failed SNR model), but correctly derived given the (wrong) assumption. Large margin is a feature: the mechanism works better than the pessimistic bound predicted. |
| Token-level ridge (linear baseline) | Not formally predicted | 98.3% | N/A | Key unpredicted finding. Paper marks as "Unpredicted" -- honest. |
| Sequence-level ridge ~96% | ~96% (Finding #276) | 100% | YES | 4pp above; small test set (50 sequences). Consistent with prior. |
| PPL within 5% of oracle | <= 1.05 ratio | 1.00 ratio | YES | Correctly marked tautological. |
| Probe latency < 1ms | < 0.01ms raw compute | 0.170ms measured | YES | 2580x above raw prediction but 5.9x under budget. Dispatch overhead correctly explained. |
| Mixed routing accuracy | Not formally predicted | 100% | N/A | Unpredicted; Hoeffding's inequality over 128 tokens explains this post-hoc. |

The prediction table is honest about what was predicted, what was not, and where predictions failed. This meets the standard for Type 2 guided exploration.

## NotebookLM Findings

NotebookLM was not used for this re-review. Analysis is from manual deep review of the revised documents and code.

## Novelty Assessment

**Delta over existing work:** The primary novel finding is that per-token hidden states from a ternary base model are linearly separable at 98.3% across 5 domains, making the MLP probe (the experiment's original hypothesis) unnecessary. This is a clean negative result about MLP probes and a positive result about linear separability at token granularity.

This was not predictable from prior work:
- Finding #276 showed 96% at sequence level with mean-pooling. Token-level was unknown.
- The SNR analysis predicted substantial degradation. It did not materialize.
- The falsification of the isotropic noise model is genuinely informative.

The recommendation to use ridge regression for per-token routing (PAPER.md line 249) is the honest, practical conclusion.

**Prior art:**
- X-LoRA (arXiv 2402.07148): jointly-trained per-layer MLP gating. Different setting (joint training vs post-hoc).
- PHATGOOSE (arXiv 2402.05859): post-hoc per-module gating. Similar post-hoc approach but per-layer, not single-probe.
- Finding #276: Extended from sequence to token level. Genuine extension.

## Remaining Issues (Non-blocking)

### 1. K785 remains a tautological pass
The fix correctly labels it as tautological, but the kill criterion itself was never reformulated. K785 measures nothing because routing accuracy is 100%. This is acknowledged, not hidden. For a SUPPORTED finding, this is acceptable -- the real value is K784 (token-level accuracy) and K786 (latency), both of which are genuine passes.

### 2. Oracle-worse-than-base is analyzed but unresolved
The three proposed causes (LORA_SCALE, cold-start, distribution shift) are plausible but untested. The experiment does not determine which cause is operative. This is appropriately listed as a limitation, not claimed as resolved. For a Type 2 exploration, this is an open question for future work, not a deficiency.

### 3. Model configuration mismatch is documented but could bite at scale
Phase 1 extracts from raw BitLinear; Phase 3 probes on unpacked-bf16. At 100% accuracy this is benign, but the fact that it works may be coincidental rather than guaranteed. The code at line 780 (`get_base_hidden`) zeros the adapter weights but still uses the unpacked-bf16 nn.Linear layers, which differ numerically from BitLinear forward passes. Documented in Limitation 7. Acceptable for micro.

### 4. Finding reframing
The finding status SUPPORTED is appropriate for a Type 2 guided exploration that narrowed the unknown (per-token separability is high, MLP unnecessary). The finding should be reframed as a domain classification result, not a routing quality result, given that segment-isolated routing with these adapters hurts PPL. PAPER.md lines 138-143 already states this: "The experiment's real success is accurate token-level domain classification, not PPL improvement from routing." The reframing was applied.

## Macro-Scale Risks (advisory)

1. **K=5 is trivially separable.** At K=50+, domain cluster overlap increases. The legal-finance confusion (cos=0.981) at K=5 foreshadows degradation. The linear separability result may not hold.

2. **Segment-isolated PPL degrades vs per-sequence.** This limits the practical utility of per-token routing. Per-token routing is only valuable if adapters actually help on short segments. At scale, adapters trained on longer contexts with appropriate LORA_SCALE may behave differently.

3. **Model config mismatch (Phase 1 vs Phase 3).** At scale, numerical drift between raw BitLinear and unpacked-bf16 representations could degrade probe accuracy below the 100% threshold, exercising K785 and Theorem 3 non-tautologically.

4. **Response-only token classification.** Inference sees all tokens. Template/instruction tokens were excluded from training. At scale, the probe needs to handle these.

## Verdict

**PROCEED**

All 6 required fixes from the first review have been correctly applied:

1. K785 tautology is explicitly acknowledged in both PAPER.md and results.json.
2. Theorem 1 renamed to Claim 1 with honest disclaimer about existence-only guarantees.
3. Failed SNR prediction (16x predicted, 0.2pp actual) has a dedicated section with correct diagnosis.
4. Model configuration mismatch between phases is documented as Limitation 7.
5. Oracle-worse-than-base is substantively analyzed with three plausible mechanisms.
6. Rahimi & Recht citation corrected to kernel approximation, not classification accuracy.

The experiment is a valid Type 2 guided exploration. It started with a proven framework (ridge regression on mean-pooled hidden states), identified a clear unknown (per-token accuracy), and narrowed that unknown conclusively. The key finding -- that per-token hidden states are linearly separable at 98.3%, making MLP probes unnecessary for K=5 domains -- is genuine, honest, and useful. The failed SNR prediction is correctly treated as a scientific finding rather than hidden.

The remaining issues (tautological K785, unresolved oracle-worse-than-base, model config mismatch) are all documented limitations, not hidden defects. Finding status SUPPORTED is appropriate.
