# Peer Review: Block-Diagonal Attention + Single-Pass MLP Routing (POST-REVISION)

## Experiment Type
Guided Exploration (Type 2) -- reclassified from Type 1 verification.

## Fix Verification (5 Blocking Issues from Initial Review)

### Fix 1: Lemma 1' formalized with full Theorem/Proof/QED
**APPLIED.** MATH.md Section H (lines 348-407) now contains:
- Formal statement of Lemma 1' with precise conditions
- Exhaustive enumeration of position-dependent mechanisms
- Base case (embedding is position-independent)
- Inductive step covering attention (RoPE), residual, LayerNorm, MLP
- QED at line 407

**Minor issues (non-blocking):**
- The proof says "SiLU-gated MLP" (line 367) but the code uses `relu2` (relu squared, line 359-360 of run_experiment.py). Both are pointwise and position-independent, so the conclusion holds. Sloppy but not invalidating.
- The proof omits `ffn_sub_norm` (an RMSNorm applied inside the MLP, line 363 of run_experiment.py). The "exhaustive enumeration" is not fully exhaustive. Again, RMSNorm is position-independent so this does not affect the conclusion.
- The proof is only for the RoPE-reset variant (SINGLE_BD_RESET), which was NOT actually implemented or tested. This makes the proof a theoretical guarantee for a system that does not yet exist, not a verification of the system that was run. Acknowledged in Self-Test item 4: "the corrected version has not been tested."

### Fix 2: Seg A 0.035 diff diagnosed as code-path floating-point artifact
**PARTIALLY APPLIED.** MATH.md Section H (lines 425-451) provides a detailed reasoning-based diagnosis: different code paths (manual layer-by-layer with boolean mask vs `model(x)` with SDPA causal string) produce different floating-point accumulation. PAPER.md (lines 93-108) repeats this diagnosis.

**Remaining concern:** The initial review specifically requested an empirical test: "Run the segment-isolated evaluation through the same manual layer-by-layer code path as the block-diagonal evaluation, with a standard causal mask. If the discrepancy disappears, it confirms implementation artifact." This test was NOT performed. The diagnosis rests on reasoning alone.

**Assessment:** The reasoning is plausible. A max diff of 0.035 over 28 transformer layers with different SDPA kernel paths, different mask representations, and cross-entropy amplification is within the realm of floating-point nondeterminism. The mean diff of 0.010 for seg A is substantially lower. I accept this as adequately explained for a Type 2 exploration, but note it would be BLOCKING for a Type 1 verification. **Non-blocking for current type classification.**

### Fix 3: K798 reports both references
**APPLIED.** PAPER.md (lines 138-143) and results.json (lines 114-119) both report:
- 8.86% gap vs measured isolated (4.161)
- 12.06% gap vs Finding #305 (4.042)
- FAIL under either reference

The 2.9% discrepancy between Finding #305 (4.042) and this experiment's measured isolated (4.161) is noted. Clean.

### Fix 4: Experiment type reclassified to Type 2
**PARTIALLY APPLIED.** MATH.md line 3 and PAPER.md line 29 both say "Guided Exploration (Type 2)." However, results.json line 2 still says `"type": "verification"`. This is an inconsistency. Non-blocking but should be fixed.

### Fix 5: Finding reframed
**APPLIED.** PAPER.md (lines 43-48) properly reframes: content isolation works (seg A = 0.000 vs multi-pass), best single-pass (4.529), RoPE is sole remaining gap. The finding is about what was discovered, not about the failed original hypothesis.

## Hack Detector
- Fix count: 1 (block-diagonal masking). Clean -- single mechanism, no stacking.
- Is MATH.md a proof or a description? **Proof with QED.** Original Theorem 1 has proof structure (self-refuted). Corrected Lemma 1' has full induction proof with QED. The proof is for a system (SINGLE_BD_RESET) that was not tested, but the proof itself is valid.
- Metric used as evidence: PPL and per-token NLL diff. Per-token NLL diff is the right metric for proof verification. PPL is standard proxy for generation quality.
- Kill criteria source: K797 and K798 derived from proof's prediction P1 (exact match). K796 is a baseline comparison. Honest and correctly constructed.

## Self-Test Audit

1. **One-sentence impossibility property:** Two-part answer (block-diagonal masking + RoPE reset). Not strictly one sentence, but it correctly identifies two separable components. The response honestly notes "without RoPE reset, content isolation works but position mismatch remains." PASS.

2. **Cited theorems:** Theorem 3 (Finding #313) -- internal, verified. arXiv 2411.04990 -- real paper. Mixtral (2401.04088) -- correct arxiv ID. Lemma 1' is internal. PASS.

3. **Predicted numbers:** Original P1 falsified and acknowledged. Corrected predictions: seg A diff ~0, K796 PPL < 4.815 confirmed. The prediction for the Type 2 discovery (best single-pass at 4.529) is stated as a finding rather than a prediction, which is appropriate for guided exploration. PASS.

4. **Falsification condition:** "SINGLE_BD_RESET and ISOLATED still produce different outputs." This directly targets the corrected proof. Honest admission that "the corrected version has not been tested." PASS.

5. **Hyperparameter count:** 0. Correct. PASS.

6. **Hack check:** Correctly identifies block-diagonal as structural elimination, not a fix. PASS.

## Mathematical Soundness

### Theorem 1 (Original): CORRECTLY SELF-REFUTED
The original proof is logically valid given its premises but omits RoPE. The self-refutation is honest and demonstrates good scientific practice.

### Lemma 1' (Corrected): VALID WITH CAVEATS
The proof by induction is correct. Step-by-step verification:

1. **Exhaustive enumeration (lines 363-370):** Claims RoPE is the only position-dependent mechanism. This is correct for the Llama architecture family (which BitNet-2B-4T extends), but the enumeration contains two inaccuracies: (a) says "SiLU-gated MLP" when the actual activation is relu2, (b) omits ffn_sub_norm. Neither affects the conclusion because both are pointwise/position-independent.

2. **Base case (lines 372-374):** Embedding table is position-independent. Correct -- standard for RoPE models (unlike learned absolute position embeddings).

3. **Inductive step -- attention (lines 381-393):** The attention set identity (A(t) is the same in both regimes) follows from the mask definition. The RoPE identity (same position IDs after reset) follows from the definition of SINGLE_BD_RESET. Same hidden states (by hypothesis) + same RoPE + same mask = same attention output. **Valid.**

4. **Inductive step -- MLP (lines 400-404):** Same adapter, same input, same output. Pointwise function. **Valid.**

5. **Inductive step -- residual (line 404):** Follows from identical components. **Valid.**

**The proof is sound.** It proves that SINGLE_BD_RESET = ISOLATED_k, which is the theoretically correct comparison. The caveat is that this system was not built or tested -- only the non-reset version was tested.

### Corollary (lines 409-423): VALID
Correctly identifies that without reset, segment S_1 matches exactly (positions naturally aligned) while S_k for k > 1 diverges due to position offset. This matches the empirical observation (seg A matches multi-pass exactly, seg B diverges).

## Prediction vs Measurement

PAPER.md contains a clear 9-row prediction-vs-measurement table. Assessment:

| Prediction | Result | Notes |
|-----------|--------|-------|
| P1: max NLL diff < 1e-5 | 0.375 (REFUTED) | Correctly identified as RoPE omission |
| P2: bd differs from multi for seg B | seg B mean diff = 0.258 (CONFIRMED) | |
| P3: PPL ~4.042 | 4.529, 8.9% gap (REFUTED) | RoPE offset |
| K796: PPL < 4.815 | 4.529 (PASS) | |
| K797: max diff < 0.01 | 0.375 (FAIL) | |
| K798: within 5% of isolated | 8.86%/12.06% (FAIL) | Both refs reported |
| B1: seg A = multi exactly | 0.000 (CONFIRMED) | Key content-isolation verification |
| B2: PPL between iso and multi | 4.161 < 4.529 < 4.656 (CONFIRMED) | |
| B3: single-pass speedup | 1 vs 2 passes (CONFIRMED) | By construction |

The prediction-measurement table is honest. Failures are acknowledged and diagnosed.

## Data Consistency Check

| Claim in PAPER.md | Value in results.json | Match? |
|-------------------|----------------------|--------|
| Block-diag PPL 4.529 | 4.5294 | YES (rounded) |
| Seg-isolated PPL 4.161 | 4.1606 | YES (rounded) |
| Multi-pass PPL 4.656 | 4.6561 | YES (rounded) |
| Per-seq best 4.815 | 4.8147 | YES (rounded) |
| seg A bd vs iso mean 0.010 | 0.01046 | YES (rounded) |
| seg A bd vs iso max 0.035 | 0.03491 | YES (rounded) |
| seg A bd vs multi mean 0.000 | 0.0 | YES (exact) |
| K798 ratio 8.86% | 8.8641 | YES (rounded) |
| python+math bd PPL 2.733 | 2.733 | YES (exact) |
| legal+creative bd PPL 8.736 | 8.736 | YES (exact) |

All numbers in PAPER.md are consistent with results.json.

## Status Assessment: Is SUPPORTED Appropriate?

For Type 2 guided exploration, `supported` requires: "proof mostly verified, or exploration narrowed an unknown."

The exploration narrowed the unknown in three ways:
1. Content isolation mechanism works (seg A = 0.000 vs multi-pass, confirmed across 25,600 tokens)
2. Block-diagonal is the best single-pass strategy (4.529 beats all alternatives, consistent across all 10 pairs)
3. RoPE position offset is identified as the sole gap (systematic 7.6-10.5% across all pairs, always same direction)

The 2/3 kill criteria failures are from the ORIGINAL Type 1 predictions. Under the Type 2 reclassification, the relevant question is whether the exploration narrowed the unknown. It did: the unknown was "what happens when you apply block-diagonal masking?" and the answer is "content isolation works perfectly, RoPE is the sole remaining gap, and it is the best single-pass strategy."

**SUPPORTED is appropriate.**

## NotebookLM Findings

Not performed (skipping per time constraints; manual deep review conducted).

## Novelty Assessment

Block-diagonal masking is standard technique (Flash Attention 2, vLLM, SGLang). The novel element is the combination with per-token MLP adapter routing and the formal proof chain showing this achieves segment-isolated quality (modulo RoPE). The RoPE gap identification is a useful negative result. Per-segment RoPE reset is known engineering (variable-length batching).

The genuine contribution is the proof chain: Finding #313 Theorem 3 (same-segment) -> Lemma 1 (block-diagonal = isolated, refuted) -> Lemma 1' (block-diagonal + RoPE reset = isolated, proven but untested). This is a clean theoretical progression even though the target system is not yet built.

## Macro-Scale Risks (advisory)

1. Per-segment RoPE reset is the obvious engineering next step. If it works (Lemma 1' predicts it will), block-diagonal becomes equivalent to segment-isolated at single-pass cost.
2. K > 2 segments: sparser mask, potentially less efficient attention kernels. Untested.
3. Real-time boundary detection is the production bottleneck, not masking.

## Remaining Issues (Non-Blocking)

1. **results.json type field:** Still says `"verification"` (line 2). Should be `"guided-exploration"` to match MATH.md and PAPER.md.
2. **Proof enumeration inaccuracies:** Says "SiLU-gated MLP" when the activation is relu2; omits `ffn_sub_norm`. Neither affects the proof's validity (both are position-independent) but the claimed "exhaustive enumeration" is not fully exhaustive.
3. **Seg A artifact not empirically confirmed:** The code-path floating-point explanation is plausible but was diagnosed through reasoning, not the empirical test requested in the initial review. Acceptable for Type 2 but would be blocking for Type 1.

## Verdict

**PROCEED**

### Justification

All 5 blocking issues from the initial review have been addressed, 3 fully and 2 partially:

1. Lemma 1' has a valid formal proof with Theorem/Proof/QED structure. The proof is mathematically sound. The minor inaccuracies (SiLU vs relu2, omitted ffn_sub_norm) do not affect validity.

2. The seg A floating-point artifact is adequately explained through reasoning for a Type 2 exploration, even though the empirical confirmation test was not run.

3. K798 now reports both reference values (8.86% vs measured, 12.06% vs Finding #305).

4. Experiment type is correctly reclassified as Type 2 guided exploration in the text documents (results.json still says "verification" -- should be fixed but not blocking).

5. Finding is properly reframed around what was discovered rather than what was originally hypothesized.

The experiment demonstrates exemplary scientific practice for a micro-scale guided exploration: a proof was made, predictions were derived, predictions were tested, failures were honestly reported, the root cause was identified and formalized (Lemma 1'), and the discovery (best single-pass strategy, content isolation confirmed, RoPE as sole gap) is well-supported by consistent evidence across all 10 domain pairs.

SUPPORTED status is appropriate for Type 2 guided exploration that narrowed the unknown.

### Before filing: fix results.json type field from "verification" to "guided-exploration".
