# Peer Review: Single-Pass MLP Mixed-Adapter Routing (RE-REVIEW)

## Re-Review Context

This is a re-review after the first adversarial review raised 3 blocking issues.
The researcher applied all 3 fixes. This review verifies the fixes and checks
for any new issues.

## Experiment Type
Verification (Type 1 with corrected proof)

## Fix Verification

### Fix 1: Cross-segment max-diff reporting
**Status: APPLIED CORRECTLY.**

PAPER.md (lines 86-89) now reports three rows for cross-segment:

| Metric | Same-segment | Cross-segment |
|--------|-------------|---------------|
| Mean NLL diff | 0.000 | 0.068 |
| Max NLL diff (per-seq mean) | 0.000 | 0.170 |
| Max NLL diff (per-token) | 0.000 | 4.125 |

PAPER.md (lines 96-102) explicitly discusses the per-token outlier: "the true
per-token max NLL diff is 4.125 (from results.json max_per_token_abs_diff),
meaning some individual tokens experience NLL differences of 100-400%."

The same-segment per-token max of 0.000 is correctly inferred: since
same_seg_diffs stores per-sequence means of absolute values, and all means are
exactly 0.000, every individual same-segment token must also be 0.0. Sound
reasoning.

### Fix 2: Per-pair K793 failures acknowledged
**Status: APPLIED CORRECTLY.**

PAPER.md prediction table (line 19) now reads:
"PARTIAL -- global avg passes, but 4/10 pairs exceed 1% (math+legal 1.14%,
medical+legal 1.28%, medical+creative 1.32%, legal+creative 1.04%)"

The 4 failing pairs are enumerated with their exact percentages. The prediction
match column honestly says "PARTIAL" rather than "YES." Limitation #4 also
discusses this.

### Fix 3: Theorem 2 downgraded to Conjecture
**Status: APPLIED CORRECTLY.**

MATH.md line 261 now reads:
"Conjecture 2 (Bounded Divergence -- Informal)."

Lines 272-278 explicitly state: "This is NOT a formal theorem. The bound is
vacuous at L=30: the recurrence solves to eps_L <= delta_MLP * ((1+L_attn)^L-1)
/ L_attn, which with L_attn ~ 1 gives ~10^9 * delta_MLP -- orders of magnitude
larger than the empirical divergence (0.61% PPL)."

This is honest and precise.

## Hack Detector
- Fix count: 1 (single-pass mixed adapter). No stacking. CLEAN.
- Is MATH.md a proof or a description? Contains two proofs with QED (Theorem 1 corrected: non-equivalence; Theorem 3: same-segment exact match). Conjecture 2 is honestly labeled as informal with vacuous bound. The proof of non-equivalence is constructive and correct. The same-segment exact match proof is correct.
- Metric used as evidence: Per-token NLL diff (directly tests the proof). PPL ratio (proxy but well-motivated). Same-segment exact match (0.000) is the strongest evidence -- it directly confirms the theorem.
- Kill criteria source: K793 derived from proof prediction (small divergence, 1% conservative threshold). K794 derived from Finding #312 measurement. K795 trivial by construction.

## Self-Test Audit

All 6 items present and correctly answered. No changes since first review. PASS.

1. One property (MLP token-independence) with honest caveat about indirect propagation.
2. Real, correctly cited papers (Vaswani, Mixtral, Switch, LoRA).
3. Specific falsifiable numbers (K793 < 1%, K794 near 4.656 and < 4.815).
4. Falsification targets the proof (exact match would contradict non-equivalence).
5. Zero hyperparameters (oracle assignment).
6. Not a fix stack -- direct verification.

## Mathematical Soundness

### Theorem 1 Corrected (Non-Equivalence): SOUND

The proof shows that in multi-pass (pass k), token s with sigma(s) != k gets
adapter k's MLP output, producing a different residual than in single-pass where
sigma(s) is applied. At layer l+1, attention over these different residuals
breaks equivalence. The proof is constructive with a concrete 2-token example.
QED present.

### Conjecture 2 (Bounded Divergence): HONEST ABOUT LIMITATIONS

Now correctly labeled as a conjecture with an informal argument. The vacuous
bound (~10^9 * delta_MLP at L=30) is explicitly acknowledged. The empirical
divergence (0.61%) is vastly smaller, which the text correctly identifies as
meaning the bound provides "directional intuition" only. No claim of a tight
guarantee. This is the correct framing.

### Theorem 3 (Same-Segment Exact Equivalence): SOUND

For tokens at positions 0..boundary-1, causal masking ensures they only attend
to positions with the same adapter in both single-pass and multi-pass regimes.
By induction over layers: identical embeddings -> identical attention (base
weights, same context) -> identical MLP (same adapter) -> identical residuals.
QED present. Empirically verified (0.000 diff across all 200 sequences, 25,600
same-segment tokens).

### Hidden Assumptions Check

1. No dropout: verified (code uses dropout=0.0). HOLDS.
2. Deterministic eval: MLX with no stochastic components. HOLDS.
3. RMSNorm is token-independent: BitNet-2B uses RMSNorm (per-token norm). HOLDS.
4. No cross-token MLP ops: standard SiLU-gated (actually relu2 for BitNet) MLP. HOLDS.
5. No KV cache: full forward passes without caching. HOLDS.

No new hidden assumptions found.

## Prediction vs Measurement

PAPER.md contains a prediction-vs-measurement table (lines 16-26). After
revision, the table is honest and detailed:

| Prediction | Measured | Match? | Assessment |
|-----------|----------|--------|------------|
| P1: ratio < 1% (global) | 0.61% | PARTIAL | Global passes; 4/10 pairs exceed 1%. Honestly reported. |
| P2: single < 4.815 | 4.684 | YES | Clear pass with margin. |
| P3: assignments identical | By construction | YES | Trivial but correct. |
| Same-segment exact match | 0.000 (mean and per-token max) | YES | Core theorem verification. Strongest evidence. |
| Cross-segment divergence | mean 0.068, per-seq-max 0.170, per-token-max 4.125 | YES | Non-zero as predicted. Both scales reported. |
| Divergence proportional to adapter distance | legal/creative > python/math | YES | Consistent with Conjecture 2's delta_MLP dependence. |

The per-pair breakdown (PAPER.md lines 66-77) shows the full data.

The medical+legal anomaly (single-pass 5.499 > per-seq 5.472) is acknowledged
in Limitations and the pair-by-pair table. This is one pair out of ten and the
reversal is 0.5% -- not structurally concerning.

## New Issues Found in Re-Review

### Issue 1: Per-token max attribution (ADVISORY, non-blocking)

The `max_per_token_abs_diff` of 4.125 is computed globally (line 639-643 of the
code) but is not tracked per-pair. We cannot tell which domain pair produces this
extreme outlier. Given that legal+creative has the largest PPL divergence
(1.04%), it is likely from that pair, but this is not confirmed. For a
verification experiment this is acceptable -- the same-segment exact match is the
core claim and it holds perfectly.

### Issue 2: The code computes both adapter LoRA outputs for ALL tokens (ADVISORY)

Line 351-352: both lora_out_A and lora_out_B are computed for all T tokens, then
mx.where selects per-token. This is 2x LoRA compute at every MLP layer. For
verification this is fine. For production, a more efficient implementation would
compute each adapter's LoRA only for its assigned tokens. The PAPER.md
acknowledges this in the architecture section (Section G of MATH.md).

### Issue 3: K793 threshold derivation could be tighter (ADVISORY)

The 1% threshold for K793 is described as a "conservative practical threshold"
(MATH.md line 165). Since Conjecture 2's bound is vacuous, the 1% is effectively
an empirical target rather than a proof-derived threshold. This is acceptable
given that the conjecture is honestly labeled, but it means P1 is a practical
benchmark rather than a theoretical prediction. The experiment does not claim
otherwise after the revision.

None of these rise to blocking level.

## NotebookLM Findings

NotebookLM was not used for this re-review. The first review was thorough and
the fixes are targeted -- a deep review is not warranted for a fix verification.

## Novelty Assessment

Unchanged from first review. The architecture is not novel (MixLoRA, Mixtral,
Switch Transformer all implement per-token FFN routing with shared attention).
The contribution is:

1. The mathematical proof structure: same-segment exact equivalence via causal
   mask + MLP token-independence (Theorem 3). This specific theorem and its
   proof are not in the MixLoRA paper.

2. The self-correcting proof narrative: proving exact equivalence, finding the
   error, proving non-equivalence, then measuring empirical divergence. This is
   good mathematical practice and informative.

3. The observation that single-pass is arguably "more correct" than multi-pass
   (each token sees genuinely-adapted neighbors), which is a useful conceptual
   reframe.

PAPER.md cites MixLoRA as a key reference (line 48). The MATH.md could more
explicitly note that MixLoRA already implements this architecture, with the delta
being the proof structure. This remains an advisory note.

## Macro-Scale Risks (advisory)

Unchanged from first review:

1. Adapter distance scaling at N>10 domains may increase per-pair divergence.
2. Longer sequences (2048+) accumulate more cross-segment attention interactions.
3. Per-token outliers (4.125 NLL diff) could cause generation failures at scale.
4. Conjecture 2's bound is vacuous -- no formal guarantee at any scale.

These are known limitations, not blocking issues.

## Verdict

**PROCEED**

### Justification

All 3 blocking fixes from the first review have been correctly applied:

1. **Cross-segment max-diff reporting:** PAPER.md now reports both max-of-means
   (0.170) and true per-token max (4.125) with clear labeling and explicit
   discussion of the outlier magnitude.

2. **Per-pair K793 failures:** The prediction table marks K793 as "PARTIAL"
   and enumerates all 4 failing pairs with exact percentages.

3. **Theorem 2 downgrade:** Correctly relabeled as "Conjecture 2 (Bounded
   Divergence -- Informal)" with explicit acknowledgment that the bound is
   vacuous at L=30.

The core finding stands:

- **Same-segment exact match (Theorem 3):** Proven correctly, verified
  empirically (0.000 diff across 25,600 same-segment tokens). This is the
  strongest result -- a formal proof with exact empirical confirmation.

- **Cross-segment bounded divergence:** Not formally bounded (conjecture is
  vacuous), but empirically small (0.61% global PPL, though 4/10 pairs exceed
  1% and per-token outliers reach 4.125 NLL). Honestly reported after revision.

- **Practical utility:** Single-pass PPL 4.684 beats per-sequence best 4.815
  with 5x fewer forward passes. The architecture works.

The finding should be recorded as **SUPPORTED** (not CONCLUSIVE) because:
- Theorem 3 is conclusively verified (same-segment exact match)
- Cross-segment divergence lacks a tight formal bound
- 4/10 pairs exceed the 1% threshold at per-pair granularity
- Single seed, single boundary position

These caveats are appropriate for SUPPORTED status at micro scale.
