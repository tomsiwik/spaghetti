# Peer Review: N-gram Expert Mixing (Revision v2)

## NotebookLM Findings

Skipped. This is a straightforward statistical mixing experiment where code-level
verification is more productive than document analysis. The prior review
identified 5 specific bugs; this review verifies each fix against the code and
results.

## Verification of 5 Previous Fixes

### Fix 1: Backoff bug -- FIXED.
`_backoff_score()` (lines 143-177) now correctly accumulates
`backoff_weight *= self.backoff` at three fall-through points (lines 154, 165,
167) and applies the accumulated weight when returning scores (line 163) and
at the unigram fallback (lines 175-176). The standalone 5-gram PPL improved
from 8.14 to 7.64, consistent with proper backoff discounting lower-order
fallback scores.

### Fix 2: Alpha search extended -- FIXED.
Line 659 now tests `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`.
The full curve is reported, peak at alpha=0.7 (+20.80%), with a flat plateau
from 0.6-0.8 (all >20%). Alpha=1.0 (pure n-gram, +17.44%) is properly included
as a control confirming that mixing genuinely improves over either model alone.

### Fix 3: Padding asymmetry -- DOCUMENTED, NOT FIXED.
Lines 505-547 add a padding asymmetry analysis quantifying the gap: short
sequences (76% padding) have 15.84% worse neural PPL than long sequences.
The paper (lines 80-97) documents this clearly. However, the evaluation itself
still uses zero-padding without attention masking (line 468). The paper
correctly flags this as a confound. Documenting the issue rather than fixing
it is acceptable for a micro-experiment, though it weakens the quantitative
claims. Acceptable.

### Fix 4: Per-domain eval -- FIXED.
Line 452: `n_eval_seqs = len(data["all_val_seqs"])` now processes all 6,403 val
sequences (45,741 tokens). The per-domain token counts sum correctly:
a_e(15,025) + f_j(7,127) + k_o(12,340) + p_t(7,991) + u_z(3,258) = 45,741.
All 5 domains show 15-19% improvement. Domain mapping relies on dict iteration
order matching between lines 255-259 (data construction) and lines 703-708
(domain mapping), which is correct in Python 3.7+.

### Fix 5: MATH.md bound qualified -- FIXED.
MATH.md lines 106-131 now explicitly state that mixing can hurt when
`p_ng(w*) < p_nn(w*)` and provide the condition for net benefit. The 2-gram
case (34% n-gram win rate, mixing hurts) is cited as empirical evidence that
the bound does not universally hold. The "mixing can only help" claim is
removed.

## Mathematical Soundness

### Convex combination: CORRECT.
Standard result, no issues.

### Stupid backoff with corrected implementation: CORRECT.
The implementation now matches Brants et al. (2007). One minor note: the code
applies smoothing only at the unigram level (line 175), not at higher orders.
This is consistent with standard stupid backoff (smoothing only at the terminal
fallback), though MATH.md line 33 presents add-delta smoothing as applying
generally. This is a documentation inconsistency, not a code bug. The code
behavior is correct.

### Entropy-adaptive mixing: CORRECT.
Formula is sound, alpha is properly bounded in [0,1], the threshold behavior
is clean.

### Improvement bound: NOW CORRECT.
The qualified bound correctly identifies that mixing helps only at positions
where the n-gram model assigns higher probability to the correct token.

### Memory analysis: CORRECT.
14.5MB measured for a global 5-gram table matches the sparse storage analysis.
The 500x margin below 2GB is genuine.

## Factual Error in PAPER.md

PAPER.md line 149 states: "the gap narrowed compared to v1 (was 13.02% vs
11.91%, now 20.80% vs 19.35%)." The v1 gap was 13.02 - 11.91 = 1.11
percentage points. The v2 gap is 20.80 - 19.35 = 1.45 percentage points.
The gap WIDENED from 1.11 to 1.45 ppt, contradicting the claim that it
"narrowed." This is a minor factual error that does not affect any conclusion,
but should be corrected for accuracy.

## Experimental Design

### The elephant in the room: n-gram beats neural.
The 5-gram model alone (PPL 7.64) beats the neural model (PPL 9.26). This
means the headline "+20.80% improvement" is mostly the n-gram model being
better, not mixing being beneficial. The actual evidence of complementary
mixing is the delta between alpha=1.0 (pure n-gram, +17.44%) and alpha=0.7
(best mix, +20.80%) -- a 3.36 percentage point improvement from adding neural
signal to the n-gram model. The paper acknowledges this in Limitation 4 but
the framing in the kill criteria ("K1 PASS: +20.80%") overstates what the
evidence shows about the mixing mechanism specifically.

This is not a kill-worthy issue. The mechanism works: mixed > max(neural,
n-gram). The K1 criterion asks about PPL improvement, not about the source
of improvement. But for integration into the composable experts architecture,
the relevant question is whether n-gram mixing helps a *strong* neural model,
and the micro-scale evidence is ambiguous on this because the neural model is
weak.

### Padding confound remains.
The neural model's 15.84% short-vs-long PPL gap inflates the baseline that
mixing improves upon. If attention masking were applied, neural PPL would
drop (perhaps to ~8.5-8.8), reducing the mixing improvement to perhaps
14-16%. Still above the 5% kill threshold, but materially lower than claimed.

### Controls are adequate.
The full alpha sweep with alpha=1.0 (pure n-gram) serves as an effective
control proving genuine complementarity. The n-gram order ablation (2-5)
shows the expected monotonic improvement. The frac_ngram_wins metric (34%
for 2-gram, 59% for 5-gram) explains why 2-gram mixing hurts. These are
good experimental controls.

## Novelty Assessment

Not novel. N-gram + neural mixing is a textbook technique (Mikolov et al.
2011, Brants et al. 2007, parameter-golf 2024). The experiment correctly
positions itself as validation for the SOLE integration path, not as a
novelty claim. This is appropriate for a micro-experiment.

## Macro-Scale Risks (advisory)

1. **Vocabulary explosion.** At V=32K, 5-gram tables become impractical
   (potentially 30GB+). Character-level V=28 is 1000x smaller. The technique
   may need to fall back to 2-3 grams at token level, where the improvement
   was marginal (0-5%).

2. **Diminishing returns with stronger models.** The n-gram model beats the
   202K neural model. At macro scale, the neural model will capture most
   local patterns, reducing n-gram benefit. The 3.36 ppt complementarity
   signal is the relevant predictor, not the 20.80% headline number.

3. **Router-table coupling.** Selecting the right domain n-gram table requires
   knowing the domain before expert composition, creating a dependency on
   routing quality.

## Verdict

**PROCEED**

All 5 previous issues have been addressed (4 fixed in code, 1 documented as
an acknowledged confound). The mechanism works in principle: mixed PPL (7.33)
beats both pure n-gram (7.64) and pure neural (9.26), confirming genuine
complementarity. Kill criteria are met: K1 +20.80% >> 5%, K2 14.5MB << 2GB.

### Minor issues (non-blocking, should be noted in FINDINGS.md):

1. PAPER.md line 149 claims the entropy-adaptive vs fixed-alpha gap "narrowed"
   but it actually widened (1.11 ppt to 1.45 ppt). Correct the text.

2. The 20.80% headline number is dominated by the n-gram model being stronger
   than the neural model. The complementarity-specific improvement is 3.36 ppt
   (alpha=0.7 vs alpha=1.0). Frame this carefully when recording in FINDINGS.md
   to set accurate expectations for macro-scale benefit.

3. The padding confound (15.84% neural PPL inflation) means the true mixing
   improvement on a properly-evaluated neural model would be lower, perhaps
   14-16%. Still above the 5% kill threshold.
