# Adversarial Review: exp_p0_composition_benchmark

## Verdict: PROCEED (SUPPORTED)

## Verification

All kill criteria results in PAPER.md match results.json exactly. No fabrication.
Prediction-vs-measurement table present and complete. Experimental setup sound
(same n=100, seed=42 as Finding #508, proper holdout via benchmark-specific evaluation).

## Issues (all non-blocking)

### 1. MATH.md proof is fundamentally wrong
The proof predicted 68-73% GSM8K but measured 0%. The error: extrapolating from
Finding #505's 2.1% PPL degradation (smooth metric, averaged over vocabulary) to
benchmark accuracy (hard threshold, exact answer extraction). These are categorically
different — PPL absorbs interference gracefully, benchmarks don't. PAPER.md correctly
identifies this flaw in the K1408 analysis section.

**Impact:** Non-blocking — the proof failure IS the finding. The impossibility structure
in PAPER.md (O(N) perturbation vs O(1) signal, SNR ~ 1/(N-1)) is the corrected analysis.

### 2. MedMCQA merged (20%) < base (31%) — understated
PAPER.md notes 25% MCQ baseline but doesn't explicitly flag that merged performance
(20%) is WORSE than the unmodified base (31%). This means interference isn't just
failing to help — it's actively degrading the base model's existing capability by 11pp.
This strengthens the impossibility argument further.

### 3. Pre-merge method clarity
PAPER.md says "concatenate LoRA weights along rank dimension (rank 8→24, scale 8→24,
effective factor = 1.0 per adapter)." This is full-scale additive composition
W_base + ΔW_1 + ΔW_2 + ΔW_3, not averaged 1/N scaling. This is the correct test
(matches current_direction.md design), but future readers should note this is the
worst-case scenario — averaged merging (1/3 scale each) might survive partially.
However, 1/3 scaling would also reduce each adapter's individual contribution,
defeating the purpose.

### 4. Code routing at 84% is a production concern
TF-IDF router was trained on CodeAlpaca but evaluated on HumanEval function signatures.
The domain shift (natural language docstrings vs code tokens) explains the gap. Not
blocking for this experiment but worth noting for routing improvements.

### 5. Routing barely passes at 90.7% vs 90% threshold
Only 0.7pp above kill threshold. Two of three domains (code 84%, medical 89%) are
below the predicted 95%. The routing corollary in MATH.md assumed R≥0.95 but got
R=0.907. Still passes the conservative kill criterion.

## What this experiment proves

1. Standard LoRA pre-merge at full scale is catastrophically incompatible with
   benchmark accuracy (0% GSM8K, 0% HumanEval from 73%/63% solo)
2. Orthogonal adapters (Grassmannian/PoLAR) are STRUCTURALLY REQUIRED, not optional
3. TF-IDF routing works on real benchmark text (90.7%)
4. Base model is completely unaffected by adapter operations (0.0pp delta)
5. The Pierre architecture's dual composition strategy (orthogonal pre-merge for top-N,
   routing for dynamic domains) is validated by this experiment's failure mode
