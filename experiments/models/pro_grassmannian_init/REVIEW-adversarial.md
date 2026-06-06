# Peer Review: pro_grassmannian_init

## Experiment Type
**Verification.** MATH.md contains three Theorem/Proof/QED blocks (Theorems 1, 2, 3
plus Corollary 1.1). The experiment verifies quantitative predictions derived from
these proofs against the Qwen3-4B GQA architecture.

## Hack Detector
- Fix count: **1** (QR decomposition). Single construction, single guarantee. Clean.
- Is MATH.md a proof or a description? **Proof with QED.** Theorem 1 is a genuine
  (if trivial) proof. Theorems 2 and 3 are straightforward but formally stated.
- Metric used as evidence: Flattened cosine similarity `|<vec(A_i), vec(A_j)>| / (||A_i|| ||A_j||)`.
  See Mathematical Soundness for discussion of whether this metric fully verifies
  the theorem's prediction.
- Kill criteria source: **Derived from proof** (MATH.md Section D explicitly derives
  both K810 and K811 from the theorems), though the thresholds are so loose they
  function as sanity checks rather than discriminating tests.

## Self-Test Audit

1. **One-sentence impossibility property:** "QR decomposition produces exactly
   orthonormal columns, making A_i^T A_j = 0 by construction for columns from
   different blocks." -- PASS. Single property, clearly stated.

2. **Cited theorems:** Householder 1958 (real, applies), Grassmannian capacity
   floor(d/r) (standard, applies), JL Lemma (real, used for contrast only).
   Finding #132, #317 (project-internal, verifiable). -- PASS.

3. **Predicted numbers:** cos = 0.0 exactly (< 1e-6), N_max = 160, init time
   < 1s. -- PASS. Specific and falsifiable.

4. **Falsification condition:** "The proof is wrong if QR does not produce
   orthonormal columns (NumPy/LAPACK bug), or if in_features varies across
   module types in a way that reduces capacity below N=24." -- PASS, though
   the first condition is effectively unfalsifiable (it would require a LAPACK
   bug). The second condition is genuinely testable.

5. **Hyperparameter count:** 0. Correct -- the construction is fully determined
   by architecture parameters. -- PASS.

6. **Hack check:** "No. Single construction, single guarantee." -- PASS.

Self-test is complete, all answers are substantive. No blanks or evasions.

## Mathematical Soundness

**Theorem 1 (QR Orthogonality): CORRECT.**
The proof is trivially correct. QR decomposition produces Q with orthonormal columns
(Q^T Q = I). Partitioning into non-overlapping column blocks yields A_i^T A_j = 0
for i != j because each entry of A_i^T A_j is an inner product of two distinct
columns of Q. No hidden assumptions beyond the existence of QR (which is guaranteed
for any matrix with m <= d, always satisfied here since N*r <= d is a precondition).

**Corollary 1.1 (Flattened Cosine): CORRECT but with a subtle gap.**
The identity `<vec(A_i), vec(A_j)> = tr(A_i^T A_j)` is correct (standard
vec-trace duality). Since A_i^T A_j = 0, the trace is zero, so the flattened
cosine is zero. The logic is sound.

However, the EXPERIMENT measures the flattened cosine (`np.abs(np.sum(A_i * A_j))`,
which equals `|tr(A_i^T A_j)|`), not the full matrix `A_i^T A_j`. The flattened
cosine being zero is a NECESSARY but NOT SUFFICIENT condition for `A_i^T A_j = 0`.
In principle, `tr(A_i^T A_j) = 0` could hold even with nonzero off-diagonal entries
of `A_i^T A_j` that cancel in the sum. The experiment therefore measures a weaker
quantity than what the theorem guarantees.

In practice, this gap is irrelevant: the A-matrices come directly from QR
decomposition of a random matrix, so `A_i^T A_j = 0` holds exactly by
construction, and the flattened cosine measurement is merely confirming
a consequence. But a more rigorous experiment would verify
`max|A_i^T A_j| < epsilon` (the Frobenius norm or max entry of the cross-product
matrix), not just the trace.

**Severity: LOW.** The proof is correct. The metric is consistent with but weaker
than the guarantee. For a verification of QR on a new architecture, this is adequate.

**Theorem 2 (Capacity): CORRECT.** Standard linear algebra. N mutually orthogonal
r-planes in R^d requires N*r <= d, so N <= floor(d/r). Tight by construction.

**Theorem 3 (GQA Invariance): CORRECT.**
A-matrices operate in the input space of linear projections. For q/k/v projections
in GQA, the input is always the hidden state h in R^{hidden_dim}, regardless of
the number of KV heads. The argument is sound. The experiment empirically confirms
that in_features = 2560 for all of q/k/v/gate/up (and 4096 for o_proj, 9728 for
down_proj), which is consistent.

**Complexity analysis: WRONG by 300x.**
MATH.md Section G predicts: "Total: 252 * O(2560 * 384^2) ~ 9.5e10 flops.
At ~1 TFLOP/s for numpy on M5 Pro: ~0.1s." The measured time for N=24 is 32.3s,
which is 300x slower than predicted. This is not acknowledged in MATH.md (though
the kill criterion K811 = 60s catches it with margin). The discrepancy likely
comes from: (a) NumPy TFLOP/s estimate being optimistic for small matrices,
(b) memory allocation overhead for 252 QR calls, (c) Python loop overhead.
The timing prediction P5 ("< 10s") was also wrong for N=24 (32.3s). PAPER.md
acknowledges this as "PARTIAL" but the complexity analysis itself is flawed.

**Severity: LOW.** Timing is not a mathematical guarantee; it is a practical
prediction. The proof's core claims (orthogonality, capacity) are unaffected.

## Prediction vs Measurement

PAPER.md contains the required prediction-vs-measurement table. Assessment:

| Prediction | Expected | Measured | Match? | Assessment |
|-----------|----------|----------|--------|------------|
| P1: cos=0 at N=5 | 0.0 (< 1e-6) | 0.000000 | YES | Confirmed |
| P2: cos=0 at N=24 | 0.0 (< 1e-6) | 0.000000 | YES | Confirmed |
| P3: N_max = 160 | 160 | 160 | YES | Confirmed |
| P4: GQA same N_max | 160 all | q/k/v all 160 | YES | Confirmed |
| P5: Init < 10s | < 10s | 4.15s / 32.3s | PARTIAL | N=5 yes, N=24 no |
| P6: in_features = 2560 | 2560 all | 2560 for q/k/v/gate/up | YES | Confirmed |
| P7: o_proj different | != 2560 | 4096 | YES | Confirmed |
| P8: down_proj different | != 2560 | 9728 | YES | Confirmed |

7/8 predictions match exactly. P5 partially fails (N=24 timing 3.2x over prediction).
Core mathematical predictions (P1-P4) all confirmed exactly.

The table is present, specific, and honest about the partial failure. This meets
the prediction-vs-measurement requirement.

## NotebookLM Findings

Skipped. The experiment is a straightforward QR orthogonality verification on a
known architecture. The mathematical content is elementary (QR decomposition is
first-year linear algebra). The interesting aspects are the architecture detection
(quantized weight shape bug) and GQA invariance confirmation, neither of which
requires deep literature review. NotebookLM would not add material insight beyond
what the review below covers.

## Novelty Assessment

**Within the project:** This is a TRANSFER verification, not a novel result.
Finding #132 already proved Grassmannian initialization works on BitNet-2B-4T
(d=2560, MHA). This experiment confirms it works on Qwen3-4B (d=2560, GQA).
The "novel" content is Theorem 3 (GQA invariance), which is a minor observation --
A-matrices depend on input dimension, not output dimension.

**External prior art:** QR decomposition for orthogonal initialization is standard
(Saxe et al. 2014, "Exact solutions to the nonlinear dynamics of learning in deep
linear networks"). The specific application to LoRA A-matrices has appeared in
LoRA-GA (Wang et al. 2024) and related work, though typically for a single adapter
rather than multi-adapter orthogonality.

**Verdict:** Low novelty, but the experiment serves a necessary engineering purpose
(confirming transfer to the new base model, finding the quantized weight shape bug).

## Macro-Scale Risks (advisory)

1. **Orthogonality at init does not imply orthogonality after training.** The
   A-matrices are frozen (not trained), so orthogonality is preserved by design.
   However, effective interference is `||A_i B_i^T B_j A_j^T||`, which depends on
   B-matrices. Finding #132 shows B-matrix cosine ~0.03 with 17x decorrelation,
   but this was measured on BitNet-2B-4T -- needs re-measurement on Qwen3-4B.

2. **Skeleton storage at scale.** N=24 skeleton is 968 MB. At N=100, this would
   be ~4 GB. At N=160 (capacity), ~6.4 GB. These fit within 48GB, but are
   non-trivial fractions of the memory budget alongside the base model.

3. **The timing analysis is wrong.** If skeleton generation is 300x slower than
   predicted at N=24, extrapolation to N=100 or N=160 could be problematic
   (possibly multi-minute generation). This is a one-time cost but should be
   re-benchmarked before committing to large N.

## Verdict

**PROCEED**

### Justification

The experiment is a clean verification of a trivially correct mathematical result
(QR orthogonality) applied to a new architecture (Qwen3-4B GQA). The proof is
sound. 7/8 predictions match exactly. The one partial failure (timing) is a
practical concern, not a mathematical error. The finding status of SUPPORTED is
appropriate -- it would be CONCLUSIVE if the timing prediction hadn't failed and
if the metric fully verified A_i^T A_j = 0 rather than just its trace.

The experiment serves its engineering purpose: confirming that the Grassmannian
skeleton construction transfers from BitNet-2B-4T (MHA) to Qwen3-4B (GQA),
detecting the quantized weight shape bug, and generating the skeleton files
needed for downstream SFT experiments.

### Minor recommendations (non-blocking)

1. **Verify A_i^T A_j directly.** Add a check that `max(|A_i^T A_j|) < 1e-6`
   (all entries of the cross-product matrix, not just the trace). This fully
   verifies Theorem 1 rather than just Corollary 1.1.

2. **Fix the timing prediction.** The complexity analysis in MATH.md Section G
   is off by 300x. Either correct the TFLOP/s estimate or note that Python/NumPy
   overhead dominates for this workload size.

3. **K810 threshold is uselessly loose.** A threshold of 0.05 when the expected
   value is 0.0 (< 1e-6) tests nothing. Consider tightening to 1e-5 to actually
   test machine-precision orthogonality. (This is cosmetic -- the result is 0.0
   regardless.)
