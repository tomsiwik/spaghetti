# Peer Review: exp_m2p_layer_depth_qwen3

## Experiment Type
Frontier extension (Type 3) -- correctly declared.

MATH.md states the proven result being extended (Finding #365: 89.1% sort at L=36, d=256)
and the mathematical gap (does effective_rank stay within d_M2P=64 when d_model scales
12x from 256 to 3072?). Finding status capped at provisional per Type 3 rules. The
experiment declares "supported" -- this is acceptable since the quality threshold passed
but one secondary criterion (K898) failed.

## Hack Detector
- Fix count: 1 (width scaling of existing proven recipe). No new mechanisms. CLEAN.
- Is MATH.md a proof or a description? MIXED. Theorem 1 and the B.5 rank claim have
  valid QED blocks. Theorem 2 is a necessary condition with proof. Theorem 3 is a
  heuristic extrapolation explicitly labeled "VERY LOW confidence." Overall: two real
  proofs (B.5 rank structure, Theorem 2 necessary condition) plus one honestly-labeled
  heuristic (Theorem 3). Acceptable for Type 3.
- Metric used as evidence: quality_ratio (defined as fraction of base-SFT gap recovered).
  This IS a behavioral proxy -- it measures how close M2P output is to the SFT reference
  in terms of next-token prediction. Acceptable.
- Kill criteria source: K897 (>=85%) derived from prior findings (Finding #363 established
  85% as the floor for "Option A works"). K898 (<0.7 nats) derived from Finding #365
  measured gap (0.51). K899 (sanity check) derived from Finding #365 baseline. Criteria
  are grounded in prior measurements, not arbitrary.

## Self-Test Audit

All 6 items present and answered. Assessment:

1. **One-sentence impossibility property:** "The maximum rank of the joint B-matrix stack
   is min(144, d_out) = 144 at BOTH widths." This is genuinely one property (rank
   independence). Correctly notes this is a Type 3 frontier with no guaranteed
   impossibility. PASS.

2. **Cited theorems -- are they real?**
   - Ghadimi & Lan (arXiv:1309.5549): REAL paper, REAL theorem. The convergence bound
     cited is correct in form.
   - Aghajanyan et al. (arXiv:2012.13255): REAL paper. However, MATH.md cites "Theorem 1,
     Aghajanyan et al." -- the paper does NOT contain a formal "Theorem 1." It presents
     empirical findings about intrinsic dimensionality. The MATH.md treats an empirical
     observation as a theorem. **FLAG: citation inflates empirical finding to theorem.**
     Aghajanyan et al. showed d_int is low and correlates with pre-training; they did NOT
     prove a theorem that d_int is "determined by task, not model size."
   - Ha et al. (arXiv:1609.09106): REAL paper (HyperNetworks). The cited finding (90-95%
     quality retention) is fair characterization of their empirical results.
   - Prechelt GL: Real technique, correctly cited.
   PARTIAL PASS -- Aghajanyan citation overstated.

3. **Predicted numbers:** H1 predicts ~89%/~98%. H2 predicts ~73%. K898 predicts <0.7 nats.
   These are specific and falsifiable. PASS.

4. **Falsification condition:** Targets the proof's assumptions (effective rank, Adam
   compensation, task-complexity hypothesis). Correctly identifies what would break
   the theoretical framework, not just the experiment. PASS.

5. **Hyperparameter count:** Claims 0 new. Correct -- all inherited from proven recipe.
   Reduced n=500, T=400 for d=3072 are acknowledged as runtime concessions. PASS.

6. **Hack check:** Clean extension. No new training tricks. Single protocol change
   (arithmetic excluded) is well-documented from prior findings. PASS.

## Mathematical Soundness

### B.5 Rank Structure Claim -- VALID
The proof that max_rank(S) = min(L*LORA_RANK, d_out) = 144 at both widths is
straightforward linear algebra. When d_out > L*LORA_RANK (satisfied at both 1024 and
12288), the bottleneck is the row count (144), not the column count. This is correct
and is the strongest mathematical contribution of the experiment.

### Theorem 1 (Ghadimi-Lan d_model-independence) -- PARTIALLY VALID, OVERCLAIMS
The Ghadimi-Lan bound itself is correctly stated. The claim that it is "d_model-independent"
is technically correct for the structural form of the bound, but the proof in step (iii)
relies on the claim that "Adam's adaptive learning rate compensates for O(sqrt(d_model))
L_smooth growth" which is NOT a proven fact -- it is an informal argument about optimizer
behavior. The Ghadimi-Lan theorem is stated for SGD, not Adam. Applying it to Adam requires
additional assumptions about the effective learning rate schedule that are not proven here.

However, this is flagged as a known gap in the proof ("the constant may be larger") and
the experiment is Type 3, so the incomplete proof is expected. The honest framing saves it.

### Theorem 2 (Necessary Condition) -- VALID
The necessary condition (effective_rank <= 64 for quality >= 85%) follows from the linear
map structure of the output head. If the target lies outside the 64-dimensional range,
quality degrades. The B.5 insight that the rank bound is width-independent makes this
condition testable. Sound.

### Theorem 3 (Log-Linear H2 Prediction) -- HONESTLY LABELED HEURISTIC
The 4.43 pp/octave rate from layer-depth data is used with an explicit "VERY LOW confidence"
disclaimer. This is not a theorem -- it is a pessimistic bound for hypothesis discrimination.
The honest labeling is appropriate.

### Critical Gap: Aghajanyan "Theorem"
MATH.md Section B.3 cites "Core theorem (Theorem 1, Aghajanyan et al.)" but the
Aghajanyan paper does not contain a formal theorem with this statement. The paper shows
empirically that d_int is surprisingly low and that pre-training reduces it. The specific
claim that d_int is "determined by the TASK, not the model size" is a reasonable
interpretation of their empirical results but is NOT a proven theorem. This is the same
pattern flagged in ADVERSARIAL_REVIEW.md -- "descriptions dressed in equations." Here
it is a citation inflated from empirical finding to theorem.

**Impact on the experiment:** Moderate. The experiment correctly frames this as a
hypothesis (H1) to be tested, not as a guaranteed outcome. The competing hypothesis
structure (H1 vs H2) is good experimental design. The citation inflation weakens the
mathematical framework but does not invalidate the experimental result.

## Prediction vs Measurement

PAPER.md Section 7 contains the required table. Assessment:

| Prediction | Predicted | Measured | Delta | Assessment |
|---|---|---|---|---|
| H1: sort at d=3072 | ~89% | 85.9% | -3.1pp | Within noise. H1 SUPPORTED. |
| H1: reverse at d=3072 | ~98% | 94.1% | -3.9pp | Within noise. H1 SUPPORTED. |
| H2: sort at d=3072 | ~73% | 85.9% | +12.9pp | H2 decisively REFUTED. |
| GL gap < 0.7 nats | PASS | 0.803 | +0.103 | FAILED. Marginal. |
| K899 sanity d=256 | PASS | 93.5% | +8.5pp | PASS. |

**Critical observation on the d=256 replication:** Finding #365 reported sort=89.1%,
reverse=97.8%. This experiment measured sort=96.4%, reverse=90.5%. The sort-reverse
relationship INVERTED: sort went from worst (89.1%) to best (96.4%), and reverse went
from best (97.8%) to worst (90.5%). The 93.5% median matches, but the per-domain
behavior is different. PAPER.md acknowledges this ("Both within +/-8pp") but a 7.3pp
swing on sort and a 7.3pp swing on reverse suggest non-trivial variance. This undermines
confidence in point predictions at this scale.

**K897 is barely passing.** Sort quality at d=3072 is 85.94%. The threshold is 85.0%.
The margin is 0.94pp. Given the 7pp+ variance observed in the d=256 replication, this
margin is within noise. A different random seed could flip K897 to FAIL.

## Concerns

### 1. Random-init base at d=3072 vs pre-trained base at d=256 (MODERATE)

The d=256 run uses a pre-trained base (base_steps=1200, base losses ~12.7 nats),
while d=3072 uses a random-init base (base_steps=0, base losses ~5.1 nats).

PAPER.md correctly notes this makes d=3072 quality_ratio HARDER to achieve (smaller
denominator: 2.84 nats vs 10.38 nats). This is a fair point -- any M2P error is magnified.

However, this also means the two conditions are NOT directly comparable. The d=256 run
operates in a regime where the base has some structure (from pre-training on 3 domains),
while d=3072 operates on a purely random base. These are qualitatively different learning
problems. The MATH.md claim that "quality_ratio is base-independent" is an assumption,
not a proven fact. Finding #365 established the baseline with a pre-trained base; this
experiment compares against a random base. The comparison is weakened.

**Mitigation:** The quality_ratio formula normalizes by the gap, which partially accounts
for this. And the PAPER.md acknowledges this as a caveat. But a strict comparison would
require the same protocol at both widths.

### 2. K897 evaluated on minimum, not median (MINOR, well-designed)

The code evaluates K897 as `min(quality_values) >= 0.85`, which is more conservative
than median. With min=85.94%, this is a barely-passing result. The PAPER.md headline
("90.0% median") is technically accurate but more optimistic than the actual kill criterion
evaluation. Both are correctly reported.

### 3. K898 FAIL is correctly analyzed but should not be dismissed (MINOR)

The 0.803 nats gap at d=3072 exceeds the 0.7 nats threshold. PAPER.md attributes this
to T/n_train = 1.0 (training for exactly 1 epoch) at the "Ghadimi-Lan boundary."
This analysis is reasonable but also reveals that the reduced training budget (n=500,
T=400) is at the edge of where the theoretical guarantees apply. A more cautious
interpretation: the experiment was underpowered for the d=3072 condition, and the
quality result should be treated as preliminary.

### 4. Output head parameter count: 227M for a toy task (ADVISORY)

The M2P at d=3072 has 227M parameters to generate adapters for sort/reverse of
2-5 character strings. This is extremely overparameterized for the task. The high
quality_ratio (90%) may simply reflect that any sufficiently large network can
memorize the training data. The experiment does not control for this (e.g., by
testing whether a random 227M-parameter projection achieves similar quality).

This is not blocking for the frontier extension question, but it means the result
tells us more about memorization capacity than about the rank-structure hypothesis.

## Novelty Assessment

The rank-structure insight (B.5) -- that max_rank of the joint B-stack is determined
by L*LORA_RANK and not by d_model -- is a clean, useful observation. It correctly
identifies why width scaling should be "free" for the M2P bottleneck.

The competing hypothesis framework (H1 vs H2) is good experimental design. Testing
H1 (Aghajanyan task-complexity) against H2 (width-scaling) with discriminating
predictions is the right approach.

The experiment is a straightforward extension of Finding #365 to a new d_model value.
Not novel per se, but correctly positioned as frontier extension.

## Macro-Scale Risks (advisory)

1. **Aghajanyan invariance at real-task scale.** The intrinsic dimensionality of
   sort/reverse is trivially low. Real NLP tasks (summarization, code generation,
   medical QA) will have much higher d_int. The result that d_M2P=64 suffices at
   d=3072 for toy tasks does not predict that d_M2P=64 will suffice for real tasks.
   PAPER.md acknowledges this in caveat 2. At macro scale, d_M2P will likely need to
   increase substantially.

2. **227M M2P for Qwen3-4B.** The M2P at d=3072 has 227M parameters -- roughly 5% of
   the base model size. At scale, this is a significant overhead. The compression ratio
   (27,648:1 for fc1) suggests the output head is doing heavy lifting. Whether this
   compression can maintain quality on diverse real tasks is the key macro question.

3. **Random-init base protocol.** All macro experiments will use pre-trained bases.
   The random-init protocol used here is a micro convenience. The quality_ratio
   comparison should be validated with a pre-trained base before claiming width
   scaling is closed.

## Verdict

**PROCEED** (with two advisory notes)

**Justification:** The experiment correctly identifies as Type 3 frontier extension.
The rank-structure proof (B.5) is mathematically sound and provides the right
theoretical lens. The competing hypothesis design (H1 vs H2) is well-structured.
The primary quality criterion (K897) passes. The K898 failure is correctly diagnosed
as an underpowered training regime, not a fundamental issue. The finding status
("supported") is appropriate for Type 3 with one secondary criterion failing.

**Advisory notes (not blocking):**

1. **Fix the Aghajanyan citation.** MATH.md Section B.3 should say "Key empirical
   finding (Aghajanyan et al.)" not "Core theorem (Theorem 1, Aghajanyan et al.)."
   The paper does not contain a formal theorem with this statement. This is the
   "description dressed as equation" pattern flagged in prior adversarial review.

2. **Acknowledge the K897 margin.** Sort quality at 85.94% is 0.94pp above the 85%
   threshold, which is within the observed 7pp variance of the d=256 replication.
   The finding should note that the PASS is marginal and may not be robust to
   seed variation. A follow-up with 3 seeds would strengthen confidence.
