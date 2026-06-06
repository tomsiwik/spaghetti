# Adversarial Review: exp_shine_session_adapter

## Verdict: REVISE

The experiment is directionally interesting and the code is well-structured, but there
are document-code inconsistencies, a potentially confounded experimental design, and
a failed prediction that is dismissed rather than investigated. Three blocking issues
and several non-blocking observations.

---

## Type Check

**Declared type:** Frontier Extension (Type 3). Correct.

MATH.md identifies the proven result (SHINE arXiv:2602.06358, Finding #336) and the
mathematical gap (full training loop convergence on toy model). The gap is clearly
stated and cannot be closed by proof alone. Type 3 is appropriate, and finding status
is correctly capped at provisional/supported.

**However:** The Self-Test Q2 claims "SHINE arXiv:2602.06358 Theorem: M2P training with
NTP loss converges when the base LM provides informative hidden states and the adapter
rank is sufficient." SHINE is an empirical paper. It does not contain a formal theorem
with proof stating convergence. It demonstrates convergence empirically. Citing an
empirical demonstration as a "theorem" is a misrepresentation. This should be corrected
to: "SHINE demonstrates empirically that..." This is non-blocking but affects
intellectual honesty.

---

## Hack Detector

- **Fix count:** 0. Clean single-mechanism experiment. No stacked fixes.
- **Is MATH.md a proof or a description?** Description of the SHINE mechanism plus
  a gap statement. Appropriate for Type 3 (frontier extension). No formal proof
  claimed, which is honest.
- **Metric used as evidence:** PPL ratio (M2P PPL vs SFT PPL improvement). PPL is
  used as a proxy for "adapter quality." On synthetic toy data with controlled bigram
  structure, PPL improvement is a valid signal for domain adaptation. Acceptable.
- **Kill criteria source:** K832 derived from the gap question (50% of SFT quality).
  K833 is an engineering bound (5s latency). Both reasonable.

---

## Self-Test Audit

**Q1 (Impossibility property):** "There is no impossibility structure. The failure mode
is possible. K832 measures whether it occurs." -- Honest and correct for Type 3. PASS.

**Q2 (Cited theorems):** Claims SHINE has a convergence theorem. SHINE is empirical, not
a formal convergence proof. This is a misrepresentation. The experimental result from
SHINE is real; calling it a "theorem" is not. PARTIAL FAIL -- correct the citation.

**Q3 (Predicted numbers):** Three predictions (D.1, D.2, D.3). D.1 is a convergence
bound (<0.5 loss ratio), D.2 is the K832 criterion, D.3 is a latency prediction (<100ms).
All specific and falsifiable. PASS.

**Q4 (Falsification):** Three clear falsification conditions. PASS.

**Q5 (Hyperparameters):** Claims 3 hyperparameters (LR, steps, M), all inherited from
prior work. But code uses different values than the prior work: M2P_STEPS=300 (not 200
as in MATH.md line 135 "matching SFT adapter training"); M2P_DIM=128 (not the 64 from
Finding #336). The actual experiment has more tuning degrees of freedom than acknowledged.
PARTIAL FAIL.

**Q6 (Hack check):** No stacked fixes. PASS.

---

## Mathematical Framework

### BLOCKING: MATH.md-to-Code Dimension Mismatch

MATH.md Section C.4 specifies the toy model as:

> Transformer LM: 4 layers, **d=256** hidden, 4 heads, **vocab=50**

The code (run_experiment.py line 46-47) uses:

```python
TOY_D = 128   # hidden dim
TOY_VOCAB = 65
```

PAPER.md (line 113) correctly reports d=128, vocab=65. The results.json confirms
hidden_dim=128. **MATH.md is wrong about the dimensions actually used.** This means:

1. The worked example in Section F (d=16) and the complexity analysis in Section G
   (d=256) both cite the wrong production dimensions.
2. The memory analysis claims "Toy LM (d=256, L=4, vocab=50): ~2M params = 8 MB"
   but the actual model is 804K params at d=128.
3. The adapter size calculation in C.4 ("LoRA adapter (d=256, r=4, 8 matrices):
   8 * 2 * 256 * 4 = 16K params") is for the wrong dimension. Actual adapter:
   8 * 2 * 128 * 4 = 8192 params = 32 KB.

This is a document-code desync. The code ran correctly (results are from d=128),
but MATH.md's analysis applies to a different model than what was tested.
**MATH.md must be updated to match the actual experiment.**

### M2P Size Confound (non-blocking but important)

M2P has 9.18M parameters. The toy LM has 804K parameters. M2P is 11.4x larger.

The M2P parameter breakdown:
- M2P Transformer backbone (4 blocks): ~1.57M params
- Positional embeddings: ~1.5K params
- **Per-layer projection heads (4 heads, each 1024->2048):** ~8.39M params (91.4%)

The vast majority of M2P is the projection heads, not the transformer. This means
the "Memory-to-Parameter Transformer" is really a "Memory-to-Parameter Linear Projection
with a small transformer preprocessor." The transformer processes a (4, 8, 128) memory
grid (~4K values) and produces a (4, 8, 128) output, which is then mapped through a
1024->2048 linear layer per LM layer to produce adapter weights.

This is not a bug -- the SHINE paper uses projection heads too -- but it changes the
interpretation. The 66.6% SFT quality could be largely attributable to the 8.39M-param
linear projection learning a fixed domain-specific adapter, with the transformer
contributing minimally. PAPER.md Limitation #3 acknowledges the size disparity but
does not investigate this specific confound.

**Advisory:** A control experiment removing the M2P transformer blocks (direct projection
from positional embeddings to adapter weights) would isolate whether the transformer
contributes at all. This is not blocking but would strengthen the finding considerably.

---

## Prediction vs Measurement

### BLOCKING: Prediction D.1 Failed -- Insufficiently Addressed

MATH.md Prediction D.1: "M2P training loss decreases monotonically over 200 steps
(ratio final/initial < 0.5)."

Measured: ratio = 0.894 (10.6% reduction, not 50%).

PAPER.md acknowledges this as "weak convergence" and says "Despite this weak
convergence, K832 PASSES." This is problematic:

1. **D.1 predicted <0.5, measured 0.894.** This is a clear prediction failure.
   The experiment had 3 predictions. One failed. The paper treats this as
   unimportant because the downstream metric (K832) passed anyway. But prediction
   failures should be investigated, not dismissed.

2. **Why did D.1 fail?** The paper does not investigate. Possible explanations:
   - 300 steps is insufficient (but the prediction said 200 steps)
   - Learning rate 1e-3 is wrong for this scale
   - The NTP loss through M2P is a weak training signal for this architecture
   - The M2P transformer is not contributing (see projection head confound above)

3. **How can K832 pass while D.1 fails?** If M2P loss only dropped 10.6% but the
   generated adapter captures 66.6% of SFT quality, this suggests the M2P was
   already generating useful adapters early in training. This would be consistent
   with the projection heads learning a fixed mapping and the transformer contributing
   minimally. The paper does not explore this hypothesis.

**Required fix:** PAPER.md must investigate the D.1 failure. At minimum, report:
(a) M2P PPL at step 0 (is the initial adapter already useful?), and (b) explain
the discrepancy between weak convergence and strong downstream performance.

### Prediction Table

| Prediction | Predicted | Measured | Match |
|-----------|-----------|----------|-------|
| D.1: loss ratio < 0.5 | < 0.500 | 0.894 | FAIL |
| D.2: M2P PPL < threshold | < 57.76 | 50.65 | PASS |
| D.3: gen time < 100ms | < 100ms | 1.15ms | PASS |
| E.1: domain-discriminative hiddens | cos < 0.99 | 0.42 | PASS |
| E.2: SFT shows improvement | PPL < 0.95x base | 0.459x base | PASS |

PAPER.md does contain this table. The D.1 FAIL is honestly reported but inadequately
investigated.

---

## Kill Criteria Assessment

**K832 (50% of SFT quality):** The threshold itself is reasonable for a frontier
extension. The measurement (66.6%) is above threshold (50%). However:

1. The threshold assumes SFT is a meaningful ceiling. SFT PPL = 36.36 vs base
   PPL = 79.16, a 54.1% reduction on synthetic bigram data. This is a huge
   improvement on trivially structured data. The absolute PPL values (79 and 36
   on vocab=65 character-level data) suggest the base model has not converged
   on the training distribution, and SFT dramatically fits it.

2. On real text with a pre-trained LM, the delta would be much smaller (cf.
   Finding #333: 13.4% improvement). The 50% threshold on a 54% improvement
   is much easier to hit than 50% of a 13.4% improvement.

3. **Single-context evaluation:** K832 is measured using a single context sequence
   (train_data[0]) to generate the adapter. There is no variance estimate. The
   result could be context-dependent. Running evaluation with multiple contexts
   and reporting mean +/- std would be more rigorous.

**K833 (< 5s generation):** A 1.15ms measurement against a 5s threshold is 4,347x
margin. This is not a meaningful kill criterion -- it would pass with any architecture
on this hardware at this scale. It provides zero discriminative signal.

---

## Experimental Design Issues

### 1. No statistical uncertainty on K832

The K832 measurement (M2P PPL = 50.65) is reported as a point estimate. There is no
confidence interval or variance estimate. The experiment uses a single trained M2P,
a single context for adapter generation, and a single evaluation on 50 val sequences.

Without variance, we cannot assess whether the result is robust. If M2P PPL has high
variance across contexts or random seeds, the 66.6% result may be unstable.

**Recommendation:** Run with 3 different contexts and report mean +/- std of M2P PPL.

### 2. M2P trains longer than SFT but doesn't converge

SFT trains for 200 steps. M2P trains for 300 steps (50% more). MATH.md (line 295)
says "matching SFT adapter training" about the 200 steps, but the code uses 300.
The extra steps give M2P an advantage in optimization time. Despite this, D.1 still
fails.

### 3. Hidden state separation check is too weak

Assumption E.1 uses cos < 0.99 as the threshold for "informative hidden states."
The measured value is 0.42, which is well below the threshold. But the threshold
is essentially vacuous -- nearly any non-degenerate model would produce hidden
states with inter-domain cosine < 0.99. A more meaningful check would compare
inter-domain vs intra-domain cosine similarity, which the code computes for
inter-domain only (no intra-domain measurement).

### 4. Synthetic data is too easy

The synthetic domains have extremely distinctive bigram patterns (even-to-odd
transitions for "medical," mod-4 skips for "code"). The inter-domain cosine of
0.42 shows the domains are very well-separated. This is far easier than real text
domains where the hidden state distributions would overlap substantially.

On such clean synthetic data, even a linear projection (the 91.4% of M2P that is
projection heads) could learn to map domain-specific hidden states to useful adapters.
The experiment does not demonstrate that the M2P *transformer* is necessary.

---

## Novelty Assessment

The experiment is a direct application of SHINE (arXiv:2602.06358) to a toy setting.
No novelty is claimed and none exists. The contribution is: "SHINE training loop
converges on a toy LM in MLX." This is appropriate for an infrastructure validation
experiment.

---

## Macro-Scale Risks (advisory)

1. **M2P projection head scaling:** At d=2048 (Qwen3-4B), each projection head
   would be (M*m2p_dim) -> (4*d*r) = (8*m2p_dim) -> (4*2048*4) = 32,768 outputs.
   If m2p_dim matches the LM dim (m2p_dim=2048), each head is 16K -> 32K = 524M
   params. Four heads = 2B params just for projection. This is explicitly infeasible
   as acknowledged in PAPER.md. The SHINE paper's approach of shared projection
   across layers is the known solution.

2. **Real text domain separation:** Toy domains have cosine 0.42. Real text domains
   (medical vs legal vs code) may have higher cosine in pre-trained LM hidden
   states. The M2P may not receive enough domain signal.

3. **Single-domain M2P:** This experiment trains a separate M2P per domain. The SHINE
   vision is a single M2P that generates domain-appropriate adapters for *any* context.
   The current experiment does not test this -- it tests a much easier problem
   (overfit M2P to one domain's contexts).

---

## Required Fixes (REVISE)

### Blocking Fixes

1. **MATH.md dimension correction (blocking).** Update Section C.4 to reflect the
   actual model: d=128, vocab=65, ~804K params. Update the worked example (Section F)
   and complexity analysis (Section G) accordingly. The adapter parameter count,
   memory analysis, and M2P overhead calculations all cite wrong numbers.

2. **Investigate D.1 prediction failure (blocking).** PAPER.md must explain why D.1
   failed (loss ratio 0.894, not <0.5) while K832 passed (66.6% of SFT). At minimum:
   - Report M2P PPL at step 0 (before any training). If the initial adapter already
     helps, the result is about initialization, not training.
   - State explicitly whether this falsifies the convergence prediction or whether the
     prediction was poorly calibrated.

3. **Self-Test Q2 correction (blocking).** Replace "SHINE arXiv:2602.06358 Theorem"
   with "SHINE arXiv:2602.06358 demonstrates empirically." SHINE does not contain a
   formal convergence theorem.

### Non-Blocking Recommendations

4. **Report M2P PPL variance across contexts.** Use 3-5 different context sequences
   to generate adapters and report mean +/- std of M2P PPL. This would show whether
   K832 is robust.

5. **Ablation: remove M2P transformer blocks.** Run a control where the projection
   heads receive fixed positional embeddings (no transformer processing). If K832
   still passes, the transformer is not contributing.

6. **MATH.md Self-Test Q5:** Acknowledge that M2P_STEPS was changed from 200 (as
   predicted in D.1) to 300 (as implemented). This is an unacknowledged
   hyperparameter change that makes the prediction harder to evaluate.
