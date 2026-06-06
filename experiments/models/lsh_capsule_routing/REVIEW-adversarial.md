# Peer Review: LSH Capsule Routing (Post-Revision)

## Review Context

This is a post-revision review. The first review issued a REVISE verdict with 6 fixes (2 blocking, 4 non-blocking). This review assesses whether all fixes were adequately applied and whether the experiment is now sound.

## Revision Checklist

| Fix | Required | Status | Assessment |
|-----|----------|--------|------------|
| 1. Add softmax-no-balance-loss control | Blocking | DONE | Correctly implemented via `zero_aux_loss` monkey-patch. Cleanly removes the confound. |
| 2. Add uniform routing baseline | Blocking | DONE | Uses `capsule_moe_uniform` model. Establishes the correct floor. |
| 3. Correct FLOP scaling table | Non-blocking | DONE | MATH.md Section 3.3 and PAPER.md FLOP section now clearly distinguish actual implementation cost O(T*G*d) from theoretical binary LSH cost O(T*d). Tables show the 4x ratio at all N for the actual implementation. |
| 4. Correct isotropy assumption | Non-blocking | DONE | MATH.md Section 5.1 now explicitly states "post-RMSNorm activations are NOT isotropic" and Section 6, Assumption 2 correctly characterizes RMSNorm as norm-normalization without isotropy guarantee. Utilization std 0.29-0.33 is now correctly interpreted as evidence of anisotropy. |
| 5. Report paired statistical test | Non-blocking | DONE | `paired_ttest` function implemented in `run_experiment.py`. Results reported in PAPER.md with p-values for all configs vs softmax_no_bal. No config achieves p<0.05. |
| 6. Soften language | Non-blocking | DONE | PAPER.md title is now "Research Digest (Revised)". Claims changed from "LSH beats softmax" to "statistically indistinguishable." HYPOTHESES.yml third evidence entry reflects the honest null finding. |

All 6 fixes applied. The blocking fixes are correctly implemented.

## NotebookLM Findings

Skipped -- the material is tractable for direct manual review and all key documents have been read in full.

## Mathematical Soundness

### What holds

1. **SimHash foundation (Section 1).** Standard result, correctly cited. The argmax-over-G-projections extension to Voronoi partitioning is correct.

2. **Vote accumulation (Section 2).** The sum constraint sum_g v_g(x) = T is trivially correct. The lexicographic ranking (votes * 1000 + score) is standard and correctly implemented in code (line 128 of lsh_capsule_routing.py).

3. **Gradient flow analysis (Section 2.3).** Correctly identifies that selection is non-differentiable but weighting and expert computation are differentiable. The analogy to Switch Transformer top-1 routing is apt.

4. **Worked example (Section 2.4).** Spot-checked: s_0 = 0.21 + 0.07 + 0.25 + 0.36 = 0.89, s_1 = 0.19 + 0.38 + 0.09 + 0.04 = 0.70. Softmax: exp(0.89)/(exp(0.89)+exp(0.70)) = 2.435/4.449 = 0.547, rounds to 0.548. Correct.

5. **Parameter count (Section 4).** 4 layers * 8 * 64 = 2,048 routing params saved. 202,112 vs 204,160 = 1.0% difference. Verified against results.json param counts.

6. **Corrected isotropy discussion (Section 5.1, 6).** Now correctly states RMSNorm does not guarantee isotropy, and the utilization std of 0.29-0.33 is evidence of anisotropy. This was the key mathematical error in the original submission and is now fixed.

7. **FLOP comparison (Section 3.3).** The corrected table honestly shows the 4x ratio (T:1) at all N for the actual implementation. The binary LSH scaling is clearly labeled as "not implemented" and "theoretical." This was the other key correction and is now handled properly.

### Remaining minor issue

The module docstring in `lsh_capsule_routing.py` (lines 2-3 and line 232) still says "O(T*d) time, independent of the number of experts N" and "Routing cost is O(T*d) independent of N." This contradicts the corrected MATH.md and PAPER.md, which both acknowledge the actual cost is O(T*G*d). This is cosmetic (comments in code, not claims in the paper) but creates confusion for anyone reading the source. Not blocking.

## Novelty Assessment

### Prior art

- **Hash Layers (Chen et al., NeurIPS 2021):** Direct prior art. Demonstrated hash-based routing competitive with Switch Transformer at scale. The LSH experiment applies random (not learned) hash functions to capsule groups. Correctly cited.
- **PEER (He et al., DeepMind 2024):** Product-key retrieval for 1M+ experts. More sophisticated structured hashing. Correctly cited.
- **SimHash (Charikar 2002):** Foundation. Correctly cited.

### Delta over existing work

The delta is modest: applying random-projection LSH (rather than learned hash functions as in Hash Layers) to capsule groups (rather than FFN experts). The revised paper correctly frames this as a micro-scale validation of a known principle, not a novel contribution. The honest null finding ("all routing is equivalent at G=8") is itself the contribution -- it tells us where to look next.

## Experimental Design

### What the revision got right

1. **The softmax-no-balance-loss control is decisive.** It reveals that the original "LSH beats softmax" claim was entirely explained by the balance loss handicap. With balance loss removed, softmax matches LSH. This was the critical missing control and it fundamentally changed the narrative. The researcher handled this honestly.

2. **The uniform baseline is devastating (in a good way).** Uniform routing at -0.85% vs softmax_no_bal (p=0.295) proves that routing quality is irrelevant at G=8 with homogeneous data. This transforms the experiment from "LSH is good" to "routing does not matter at this scale" -- a much more honest and useful finding.

3. **Statistical tests are correctly reported.** No config achieves p<0.05 with 3 seeds. The paper correctly states this and avoids overclaiming. The directional p-values (T=1 at p=0.088, T=4 at p=0.106) are noted but not misinterpreted.

4. **The three-factor explanation (Section "Why Results Are Indistinguishable") is well-reasoned.** G=8 too small, homogeneous data, short training -- all plausible and honestly stated.

### Remaining design considerations (non-blocking)

1. **The paired t-test implementation (lines 41-70 of run_experiment.py) uses a closed-form formula for df=2 only.** This is correct for n=3 seeds but would fail silently for other seed counts (the normal approximation fallback on line 69 is rough). Since the experiment was only run with 3 seeds, this is acceptable but fragile.

2. **The `zero_aux_loss` monkey-patch (line 116-118, applied line 192-193) is clean** -- it replaces `model.aux_loss` with a method returning 0.0, ensuring the softmax_no_bal and all LSH models train on identical loss functions. This correctly removes the balance loss confound.

3. **All models use the same `mx.random.seed(seed)` before construction (line 187).** This means the random projections in the LSH router and the initial weights of all models are deterministic per seed. However, different models have different architectures, so the random state after construction differs. This is standard practice and not a concern.

## Hypothesis Graph Consistency

The HYPOTHESES.yml node `exp_lsh_capsule_routing` has:
- Kill criterion 1: "LSH routing quality >3% worse than learned softmax routing" -- PASSES (best LSH is -1.34% better, worst is -0.33% better)
- Kill criterion 2: "LSH requires >4 hash tables to match softmax quality" -- PASSES (T=1 suffices)
- Status: "proven"
- Third evidence entry correctly reflects the revised finding

The "proven" status is appropriate given the kill criteria. The kill criteria ask "is LSH routing viable?" not "is LSH routing superior?" The null finding (all routing equivalent) means LSH passes the viability threshold -- it does not degrade quality.

## Macro-Scale Risks (advisory)

1. **Diverse data will differentiate routing methods.** The paper correctly predicts this. At G=64+ with code/prose/math, learned routing should outperform random partitions. This is the primary macro risk.

2. **Load imbalance scales with G.** Utilization std 0.29-0.33 at G=8 with 4 tables. At G=256, the variance of random Voronoi cell sizes will create extreme imbalance. No balance loss can fix this since routing is fixed. Some experts may receive near-zero traffic.

3. **The O(T*d) scaling advantage requires binary LSH.** The paper now correctly states this. Implementing binary LSH (sign(r^T x) producing bit vectors with precomputed hash tables) is a necessary next step before any macro FLOP claims.

4. **Non-stationary representations.** Hidden-state distributions shift during training. Fixed hash functions that partition well at initialization may produce poor partitions later. This is acknowledged but untested.

## Verdict

**PROCEED**

The revisions are thorough and intellectually honest. All 6 fixes from the first review were applied correctly. The two blocking fixes (softmax-no-balance-loss control, uniform baseline) fundamentally changed the narrative from an overclaimed positive result to an honest null finding, which is the correct interpretation of the data. The corrected FLOP analysis, isotropy discussion, statistical tests, and softened language all meet the standard set by the first review.

The one remaining cosmetic issue (module docstring still claims O(T*d)) should be fixed but does not block proceeding.

The experiment establishes that at micro scale (G=8, homogeneous character-level data, 500 steps), routing quality is irrelevant -- all methods from uniform to learned softmax to LSH produce statistically indistinguishable results. This is a useful null result that correctly identifies where to look next (larger G, diverse data, binary LSH implementation). The kill criteria are legitimately passed: LSH routing is viable (not degraded) and requires only T=1.

The architectural value claim (zero calibration, zero routing parameters, instant expert addition) remains valid in principle and awaits macro-scale validation with diverse data where routing quality actually matters.
