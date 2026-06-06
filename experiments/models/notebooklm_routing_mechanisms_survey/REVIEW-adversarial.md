# Peer Review: Routing Mechanisms Survey (NotebookLM)

## NotebookLM Findings

Skipped. This is itself a NotebookLM-powered survey, not an empirical experiment. The review focuses on verifying mathematical claims, citation accuracy, and whether recommendations are properly justified.

## Mathematical Soundness

### Angular Concentration Bound (MATH.md Section 0)

The bound cited is:

> max_{i!=j} |cos(w_i, w_j)| >= sqrt((N - d) / (d(N-1)))

This is presented as coming from L2R (arxiv 2601.21349). The bound is a variant of the Welch bound / Rankin bound for equiangular lines. Let me check the arithmetic:

- At d=64, N=100: sqrt((100-64)/(64*99)) = sqrt(36/6336) = sqrt(0.00568) = 0.0754. **Correct.**
- At d=64, N=500: sqrt((500-64)/(64*499)) = sqrt(436/31936) = sqrt(0.01365) = 0.1168. **Correct.**

**Issue 1: The bound is applied incorrectly in context.** This bound says the *maximum pairwise cosine among N random unit vectors* must be at least this value. But router weight vectors are *learned*, not random. Learned routers can achieve better packing than random vectors (up to the Welch bound itself, which is the *minimum possible* maximum cosine). The survey conflates a random-vector lower bound with a fundamental limit on routing discriminability. The actual Welch bound (the theoretical best packing) is:

max |cos| >= sqrt((N-d) / (d(N-1)))

This IS the Welch bound -- it applies to *any* set of N vectors in R^d, not just random ones. So the bound is actually correct as a fundamental limit, not just a random-vector statement. My initial objection was wrong. The minimum achievable maximum pairwise cosine for any N unit vectors in R^d cannot be smaller than this. **The bound holds and is applied correctly.**

**Issue 2: The practical significance is overstated.** At d=64, N=100, the bound gives max |cos| >= 0.075. This is a tiny cosine -- for comparison, the project's own orthogonality threshold is 0.05. The survey argues this "means router embeddings become increasingly indistinguishable" but 0.075 is well within discriminable range. The real problem would emerge at much larger N or much smaller d. At d=64, N=500, max |cos| >= 0.117 -- still quite small. The angular concentration argument for SIPS is theoretically sound but the urgency at N=100 is overstated.

**Issue 3: d_route confusion.** The bound is computed at d_route=64, but the current router operates at d=2560 (full hidden dim). At d=2560, N=100: sqrt((100-2560)/(2560*99)) is imaginary (N < d), meaning the bound is vacuous -- there exist perfect orthogonal packings. The angular concentration problem only emerges when N > d_route. With d=2560 and N=100, there is no theoretical barrier. SIPS becomes relevant only if a low-rank projection (r_route << N) is used, which creates the very problem it solves. This is a circular argument: "we need SIPS because low-rank projection concentrates angles, and SIPS uses low-rank projection."

**Verdict on angular concentration argument:** The math is correct but the framing obscures a crucial detail. At the current routing dimensionality (d=2560), angular concentration is not a problem for N=100 or even N=500. It becomes a problem only if routing is projected to a low-rank space (r_route=16 as proposed). SIPS mitigates a problem introduced by its own architectural choice. This does not invalidate R1 -- there are good reasons to use low-rank projection (parameter efficiency, FLOP reduction) -- but the motivation should be "low-rank routing is cheaper AND SIPS keeps it accurate" not "angular concentration forces us to adopt SIPS."

### FLOP Calculations (Section 4)

- Softmax at N=100, d=2560: O(d*N) = O(256,000). **Correct.**
- SIPS at r_route=16: O(d*r_route + N*r_route) = O(40,960 + 1,600) = O(42,560). **Correct.**
- "6x cheaper" claim: 256,000 / 42,560 = 6.01. **Correct.**
- "30x at N=500": O(2560*500) / O(2560*16 + 500*16) = 1,280,000 / 48,960 = 26.1x. **The claim of 30x is wrong; the actual ratio is ~26x.** Minor overstatement.

### Worked Example (Section 5)

- Step 1: W_proj in R^{8x64}, z = W_proj @ x. Cost 64*8=512. **Correct.**
- Step 2: w_i^T z costs 8 FLOPs each, 8 experts = 64. **Correct.**
- Step 3: Gumbel sampling + adds = 16. **Reasonable approximation.**
- Step 4: 8 sigmoid ~40 FLOPs. **Reasonable (sigmoid is ~5 FLOPs each).**
- Total: 632 FLOPs. **Consistent with above.**
- Expert forward: 2*64*4=512. **Correct for rank-4, d=64 toy example.**
- Routing/expert ratio at k=2: 632/1024 = 0.617. **Correct.**

The scale-up calculation has an error: "2560*16 + 100*16 + 100 + 100*5 = 40,960 + 1,600 + 600 = 43,160." The 100 (Gumbel samples) + 100*5 (sigmoid) = 600 is consistent, but 40,960 + 1,600 + 600 = 43,160. **Correct.**

### Memory Budget Table (Section 3.3)

The table claims 45.2 MB runtime buffer per adapter. At N=853 this gives 38.6 GB. But ternary adapter *weights* are ~1.9 KB -- the 45.2 MB figure appears to be runtime activation memory, not weight storage. The paper should clarify whether all 853 adapters need simultaneous runtime buffers or only the top-k selected ones (k=2). If only k=2 are active, runtime buffer cost is ~90 MB regardless of N, making the table misleading.

### Grassmannian Collapse Bound

The statement "P(collapse) <= exp(-c * min_gap(Lambda))" attributed to arxiv 2602.17798 is presented without derivation. The Bingham distribution concentration is a known result in directional statistics, so the form is plausible. However, the constant c is unspecified, making the bound non-falsifiable. Without knowing c, one cannot verify whether the guarantee is tight or vacuous at practical parameter settings.

### Gumbel-Sigmoid Collapse Probability

"P(all-same) ~ (1/2)^(N-1) = 2^{-24} ~ 6e-8 at N=25." This assumes balanced logits (each gate independently 50/50), which is a strong assumption. In practice, learned routers do not produce balanced logits. If one expert's logit is significantly higher, collapse probability increases. The bound is correct under stated assumptions but the assumptions are unrealistic.

## Novelty Assessment

This is a survey, not a novel contribution. The value lies in synthesis and recommendations, not in new methods. The key question is whether the synthesis is accurate and complete.

**Missing alternatives:**

1. **Mixture-of-Depths (arxiv 2404.02258)** -- tokens can skip entire layers, a form of dynamic compute allocation. Relevant to the entropy gating discussion (R3) and not cited.

2. **Expert Choice routing (arxiv 2202.09368, Zhou et al.)** -- experts choose tokens instead of tokens choosing experts. Achieves perfect load balance by construction. Not considered despite being directly relevant to F2 (load imbalance).

3. **Soft MoE (arxiv 2308.00951, Puigcerver et al.)** -- fully differentiable routing via soft assignment. Eliminates discrete selection entirely. Relevant as a contrast to the Gumbel approach.

These omissions do not invalidate the recommendations but leave the survey incomplete.

**Prior art correctly identified:** The survey properly identifies L2R, Grassmannian MoE, LD-MoLE, and CoMoL as the most relevant mechanisms. The elimination of X-LoRA, PHATGOOSE, SpectR, and CLONE is well-justified based on proven findings (per-layer routing killed, Metal compatibility requirements).

## Experimental Design

As a survey, the "experiment" is the quality of analysis and recommendations. The kill criterion (K1: 3+ actionable recommendations with arxiv citations) is appropriate for a survey.

**Are there 5 actionable recommendations?** Yes:
- R1 (SIPS + Gumbel-sigmoid): Cited, implementable, clear kill criterion. **Actionable.**
- R2 (Hierarchical cluster-then-route): Cited, builds on proven Finding #116, clear kill criterion. **Actionable.**
- R3 (LD-MoLE dynamic-k): Cited, clear delta over current entropy gating, clear kill criterion. **Actionable.**
- R4 (CoMoL core-space): Cited, but acknowledged as speculative/architectural change. **Borderline actionable** -- the incompatibility with Grassmannian skeleton is correctly flagged but the "hybrid approach" is hand-waved.
- R5 (MoLoRA per-token): Cited, but explicitly marked low-priority and conditional on mixed-domain data that does not yet exist. **Actionable only conditionally.**

**Conservative count: 3 clearly actionable (R1-R3), 2 conditional (R4-R5).** K1 threshold of 3+ is met.

**Concern: Kill criteria are self-serving.** Each recommendation has a kill criterion, but they are phrased to be hard to trigger:
- R1 killed only if "Gumbel-sigmoid alone maintains oracle-matching quality at N=100 without SIPS." This requires proving a negative (that angular concentration is NOT a problem).
- R3 killed only if the expert-count distribution is bimodal. This is empirically falsifiable, which is good.
- R4's kill criterion (within-cluster cosine > 0.05) is concrete and testable. Good.

### Does the Survey Account for Proven Results?

**Gumbel-sigmoid:** Yes, thoroughly integrated. The survey correctly positions it as the proven baseline and frames recommendations as extensions.

**Entropy gating:** Yes, R3 explicitly proposes upgrading it.

**Grassmannian skeleton:** Yes, correctly identified as an incompatibility with CoMoL (R4) and a natural fit for R2 (hierarchical routing).

**Per-layer routing killed:** Yes, correctly used to eliminate X-LoRA, PHATGOOSE, SpectR.

**One gap:** The survey does not engage with Finding #118 ("Routing moot without specialization") at sufficient depth. If expert quality matters more than routing quality, then optimizing the router (R1-R3) may yield diminishing returns. The survey lists this finding but does not discuss its implications for the recommendations.

## Macro-Scale Risks (advisory)

1. **SIPS hyperparameter C:** The saturation bound C is a critical hyperparameter with no guidance on how to set it. Too small = information loss, too large = no benefit. This will need careful tuning at macro scale.

2. **The circular motivation for SIPS** (see Mathematical Soundness): At d=2560, angular concentration is not a problem. SIPS becomes necessary only if low-rank projection is adopted first. The macro experiment should test whether full-rank Gumbel-sigmoid works at N=100 before introducing SIPS.

3. **CoMoL hybrid (R4)** is entirely speculative. Any macro implementation would be novel research, not validation of a published method.

4. **Runtime buffer accounting:** The survey's memory table assumes all N adapters need simultaneous buffers. If only top-k are active, the scaling story changes entirely.

## Verdict

**PROCEED**

The survey meets its kill criterion (3+ actionable recommendations with arxiv citations), the mathematical analysis is largely correct (with caveats below), eliminated mechanisms are correctly dismissed based on proven findings, and the recommendations logically build on the existing architecture.

**Required acknowledgments (not blocking, but should be noted):**

1. The angular concentration argument for SIPS (R1) is circular at d=2560. The honest framing is: "low-rank projection reduces routing cost; SIPS prevents accuracy loss from the lower dimensionality." Fix in MATH.md Section 0 or PAPER.md R1 motivation.

2. The "30x at N=500" claim should be corrected to ~26x.

3. The memory table should clarify whether runtime buffers scale with N (all loaded) or k (only active). This materially affects the N=853 feasibility analysis.

4. Finding #118 (routing moot without specialization) deserves explicit engagement -- if expert quality is the bottleneck, routing optimization has bounded upside.

5. Mixture-of-Depths, Expert Choice, and Soft MoE should be acknowledged as surveyed-but-excluded (with brief justification) rather than simply missing.

These are improvements to the survey document, not blockers for proceeding with the recommended experiments (R1-R3).
