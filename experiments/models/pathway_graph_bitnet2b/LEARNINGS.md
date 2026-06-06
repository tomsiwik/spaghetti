# LEARNINGS: Pathway Graph Construction on BitNet-2B

## 1. What We Learned

**0-dim PH on sparsified co-activation graphs does not detect meaningful structure.**
The random baseline control showed that graphs with the same edge density and weight
distribution produce MORE high-persistence features than the real data. The entire
persistence result was a sparsification artifact.

**The impossibility structure:** Any graph sparsified at the Nth percentile creates
~(1-N/100)*k disconnected components. These components then merge during the PH
filtration, producing high-persistence features regardless of whether the underlying
graph has meaningful structure. This is a fundamental methodological problem with
the approach, not specific to BitNet-2B.

**What IS meaningful:** The domain structure analysis (inter-domain cosine similarities)
showed real signal: legal-finance cluster (cos=0.44), code-finance most dissimilar
(cos=-0.51). This independently confirms Finding #217's domain categorization.

## 2. How It Connects to Prior Work

| Finding | Connection |
|---------|-----------|
| #217: Domain categories | **Independently confirmed**: legal-finance cluster in activation space |
| 1812.09764 (Neural Persistence) | Used weight-space filtrations, NOT sparsified co-activation. Different approach may work. |
| 2506.01042 (Neural Topology Probing) | Builds graph representations differently. Our construction may be wrong. |

## 3. What It Means for the Architecture

The pathway preservation framework (sheaf theory + PH) needs a fundamentally different
graph construction method before it can be applied. The current approach (SVD projections
→ co-activation → sparsification → PH) fails at the graph construction step.

Possible fixes:
1. **Weight-space PH** (as in 1812.09764): compute PH directly on weight matrices, not activations
2. **Rips complex on activation POINTS** (not co-activation): treat each input's activation
   vector as a point in R^d, build Rips complex directly
3. **Higher-dimensional PH**: H_1 (loops) and H_2 (voids) may capture structure that
   H_0 (components) cannot
4. **No sparsification**: use the full weighted graph with a filtration that doesn't
   require artificial edge removal

## 4. What Surprised Us

1. **Random baseline kills the finding completely.** The reviewer's concern was exactly right.
   Without the baseline, 91 high-persistence features looked compelling.
2. **Persistence rank is independent of SV rank (rho=-0.07).** Even though the persistence
   values are artifacts, the RANK ordering is completely independent of spectral importance.
   This suggests something interesting MAY be there but our measurement method is wrong.
3. **The adversarial review was more valuable than the experiment itself.** The random
   baseline control should have been in the original design.

## 5. What We'd Do Differently

1. **Always include a random baseline for PH experiments.** This is now a mandatory control.
2. **Don't sparsify.** Use the full weighted graph with a Vietoris-Rips or sublevel
   filtration that doesn't require artificial edge removal.
3. **Start with weight-space PH**, which has more established methodology (1812.09764).
4. **Test multiple layers** to understand if topology varies across the network.

## 6. NotebookLM Consultation

Skipped — the kill was clear from the random baseline control. No NotebookLM query needed.

## 7. Recommended Follow-ups

1. **exp_persistence_diagram_diff** (P1, next in pipeline) — may need redesign:
   use weight-space PH instead of co-activation PH
2. **Weight-space neural persistence** — apply 1812.09764's filtration directly to
   BitNet-2B weight matrices (no activation collection needed)
3. **Activation-space Rips complex** — treat activation vectors as points, compute
   PH on the point cloud directly (no SVD, no co-activation graph)
