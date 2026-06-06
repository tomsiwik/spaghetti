# Peer Review: OSRM-Constrained Adapter Init

## NotebookLM Findings
Skipped -- no PAPER.md exists; the experiment has only MATH.md, code, and results. The material is compact enough to review directly.

## Mathematical Soundness

**MATH.md derivations: Correct.** The OSRM algorithm is faithfully implemented from arXiv:2505.22934. The leave-one-out covariance S_{-i}, eigendecomposition, and selection of bottom-r eigenvectors all match the paper's prescription. The cross-activation metric is properly normalized.

**Implementation matches math: Verified.** Lines 245-258 of `run_experiment.py` implement `np.linalg.eigh(S)` (ascending order) and select `eigenvectors[:, :rank]` (bottom-r), which is correct. Covariance computation at lines 176-180 properly centers and divides by (k-1).

**One subtle issue: same A across all layers.** Lines 658-677 apply a single A matrix (computed from final-layer hidden states) to all 30 transformer layers. OSRM's covariance is computed from final-layer mean-pooled representations. Intermediate layers have different feature distributions, so the "minimal cross-domain variance" guarantee strictly applies only to the final layer. This is acknowledged implicitly ("for simplicity") but should be flagged: a per-layer covariance approach could yield different results. However, this is a limitation shared across all three conditions and does not invalidate the comparison.

**down_proj escape hatch.** Lines 662-668: `mlp.down_proj` (d_in=6912) gets random init in ALL conditions, including OSRM. This is correct -- the covariance was computed in d=2560 space -- but it means a substantial fraction of LoRA parameters (the down_proj adapters) are identical across conditions. This dilutes any signal from init differences on the attention/FFN-up/gate projections.

**Hidden state extraction is sound but coarse.** 100 samples per domain, mean-pooled over sequence length, from the final norm layer. This is a reasonable approximation for covariance estimation at micro scale. The eigenspectrum analysis (bottom-r/top-r ratio) is a good diagnostic.

## Novelty Assessment

**Prior art: Finding #68 already predicted this result.** The MATH.md itself cites finding #68, which showed that weight-space orthogonality does not imply data-space orthogonality, and that composition works via constructive cross-domain transfer rather than orthogonal isolation. The experiment was designed knowing this and proceeded anyway to test whether *data-aware* orthogonality (OSRM) could succeed where *geometry-only* orthogonality (Grassmannian) failed.

**The experiment is a reasonable follow-up to #68**, not a repetition. Finding #68 showed weight-orthogonality fails to produce data-orthogonality. OSRM directly targets data-orthogonality via covariance constraints. Testing whether this "fixes" the gap is a natural next step. The result that it does NOT fix it (cross-activation 15% lower but PPL unchanged) is a genuinely useful finding.

**OSRM paper (arXiv:2505.22934) context.** The OSRM paper reports +12.78% improvement for multi-adapter composition. The key difference: OSRM was tested with adapters trained on *disjoint* task types where cross-domain interference is the dominant failure mode. In this experiment, the 5 domains may share enough linguistic structure that cross-activation is not the bottleneck -- 1/N averaging already regularizes sufficiently.

## Experimental Design

**Controls are adequate.** Three conditions (random QR, Grassmannian AP, OSRM), same training procedure, same data, same hyperparameters. The only variable is A-matrix initialization. This is a clean comparison.

**Kill criteria are appropriate and correctly evaluated.**
- K1 (individual quality not degraded): PASS. OSRM adapters match random within 1%.
- K2 (merge quality better than random): FAIL. OSRM merge is 0.8% *worse* than random.
- The kill on K2 is justified.

**The cross-activation diagnostic is informative.** OSRM achieves 15% lower pre-training cross-activation (0.071 vs 0.084) but this does not translate to better composed PPL. This is the key finding: reducing cross-activation at init time is insufficient because (a) B matrices are unconstrained and learn to compensate, and (b) cross-activation may not be the bottleneck for composition quality at this scale.

**Potential confound: frozen A.** A matrices are frozen during training (only B is trained). This is a deliberate design choice matching the OSRM protocol, but it means the experiment tests "constrained init with frozen A" rather than "constrained init with trainable A". If A were trainable, all three conditions might converge to the same solution regardless of init. The frozen-A choice actually gives OSRM its *best* chance to show a difference, making the negative result more convincing.

**Missing control: What if all three A inits are functionally equivalent at d=2560, rank=16?** The ratio rank/d = 16/2560 = 0.6% means any rank-16 subspace captures a negligible fraction of the total variance. The MATH.md correctly identifies this risk in "What Could Break This" point 2 (flat eigenspectrum). The eigenspectrum data is collected but not reported in results.json -- this would have been valuable to include to confirm or deny the "flat spectrum" hypothesis.

**Composition method: arithmetic mean.** Lines 543-545 merge via `mx.mean(stacked, axis=0)`, which averages both A and B matrices. This is the standard 1/N uniform merge. Averaging A matrices from different init methods (random A1 + random A2 vs. OSRM A1 + OSRM A2) produces different merged subspaces, but at rank 16 in d=2560, the merged A is a low-rank approximation regardless.

## Macro-Scale Risks (advisory)

1. **OSRM covariance computation does not scale.** Computing d x d covariance matrices (d=2560 here, d=4096+ at scale) requires O(d^2) memory and O(kd^2) compute per domain. With many adapters, this becomes expensive. However, this is moot since the mechanism itself failed.

2. **The finding that init method does not matter at micro scale needs verification at macro.** At larger rank (r=64+) or with more domain-diverse adapters, the init subspace might matter more. The eigenspectrum ratio would be the diagnostic to watch.

3. **The 1/N merge strategy itself may be more important than any init strategy.** Finding #68's "constructive cross-domain transfer" hypothesis suggests the merge mechanism, not the init, is what matters for composition quality. Future work should focus there.

## Verdict

**KILL -- justified.**

The kill is well-supported by the evidence:

1. K2 fails cleanly: OSRM composed PPL (8.38) is worse than random (8.31), not better. The margin is small (-0.8%) but in the wrong direction.
2. The pre-training cross-activation reduction (15% lower for OSRM) does not translate to improved composition, confirming MATH.md's own "What Could Break This" predictions.
3. Finding #68 already established that orthogonal isolation is not the mechanism behind successful composition. This experiment extends that finding from weight-space orthogonality to data-space orthogonality: neither matters because composition works via constructive transfer, not interference avoidance.
4. All three init methods produce nearly identical results (within 1% individual, within 1% composed), suggesting that at rank=16/d=2560, the specific choice of rank-16 subspace for A is irrelevant to final quality.

**Key finding to record:** Data-aware A-matrix initialization (OSRM) reduces cross-domain activation by 15% but does not improve composition quality. Combined with finding #68, this closes the "orthogonality hypothesis" for adapter composition: neither weight-space nor data-space orthogonality is the mechanism behind successful multi-adapter merging. The mechanism is constructive cross-domain transfer regularized by 1/N scaling.

**Non-blocking issues:**
1. Eigenspectrum data was collected but not included in results.json. Would strengthen the analysis to show whether the spectrum is flat (explaining why init does not matter).
2. No PAPER.md was written. For a killed experiment this is acceptable but the finding deserves documentation.
3. The timestamp in results.json shows "2026-03-28" (date only), suggesting the full timestamp was overwritten during cleanup. Minor.
