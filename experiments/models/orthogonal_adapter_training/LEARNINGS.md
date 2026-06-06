# LEARNINGS: Orthogonal Adapter Training (OPLoRA on Ternary)

## Core Finding

Eliminating 99.9% of direction interference via OPLoRA orthogonal projection
only recovers ~20% of MMLU math behavioral degradation. The dominant failure
mechanism (~80%) is **capacity interference** from ternary weights' flat singular
spectrum (gap ratio 1.003–1.018), which invalidates all subspace-based knowledge
preservation methods. This is a structural property of ternary quantization, not
a hyperparameter problem.

## Why This Happened

### Ternary Flat Spectrum Destroys Subspace Assumptions

OPLoRA's guarantee—preserving top-k singular triples—is mathematically correct
and was verified to numerical precision (rho_k: 0.012 → 0.000012). But the
guarantee is *vacuous* for knowledge preservation when the singular spectrum is
flat: if sigma_15 ≈ sigma_16 ≈ sigma_17 ≈ ... (gap ratio ~1.003), then "top-16"
is an arbitrary partition, not a meaningful knowledge boundary.

FP16 models have clear spectral gaps (power-law decay), so the top-k directions
genuinely concentrate knowledge. Ternary weights {-1, 0, 1} × scale have
fundamentally different spectral structure—quantization to three values
homogenizes the singular value distribution. This was previously observed in
Finding #37 (PiSSA: "Ternary SVD spectrum too flat, 32.8% variance at rank-8").

### Three-Mechanism Interference Decomposition (Complete)

Building on Finding #268/#269 (DARE density/direction decomposition), we now
have a complete picture:

| Mechanism | Cause | Fix | Status |
|-----------|-------|-----|--------|
| **Density interference** | Too many non-zero perturbation entries | DARE p=0.5 | **Solved** (code gen: 80%→90%) |
| **Direction interference** | Perturbation along base model's principal directions | OPLoRA projection | **Solved** (rho_k: 99.9% reduction) |
| **Capacity interference** | ANY perturbation on a flat-spectrum model disrupts distributed knowledge | None (structural) | **Unsolved, dominant** |

Capacity interference explains ~80% of MMLU math degradation. It cannot be
fixed by ANY method that assumes knowledge concentration in a subspace, because
the flat spectrum means knowledge is uniformly distributed across ALL directions.

### Why GSM8K Improved (+14pp, Best Result)

Procedural reasoning (step-by-step math) benefits from orthogonal constraints
because it uses the adapter's learned direction constructively. The constraint
prevents cross-domain interference from corrupting reasoning chains. This is
consistent with the NTP vs SFT finding (#265): NTP adapters preserve reasoning
precisely because they maintain directional coherence.

## Confirming Evidence

- **arXiv:2510.03262** (Rethinking Inter-LoRA Orthogonality): Found that strict
  orthogonality between adapters "brings little benefit" for semantic
  compositionality and "in some cases even reduces quality." Our result is
  the same finding in a different domain: orthogonality fixes the math but
  not the behavior, because the underlying assumption (spectral concentration)
  is violated.

- **Finding #37** (PiSSA, our experiment): PiSSA init was incompatible with
  ternary because "SVD spectrum too flat (32.8% variance at rank-8)." Same
  root cause—ternary flat spectrum invalidates SVD-based methods.

- **Finding #270** (our experiment): Direction interference is ~20% of MMLU
  degradation. Directly confirmed by this experiment (5pp out of 25pp).

- **arXiv:2504.07448** (LoRI, COLM 2025): Cross-task interference persists
  even with orthogonality constraints. Pruning + orthogonality helps some tasks
  but compromises others—consistent with our density/direction/capacity decomposition.

## Contradicting Evidence

- **OPLoRA (arXiv:2510.13003):** Reports strong knowledge preservation on
  LLaMA-2 7B and Qwen2.5 7B (both FP16 with clear spectral gaps). The method
  genuinely works when the spectral gap assumption holds. Our failure is
  ternary-specific, not a refutation of OPLoRA for FP16 models.

- **SC-LoRA (arXiv:2505.23724):** Data-driven subspace selection (least
  principal directions of knowledge features) succeeds on FP16 models. Uses
  activation features rather than weight SVD to identify knowledge subspace.
  This approach might circumvent the flat-spectrum problem because it doesn't
  depend on weight spectral gaps—but this is speculative for ternary.

## Alternative Approaches

1. **Routing (already proven, Finding #185):** Don't compose the math adapter
   for MMLU-like queries. Energy-gap routing already achieves 88% accuracy and
   +133% math correctness. This sidesteps capacity interference entirely by
   selecting the right adapter rather than composing all of them.
   *Status: deployment-ready, proven in our architecture.*

2. **Data-driven subspace identification (SC-LoRA, arXiv:2505.23724):** Instead
   of using weight SVD (broken for ternary), identify knowledge-bearing
   directions from activation features on calibration data. If knowledge
   concentrates in activation space even when it doesn't in weight space,
   projection in activation space could work.
   *Status: untested for ternary, would require significant implementation.*

3. **Spectral re-parameterization (SeLoRA, arXiv:2506.16787, ACL 2025):**
   Re-parameterizes LoRA from a sparse spectral subspace using frequency
   bases (e.g., DCT). This avoids SVD entirely and reduces parameter redundancy.
   May offer a different lens on the flat-spectrum problem.
   *Status: untested for ternary composition.*

4. **Accept the tradeoff + routing:** Orthogonal training gives the best GSM8K
   (+14pp) and best MMLU overall (41%) of any composition method tested.
   Combined with routing to avoid composition on knowledge-retrieval queries,
   this is already the best achievable configuration.
   *Status: deployment-ready combination of existing findings.*

## Implications for Next Experiments

1. **Subspace-based methods are dead for ternary.** Any method requiring
   spectral concentration of knowledge in weight space (OPLoRA, PiSSA, MDM-OC,
   OrthoMerge) will fail on ternary weights. Do not pursue further weight-SVD
   approaches. This eliminates a class of methods.

2. **Routing is the answer for knowledge preservation.** Capacity interference
   is structural—it cannot be fixed by changing how we compose, only by
   choosing WHEN to compose. The routing infrastructure (energy-gap, Gumbel-
   sigmoid, entropy gating) is the correct solution.

3. **GSM8K +14pp is the architecture's strongest result.** Orthogonal
   training + DARE p=0.5 + NTP adapters gives the best procedural reasoning
   improvement. The P0 deployment track should use this configuration.

4. **The flat-spectrum property is fundamental to ternary.** This isn't a
   BitNet-2B-4T quirk—any {-1, 0, 1} × scale weight matrix will have a
   flatter spectrum than its FP16 counterpart. This constrains the entire
   ternary composition research direction.

## Recommended Follow-Up

**No new orthogonal/subspace experiment recommended.** The impossibility
structure is clear: flat spectrum + low-rank adapter = unprotectable knowledge.

The P0 deployment track should proceed with:
- NTP adapters (Finding #265: 30pp GSM8K gap, p=0.003)
- DARE p=0.5 (Finding #269: fixes density interference)
- Routing for knowledge queries (Finding #185: 88% accuracy)
- Optionally orthogonal training for procedural reasoning adapters (this finding: +14pp GSM8K)

If SC-LoRA's activation-space approach (arXiv:2505.23724) is later tested, it
should be motivated by a specific deployment need for knowledge-task composition
without routing, not as a general interference fix.
