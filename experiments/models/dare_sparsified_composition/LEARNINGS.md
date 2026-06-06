# LEARNINGS: DARE Sparsified Adapter Composition

## Core Finding

DARE sparsification at p=0.5 recovers code gen OOD performance (80%->90%) and
preserves GSM8K reasoning gains (+6pp), but MMLU math degradation (-25pp) persists
across ALL drop rates. This reveals two distinct interference mechanisms: **density
interference** (fixable by sparsification) and **direction interference** (requires
orthogonal methods). Optimal p for ternary adapters is 0.5, not the 0.9 recommended
for FP16 in the original DARE paper.

## Why This Happened

### Density vs Direction Interference

DARE zeros out random entries in the adapter delta, reducing the number of perturbed
output dimensions. For code gen, the OOD degradation was caused by too many irrelevant
weight perturbations — a density problem. Halving the density (p=0.5) eliminated
the interference while the unbiased estimator property preserved in-distribution effect.

For MMLU math, the degradation is caused by the *direction* of the perturbation in
weight space disrupting stored knowledge representations. Sparsifying doesn't change
the direction of surviving entries; it just reduces how many there are. Knowledge
recall requires specific activation patterns that are disrupted even by sparse
perturbations in the wrong direction.

This density/direction decomposition explains why:
- Code gen (format task): recovered by DARE (density-dependent)
- GSM8K (reasoning): partially preserved (reasoning uses adapter direction constructively)
- MMLU math (knowledge recall): unaffected by DARE (direction-dependent)

### Why p=0.5, Not p=0.9

The original DARE paper (Yu et al., 2311.03099) used FP16 models with typical LoRA
scales of 1-4. Our setup uses scale s=20 on BitNet-2B. The effective perturbation
magnitude on surviving entries is s/(1-p):
- p=0.5: effective = 40 (manageable)
- p=0.9: effective = 200 (destructive — variance cost overwhelms)
- p=0.95: effective = 400 (kills GSM8K entirely, -14pp vs base)

The variance of the DARE estimator is delta^2 * p/(1-p). At p=0.9 this is 9x the
signal, compared to 1x at p=0.5. For a 2B parameter model, the law of large numbers
doesn't average out this variance sufficiently at high drop rates.

## Confirming Evidence

- **Yu et al. (arXiv:2311.03099):** Original DARE paper shows sparsification preserves
  model merging quality. Our unbiasedness and in-distribution preservation results
  confirm their theoretical guarantees extend to ternary base models.

- **LoRA Land (arXiv:2405.00732):** 25+ LoRA study showing task-specific LoRAs transfer
  poorly across tasks. Supports our finding that adapter composition degrades OOD
  benchmarks — DARE partially mitigates but doesn't eliminate this.

- **LoRI (arXiv:2504.07448, COLM 2025):** Directly addresses cross-task interference
  in multi-task LoRA adaptation. Confirms that pruning methods (DARE, TIES) improve
  some tasks but often compromise others — consistent with our density-only fix.

- **Sparse Adapters for Merging (arXiv:2507.07140):** Sparse adapters yield superior
  in-distribution performance post-merging, but held-out (OOD) performance remains
  challenging for all methods. Directly supports our observation that DARE helps
  in-distribution but has limited OOD benefit beyond density reduction.

## Contradicting Evidence

- **DARE paper recommends p=0.9:** Our optimal is p=0.5. Not a true contradiction —
  the difference is explained by our higher adapter scale (s=20 vs typical s=1-4)
  creating larger effective perturbation at high drop rates.

- **TIES-Merging (arXiv:2306.01708) claims superiority over random dropping:** TIES
  uses magnitude-based pruning + sign resolution, which should be more principled
  than DARE's random masking for low-rank deltas. However, NotebookLM sources note
  TIES can "severely degrade sentence-level fluency and narrative coherence" due to
  its sign-enforcement mechanism. For our behavioral-first approach, TIES's fluency
  degradation may be worse than DARE's variance cost. Not tested in this experiment.

## Alternative Approaches (with paper references)

### 1. Orthogonal Projection Methods (most promising for direction interference)

- **OPLoRA (arXiv:2510.13003):** Projects LoRA updates orthogonally to top singular
  vectors of frozen weight matrices. Prevents catastrophic forgetting by ensuring
  adapter updates don't interfere with dominant pre-trained subspaces. This directly
  addresses our MMLU math direction interference — if the adapter delta is orthogonal
  to knowledge-critical subspaces, composition cannot disrupt knowledge recall.

- **MDM-OC (arXiv:2507.20997):** Orthogonal delta merging ensures task-specific updates
  occupy independent subspaces with mathematical non-interference guarantees. Provides
  exactly the structural guarantee our architecture needs.

- **OrthoMerge (arXiv:2602.05943):** Merges models on the orthogonal group manifold,
  preserving pre-trained knowledge. Theoretically grounded approach to the direction
  interference problem.

- **DO-Merging (arXiv:2505.15875):** Decouples magnitude and direction components,
  merging them independently. Addresses the magnitude variance problem that causes
  DARE's rescaling to be destructive at high p.

### 2. Interference-Aware Training (fix at training time, not merge time)

- **LoRI (arXiv:2504.07448):** Reduces cross-task interference during multi-task
  LoRA training itself, rather than post-hoc fixing during merge. Would require
  retraining our adapters with interference-aware loss.

### 3. Contrastive/Retrieval Routing (avoid merging entirely)

- **LoRAuter (arXiv:2602.21222):** Routes to task representations rather than
  individual adapters. Avoids composition interference by selecting rather than
  merging. Already explored in exp_lorauter_task_routing (Finding #244).

## Implications for Next Experiments

1. **DARE is deployment-ready at p=0.5.** For the P0 deployment track, DARE p=0.5
   should be the default composition method. It recovers code gen OOD, preserves
   reasoning gains, and has zero inference overhead (pre-merge operation).

2. **MMLU math requires orthogonal methods.** The persistent -25pp degradation cannot
   be fixed by any sparsification approach. OPLoRA-style orthogonal projection during
   training, or MDM-OC-style orthogonal delta merging, are the principled paths.

3. **The density/direction decomposition is the key theoretical contribution.** Future
   experiments should classify interference by type before choosing a fix:
   - Density interference -> DARE (or magnitude pruning)
   - Direction interference -> Orthogonal projection (OPLoRA, MDM-OC)

4. **Scale-dependent optimal p.** The relationship s/(1-p) < threshold should be
   tracked. If adapter scales change (e.g., per-domain optimal scales from Finding
   #249), the optimal DARE drop rate changes too.

## Recommended Follow-Up

**exp_orthogonal_adapter_training:** Train LoRA adapters with OPLoRA-style orthogonal
projection constraint (arXiv:2510.13003). Motivation: DARE fixes density interference
but MMLU math -25pp persists as direction interference. OPLoRA's orthogonal constraint
ensures adapter updates don't disrupt knowledge-critical subspaces. This would test
whether combining DARE (post-hoc sparsification) + OPLoRA (training-time orthogonal
constraint) eliminates BOTH interference mechanisms.

**NOT recommended:** TIES-Merging comparison. While the adversarial review flagged
this gap, TIES's known fluency degradation and our prior exp_lora_merging_bakeoff
(which showed simple average dominates for orthogonal adapters) suggest limited
upside. The orthogonal projection direction is more promising.
