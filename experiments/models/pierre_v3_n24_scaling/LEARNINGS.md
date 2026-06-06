# LEARNINGS.md: Pierre v3 N=24 Scaling

## Core Finding

A-matrix orthogonality (Grassmannian skeleton) is likely the sole operative
interference-prevention mechanism at N=24. B-matrix null-space projection
fails its theoretical bound (36.1% vs 85.6% predicted) due to directional
correlation, yet composition quality is robust (PPL 0.87x worst single).
This decoupling — null-space fails, composition succeeds — is the central
insight of this frontier extension.

## Why This Happened

### 1. B-matrix directional correlation breaks the uniform gradient assumption

The null-space preservation bound (d - Kr)/d = 85.6% assumes the test
gradient is uniformly distributed in R^d. When all 24 adapters fine-tune
the same base model on text data, their B-matrices learn correlated
directions despite having maximal rank (368/368). This is not rank
deficiency — it is directional alignment.

This matches LoRI (arXiv:2504.07448), which freezes A as random projection
and sparsifies B precisely because B encodes task-specific information that
correlates across similar tasks. Their finding that 90% B-sparsity
preserves performance suggests most B-matrix capacity is redundant across
tasks — consistent with our observation that B-matrix null-space projection
captures disproportionate gradient energy.

OSRM (arXiv:2505.22934, ACL 2025) addresses the same root cause from the
opposite direction: constraining LoRA subspaces BEFORE fine-tuning so that
one task's updates have minimal transformation capacity on other tasks'
data. Our Grassmannian A-matrix initialization achieves a similar effect
through the A-matrices, which may explain why composition survives despite
B-matrix correlation.

### 2. A-matrix orthogonality IS the interference barrier

The Grassmannian skeleton enforces near-orthogonality of A-matrices (mean
|cos| = 0.024, max 0.089 across all 276 pairs at N=24). Since ΔW = BA,
if A-matrices are orthogonal, the COLUMN SPACES of different ΔW matrices
are approximately orthogonal — meaning different adapters modify different
subspaces of the output, regardless of B-matrix correlation.

This aligns with Cao et al. (arXiv:2508.11985), who show that naive LoRA
summation works when adapter deltas are approximately orthogonal, with RMS
cosine similarity between deltas correlating linearly with PPL degradation.
Our mean |cos| = 0.024 is well within their "safe summation" regime.

### 3. Within-cluster misrouting is PPL-benign

3 of 4 misrouted domains actually IMPROVE PPL. Medical misroutes to
health_fitness (5.29 vs 7.09 oracle), cooking to cybersecurity (3.22 vs
3.30), creative_writing to agriculture (23.14 vs 26.30). This replicates
Finding #287 at N=5 and confirms the pattern at N=24.

MoLoRA (arXiv:2603.15965, Microsoft) provides theoretical grounding: when
adapters share overlapping expertise (as semantically related domains do),
routing to a "wrong" but related adapter can provide useful features. Their
per-token routing shows that domain boundaries are fuzzier than discrete
labels suggest.

## Confirming Evidence

- **arXiv:2508.11985** (Naive LoRA Summation): RMS cosine between deltas
  predicts composition quality. Our cos=0.024 predicts near-perfect
  summation, confirmed by K723 PASS.
- **arXiv:2504.07448** (LoRI): Freezing A and sparsifying B works because
  A provides subspace isolation while B is task-correlated — same
  decomposition we observe (A orthogonal, B correlated).
- **arXiv:2604.01152** (Brainstacks): Null-space SVD projection for
  continual learning. Our N=5 results (Finding #273: 95.2%) confirmed
  their approach. N=24 reveals the scaling limit of the uniform gradient
  assumption.
- **Finding #54**: 24/24 adapters specialize with mean |cos| = 0.0238,
  confirming Grassmannian skeleton maintains orthogonality at scale.
- **Finding #287**: Pierre pipeline at N=5 showed 99.6% routing, 0% PPL
  degradation — the N=5 baseline that this experiment extends.

## Contradicting Evidence

- **arXiv:2505.22934** (OSRM): Argues that pre-training subspace
  constraints are necessary for robust merging. Our results suggest
  A-matrix orthogonality alone may suffice, without constraining B. However,
  OSRM operates in weight-space merging (permanent), while we use
  activation-space composition (dynamic) — the comparison may not be direct.

- **arXiv:2602.21919** (NESS): Learning in null-space using small singular
  values for continual learning. Their approach succeeds at scale,
  suggesting null-space methods CAN work — but they constrain updates to
  the null-space during training, not post-hoc projection. Our post-hoc
  SVD projection may be the wrong timing.

- **The 36.1% preservation is NOT zero.** The ablation experiment has not
  been run. It remains possible that partial null-space protection (36.1%
  vs 0%) contributes meaningfully to K723 PASS. The claim that "A-
  orthogonality alone suffices" is a hypothesis, not a finding.

## Alternative Approaches

### For scaling null-space preservation:

1. **Train-time null-space constraint** (arXiv:2602.21919, NESS): Project
   gradient updates into the null-space DURING training, not post-hoc.
   This ensures adapters never enter each other's subspaces. Proven to
   scale in continual learning settings.

2. **Sparse B-matrices** (arXiv:2504.07448, LoRI): Instead of null-space
   projection, sparsify B-matrices with task-specific masks. 90% sparsity
   preserves performance while reducing cross-task interference. Could
   replace null-space SVD entirely.

3. **Gradient space splitting** (arXiv:2505.22370, SplitLoRA): Partition
   gradient space into primary (existing knowledge) and minor (new task)
   subspaces. New adapters train only in the minor subspace.

### For improving routing accuracy:

4. **Per-token routing** (arXiv:2603.15965, MoLoRA): Replace sequence-level
   ridge routing with per-token gating. Avoids the "slice domain" problem
   entirely — tokens route themselves regardless of domain labels.

5. **Task-representation routing** (arXiv:2601.21795, LoRAuter): Route
   using task embeddings from small validation sets, not individual sample
   embeddings. More robust than single-sample centroid matching.

## Implications for Next Experiments

### The A-orthogonality hypothesis is the critical question.

If confirmed by ablation: null-space projection can be dropped from the
pipeline, simplifying Pierre to {Grassmannian init → ridge router → NRE
compose}. This is a 3-component system vs 4-component. Simpler = faster
= fewer failure modes.

If refuted (composition degrades without null-space): the issue becomes
scaling null-space to N=24. Train-time projection (NESS) or sparse B
(LoRI) are the proven alternatives.

### Routing needs data quality, not architecture changes.

6/7 genuine domains route at >80%. The failure is in 17 slice-based
domains with semantically overlapping training data. SFT adapters
(exp_sft_24_domain_adapters) will produce more distinctive hidden
representations due to instruction-format diversity, likely improving
routing without any architectural change.

### The compose weight [0.7, 0.3] sensitivity gap matters at macro scale.

The undeclared hyperparameter should be derived from router confidence
scores. At macro scale with SFT adapters, router confidence will differ
from NTP adapters, and the optimal compose weight may shift.

## Recommended Follow-Up

### 1. Null-Space Ablation (highest priority)

**Motivation:** This experiment's central ambiguity — does composition
quality come from A-orthogonality alone, partial null-space (36.1%), or
both?
**Literature:** LoRI (arXiv:2504.07448) shows frozen-A + sparse-B works
without null-space. OSRM (arXiv:2505.22934) shows pre-training constraints
suffice. Both suggest null-space may be redundant when A is orthogonal.
**Design:** Run Pierre compose at N=24 with null-space projection disabled
(P=I). Compare PPL to current results. If PPL unchanged → null-space is
redundant. If PPL degrades → quantify how much.

### 2. SFT Adapter Training (already planned as P0)

**Motivation:** Finding #296 shows routing failure concentrates in
slice-based domains with non-distinctive representations.
**Literature:** MoLoRA (arXiv:2603.15965) shows per-token routing with
instruction-tuned adapters achieves strong separation. SFT format produces
more distinctive hidden states than NTP.
**Design:** exp_sft_24_domain_adapters — retrain as SFT, then re-run
routing evaluation.
