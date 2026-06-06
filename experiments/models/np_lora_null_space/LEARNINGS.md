# Learnings: exp_np_lora_null_space_composition

## Core Finding

Null space projection (NP-LoRA) is a complete no-op when adapters are initialized
on the Grassmannian, and provides zero measurable benefit even on random A matrices
at d/r=32. Pre-hoc orthogonality (Grassmannian init) dominates post-hoc projection
in every dimension: effectiveness, compute cost, and simplicity.

## Why This Happened (Literature-Grounded)

Three independent mechanisms explain this result:

**1. Grassmannian init eliminates the target signal.** When A_i @ A_j^T = 0 by
construction, the cross-term tr(Delta_i^T Delta_j) = tr((B_i^T B_j)(A_j A_i^T)) = 0
regardless of B correlation. NP-LoRA's SVD-based projector has nothing to project
away -- the interference subspace is empty. This is a direct mathematical consequence,
not an empirical observation.

**2. Johnson-Lindenstrauss makes random subspaces near-orthogonal at high d/r.**
FlyLoRA (arxiv 2510.08396) showed that random A matrices satisfy JL-lemma bounds:
at d/r=32, random subspaces have |cos| < 0.001, making interference negligible.
Our measurement (|cos| = 9.34e-4 for random vs 2.54e-7 for Grassmannian) confirms
this -- a 3700x ratio that produces only 0.03% PPL difference (1.5595 vs 1.5580).

**3. SVD-based null space projection scales catastrophically.** The O(N^3 * L * d^2)
complexity means each adapter requires SVD of an (N-1, d_out*d_in) matrix per layer.
At N=50: 318 seconds (318x over threshold). At macro scale (d=2560, L=100), memory
alone would require ~2.5GB per matrix * 100 layers * 50 adapters = 12.5TB.

## Confirming Evidence

**NP-LoRA's own domain validates our finding indirectly.** NP-LoRA (arxiv 2511.11051)
was designed for diffusion model LoRA fusion (subject + style), where independently
trained adapters naturally occupy overlapping, non-orthogonal subspaces. The paper
explicitly states that standard weight-based fusion is "fundamentally flawed" due
to geometric overlap. Our experiment shows this overlap doesn't exist when you
control initialization -- NP-LoRA solves a problem that Grassmannian init prevents
from occurring.

**"Rethinking Inter-LoRA Orthogonality in Adapter Merging" (arxiv 2510.03262)**
found that enforced orthogonality does not actually lead to semantic disentanglement.
This is consistent with our finding: geometric orthogonality (which Grassmannian
provides cheaply) is necessary but not sufficient. Post-hoc geometric fixes like
NP-LoRA address the wrong level of abstraction.

**Our own SOLE experiments** confirm that at d=256/r=8, composition ratio is 1.0224
(Grassmannian) vs 1.0227 (random) -- practically identical, validating that
interference is negligible at these dimensions regardless of initialization strategy.

## Contradicting Evidence

**NP-LoRA does work in its intended domain.** The original NP-LoRA paper shows
meaningful improvement for diffusion model subject+style fusion where adapters are
independently trained without orthogonal initialization. The key difference: diffusion
LoRAs are typically low-rank (r=4-16) in moderate dimensions (d=320-1280), with
d/r ratios of 20-320, but critically, they are trained on overlapping visual features
(faces, textures) that create genuine subspace alignment.

**JL-lemma breakdown after training.** Our SOLE adversarial review found that
while untrained random initializations show near-perfect orthogonality, training
can inflate functional cosine similarity to 0.703 -- up to 35.6x worse than random
JL predictions. This means NP-LoRA could become relevant if Grassmannian init's
A-orthogonality guarantee doesn't hold through training (it does in our architecture
because A is frozen, but non-frozen-A designs would face this risk).

**Low d/r regimes.** At d/r < 10 (e.g., d=64/r=16), random subspaces would show
substantial interference, and NP-LoRA could provide measurable improvement. Our
experiment only tested d/r=32. This is acknowledged in the paper's limitations but
is irrelevant to our architecture (d=256/r=8 minimum).

## Alternative Approaches (What We Could Try Instead)

The question "how to ensure interference-free composition" is now settled for our
architecture (Grassmannian init is the answer). The remaining composition challenges
are about **functional interference** (adapters that disagree about what to output)
rather than **geometric interference** (adapters whose weight updates overlap).

Literature suggests several approaches for functional composition:

**1. Dynamic routing (most promising for us):**
- LoRAuter (arxiv 2602.21222): retrieval-weighted output-space fusion achieved
  101.2% of oracle performance on in-domain tasks. Uses vector DB of training
  examples for task similarity scoring.
- MoLoRA (arxiv 2603.15965): per-token routing, Qwen3-1.7B + 4 adapters > 8B
- CLONE (arxiv 2506.02847): MoE router for dynamic LoRA selection at edge
- These address the real bottleneck (which adapter to use when) rather than the
  solved problem (how to merge weights cleanly).

**2. Training-time regularization:**
- OMoE (arxiv 2501.10062): Gram-Schmidt orthogonality constraint on Stiefel manifold
- MoDE (arxiv 2402.00893): mutual distillation to prevent expert overspecialization
- CDSP-MoE: gradient-conflict-driven subspace pruning (structural emergence)
- These are relevant for future work on jointly-trained expert systems.

**3. Post-hoc merging methods (for when we can't control initialization):**
- TIES merging: sign-aware majority consensus, but sacrifices fluency
- Task Arithmetic (arxiv 2212.04089): delta vectors as reusable modules
- LoRA-LEGO: rank-wise parameter clustering into Minimal Semantic Units
- Sub-MoE: joint SVD for shared U-matrix extraction (96% zero-shot at 25% compression)
- These are relevant for third-party adapter integration.

## Implications for Next Experiments

1. **Grassmannian skeleton is definitively validated.** No further experiments on
   post-hoc interference reduction are needed. The pre-hoc approach is both cheaper
   (O(1) at composition) and more effective (zero interference by construction).

2. **The composition bottleneck is now routing, not merging.** Future experiments
   should focus on which adapter to activate for which input, not on how to clean
   up weight-space interference after merging. The L2R Gumbel-sigmoid routing and
   TTT expert selection experiments are correctly prioritized.

3. **JL-lemma gives us a safety margin.** Even if Grassmannian init were removed
   (e.g., for capacity reasons beyond N=d/r), random A at d/r >= 16 provides
   near-equivalent interference suppression. This is useful insurance for scaling
   beyond N_max = d/r adapters.

4. **NP-LoRA could be revisited only if:** (a) we move to non-frozen A matrices,
   (b) at low d/r ratios, AND (c) an efficient approximate null space method exists
   (e.g., randomized SVD, iterative projection). All three conditions must hold.
   Currently none do.

5. **The OSRM finding (arxiv 2505.22934) remains relevant.** Weight-space
   orthogonality != data-space orthogonality. Our Grassmannian init ensures the
   former; the effective-delta-cosine experiment (exp_bitnet_effective_delta_cosine)
   will test whether this translates to the latter.
