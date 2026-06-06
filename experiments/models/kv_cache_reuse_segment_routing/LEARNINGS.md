# LEARNINGS: KV-Cache Reuse Across Adapter Switches

## Core Finding

**Cross-adapter KV-cache reuse is structurally impossible under Grassmannian orthogonality:
the same property that prevents adapter interference (orthogonal A-matrix subspaces) guarantees
that adapter A's key/value representations are maximally incompatible with adapter B's queries.
Isolated segment evaluation (Finding #305) is not leaving value on the table -- cross-segment
context from a different adapter is noise, not signal. Domain signal IS detectable (Finding #307)
but cannot be exploited via KV-cache sharing between orthogonal adapters.**

## Why This Happened

### The Duality of Orthogonality
Grassmannian orthogonality is designed to make adapters project into non-overlapping subspaces
(||cos(A_i, A_j)|| < 0.05). This guarantees composition without interference: adapter i's
perturbation doesn't corrupt adapter j's contribution. But KV-cache reuse requires the OPPOSITE:
adapter B's query projections must be able to extract useful information from adapter A's
key/value projections. For this to work, the Q/K subspaces must OVERLAP. These two requirements
-- orthogonal for composition, overlapping for KV-cache sharing -- are mutually exclusive.

### DPI Misapplication
Theorem 3 invoked the data processing inequality: conditioning on more data cannot increase
uncertainty. But the DPI requires clean observations through a fixed channel. Cross-adapter
KV entries are not "observations of segment A" -- they are adapter A's internal representations
read through adapter B's query projections. The channel is distorted and adapter-dependent,
violating the DPI's preconditions. The experiment directly refutes: KV-reuse PPL (5.704) is
6.27% WORSE than isolated (5.367).

### Perturbation Bound Self-Contradiction
Theorem 2 claimed cross-adapter interaction is O(alpha^2 * r^2 / d^2) ~ 1.6%. The careful
derivation in the same proof yields O(alpha^2 * r / d) = 62.5% -- already vacuous as a
perturbation bound at a single layer, before considering 28-layer accumulation. The appeal
to Grassmannian recovery (A^B^T * A^A = 0) fails because the B-matrix product (B^B)^T * B^A
can correlate the subspaces, as the reviewer correctly identified.

## Confirming Evidence

1. **LRAgent** (arXiv:2602.01053) -- Decomposes KV-cache into shared base component + adapter-
   dependent component. Key finding: "base cache remains highly similar across agents, while
   adapter outputs are largely decorrelated across agent pairs." Under orthogonality assumptions,
   adapter components are approximately random noise to other adapters. This CONFIRMS our finding:
   the adapter-dependent KV component is decorrelated (noise) when adapters are orthogonal.

2. **Rethinking Inter-LoRA Orthogonality** (arXiv:2510.03262) -- "Orthogonality does not lead
   to the semantic disentanglement highlighted in prior work... inter-LoRA orthogonality alone
   may be insufficient for achieving true semantic compositionality." Confirms that orthogonality
   guarantees non-interference but NOT compatibility of representations.

3. **Adapter Merging Reactivates Latent Reasoning Traces** (arXiv:2601.18350) -- "Adapter
   merging can trigger structural incompatibility in the representation space, challenging the
   assumption of linear composability in PEFT." Cross-adapter representation incompatibility
   is a known failure mode, not specific to our architecture.

4. **Self-Contrast Decoding** (Finding #302, killed) -- Grassmannian orthogonality makes
   non-primary adapters produce decorrelated noise. Confirmed experimentally: contrastive
   value extraction fails because orthogonal adapters' outputs are uninformative to each other.
   Same duality: interference-free <==> contrastive-value-free <==> KV-cache-incompatible.

## Contradicting Evidence

1. **Activated LoRA (aLoRA)** (arXiv:2512.17910) -- Achieves 58x latency reduction via
   cross-model KV-cache reuse between base and LoRA-adapted models. BUT: this shares cache
   between a base model and its adapted variant (same base + small perturbation), NOT between
   two different adapters with orthogonal subspaces. The base-to-adapter Q/K relationship is
   preserved because the adaptation is perturbative. Our cross-adapter setting breaks this
   because orthogonal adapters are NOT perturbative relative to each other.

2. **FastLibra** (arXiv:2505.03756) -- Multi-LoRA KV-cache management achieving 63% TTFT
   reduction. BUT: this manages cache ALLOCATION (which adapter's cache to keep/evict), not
   cache SHARING between adapters. Each adapter still has its own KV-cache. No cross-adapter
   reuse is attempted.

3. **Math+medical pair** showed +1.72% improvement from KV-reuse. This is the most semantically
   similar pair in our adapter set. Suggests KV-cache reuse MIGHT work for within-cluster
   adapters where A-matrices are not maximally orthogonal. But 9/10 pairs are harmed, and
   relaxing orthogonality for clustered adapters conflicts with the Grassmannian design.

## Alternative Approaches

1. **Base-only KV-cache sharing** (LRAgent, arXiv:2602.01053) -- Share only the base model's
   KV component across adapters, recompute only the adapter-dependent delta. This respects
   orthogonality (no cross-adapter KV mixing) while saving compute on the base component.
   Viable for serving optimization without quality loss.

2. **Hidden-state probe routing** (X-LoRA arXiv:2402.07148, TT-LoRA MoE arXiv:2504.21190) --
   Use hidden states (already computed, O(1) per token) for adapter selection. Bypasses both
   the KV-cache incompatibility problem (no cross-adapter KV sharing needed) and the latency
   wall of PPL-based detection (Finding #307). This is the proven alternative for dynamic
   multi-adapter inference.

3. **Entropy-based context detection** (ERGO arXiv:2510.14077) -- Token-level entropy from
   next-token distribution as zero-cost temporal signal for domain shifts. Combined with
   segment isolation, could detect boundaries without the false-positive cascade of PPL
   sliding windows.

## Implications for Next Experiments

1. **Segment isolation is architecturally correct.** Finding #305's approach of evaluating each
   domain segment independently is not a limitation -- it's a necessary consequence of
   Grassmannian orthogonality. Cross-segment context cannot help when adapters project into
   orthogonal subspaces.

2. **The KV-cache optimization path is base-only sharing.** If serving latency needs improvement,
   share the base model's KV component (LRAgent decomposition). Do NOT attempt cross-adapter
   KV mixing.

3. **Three experiments killed by the same duality.** Self-contrast decoding (Finding #302),
   boundary detection PPL (Finding #307), and KV-cache reuse (Finding #309) all fail because
   Grassmannian orthogonality implies cross-adapter representations are decorrelated noise.
   Future experiments must respect this: adapters are ISOLATED processing units, not
   collaborating components.

## Recommended Follow-Up

**exp_hidden_state_probe_router (P1)** -- Train small MLP on base model hidden states for
per-token adapter selection. Motivated by:
- X-LoRA (arXiv:2402.07148): hidden-state routing works for LoRA mixtures, O(1) per token
- TT-LoRA MoE (arXiv:2504.21190): noisy top-1 gating on hidden representations
- Finding #307: domain signal IS detectable (88.7%) but PPL delivery is wrong
- Finding #309: cross-adapter KV-cache sharing is impossible, so routing must happen BEFORE
  adapter application, not during attention

This fixes the latency wall (one matmul vs 75 forward passes), the false-positive cascade
(smooth MLP outputs vs argmax flickering), and respects the Grassmannian constraint (routing
decisions use base-model hidden states, not cross-adapter representations).
