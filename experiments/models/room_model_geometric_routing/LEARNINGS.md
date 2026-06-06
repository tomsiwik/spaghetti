# LEARNINGS: exp_room_model_geometric_routing (KILLED)

## Core Finding

**Adapter output norms ||h @ A_i @ B_i|| are structurally incapable of domain routing:
31.2% best accuracy (layer 29 only) vs 98.3% ridge router on identical data. Mean weight
on correct domain (0.201) is indistinguishable from random (0.200). The failure is not
insufficient training — it is a structural incompatibility between Grassmannian A-matrices
(designed for interference prevention via orthogonality) and routing (which requires
domain alignment). Room model is now fully dead: Piece A (pre-summing) killed in #302,
#303, #315; Piece B (geometric routing) killed in #316.**

## Why This Happened

### 1. Random Projections Destroy Domain Signal (Structural)

Grassmannian A-matrices are random orthogonal projections from R^2560 to R^16. The
Johnson-Lindenstrauss lemma guarantees distance preservation WITHIN a single projection,
but routing requires CROSS-projection comparison: "is ||h @ A_i @ B_i|| > ||h @ A_j @ B_j||?"
JL provides no guarantee for this comparison. Each A_i projects h into a different random
16D subspace — the resulting norms are incommensurable.

The reviewer correctly identified this as a misapplication of JL (Review issue #2): the
lemma applies per-adapter, not across adapters. The routing decision compares apples
to oranges — norms computed through unrelated random projections.

### 2. B-Matrix Training Does Not Compensate

The core hypothesis was that B_i, trained on domain-i data, would amplify domain-specific
components enough to rescue routing. This failed because:

- B_i is trained to minimize language modeling loss, not to maximize domain discrimination
- All 5 domains share ~80% of language modeling structure (syntax, common vocabulary)
- B_i learns the SHARED structure (which reduces loss the most), not the 20% domain-specific signal
- Result: B-norms vary by adapter (math 1.83 >> code 1.59) but this reflects training
  dynamics, not domain alignment. Math adapter dominates ALL domains because it has
  the largest B-norms, not because inputs are math-like.

### 3. Norm Aggregation Discards Directional Information

Taking ||c_i|| from c_i = h @ A_i @ B_i collapses a d_out-dimensional vector to a scalar,
losing all directional structure. The ridge router operates in the full 2560D space and
can exploit any 5D subspace for routing. The geometric router is constrained to N=5
independent random 16D subspaces, then further compressed by norms — a massive information
bottleneck (Review issue #5).

## Confirming Evidence

1. **SEQR (2509.18093):** Formalizes unsupervised LoRA routing as activation norm
   maximization (||BAx||). Their framework explicitly acknowledges that raw norm-based
   routing (ARROW) achieves accuracies "just above random chance." SEQR improves via
   QR-based efficient search but still maximizes norms — confirming the signal is weak.
   Our 31.2% is consistent with their "just above random" characterization.

2. **SpectR (2504.03454):** Dynamic token-level LoRA composition via spectral routing.
   SpectR uses SVD of adapter matrices (not raw norms) and achieves +4pp average routing
   accuracy improvement over ARROW. The fact that spectral decomposition (directional
   information) outperforms norms (scalar aggregation) confirms our Limitation #2:
   norms discard the discriminative directional structure.

3. **Rethinking Inter-LoRA Orthogonality (2510.03262):** Empirical analysis reveals that
   inter-LoRA orthogonality "does not lead to the semantic disentanglement highlighted in
   prior work." Directly confirms our finding: orthogonality (Grassmannian) is a geometric
   property, not a semantic one. Orthogonal adapters are not domain-separated adapters.

4. **Naive LoRA Summation (2508.11985):** Tests superposition principle for independently
   trained LoRA modules. Found that RMS cosine similarity between LoRA deltas correlates
   linearly with merging quality degradation. The domains with highest cosine similarity
   (most shared structure) merge worst — confirming that shared language modeling structure
   dominates over domain-specific signal in adapter geometry.

5. **Finding #302 (Room Model POC):** A-only routing gave 14% (near random). This
   experiment's A-only result of 16% reproduces that baseline. B-matrix adds only +15pp
   at best (layer 29), confirming B does not fundamentally change the routing landscape.

6. **Finding #310 (Ridge Router):** 98.3% accuracy with 13K parameters and 0.17ms latency.
   The 67pp gap between ridge (98.3%) and geometric (31.2%) quantifies how much information
   is lost by constraining routing to adapter geometry rather than learning from hidden states.

## Contradicting Evidence

1. **FlyLoRA (2510.08396):** Claims frozen random A-matrices serve as implicit routers
   via JL-lemma. However, FlyLoRA uses the projection structure DURING TRAINING to
   specialize adapters, not for post-hoc routing. The "implicit routing" is that each
   adapter learns to handle what its random projection can see — not that the projection
   norms discriminate domains at inference. Our experiment tests the latter, which fails.

2. **ARROW (norm-based routing from SEQR framework):** ARROW does use activation norms
   for routing and "works" in some settings. But SEQR's own analysis shows ARROW's
   accuracy is "just above random" — matching our result. ARROW "works" only in the
   sense that it slightly beats random, not that it provides reliable routing.

3. **Latent Geometry-Preserving Composition (2410.09908):** Proposes geometry-aware adapter
   composition using latent prototype vectors. But this uses learned task prototypes and
   sparse reconstruction, not raw adapter output norms. The geometry they preserve is
   task-level, not adapter-level — a fundamentally different signal.

## Alternative Approaches (All Paper-Backed)

1. **Spectral routing (SpectR, 2504.03454):** Uses SVD of adapter weight matrices to
   create per-expert prototype vectors. Routes via dot product of input with prototypes.
   Preserves directional information lost by our norm aggregation. +4pp over ARROW.
   Could be tested as zero-training alternative to ridge router.

2. **Task representation routing (LORAUTER, 2601.21795):** Routes via task embeddings
   derived from small validation sets, not adapter geometry. At 1500+ adapter scale,
   outperforms geometry-based methods. Confirms that routing should use external signal
   (data-derived), not internal geometry.

3. **Ridge regression router (Finding #310, proven):** 98.3% accuracy, 0.17ms, 13K params.
   Already proven in our pipeline. Uses learned linear map from hidden states — the correct
   approach. Calibration cost is minimal (50 sequences, one forward pass).

4. **Fine-grained spectral-aware routing (2603.01526, ICLR 2026):** Dimension-specific
   weights instead of scalar routing weights, with spectral-aware regularization. Separates
   high-SV shared knowledge from low-SV noise. More sophisticated than our norm-based
   approach but requires training.

## Implications for the Project

1. **Adapter geometry is for COMPOSITION, not ROUTING.** Grassmannian orthogonality ensures
   adapters don't interfere (composition). Domain discrimination requires alignment
   (routing). These are structurally different objectives. Do not attempt to extract
   routing signal from adapter weights again.

2. **Room model is fully dead.** Piece A (pre-summing) killed 3x (#302, #303, #315).
   Piece B (geometric routing) killed in #316. Both core premises of the room model are
   structurally invalid. No resurrection path exists without abandoning both premises.

3. **Ridge router confirmed as the correct mechanism.** 98.3% accuracy from 13K learned
   parameters dominates all geometry-based alternatives. The cost (50 calibration sequences)
   is negligible. No reason to pursue parameter-free routing — the parameters are cheap
   and the information gain is 67pp.

4. **Norm-based signals are weak for adapter routing generally.** SEQR (2509.18093) and
   SpectR (2504.03454) independently confirm that activation norms provide minimal routing
   signal. Directional/spectral methods outperform norms. This is a known result in the
   literature, not specific to our setup.

## Recommended Follow-Up

No follow-up on geometric routing or the room model. Both are conclusively dead.

The confirmed path forward is the factored LoRA chain (#304 -> #305 -> #312 -> #313 -> #314):
- **exp_rope_reset_block_diagonal** (P1): Per-segment RoPE position reset to close the
  8.9% gap from Finding #314. Lemma 1' predicts exact equivalence with reset. Standard
  technique: Block-Attention (2409.15355) implements this as "position re-encoding."
- **exp_ridge_router_single_pass_e2e** (P2): Connect Finding #310's ridge router to
  the single-pass block-diagonal architecture from Finding #314. Completes the e2e pipeline.
