# Peer Review: Text-to-LoRA Hypernetwork

## NotebookLM Findings

Skipped -- experiment is already killed and the issues are tractable without deep-review tooling.

## Mathematical Soundness

### 1. NN retrieval: embedding and A-matrix handling (Question 1)

The embeddings are **mean-pooled last-layer hidden states** (line 306: `mx.mean(x[0].astype(mx.float32), axis=0)`). This is clearly documented in PAPER.md Limitations (item 5) and is a reasonable choice given that the base model has no [CLS] token.

**Critical implementation detail in NN PPL evaluation:** When using the NN adapter, the code correctly sets **both A and B matrices from the NN domain** (lines 551-558: `set_lora_a(model, skeleton_data, nn_di)` then loads NN domain's `adapter.npz`). This means the NN evaluation is testing a complete adapter swap (use neighbor's full A+B), which is the correct thing to measure. The 1.28x mean PPL ratio is properly measured.

**Minor concern:** The embedding cosine similarity is computed in the raw hidden-state space (2560-dim), not in a task-relevant subspace. The fact that semantically plausible pairings emerge (legal->politics, sports->health_fitness) is encouraging but the high baseline inter-domain cosine (mean 0.80, range 0.64-0.87) suggests the embeddings are not highly discriminative. This could break for domains that are semantically distant but share data-distribution properties (e.g., two domains both requiring precise numerical reasoning). Not a kill-worthy issue for micro.

### 2. Orthogonal projection implementation (Question 2)

The projection in Phase 5 (lines 577-615) operates on **flattened B-vectors** (the full ~10.9M concatenated B-matrix), not on individual per-layer per-projection B matrices. This differs from the MATH.md formulation (Section 4), which defines projection per-projection per-layer.

**Is this correct?** The flattened version is actually a **stronger** test. Per-projection projection would only enforce orthogonality within each (l, p) slice. The flattened version enforces orthogonality across all parameters simultaneously. If the flattened version fails (as it does), the per-projection version would also fail for the same reason (the hypernetwork output is a convex combination of training adapters).

**The Gram-Schmidt is applied correctly.** Lines 593-597 subtract the projection of `proj` onto each basis vector `b_i`, iterating over all 23 other adapters. However, this is **not numerically stable Gram-Schmidt** -- it uses classical Gram-Schmidt without re-orthogonalization. For 23 vectors in ~10.9M-dimensional space, this is fine (the basis is far from ill-conditioned; the Grassmannian construction ensures near-orthogonality with mean |cos|=0.024).

**Should it project against A or B?** The experiment projects against B-matrices, which is correct for testing "does the generated adapter add novel information to the adapter pool?" The A-matrices are frozen Grassmannian skeletons shared across adapters (domain-indexed), so projecting against A would test the wrong thing. The composition safety guarantee comes from A-orthogonality (Grassmannian skeleton); B-orthogonality is about information novelty, which is what K2 tests.

### 3. Hypernetwork loss function

The loss function (line 425-427) minimizes MSE between `coeffs @ Y_train` and `Y_target` in the **normalized** B-space (`Y_norm`). The prediction is then reconstructed in the **unnormalized** space (line 449: `Y_pred = coeffs_np @ Y_train_np` uses raw `Y[train_mask]`). This is correct -- normalization is for training stability, cosine similarity is scale-invariant so unnormalized comparison is valid.

### 4. "24 examples is too few" argument (Question 3)

The argument is mathematically sound but **incomplete in its framing.** The PAPER.md claims "The embedding->B mapping has far more degrees of freedom (10.9M output params) than constraints (24 examples)." This is true for the direct-generation T2L architecture, but the **actual architecture used is NOT direct generation** -- it is a softmax convex combination over 23 training adapters (output dim = 23 coefficients, not 10.9M).

The real bottleneck is not input-output dimensionality but **manifold coverage**. With 23 basis adapters, the hypernetwork can only produce points in the convex hull of those 23 adapters. The LOO test asks: "Can the held-out adapter be reconstructed as a convex combination of the other 23?" The answer is almost always no, because trained LoRA adapters are not convex combinations of each other.

**Could a different architecture work?** Possibly. Consider:
- A hypernetwork that predicts **residuals** (B_new = mean_B + delta(embedding)), trained with L2 regularization. This requires the same 24 examples but doesn't constrain output to the convex hull.
- A variational approach that learns a latent manifold of adapters.

However, with only 24 training pairs, ANY supervised approach mapping 2560-dim embeddings to 10.9M-dim outputs is doomed. The argument is fundamentally correct even if the specific architecture choice makes it worse than necessary.

### 5. Convex-combination architecture makes K2 FAIL inevitable (Question 4)

**Yes, this is the most important finding of the review.** The hypernetwork outputs `softmax(head(h))` (line 389), producing non-negative coefficients that sum to 1. The predicted adapter is therefore:

    B_pred = sum_i alpha_i * B_i,  where alpha_i >= 0, sum(alpha_i) = 1

This means B_pred lies in the **convex hull** of {B_1, ..., B_23}. When you then project B_pred against {B_1, ..., B_23} via Gram-Schmidt, you remove all components along those directions. Since B_pred is literally a linear combination of those vectors, the projection removes essentially everything.

The 0.59% retention is not zero only because:
1. Classical Gram-Schmidt is imperfect (numerical residuals).
2. The 23 basis vectors are not exactly orthogonal (mean |cos| = 0.024, so the span is slightly less than 23-dimensional).

**K2 FAIL is therefore a mathematical tautology, not an empirical finding.** The paper acknowledges this ("by construction its output lives in their span") but does not flag it as a design error. This means K2 was never a meaningful test for this architecture -- it can only pass if the hypernetwork generates adapters **outside** the training adapter span, which a softmax-weighted combination cannot do.

**Correct test:** If the goal is to test whether orthogonal projection destroys hypernetwork output, the hypernetwork must be capable of generating adapters outside the training span. Options: (a) use a direct-generation architecture (MLP -> full B-vector), (b) use a residual architecture (convex combination + additive correction), (c) use unconstrained linear combination (remove softmax, allow negative and >1 coefficients).

## Novelty Assessment

**Prior art:** Text-to-LoRA (arxiv 2506.06105) is correctly cited as the primary reference. The experiment is an honest attempt to reproduce T2L's mechanism at micro scale with a known-insufficient training set.

**Delta over T2L:** Negative -- T2L works because it trains on thousands of diverse (task, adapter) pairs. The experiment confirms this prerequisite cannot be circumvented at N=24.

**NN retrieval baseline:** This is essentially adapter retrieval by embedding similarity, which is a standard technique in multi-adapter serving (e.g., CLONE, arxiv 2506.02847, which uses MoE-style routing over LoRA adapters). The finding that NN retrieval gives 1.28x PPL is useful but not novel -- it is a weaker version of what the softmax router already achieves (oracle-matching quality at N=24, per FINDINGS.md).

**Relationship to softmax router:** The PAPER.md notes that NN retrieval is "essentially what the softmax router already does... but without requiring any router training." This is correct, but the softmax router is already proven to work (gamma = oracle at N=24, 0.0% gap) with only 330K trainable params and 500 training steps. The NN retrieval finding is therefore **redundant** with existing results. Its only value is as a fallback for when no router is available, which is a narrow use case given router training is already cheap.

## Experimental Design

### What the experiment tests vs. what it claims

The experiment cleanly tests three things:
1. **NN retrieval viability** -- properly tested with actual PPL evaluation on 6 domains. Controls: base PPL and trained PPL. Adequate.
2. **Hypernetwork generalization** -- properly tested with LOO cross-validation. B-cosine is the right metric. 500 training steps may be insufficient but the loss plateaus confirm convergence.
3. **Orthogonal projection on convex-combination output** -- this is a tautological test (see Section 5 above). K2 FAIL is uninformative.

### Missing controls

- No comparison of NN retrieval to **random** adapter selection. This would quantify how much of the 1.28x comes from semantic matching vs. general adapter benefit. Given that any adapter improves over base (all NN PPLs < base PPLs), the margin attributable to semantic matching is unclear.
- No ablation of embedding type (mean-pool vs. last-token vs. attention-pool).
- Only 6 of 24 domains evaluated for PPL (acknowledged in Limitations).

### Evaluation concerns

- 10 validation samples per domain is very small. PPL estimates will have high variance. However, relative comparisons (NN/trained ratio) are more robust than absolute values.
- Single seed throughout (acknowledged, justified by prior CV=0.5% finding).

## Macro-Scale Risks (advisory)

1. **NN retrieval scales poorly with adapter count.** At N=500+, linear search over embeddings is trivial, but the semantic clustering assumption may break. Many domains will be equidistant in embedding space.
2. **NN retrieval with mismatched A-matrices.** Currently, the NN eval swaps both A and B from the neighbor. In production with Grassmannian skeletons, each domain has its own A. Using neighbor's (A, B) pair means the new domain doesn't get its own subspace -- this conflicts with the Grassmannian guarantee.
3. **Hypernetwork at scale.** The T2L architecture requires thousands of training pairs. Building this training set means first training thousands of adapters, which is expensive. Not a viable path for the SOLE architecture.

## Verdict

**KILL** -- confirmed.

The experiment correctly identifies its own failure modes and the kill on K2 is valid, though for the wrong reason. Specific findings:

1. **K1 PASS (1.28x) is legitimate** but redundant with the softmax router result (gamma = oracle at N=24). The NN retrieval finding adds marginal value as a zero-training fallback.

2. **K2 FAIL (0.45% retention) is a mathematical tautology**, not an empirical finding. A softmax-weighted combination of training adapters projected against those same adapters must yield near-zero by construction. The experiment does not actually test whether generated adapters survive orthogonal projection -- it tests whether convex combinations of a basis survive projection against that basis (they cannot). This should be explicitly noted in FINDINGS.md.

3. **Hypernetwork B-cosine ~0.036 is genuine failure** and correctly attributed to insufficient training data (24 pairs). However, the convex-combination architecture also contributes: even with perfect coefficients, the LOO target may not lie in the convex hull of the other 23 adapters.

4. **The "data scale is the fundamental bottleneck" claim is correct** but the convex-combination architecture is an additional, independent bottleneck that the paper underemphasizes. A direct-generation hypernetwork would fail differently (overfit to 24 examples) but would at least make K2 a meaningful test.

No revisions warranted -- the experiment is dead and should remain killed. The NN retrieval finding is already subsumed by the softmax router results. The hypernetwork direction requires orders of magnitude more training data than the SOLE architecture can provide.

## Audit-Rerun Closure Review (2026-04-18)

Experiment re-queued under `audit-2026-04-17-rerun/lora-scale`. Researcher applied a **mathematical closure**, not a rerun. Reviewing the closure for soundness.

### Adversarial checklist on closure
- (a) results.json verdict=KILLED; proposed status=killed. **CONSISTENT.**
- (b) K2 fail=true, status=killed. **CONSISTENT.**
- (c) PAPER.md Addendum verdict line: "KILLED preserved, no rerun required." **CONSISTENT.**
- (d) `is_smoke` not set; this is a closure, not a run. N/A.
- (e) MATH.md KCs unchanged (K1 thr 3.0, K2 thr 0.5, K3 48GB). Addendum is in PAPER.md only. **No KC-swap.**
- (f) Tautology sniff test: K2 FAIL IS tautological — but the tautology supports a KILL verdict, not a bogus PASS. Safe direction. Further, the existing Section 5 of this review already calls out the tautology. **OK.**
- (g) K#217 numeric ID matches DB/PAPER/evidence. **OK.**
- (h)-(m) code-level antipatterns: no code changes in this iteration. N/A.
- (i) LORA_SCALE=20 in original run is flagged — but the Addendum theorem proves scale-invariance, so the kill holds at s=5. **Documented, OK.**
- (r) PAPER.md has KC table. **OK.**

### Scale-invariance theorem soundness
The theorem reads:
1. `B_new = Σ α_i B_i`, `α_i ≥ 0`, `Σ α_i = 1` → `B_new ∈ span{B_i}` by construction. ✓
2. Orthogonal projection against `span{B_i}` of any element in that span equals that element → `B_proj = 0` in exact arithmetic. ✓
3. `span{s·B_1, …, s·B_N} = span{B_1, …, B_N}` for any `s ≠ 0` (span is closed under uniform scaling). ✓
4. Therefore `ρ = ‖B_proj‖²/‖B_new‖²` is independent of `s`. ✓

The theorem is mathematically watertight. The 0.45% residual is numerical (imperfect classical Gram-Schmidt + 0.024 mean inter-adapter cosine), not signal.

### Candidate antipattern promotion
Researcher flagged `ap-convex-hull-projection-tautology` as a new candidate. Concur: this is a **pattern-level** design error — K2 on a generator whose output is a linear combination of the basis it projects against is tautological. The prior review (Section 5) already identified this; the addendum now formalizes it as a promotable antipattern. **Recommend analyst promote if a second instance surfaces.**

### Verdict on closure
**KILL preserved.** No rerun needed. Closure is mathematically complete, consistency is intact, and the existing REVIEW body already endorses KILL for the same underlying reason.
