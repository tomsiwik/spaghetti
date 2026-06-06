# Peer Review: lora_rank_composition

## NotebookLM Findings

Skipped -- the experiment is already self-killed with both kill criteria triggered.
The review focuses on whether the kill is correct, the findings are sound, and the
secondary claims (effective rank saturation, delta norm scaling) hold up.

## Mathematical Soundness

### What holds

1. **Effective rank definition is correct.** The Shannon-entropy-based effective
   rank r_eff = exp(H(p)) where p_i = s_i / sum(s_j) is standard (Roy & Vetterli
   2007). Properties stated in MATH.md Section 4 are correct: 1 <= r_eff <= r,
   with equality at the extremes.

2. **Shared fraction at N=2 with orthogonal deltas.** Section 6 correctly derives
   that when cos(dW_A, dW_B) ~ 0, shared fraction ~ 0.5. The empirical results
   (0.499-0.509) confirm this. The math is straightforward vector algebra.

3. **Parameter count formula.** P(r) = n_layer * 10 * d * r is correct for LoRA
   on fc1 (d -> 4d) and fc2 (4d -> d): per layer = d*r + r*4d + 4d*r + r*d =
   10*d*r. The table values check out.

4. **Kill criteria evaluation.** The gap range (0.70pp < 1pp) and r-squared
   (0.156 < 0.2) are correctly computed from the data. The r-squared calculation
   in the test code uses log2(rank) vs cosine similarity, which is reasonable.

### What does NOT hold

5. **The alpha/r confound is a fatal experimental design flaw.** The experiment
   uses fixed alpha=1.0 across all ranks. The LoRA scaling is alpha/r, meaning:
   - r=2: effective scale = 0.5
   - r=4: effective scale = 0.25
   - r=64: effective scale = 0.015625

   This is a 32x difference in gradient multiplier. The paper acknowledges this
   produces decreasing delta norms (15.62 at r=2 vs 10.40 at r=64) and even
   notes that "these effects approximately cancel." But this is not a controlled
   experiment -- you are simultaneously varying TWO things (rank and effective
   learning rate). The flat composition curve could be because:
   - (a) Rank truly does not matter, OR
   - (b) Low rank with large scale and high rank with small scale happen to
     produce similar quality by coincidence

   The standard practice in LoRA literature (Hu et al. 2022) is to set alpha=r
   or alpha=2*r so that the effective scale is constant across rank sweeps. The
   paper lists this as Limitation 3 but it should have been the PRIMARY
   experimental design choice.

   **Severity**: This does not invalidate the kill (the hypothesis was about rank
   constraining composition, and the experiment shows it does not at micro scale
   regardless of confound). But it weakens the "rank is irrelevant" conclusion
   and makes the effective rank saturation finding harder to interpret.

6. **Rate-distortion framing is an analogy, not a theorem.** Section 3 draws a
   parallel between Shannon rate-distortion and LoRA rank vs composition quality.
   This is fine as motivation but the paper never provides a formal connection.
   Rate-distortion theory applies to source coding with a well-defined distortion
   measure; composition quality is not a distortion measure in the information-
   theoretic sense. The "prediction" of a critical rank r* is intuitive but not
   derived from the theory. This is acknowledged implicitly but should be more
   explicit.

7. **Effective rank computation concatenates across layers then averages per
   matrix.** The code computes SVD per delta matrix (per layer, per fc1/fc2) and
   then averages. This is fine as a summary statistic, but the effective rank of
   the full stacked delta system could differ. The saturation at ~8 is across 8
   individual matrices (4 layers x 2 MLP weights), each of which saturates at
   ~8. Since these are applied independently, the "task dimensionality is ~8"
   claim should be "per-layer MLP task dimensionality is ~8."

## Novelty Assessment

### Prior art

The paper cites Biderman et al. (2024) on LoRA rank constraints and InfLoRA on
orthogonality. These are the right references.

**Missing reference**: "LoRA-Hub: Efficient Cross-Task Generalization via Dynamic
LoRA Composition" (Huang et al., 2023) composes multiple LoRA adapters and
discusses rank effects on composition. The experiment should cite this.

**Missing reference**: "Tied-LoRA" (Renduchintala et al., 2024) explores
parameter sharing across LoRA adapters, relevant to the shared fraction finding.

### Delta over existing work

The effective rank saturation finding is mildly novel in the composition context.
Prior work (Biderman et al.) showed LoRA rank acts as implicit regularization;
this experiment shows the same phenomenon manifests in composition quality metrics.
The delta is small but the experiment was designed as a sweep, not a novelty claim.

## Experimental Design

### Does it test what it claims?

Partially. The experiment claims to test whether rank constrains composition
quality. It does test this, but with the alpha/r confound noted above.

### Controls

- Joint training baseline: present and correct (trains on both domains)
- Multiple seeds (3): adequate for a micro experiment
- Multiple composition methods (task arithmetic + concat+calibrated): good
- Monotonicity check on effective rank: present

### Missing controls

- **Alpha=r sweep**: The most important missing control. Without it, you cannot
  separate rank effects from scale effects.
- **Single-domain quality vs rank**: The experiment measures composition quality
  but does not report single-domain LoRA quality at each rank. If single-domain
  quality is also flat across ranks, the finding is trivially about the task, not
  about composition.

### Confounds

The alpha/r scaling issue is the primary confound. The paper acknowledges it in
limitations but does not run the control.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry matches the experiment:
- Kill criteria: "rank has no effect on composition quality (all ranks within 1%)"
  and "orthogonality does not correlate with rank (r^2 < 0.2)"
- Both triggered, status correctly set to "killed"
- Evidence entries accurately summarize findings

The experiment `blocks: [exp_lora_procrustes_linear]` -- this is correct since
this rank sweep informs whether the Procrustes approach needs rank-specific
handling.

## Macro-Scale Risks (advisory)

1. **The alpha/r confound becomes critical at macro scale.** If repeated with
   diverse domains and higher inherent dimensionality, the same confound will
   obscure results. Must use alpha=r (or alpha=2r) for the macro version.

2. **Effective rank saturation ceiling will change.** The ~8 ceiling is task-
   specific. At macro scale with BPE tokenization and diverse domains (code vs
   prose), effective rank could be 50-200. The saturation curve shape may differ.

3. **The "composition quality is dominated by composition mechanism" finding may
   not transfer.** At micro scale, the 1-2% gap is small. At macro scale with
   domain interference, rank could become the bottleneck if domains need many
   orthogonal directions.

## Verdict

**PROCEED**

The experiment correctly kills its own hypothesis at micro scale. The self-kill is
honest and well-documented. The secondary findings (effective rank saturation,
shared fraction stability, delta norm vs alpha/r scaling) are informative for
future work.

The alpha/r confound is a real flaw but does not invalidate the kill -- even if
you fixed alpha=r, the task's low inherent dimensionality (~8) means rank
sensitivity is unlikely to appear at micro scale. The paper correctly identifies
this as the root cause.

The experiment advances the project by:
1. Establishing that rank sweeps are uninformative at micro scale (saving future
   compute on similar sweeps)
2. Providing the effective rank saturation measurement as a task complexity proxy
3. Confirming rank-independence of orthogonality, shared fraction, and dead rate

**Advisory for future work (not blocking):**
1. Any macro-scale rank sweep MUST use alpha=r to isolate rank from scale.
2. Report single-domain quality alongside composition quality to separate task
   effects from composition effects.
3. Cite LoRA-Hub (Huang et al., 2023) for LoRA composition context.
