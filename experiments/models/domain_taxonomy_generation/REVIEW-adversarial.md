# Peer Review: Domain Taxonomy Generation (v2 -- Post-Revision)

## Context

This is a re-review after the original REVISE verdict with 4 required fixes.
The experiment is infrastructure/tooling work (domain taxonomy for SOLE expert
planning), not a core scientific claim. Reviewed against that scope.

## NotebookLM Findings

Skipped. The math is cosine similarity and pair counting -- no derivations
requiring deep verification. The issues are methodological.

## Assessment of the 4 Fixes

### Fix 1: Negative Control Taxonomy -- ADEQUATE

The negative control (30 base domains x 9 paraphrases = 270) is well-designed.
The key result is that K2 discriminates (good: 3.0% PASS vs bad: 72.2% FAIL,
24.4x separation) while K1 does not (both pass). This is reported honestly.

The negative control validates that K2 catches a specific failure mode:
near-duplicate domains that differ only in phrasing. It does not validate
detection of subtler problems (e.g., domains that are semantically distinct but
produce similar experts), but that was never the claim.

One concern: the negative control is maximally easy to detect. 9 paraphrases
of each domain is extreme redundancy. A harder negative control would be 3
paraphrases of 90 domains, or domains drawn from a single narrow field
(e.g., 270 subfields of "programming"). The 24.4x separation ratio could
shrink dramatically with a more realistic failure mode. However, this is an
advisory note, not a blocking concern -- the current negative control
demonstrates K2 has nonzero discriminative power, which is what was required.

### Fix 2: Tightened Kill Criteria -- ADEQUATE

The old criteria (K1: 30% at cos>0.7, K2: 20% at cos>0.85) were vacuous.
The new criteria (K1: 5% at cos>0.5, K2: 5% at cos>0.7) pass with margins
of 11x and 1.7x respectively. The 1.7x margin on K2 is genuinely meaningful
-- it signals that scaling to 500 domains could approach failure, which is
exactly the kind of warning a metric should provide.

The HYPOTHESES.yml kill criteria (lines 1068-1069) have been updated to match
the tightened thresholds. Consistent.

Minor: the first evidence entry (lines 1072-1079) still references the old
vacuous numbers ("K1 PASS: 0.01% of pairs have cos>0.7 (3000x below 30%
threshold)"). This creates confusion when reading the evidence chronologically.
Not blocking, but should be cleaned up.

### Fix 3: Pilot-50 Proxy Validation -- ADEQUATE (reframed)

The original review asked for LoRA weight-cosine validation against pilot-50
trained experts. Instead of performing that computation, the researcher
confirmed the r=0.034 null correlation and documented it as a fundamental
limitation. This is actually a stronger response: rather than doing a
computation that might produce ambiguous results at N=50, they acknowledged
the proxy's limitation directly and repeatedly (MATH.md lines 96-107,
PAPER.md lines 148-183, Limitations section).

The reframing as "necessary-but-not-sufficient" (MATH.md line 105) is
logically sound: domains identical in embedding space would produce similar
experts (necessary), but distinct embeddings do not guarantee distinct experts
(not sufficient). This is the correct interpretation.

### Fix 4: PAPER.md Rewrite -- ADEQUATE

The paper is now notably honest. Key improvements:
- Negative control table with clear "Discriminates?" column (PAPER.md line 84-89)
- Old vs new kill criteria comparison table (lines 104-109)
- Explicit "does NOT predict" language for the proxy (line 149, bold)
- The devastating self-assessment: "You could replace the entire embedding
  analysis with 'I looked at the names and they seem different' and get
  identical predictive power" (lines 162-163)
- Status assessment: "infrastructure with weak validation" (line 245)
- Limitation 5 honestly flags: "The title says 'generation' but there is no
  algorithmic generation" (lines 215-217)

## Mathematical Soundness

Trivially correct. The only mathematical concern from the original review
remains: the scaling claim that overlap fraction follows O(k^2/N^2) (MATH.md
line 87) has no empirical support and likely breaks as domain granularity
increases. This is flagged in the assumptions (MATH.md line 137: "K2 margin
is only 1.7x, so adding fine-grained domains risks pushing it over"). Not
blocking for infrastructure.

## Novelty Assessment

Unchanged from the original review. This is infrastructure, not a research
contribution. That is fine within the project scope. The taxonomy artifact
(270 domains, 35 categories, 6 supercategories) is useful for downstream
expert planning.

## Experimental Design

The revised experiment adequately tests a limited claim: "270 domain names
are semantically distinct (by K2), and the K2 metric can detect redundancy
(by negative control)." The paper does not overclaim beyond this.

The experiment does NOT test and does not claim to test:
- Whether these domains produce useful experts
- Whether embedding proximity predicts weight-space proximity
- Whether the taxonomy is optimal

This scope honesty resolves the original review's main objection.

## Remaining Issues (non-blocking)

1. **K1 is dead weight.** The paper demonstrates K1 does not discriminate
   (both good and bad taxonomies pass). K1 should either be replaced with
   something useful (e.g., within-cluster overlap) or dropped entirely.
   Currently it is reported alongside K2 as if both contribute to the
   conclusion, but only K2 does work.

2. **Stale evidence entry in HYPOTHESES.yml.** The v1 evidence (lines
   1072-1079) references the old vacuous criteria. Should be annotated as
   superseded or removed.

3. **The "generation" misnomer.** The title says "Domain Taxonomy Generation"
   but the taxonomy is hand-crafted. The paper acknowledges this (Limitation
   5). If there were a future version, algorithmic generation (e.g., LLM
   expansion with overlap filtering) would justify the title.

## Macro-Scale Risks (advisory)

1. **K2 margin at 1.7x will not survive scaling to 500.** Adding 230 more
   fine-grained domains within existing categories will push nearest-neighbor
   cosines higher. The taxonomy may need category splitting or a cap on
   within-category count.

2. **Training data quality is the real bottleneck.** The taxonomy provides
   domain names; generating 1000 quality examples for niche domains like
   "volcanology" or "zig_programming" is a qualitatively harder problem.

3. **Domain utility variance.** Some domains are broad ("python_general"),
   others narrow ("zig_programming"). Equal-weight distillation will produce
   experts of wildly varying utility.

## Verdict

**PROCEED**

The four fixes adequately address the original review's concerns:

- Fix 1 (negative control): K2 demonstrated discriminative power with 24.4x
  separation. Adequate.
- Fix 2 (tightened criteria): margins reduced from 3000x/infinity to
  11x/1.7x. Meaningful.
- Fix 3 (proxy validation): r=0.034 honestly documented as a fundamental
  limitation rather than hand-waved. Better than what was originally requested.
- Fix 4 (paper rewrite): notably honest about what the experiment does and
  does not show.

The experiment is infrastructure with weak-but-real validation. The claim is
appropriately scoped: "270 domain names are semantically distinct, and K2 can
detect redundancy." The status of "supported" (not "proven") is appropriate.
The taxonomy artifact is useful for downstream planning regardless of the
metric's limitations.

The experiment's greatest strength is its honesty about its limitations.
