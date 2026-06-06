# Peer Review: Self-Routing Adapters (Revision Review)

## Scope

This is a revision review. The previous adversarial review requested 7 specific fixes.
Each is verified below against MATH.md and PAPER.md as revised.

## Fix Verification

### Fix 1: Added 6000-step Gumbel baseline (90.41% top-2) as correct comparison point

**VERIFIED.** PAPER.md Section "Comparison with trained Gumbel-sigmoid router" now explicitly
states that the 6000-step router reaches 90.41% top-2 and 83.67% top-1, surpassing centroid
routing on both metrics. The comparison table includes both the 3000-step (86.3%) and 6000-step
(90.4%) baselines. MATH.md computational cost table also includes the 6000-step row. The
narrative no longer claims centroid routing "beats" Gumbel-sigmoid -- it correctly frames the
centroid method as matching the undertrained router but losing to the properly trained one.

### Fix 2: Added SE=1.52pp at n=490; noted 0.81pp gap not significant

**VERIFIED.** PAPER.md includes a "Statistical context" table showing SE = 1.52pp for n=490,
the centroid-vs-3000-step gap of +0.81pp labeled "No (< 1 SE)", and the 6000-step-vs-centroid
gap of +3.27pp labeled "Marginal (~2 SE)". Limitation 7 notes "With 490 samples, only gaps
>3pp are reliably detectable." The statistical framing is honest.

### Fix 3: Labeled per-domain analysis as activation-norm-specific

**VERIFIED.** PAPER.md Key Finding 1 now includes the explicit disclaimer: "Note: this
per-domain analysis is for the activation-norm method specifically, not the centroid method."
This prevents readers from attributing the 40/49 zero-accuracy failure pattern to the
centroid method.

### Fix 4: Reframed "zero params" to "closed-form/non-learned" (125K computed floats, 500KB)

**VERIFIED.** MATH.md Method D section now states: "This method has zero *learned* parameters
but requires 125,440 *computed* parameters (the stored centroids) plus 20 labeled examples
per domain for centroid estimation. The honest comparison with Gumbel-sigmoid (659K learned
params) is: closed-form solution from labeled data vs. gradient-optimized solution from
labeled data." PAPER.md uses "closed-form baseline" and "non-learned baseline" consistently.
The computational cost table labels centroid params as "125K computed" and Gumbel params as
"659K learned." The misleading "zero params" / "zero cost" language has been eliminated.

### Fix 5: Reframed narrative: centroid is cheap baseline/fallback, not replacement for learned routing

**VERIFIED.** PAPER.md Key Finding 2 heading now reads "Hidden-State Centroids as Closed-Form
Baseline" (not "the winner" or replacement). The "Honest framing" paragraph explicitly lists
four value propositions: zero training cost, instant deployment, initialization for learned
routers, and fallback when training budget is limited. The conclusions state: "it is surpassed
by a properly trained 6000-step Gumbel-sigmoid router (90.41% top-2)." The narrative no
longer overclaims.

### Fix 6: Added experiment DB section (exp_self_routing_adapters, K1 id=249)

**VERIFIED.** PAPER.md includes an "Experiment DB" section with experiment ID
`exp_self_routing_adapters`, kill criterion ID 249, status SUPPORTED (with caveats), and
dependency on the Gumbel-sigmoid ablation. The kill criteria assessment table correctly shows
K1 PASSES on the centroid method while acknowledging the winning method is not adapter-weight-
based.

**Note:** The experiment is still not present as a node in HYPOTHESES.yml. The previous review
said "Add the experiment to HYPOTHESES.yml or link it to an existing routing node." The paper
added an inline DB section, which partially addresses this, but the experiment remains orphaned
from the hypothesis graph. This is a minor bookkeeping issue, not a blocking concern.

### Fix 7: Separated concentration theorem (random subspaces, theoretical) from empirical observation (trained B-matrices behave similarly)

**VERIFIED.** Both MATH.md and PAPER.md now clearly separate theory from empirics. MATH.md
has two explicit subsections: "Theoretical intuition: Concentration of Measure" (with the
theorem stated for random subspaces) and "Empirical observation: B-matrices behave like random
subspaces" (with measured routing accuracy as the actual evidence). The "Important caveat"
paragraph states: "This theorem requires both the subspace S and vector u to be random... The
B-matrices are *trained* on domain-specific data and are NOT random subspaces. A sufficient
condition for B-matrix routing to work would be if domains induced *structured* hidden states
that aligned differently with each adapter's learned subspace. The concentration theorem does
not rule this out in principle." PAPER.md mirrors this: "While this theorem strictly requires
random subspaces and random vectors, the empirical results confirm that trained B-matrices
behave similarly... The theorem provides the correct intuition, but the empirical failure is
the actual evidence."

This is exactly the separation requested. The paper no longer conflates the theorem with the
empirical observation.

## New Issues Check

### Mathematical soundness

No new mathematical errors introduced. The SE calculation sqrt(0.87 * 0.13 / 490) = 1.52pp
is correct. The concentration theorem statement remains standard. The FLOPs accounting is
consistent between MATH.md and PAPER.md.

### Novelty framing

The revised paper correctly cites Prototypical Networks (Snell et al., NeurIPS 2017) and
frames centroid routing as NCC applied to adapter selection. The claim of novelty is modest
and appropriate: the systematic negative result on B-matrix routing is the primary
contribution, with centroid routing positioned as a known technique applied to a new context.

### Hypothesis graph consistency

The K1 criterion ("implicit routing accuracy >= 50%") is somewhat awkwardly passed by a
method (centroid cosine) that is not "implicit" in the originally hypothesized sense. The
paper acknowledges this pivot explicitly. The status "SUPPORTED (with caveats)" is honest.
The experiment should eventually be added to HYPOTHESES.yml as a proper node, but this is
bookkeeping.

### Claims vs evidence alignment

No overclaiming detected. The conclusions are well-calibrated:
1. B-matrix self-routing is killed (clear evidence)
2. Centroid routing is a strong baseline (not a replacement)
3. Training budget is the bottleneck (supported by ablation data)

### One residual concern (non-blocking)

The "What Would Kill This" section opens with "Properly trained routers already beat centroid
routing (6000-step Gumbel at 90.41%)." This is accurate and honest, but it raises a framing
question: if the method is already beaten by the comparison, should the status be "SUPPORTED"?
The answer is yes, because the value proposition is explicitly scoped to zero-training-cost
scenarios, and 87.14% top-2 is a legitimately useful fallback. The paper is clear about this.

## Macro-Scale Risks (advisory)

Unchanged from the previous review. The key risks remain:
1. Mixed-domain text where mean-pooled hidden states produce ambiguous centroids
2. Per-token routing (untested; individual tokens may not cluster as cleanly)
3. Domain overlap at N>100 where centroid space becomes crowded
4. The 6000-step Gumbel-sigmoid already beats this, so the "zero training cost" value
   proposition must be weighed against the cost of computing and caching centroids

These are acknowledged in the paper's Limitations section (items 4-6) and "What Would Kill
This" section. No new macro risks identified.

## Verdict

**PROCEED**

All 7 requested fixes have been properly applied. The revised paper is honest about its
claims, correctly separates theory from empirics, includes appropriate statistical context,
and positions centroid routing as a cheap baseline rather than a replacement for learned
routing. The mathematical content is sound. No new issues were introduced during revision.

Minor bookkeeping: add `exp_self_routing_adapters` as a node in HYPOTHESES.yml when
convenient. This is not blocking.
