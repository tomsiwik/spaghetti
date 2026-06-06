# Re-Review: Sheaf Cohomology Dimension Estimation

## Revision Check

**Fix 1: Betti formula presentation.**
ADDRESSED. MATH.md line 283 now correctly shows "dim(H^1) = E - |V| + c = 6 - 5 + 2 = 3"
using all 5 vertices and c=2 connected components. PAPER.md line 62 matches. The
equivalent K_4 subgraph calculation (6 - 4 + 1 = 3) is given as a parenthetical
consistency check, not as the primary derivation. Clean fix.

**Fix 2: Rank Budget Bound demoted to conjecture.**
ADDRESSED. MATH.md line 83 now reads "Conjecture (Rank Budget Bound)" instead of
"Theorem." Lines 87-94 explicitly state "this has NOT been formally proven" and
identify the specific gap: "A formal proof would need to establish a map from scalar
H^1 cycles to rank deficiency in the vector-valued restriction maps -- this remains
an open question." This is honest and well-framed.

**Fix 3: Scalar-vs-vector H^1 distinction.**
ADDRESSED. Three places now clarify the distinction:
- MATH.md lines 96-101: "The scalar H^1 = 3 counts independent CYCLES in the nerve
  graph, not independent directions of incompatibility in R^{2560}."
- PAPER.md lines 78-81: Bold "Important" clarification in the results section.
- PAPER.md lines 162-169 (Implications): "lower bound on the NUMBER of independent
  conflicts, NOT a proven rank budget in R^{2560}."
This was the most important fix and it is done thoroughly.

**Fix 4: Curry citation corrected.**
ADDRESSED. MATH.md lines 5 and 48 now cite "Curry, 1303.3255" (Sheaves, Cosheaves
and Applications). PAPER.md line 36 matches. The old incorrect ID (2012.37428)
is gone.

**Fix 5: P3 acknowledged as ill-formed.**
ADDRESSED. MATH.md lines 191-198 strike through the old P3 and add a clear
explanation: "This prediction was ill-formed. The topological H^1 (Betti number)
depends entirely on the Cech nerve, which is determined by PPL-based specialization
rankings -- these are layer-independent by construction." PAPER.md line 17 echoes
this in the prediction table. Good intellectual honesty.

## Remaining Issues

**Minor (not blocking):**

1. **Self-Test P3 inconsistency.** The Self-Test section (MATH.md lines 325-326)
   still lists P3 as "dim(H^1) peaks at intermediate layers" without noting it
   was acknowledged as ill-formed. The main body handles this correctly (lines
   191-198), but the Self-Test repeats the stale prediction. Should add a note
   like "(ill-formed -- see section E)" for consistency.

2. **Self-Test Q1 still overpromises.** "Non-trivial H^1 identifies the EXACT
   obstruction directions" -- as clarified by Fix 3, scalar H^1 counts cycles,
   not directions in representation space. The word "EXACT" is misleading given
   the scalar-vs-vector gap. This is a diagnostic experiment; the impossibility
   property is not yet established. Acceptable for guided exploration but the
   phrasing should match the now-honest body text.

3. **Borsuk nerve theorem still cited but not used.** Self-Test Q2 cites
   "Cech nerve theorem (Borsuk 1948) -- nerve captures homotopy type of the cover."
   As noted in the original review, this theorem requires contractible intersections,
   and for discrete sample sets the theorem is trivially satisfied but uninformative
   (nerve has same homotopy type as the union = full sample set). The Betti number
   is computed directly from the nerve graph, not via the nerve theorem. The citation
   is technically not wrong but is vestigial. Not blocking.

**None of these are blocking.** The core mathematical claims are now honest, the
conjecture is properly labeled, and the scalar-vs-vector distinction is clear.

## Verdict

**PROCEED**

All five required fixes have been substantively addressed. The mathematical claims
are now honest: the Rank Budget Bound is a conjecture (not a theorem), the scalar
H^1 = 3 is clearly distinguished from representation-space obstruction dimensions,
the Betti formula is correctly presented, the Curry citation is fixed, and the
ill-formed P3 prediction is acknowledged. The remaining issues are minor
presentation inconsistencies in the Self-Test section that do not affect the
validity of the finding. The core result -- non-trivial Cech nerve topology at
k=2 that collapses at k=3, with H^1 = 3 identifying 3 independent conflict
cycles -- is sound, honestly presented, and useful for guiding bridge adapter design.
