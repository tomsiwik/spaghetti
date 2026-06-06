# Peer Review: Hash Function Load Balance

## NotebookLM Findings

Skipped -- this is a pure engineering benchmark (no ML mechanism to deep-review). The mathematical content is elementary (balls-into-bins, Dirichlet distribution, Jain's index) and can be verified directly.

## Mathematical Soundness

**Variance formula (MATH.md line 29):**
The formula `Var[L_i] = T * (N-1) / (N^2 * (NV + 1))` is presented as following from a symmetric Dirichlet model. This is approximately correct for the arc-length distribution of N experts with V virtual nodes each, where arc fractions follow Dir(V, V, ..., V) (N parameters, each V). The variance of a single component is V(NV-V) / ((NV)^2(NV+1)) = (N-1)/(N^2(NV+1)), and multiplying by T^2 for the load (then noting Load = T * fraction, so Var[Load] = T^2 * Var[fraction]). The formula in MATH.md writes T instead of T^2, which is dimensionally incorrect: if L_i is a count in [0, T], variance should scale as T^2, not T. However, this formula is not used in the experiment code or conclusions, so it is cosmetic. **Minor error, non-blocking.**

**Max/min ratio approximation (MATH.md line 32-41):**
The approximation `R ~ (1 + c*sqrt(ln(N)/(NV))) / (1 - c*sqrt(ln(N)/(NV)))` with c = sqrt(2) is a standard balls-into-bins max-deviation bound. The code (line 459) correctly implements this as `c = sqrt(2*ln(N)/(N*V))` then `(1+c)/(1-c)`. The predicted values in the table (e.g., R=1.155 at V=100/N=8) are consistent with this formula. **Correct.**

**Jain's fairness index (MATH.md line 62, code line 161):**
Standard formula, correctly implemented. The relation J = 1/(1+sigma^2) holds when sigma is the CV. **Correct.**

**FNV1a analysis (MATH.md lines 43-57):**
The claim that FNV1a has insufficient mixing for structured input is well-known in the hashing literature. The specific failure mode -- adjacent expert IDs producing correlated ring positions -- is a legitimate concern for the multiply-XOR structure of FNV1a on short, structured keys. **Sound reasoning.**

**Ring arc computation (code lines 288-301):**
There is a subtle issue. Line 293: `prev_pos = positions[-1] - total_ring` when `i=0`. This computes the wraparound arc. Since `total_ring = 0xFFFFFFFF` (not 0x100000000), the modular arithmetic on line 294 `(positions[i] - prev_pos) % total_ring` uses mod (2^32 - 1) rather than mod 2^32. This introduces a 1-part-in-4-billion error per arc, completely negligible. **Technically imprecise, practically irrelevant.**

## Novelty Assessment

**This is not a novelty-driven experiment.** It is an engineering characterization: which hash function should SOLE use? The result (FNV1a is bad for short structured keys, xxHash/MurmurHash3 are good) is well-established in the systems engineering literature. Consistent hashing load balance has been studied extensively since Karger et al. 1997, and the virtual-node mitigation is standard (Stoica et al. 2001 "Chord", DeCandia et al. 2007 "Dynamo").

The novelty, such as it is, lies in quantifying this for the specific SOLE use case (hashing `(expert_id, virtual_node_index)` packed as 12 bytes). This is appropriate for a micro-experiment -- it provides the specific numbers needed for an architectural decision.

**No reinvention concern.** The experiment correctly uses existing libraries (xxhash, mmh3) rather than reimplementing hash functions.

## Experimental Design

**Strengths:**
1. Clean isolation of the variable under test (hash function only, identical ring logic).
2. Adequate sweep: 5 functions x 6 N values x 4 V values x 3 seeds = 360 configs.
3. 1M queries per config provides tight estimates (CV of load counts ~ 1/sqrt(1M/N) ~ 0.3% at N=8).
4. Multiple complementary metrics (max/min ratio, Jain index, CV, per-expert percentages).
5. Runtime is 32 seconds -- appropriate for the question being asked.

**Weaknesses and concerns:**

1. **K2 violation count discrepancy.** The `results_summary.json` reports 73 total K2 violations, while PAPER.md claims "37 non-FNV1a violations." The arithmetic checks out (73 - 36 FNV1a configs = 37 non-FNV1a), but the PAPER should explicitly state both numbers. The HYPOTHESES.yml evidence also says 37, omitting FNV1a violations from the count. This is not wrong but is selectively presented.

2. **Status marked "proven" despite K2 KILL.** The results_summary.json correctly says `"overall": "KILL"` because K2 fails. But HYPOTHESES.yml marks the experiment as `status: proven`. The PAPER reframes the K2 failure as a "nuanced kill" and changes the recommendation to V>=500. This is intellectually honest (the K2 criterion was arguably too strict), but the status should reflect the actual kill criteria outcome. Either: (a) mark as "supported" with caveats, or (b) explicitly revise the kill criteria before declaring proven. As written, the experiment KILLED itself by its own criteria and then declared victory anyway.

3. **Only 3 seeds.** For characterizing the tail behavior (max/min ratio), 3 seeds is thin. The max/min ratio is a max-of-N statistic, which has high variance. At N=128 with V=200, a single outlier arc can swing the ratio significantly. The "mean over 3 seeds" may not capture the true distribution. That said, the qualitative conclusion (FNV1a is much worse) is robust -- the gap between FNV1a (R~2-5) and alternatives (R~1.1-1.5) is too large to be a seed artifact.

4. **Query distribution is uniform, but this is correct.** The PAPER's Limitations section (item 1) notes this and correctly explains that the hash is applied to expert_id/virtual_node pairs during ring construction, not to queries. Queries are routed by position on the ring, so uniform queries test arc balance, which is exactly what matters. This is a strength, not a limitation.

5. **No confidence intervals reported.** With only 3 seeds, confidence intervals would be wide, but they should be reported to show the uncertainty. The PAPER reports means without any dispersion measure.

## Hypothesis Graph Consistency

The experiment is listed under `exp_hash_function_load_balance` in HYPOTHESES.yml with two kill criteria:
- K1: xxHash/MurmurHash3 >= 1.8x at N=8 --> PASS (correctly assessed)
- K2: any hash > 1.3x at N>=16 with V>=200 --> KILL (73 violations)

The status is `proven` but should be `supported` at best, given K2 failure. The evidence note in HYPOTHESES.yml accurately describes the nuance but the status field does not reflect it.

The experiment correctly depends on `exp_hash_ring_remove_expert` and the motivation chain is clear.

## Macro-Scale Risks (advisory)

1. **SOLE currently uses N<=50 (pilot 50).** At these scales with V=500 and xxHash32, the experiment predicts R < 1.20, which is excellent. No macro risk at current scale.

2. **At N>1000 (the vision mentions 122K+ experts), 32-bit hash space (4B positions) may show collisions.** With N=1000, V=500, that is 500K virtual nodes on a 4B ring -- still 8000 positions per virtual node, fine. At N=10000, V=500, that is 5M virtual nodes, still fine. 32-bit is adequate up to ~100K virtual nodes total.

3. **The real routing concern at scale is not hash balance but semantic relevance.** Hash routing is content-agnostic by design. The hash function choice affects balance, not quality of routing. This experiment correctly scopes to balance only.

## Verdict

**PROCEED** (with minor revisions)

The core finding is sound, well-tested, and actionable: replace FNV1a with xxHash32, increase V to 500. The experimental design is clean, the code is correct, and the conclusion follows from the data.

However, the following should be addressed:

1. **Fix HYPOTHESES.yml status.** Change from `proven` to `supported` with a note that K2 was technically triggered but resolved by increasing V to 500. Alternatively, retroactively revise K2 to "any hash > 1.3x at N>=16 with V>=500" and re-evaluate (which would pass). Either approach is honest; the current "proven despite self-KILL" is not.

2. **Fix the variance formula in MATH.md.** Line 29 should read `Var[L_i] = T^2 * (N-1) / (N^2 * (NV + 1))` (T^2 not T), or equivalently express it in terms of the fraction `f_i = L_i/T` with `Var[f_i] = (N-1)/(N^2(NV+1))`.

3. **Add dispersion to reported means.** Report standard deviation or range across the 3 seeds for key numbers (especially Table 1 in PAPER.md). The qualitative conclusions will hold, but scientific reporting requires it.

4. **Clarify the 73 vs 37 violation count.** State explicitly in PAPER.md: "73 total K2 violations, of which 36 are FNV1a (expected) and 37 are from the four alternative hash functions."

None of these are blocking. The architectural decision (xxHash32 + V>=500) is correct and well-supported.
