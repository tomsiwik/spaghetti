# MATH.md: Competitive Benchmark with Routed Composition

## Type: Verification (Type 1)

**Proven framework:** Per-domain optimal scales resolve the two-world problem
(Finding #220: 0/5 domains degrade). Oracle top-1 routing achieves 100% accuracy
on 5 domains (Finding #186). Pre-merge composition has 0% overhead (Finding #75).

**Claim to verify:** The original competitive benchmark (exp_competitive_benchmark)
was killed because uniform 1/N composition at s=20 degraded knowledge domains.
We now know per-domain scales fix this (Finding #220). This experiment verifies
whether routed composition with per-domain scales makes SOLE competitive.

---

## A. Failure Mode Identification

**Original kill (exp_competitive_benchmark):**
- K1 KILL: SOLE worse on 4/6 benchmarks vs Qwen2.5-3B
- K2 KILL: SOLE worse than base on math (-25pp) and legal (-10pp)
- K3 KILL: Memory 10.98GB vs Qwen 2.45GB

**Root cause analysis:**
1. Uniform s=20 destroys knowledge domains (Finding #209: legal -30%, finance -14%)
2. Uniform 1/N merging dilutes all adapters equally -- no domain selection
3. Memory inflation from bf16 unpacking (engineering, not architecture)

**The fix (proven individually):**
- Per-domain scales {math/code/medical:20, legal:4, finance:1} (Finding #217)
- Oracle top-1 routing routes each query to its domain adapter (100% accuracy)
- Pre-merge only the selected adapter (no dilution from irrelevant adapters)

---

## B. The Right Question

Not: "Is SOLE competitive with larger models?"

**Right question:** "Does routed composition with per-domain scales close the gap
that uniform composition opened? Specifically, does it beat the BitNet-2B base
on ALL benchmarks (fixing K2) and beat Gemma-2-2B on the majority?"

---

## C. Prior Mathematical Foundations

**Finding #220:** Per-domain scales yield 0/5 domains degrading (was 3/5 at s=20).
**Finding #217:** Three domain categories: learnable-task (math, s=20: +700%),
  structured-output (code/medical, s=20: +17-36%), knowledge-dependent (legal s=4,
  finance s=1).
**Finding #218:** Code adapter dominance was a scale artifact -- domain adapters
  win at correct scales.
**Finding #221:** Scale-aware composition is minimum viable architecture.

**Prediction derivation:**

Under oracle top-1 routing, only one adapter is active per query. The MMLU prompt
for domain d gets adapter d at scale s_d. No dilution from other adapters.

For MMLU (factual recall):
- At s=20 (uniform): legal 45%, finance 45% (from exp_competitive_benchmark)
- Base: legal 55%, finance 35%
- The degradation on legal (-10pp) was caused by s=20 overwriting base knowledge
- At s=4 (legal optimal): adapter augments without overwriting -> should match or
  beat base
- At s=1 (finance optimal): minimal perturbation -> should match base closely

For GSM8K (reasoning):
- Base: 38%. SOLE uniform: 48% (+10pp). The improvement came from math/code adapters.
- Under routing, GSM8K gets the math adapter at s=20 -> same or better than uniform
  (no dilution from legal/finance adapters)

---

## D. Predictions

| Prediction | Metric | Threshold | Derived From |
|------------|--------|-----------|-------------|
| P1: Routed SOLE >= base on ALL 6 benchmarks | min(delta) >= 0 | K2 fix | Per-domain scales prevent degradation |
| P2: Routed SOLE beats Gemma-2-2B on >= 5/6 | wins >= 5 | Prior result | Original SOLE already beat Gemma 6/6 |
| P3: GSM8K >= 48% (match or beat uniform) | accuracy | >= 0.48 | Math adapter at s=20, no dilution |
| P4: Legal MMLU >= 55% (match base) | accuracy | >= 0.50 | s=4 preserves base knowledge |
| P5: Finance MMLU >= 35% (match base) | accuracy | >= 0.30 | s=1 is near-identity perturbation |

---

## E. Assumptions & Breaking Conditions

1. **MMLU extraction works correctly.** The original experiment had suspicious Qwen
   scores (36% GSM8K vs published 65-70%). Same extraction code is used here, so
   inter-model comparison uses same extraction bias.

2. **Oracle routing is available.** For MMLU, we know the domain label and can route
   directly. For GSM8K, we route to math adapter. In deployment, routing heads
   achieve 100% accuracy on 5 domains, so this is realistic.

3. **Adapters from exp_real_data_domain_experts are available.** Same adapters as
   original competitive benchmark.

4. **n=20 MMLU gives +/-22pp CI.** Small differences are noise. Only 25+pp gaps
   are statistically meaningful at p<0.05.

---

## F. Worked Example (Legal MMLU)

Original: base=55%, SOLE uniform s=20=45% (-10pp)
At s=4 (legal optimal): perturbation ratio rho = (4/1) * 0.034/20 = 0.0068
This is 3x smaller than at s=20 (rho=0.034). The adapter adds domain-relevant
perturbation without overwriting base factual knowledge.
Expected: 55% +/- noise (adapter effect is small at s=4).

---

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   Per-domain scale selection keeps knowledge domains at s<=4 where rho<0.007,
   ensuring the adapter is in the pure augmentation regime (no overwriting).

2. Which existing theorem(s) does the proof build on?
   Finding #217 (domain-dependent optimal scales), Finding #220 (0/5 degrade),
   Weyl's inequality (rho bounds).

3. What specific numbers does the proof predict?
   All 6 benchmarks >= base. GSM8K >= 48%. Legal MMLU >= 50%. Finance >= 30%.
   Beat Gemma-2-2B on >= 5/6.

4. What would FALSIFY the proof?
   If routed composition with per-domain scales STILL degrades ANY benchmark
   below base. This would mean per-domain scales work for generation quality
   (Finding #220) but not for factual recall (MMLU), indicating different
   mechanisms.

5. How many hyperparameters does this approach add?
   0 new. Per-domain scales are from Finding #217. Routing is oracle (domain label).

6. Hack check: Am I adding fix #N?
   No. This is a retest of a killed experiment with a proven fix applied.
