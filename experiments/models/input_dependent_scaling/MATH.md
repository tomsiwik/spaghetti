# MATH.md: Input-Dependent Adapter Scaling

## Type: Guided Exploration

**Proven framework:** LoRA perturbation theory with domain-dependent scaling.
**Unknown:** Whether query-centroid embedding similarity predicts optimal per-query scale.

---

## A. Failure Mode Identification

**The disease:** Fixed per-domain scale lookup assumes all queries within a domain
need the same perturbation magnitude. Finding #252 killed the universal binary
{FORMAT, CAPABILITY} model -- code is noisy/non-monotonic, medical has near-zero
adapter effect. Even within math (where s*=[4,6] transition exists), per-prompt
correctness "flickers near threshold" (Finding #250 caveat).

**Degenerate behavior:** A query that is poorly matched to the training distribution
of adapter D receives the same scale as a perfectly matched query. For code domain
(Finding #252), some prompts improve with the adapter while others degrade -- the
adapter helps on in-distribution queries but hurts on out-of-distribution ones.
Fixed scale cannot distinguish these cases.

**Is this the root cause?** Partially. The deeper cause is that adapter quality varies
per-query (adapter was trained on specific data). Scale is the cheapest control
surface: routing already selects WHICH adapter (Finding #247, 90% accuracy);
scale controls HOW MUCH of that adapter to apply.

---

## B. The Right Question

**Wrong question:** "How do we find the best scale for each domain?"
(Already answered: Finding #249 gives per-domain optima.)

**Right question:** "Given a query q and domain centroid c_d, can the similarity
sim(q, c_d) predict whether q is in-distribution for adapter d, and should the
scale be modulated accordingly?"

The answer should come from the geometry of the TF-IDF embedding space:
queries close to the centroid are well-represented by the training data,
queries far from it are not.

---

## C. Mathematical Framework

### Definition 1 (TF-IDF Embedding Space)
Let phi: Text -> R^p be the TF-IDF vectorizer (p = 5000 features, bigrams,
sublinear TF). For domain d, define the centroid:

  c_d = (1/|T_d|) * sum_{t in T_d} phi(t)

where T_d is the training set for domain d.

### Definition 2 (Cosine Similarity)
For query q:

  sim(q, d) = <phi(q), c_d> / (||phi(q)|| * ||c_d)||)

This measures how well q aligns with the training distribution of adapter d.

### Definition 3 (Scale Function)
The input-dependent scale is:

  s(q, d) = s_d * f(sim(q, d))

where s_d is the fixed per-domain optimal scale (from Finding #249) and
f: [0,1] -> [0,1] is a monotone non-decreasing mapping.

### Proposition 1 (Monotone Scale-Similarity Relationship)
**Claim:** If adapter d was trained on data drawn from distribution D_d, and
the TF-IDF embedding preserves distributional similarity (i.e., queries similar
to training data in TF-IDF space are also similar in the model's representation
space), then:

  E[quality(q, s)] is maximized when s is monotonically related to sim(q, d).

**Justification:** This follows from the structure of LoRA perturbation. The
adapter output is Delta(x) = s * B @ A @ x, where A, B were trained on D_d.
For x in-distribution (high sim), B@A@x is well-calibrated and scale s_d is
appropriate. For x out-of-distribution (low sim), B@A@x may be miscalibrated
and a reduced scale hedges toward the base model.

This is NOT a formal theorem -- the TF-IDF-to-representation-space assumption
is precisely the unknown this guided exploration tests.

### Proposition 2 (Bounded Degradation)
**Claim:** Input-dependent scaling with f(sim) in [alpha, 1] for alpha > 0
cannot produce worse output than the base model (scale=0) provided:
1. The base model produces non-degenerate output at scale=0
2. alpha * s_d > 0 (minimum perturbation is non-zero)

**Note:** This does NOT guarantee improvement over fixed scaling. It only
guarantees that the modulated output stays within the interpolation between
base (s=0) and fully-adapted (s=s_d) responses.

### Choice of f(sim)
We test three mappings:

1. **Linear:** f(sim) = max(alpha, sim)
2. **Thresholded:** f(sim) = 1 if sim > tau, else alpha
3. **Softmax-weighted:** f(sim) = sum_d w_d(q) * s_d / s_d (multi-domain blend)

For this experiment, we use the simplest: **linear mapping** with floor alpha.

---

## D. Predictions

### Behavioral Predictions (from the framework)

| # | Prediction | Rationale |
|---|-----------|-----------|
| P1 | Embedding-similarity correlates with behavioral score (r > 0.3) | If TF-IDF space preserves distributional similarity, in-distribution queries should score higher |
| P2 | Per-query scaling improves over fixed on >= 2/3 domains | Modulating scale for OOD queries should reduce interference |
| P3 | Math domain shows smallest improvement (possibly none) | Math's binary eval + format activation means scale > s* all score the same |
| P4 | Code domain shows largest improvement | Code has bimodal per-prompt behavior (Finding #252) -- some prompts hurt by adapter |
| P5 | Coherence maintained (< 20% incoherent) | Scale modulation stays within [alpha*s_d, s_d], never below base |

### Quantitative Predictions

| Metric | Fixed baseline | Input-dependent predicted | Source |
|--------|---------------|-------------------------|--------|
| Math score | 0.80 | 0.70-0.80 | Finding #238; binary eval, most queries above threshold |
| Code score | 0.62 | 0.65-0.75 | Reducing scale for hurt-by-adapter prompts |
| Medical score | 0.29 | 0.28-0.32 | Near-zero adapter effect limits ceiling |

### Kill Criteria Derivation

- **K1 (#663):** If 0/3 domains improve, the similarity-scale relationship doesn't
  hold -- TF-IDF space doesn't predict adapter effectiveness.
- **K2 (#664):** r < 0.3 means embedding similarity is not predictive of behavioral
  quality, so any improvement is random.
- **K3 (#665):** > 20% incoherent output means scale modulation damages generation.

---

## E. Assumptions & Breaking Conditions

| # | Assumption | If violated |
|---|-----------|-------------|
| A1 | TF-IDF embedding preserves distributional similarity to adapter training data | Similarity is meaningless; K2 fails |
| A2 | Adapter quality varies within domain (some queries better than others) | No benefit to per-query scaling; K1 fails |
| A3 | Scale modulation in [alpha*s_d, s_d] doesn't cause degenerate generation | Incoherent output; K3 fails |
| A4 | n=10 per domain is sufficient to detect improvement | Statistical noise may swamp signal |

**A4 is the weakest assumption.** With n=10 and binary scoring (math), detecting
a 1-2 prompt difference is not statistically significant. This experiment is
directional, not definitive.

---

## F. Worked Example (TF-IDF similarity computation)

Consider a math query q = "What is 15% of 80?"

1. phi(q) = TF-IDF([15, percent, 80]) -> sparse vector in R^5000
2. c_math = centroid of 400 GSM8K training instructions
3. sim(q, math) = cos(phi(q), c_math) = 0.35 (typical for short queries)
4. sim(q, code) = cos(phi(q), c_code) = 0.02 (very different domain)
5. Routing: argmax sim = math (correct)
6. Scale: s(q, math) = 20.0 * max(0.3, 0.35) = 20.0 * 0.35 = 7.0

Compare: fixed scale would give 20.0. If this query is well-matched,
7.0 may be too low. If poorly matched, 7.0 hedges toward base.

The linear mapping f(sim) = max(alpha, sim) has the issue that typical
cosine similarities in TF-IDF space are low (0.1-0.5 for in-domain).
We may need normalization: f(sim) = max(alpha, sim / max_sim_d) where
max_sim_d is calibrated per domain.

---

## G. Complexity & Architecture Connection

**Additional compute:**
- TF-IDF vectorization: O(|q| * p) per query -- negligible
- Cosine similarity: O(p * D) per query for D domains -- negligible
- Total overhead: < 1ms per query

**No additional parameters** beyond the TF-IDF vectorizer + centroids (already
exist from Finding #247).

**Architecture connection:** This is a minimal router enhancement. Instead of
router -> (domain, fixed_scale), we get router -> (domain, modulated_scale).
The scale modulation adds zero parameters and negligible compute.

---

## Self-Test

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   This is NOT a proof-of-impossibility experiment. It is guided exploration testing
   whether TF-IDF similarity predicts adapter effectiveness (Assumption A1).

2. **Which existing theorem(s) does the proof build on?**
   LoRA perturbation structure (Delta = s * B @ A @ x); TF-IDF distributional
   semantics (Salton & Buckley 1988, term frequency-inverse document frequency).
   Finding #249 (per-domain scaling is architecturally required).

3. **What specific numbers does the proof predict?**
   P1: r > 0.3 (sim-score correlation). P2: >= 2/3 domains improve.
   P3: Math improvement minimal. P4: Code improvement largest. P5: < 20% incoherent.

4. **What would FALSIFY the proof (not just the experiment)?**
   If TF-IDF similarity anti-correlates with behavioral score (r < -0.3), the
   framework's assumption that embedding proximity predicts adapter effectiveness
   is wrong, not just imprecise.

5. **How many hyperparameters does this approach add?**
   Count: 1 (alpha, the floor on the scale mapping).
   Why can't it be derived: alpha = 0 means "fall back to base model for OOD queries"
   which may or may not be optimal. The mathematical framework suggests alpha should
   correspond to the FORMAT/CAPABILITY threshold (s*=4-6 for math), but this is
   domain-dependent and the exact mapping is unknown.

6. **Hack check:** No stacking. This is one mechanism (scale modulation by similarity)
   with one hyperparameter (alpha). The routing (Finding #247) and per-domain scales
   (Finding #249) are proven infrastructure, not additional fixes.
