# Experiment Spec: cross_domain_semantic_transfer

## Objective

Test whether weight-space LoRA composition enables genuine **semantic transfer**
-- answering queries that require simultaneous understanding of two domains --
not just sequential chaining of domain operations.

The micro predecessor (exp_cross_domain_composition) tested sequential queries
like "compute 12+34, then reverse the digits." This experiment tests queries
like "Explain how a binary search algorithm is analogous to differential
diagnosis in medicine," which require both domains active simultaneously.

## Model & Data

- **Base model:** Qwen/Qwen2.5-7B, loaded in 4-bit NF4 quantization
- **HF cache:** /workspace/hf_cache
- **Adapters:** /workspace/llm/adapters/ (pilot-50 LoRA rank-16, all-modules)
- **Eval data:** Generated at runtime by the script (cross-domain query templates)
- **Judge model:** Use the base model itself (Qwen2.5-7B) as judge to avoid
  external API costs. If quality concerns arise, fall back to a scoring rubric
  with keyword/entity detection.

## Domain Pairs (8 pairs, selected from pilot-50)

The script should discover available adapters and select from these target pairs.
Fall back to whatever is available if specific adapters are missing.

```python
DOMAIN_PAIRS = [
    # Code x Science (high semantic distance)
    ("python", "medical"),
    ("python", "physics"),
    # Code x Code (low semantic distance, control)
    ("python", "rust"),
    ("python", "sql"),
    # Science x Science (medium semantic distance)
    ("medical", "chemistry"),
    ("biology", "statistics"),
    # Code x Humanities (very high semantic distance)
    ("python", "legal"),
    ("math", "ethics"),
]
```

## Cross-Domain Query Generation

For each domain pair (A, B), generate queries at three difficulty levels.
Use templates with domain-specific terms filled in. The queries MUST require
both domains simultaneously in a single answer -- NOT sequential application.

### Level 1: Domain Translation (10 queries per pair)
"Explain [concept from domain A] using the terminology and framework of
[domain B]."

Example templates:
- "Explain {A_concept} as if you were describing it to a {B_professional}."
- "Describe {A_concept} using {B_field}'s vocabulary and mental models."
- "How would a {B_professional} understand {A_concept}?"

### Level 2: Analogical Reasoning (10 queries per pair)
"How is {A_concept} similar to {B_concept}? Draw structural parallels."

Example templates:
- "What are the structural similarities between {A_concept} and {B_concept}?"
- "If {A_concept} were a {B_field} concept, what would it be and why?"
- "Draw an analogy between {A_concept} and {B_concept}, explaining the
  mapping precisely."

### Level 3: Creative Synthesis (10 queries per pair)
"Design/propose something that combines ideas from both domains."

Example templates:
- "Design a {B_field} approach inspired by {A_concept}."
- "How could {A_concept} improve current practices in {B_field}?"
- "Propose a novel framework that combines {A_concept} with {B_concept}."

### Concept Lists (hardcoded per domain)

```python
DOMAIN_CONCEPTS = {
    "python": ["recursion", "garbage collection", "list comprehension",
               "decorator pattern", "generator functions", "duck typing",
               "GIL (Global Interpreter Lock)", "context managers",
               "metaclasses", "async/await"],
    "medical": ["differential diagnosis", "triage protocol", "drug interaction",
                "immune response cascade", "clinical trial design",
                "homeostasis", "pharmacokinetics", "diagnostic imaging",
                "evidence-based medicine", "patient history assessment"],
    "physics": ["entropy", "quantum superposition", "wave-particle duality",
                "conservation laws", "electromagnetic induction",
                "thermodynamic equilibrium", "Heisenberg uncertainty",
                "special relativity", "resonance", "gravitational lensing"],
    "rust": ["ownership model", "borrow checker", "lifetime annotations",
             "zero-cost abstractions", "pattern matching", "trait system",
             "unsafe blocks", "memory safety guarantees", "enum types",
             "concurrency model"],
    "sql": ["query optimization", "normalization", "ACID transactions",
            "indexing strategies", "join operations", "window functions",
            "stored procedures", "connection pooling", "deadlock prevention",
            "schema migration"],
    "chemistry": ["molecular bonding", "reaction kinetics", "catalysis",
                  "pH buffering", "crystallography", "electron orbitals",
                  "thermochemistry", "stereochemistry", "polymerization",
                  "electrochemistry"],
    "biology": ["natural selection", "gene expression", "mitosis",
                "ecosystem dynamics", "protein folding", "symbiosis",
                "cellular respiration", "neural plasticity", "speciation",
                "epigenetics"],
    "statistics": ["hypothesis testing", "Bayesian inference",
                   "regression analysis", "confidence intervals",
                   "p-value interpretation", "sampling distributions",
                   "central limit theorem", "survival analysis",
                   "multicollinearity", "bootstrapping"],
    "legal": ["contract law", "precedent (stare decisis)", "due process",
              "liability frameworks", "intellectual property", "tort reform",
              "statutory interpretation", "burden of proof",
              "regulatory compliance", "arbitration"],
    "math": ["proof by contradiction", "eigenvalues", "topology",
             "group theory", "optimization", "differential equations",
             "graph theory", "probability distributions",
             "linear algebra", "number theory"],
    "ethics": ["trolley problem", "utilitarianism", "deontological ethics",
               "informed consent", "algorithmic fairness", "moral hazard",
               "virtue ethics", "social contract theory",
               "consequentialism", "rights-based ethics"],
    "finance": ["portfolio optimization", "risk hedging",
                "compound interest", "derivatives pricing",
                "market efficiency", "liquidity management",
                "credit risk modeling", "arbitrage",
                "capital allocation", "volatility modeling"],
}
```

### Query Generation Procedure

```python
def generate_queries(pair, level, n=10, seed=42):
    """Generate n cross-domain queries for a domain pair at given level."""
    rng = random.Random(seed)
    domain_a, domain_b = pair
    concepts_a = DOMAIN_CONCEPTS[domain_a]
    concepts_b = DOMAIN_CONCEPTS[domain_b]

    queries = []
    for i in range(n):
        ca = concepts_a[i % len(concepts_a)]
        cb = concepts_b[i % len(concepts_b)]

        if level == 1:
            templates = [
                f"Explain {ca} from {domain_a} using the terminology of {domain_b}.",
                f"Describe {ca} as a {domain_b} expert would understand it.",
                f"Translate the concept of {ca} into {domain_b} terms.",
            ]
        elif level == 2:
            templates = [
                f"What structural similarities exist between {ca} in {domain_a} "
                f"and {cb} in {domain_b}?",
                f"Draw a detailed analogy between {ca} and {cb}.",
                f"How is {ca} fundamentally similar to {cb}? Explain the mapping.",
            ]
        else:  # level 3
            templates = [
                f"Design a novel {domain_b} approach inspired by {ca} from {domain_a}.",
                f"How could the principles of {ca} improve current {domain_b} practices?",
                f"Propose a framework combining ideas from {ca} and {cb}.",
            ]

        query = rng.choice(templates)
        queries.append({"query": query, "domain_a": domain_a, "domain_b": domain_b,
                        "concept_a": ca, "concept_b": cb, "level": level})
    return queries
```

## Procedure

### Phase 1: Setup and Query Generation (~2 min)

1. Discover available adapters in /workspace/llm/adapters/
2. Match adapter names against DOMAIN_PAIRS; skip pairs where either adapter
   is missing; log which pairs are tested
3. Generate all queries: 8 pairs * 3 levels * 10 queries = 240 queries
4. Save query list to results JSON

### Phase 2: Response Generation (~25 min)

For each query Q, generate responses from 4 configurations:

1. **base**: W only (no adapters)
2. **expert_a**: W + B_a A_a (single adapter for domain A)
3. **expert_b**: W + B_b A_b (single adapter for domain B)
4. **composed**: W + 0.5 * (B_a A_a + B_b A_b) (equal-weight merge of both)

Use `PeftModel` with `add_weighted_adapter`:
```python
# For composed:
model.load_adapter(adapter_a_path, adapter_name="expert_a")
model.load_adapter(adapter_b_path, adapter_name="expert_b")
model.add_weighted_adapter(
    adapters=["expert_a", "expert_b"],
    weights=[1.0, 1.0],  # equal contribution
    adapter_name="composed",
    combination_type="linear"
)
model.set_adapter("composed")
```

Generation parameters:
- max_new_tokens: 512
- temperature: 0.7
- top_p: 0.9
- do_sample: True
- num_return_sequences: 1
- 3 seeds: [42, 123, 456]

Memory management: load one adapter config at a time, delete and gc.collect()
between configurations. Use 4-bit NF4 quantization throughout.

### Phase 3: Automated Scoring (~15 min)

Score each response on two dimensions using automated heuristics (NOT LLM judge,
to avoid cost and to keep scores deterministic):

**M1: Domain Coverage Score (keyword-based)**
For each domain, maintain a set of 20+ domain-specific keywords/phrases.
Count how many from each domain appear in the response.

```python
DOMAIN_KEYWORDS = {
    "python": ["function", "variable", "loop", "class", "import", "def",
               "return", "list", "dictionary", "string", "integer",
               "exception", "module", "library", "syntax", "indentation",
               "interpreter", "runtime", "debug", "stack"],
    "medical": ["patient", "diagnosis", "symptom", "treatment", "clinical",
                "disease", "therapy", "prognosis", "pathology", "dosage",
                "pharmaceutical", "surgical", "anatomical", "chronic",
                "acute", "vital signs", "laboratory", "imaging", "biopsy",
                "protocol"],
    # ... similar for all domains
}

def domain_coverage(response, domain_a, domain_b):
    words_a = sum(1 for kw in DOMAIN_KEYWORDS[domain_a] if kw.lower() in response.lower())
    words_b = sum(1 for kw in DOMAIN_KEYWORDS[domain_b] if kw.lower() in response.lower())
    # Normalize by keyword list size
    cov_a = min(1.0, words_a / 5)  # 5+ keywords = full coverage
    cov_b = min(1.0, words_b / 5)
    return min(cov_a, cov_b)  # Both domains must be covered
```

**M2: Integration Score (sentence-level co-occurrence)**
Measure whether domain keywords from BOTH domains appear in the same sentences
(integrated) vs only in separate paragraphs (separated).

```python
def integration_score(response, domain_a, domain_b):
    sentences = response.split(".")
    integrated = 0
    for sent in sentences:
        has_a = any(kw.lower() in sent.lower() for kw in DOMAIN_KEYWORDS[domain_a])
        has_b = any(kw.lower() in sent.lower() for kw in DOMAIN_KEYWORDS[domain_b])
        if has_a and has_b:
            integrated += 1
    return min(1.0, integrated / 3)  # 3+ integrated sentences = score 1.0
```

**M3: Response Length Ratio**
Ratio of composed response length to base response length. Values near 1.0
are neutral; values << 1.0 suggest the model is confused (short/degenerate);
values >> 1.0 may suggest verbose padding.

**M4: Perplexity (of the response itself)**
For each configuration, compute the NTP loss of the generated response as
a rough quality signal. Lower loss = more fluent/confident.

### Phase 4: LLM-as-Judge Pairwise Comparison (~15 min)

For a subset of queries (first 5 per pair = 40 queries), use the base model
itself as a judge to compare composed vs base and composed vs best-single-expert.

Judge prompt template:
```
You are evaluating two responses to a cross-domain question.
The question requires knowledge from both {domain_a} and {domain_b}.

Question: {query}

Response A:
{response_a}

Response B:
{response_b}

Which response better integrates knowledge from BOTH domains?
Consider: (1) accuracy in both domains, (2) how well the domains are
connected, (3) depth of cross-domain insight.

Answer with ONLY "A" or "B" or "TIE".
```

Use greedy decoding (temperature=0) for deterministic judging.
Randomize A/B order to avoid position bias; run each comparison twice with
swapped order and take majority.

### Phase 5: Results Aggregation

Compute per kill criterion:

**K1 Assessment:**
```python
# Domain coverage degradation vs base
base_scores = [domain_coverage(r, da, db) for r in base_responses]
comp_scores = [domain_coverage(r, da, db) for r in composed_responses]
degradation = 1 - mean(comp_scores) / mean(base_scores)
k1_pass = degradation <= 0.20
```

**K2 Assessment:**
```python
# Win rate: composed vs best single expert
n_composed_worse = sum(1 for q in queries
                       if score(composed[q]) < max(score(expert_a[q]),
                                                    score(expert_b[q])))
k2_fail_rate = n_composed_worse / len(queries)
k2_pass = k2_fail_rate <= 0.50
```

## Kill Criteria Assessment

- **K1:** domain coverage degradation of composed vs base > 20% --> FAIL
  Measured as: mean domain coverage of composed / mean domain coverage of base.
  Both domains must appear in responses; single-domain responses score 0.

- **K2:** composed worse than best single expert on > 50% of queries --> FAIL
  Measured as: for each query, compare composed domain-coverage * integration
  score vs the same metric for the better of the two single experts.

## Output

Save results to: `/workspace/llm/results/cross_domain_semantic_transfer/results.json`

Required fields in JSON:
```json
{
  "experiment": "cross_domain_semantic_transfer",
  "timestamp": "ISO-8601",
  "runtime_seconds": 0.0,
  "model": "Qwen/Qwen2.5-7B",
  "quantization": "nf4",
  "n_domain_pairs": 8,
  "n_queries_per_level": 10,
  "n_levels": 3,
  "n_seeds": 3,
  "total_queries": 240,

  "domain_pairs_tested": [
    {"domain_a": "python", "domain_b": "medical", "n_queries": 30}
  ],

  "per_pair_results": {
    "python_medical": {
      "domain_coverage": {
        "base": {"mean": 0.0, "std": 0.0},
        "expert_a": {"mean": 0.0, "std": 0.0},
        "expert_b": {"mean": 0.0, "std": 0.0},
        "composed": {"mean": 0.0, "std": 0.0}
      },
      "integration_score": {
        "base": {"mean": 0.0, "std": 0.0},
        "composed": {"mean": 0.0, "std": 0.0}
      },
      "per_level": {
        "level_1": {"coverage_degradation": 0.0, "k2_fail_rate": 0.0},
        "level_2": {"coverage_degradation": 0.0, "k2_fail_rate": 0.0},
        "level_3": {"coverage_degradation": 0.0, "k2_fail_rate": 0.0}
      }
    }
  },

  "aggregate": {
    "mean_domain_coverage_base": 0.0,
    "mean_domain_coverage_composed": 0.0,
    "coverage_degradation_pct": 0.0,
    "mean_integration_base": 0.0,
    "mean_integration_composed": 0.0,
    "k2_fail_rate": 0.0,
    "judge_win_rate_composed_vs_base": 0.0,
    "judge_win_rate_composed_vs_best_single": 0.0
  },

  "kill_criteria": {
    "K1": {"metric": "coverage_degradation_pct", "threshold": 20.0,
           "measured": 0.0, "status": "PASS/FAIL"},
    "K2": {"metric": "k2_fail_rate", "threshold": 50.0,
           "measured": 0.0, "status": "PASS/FAIL"}
  },

  "per_query_detail": [
    {
      "query": "...",
      "domain_a": "python",
      "domain_b": "medical",
      "level": 1,
      "seed": 42,
      "responses": {
        "base": "...",
        "expert_a": "...",
        "expert_b": "...",
        "composed": "..."
      },
      "scores": {
        "domain_coverage": {"base": 0.0, "expert_a": 0.0,
                            "expert_b": 0.0, "composed": 0.0},
        "integration": {"base": 0.0, "composed": 0.0},
        "response_length": {"base": 0, "composed": 0}
      }
    }
  ]
}
```

## Constraints

- **Max runtime:** 90 minutes (generation-heavy experiment)
- **Expected GPU memory:** ~8GB with 4-bit NF4 Qwen2.5-7B + 2 LoRA adapters
- **Must support SMOKE_TEST=1:** When set, use 2 domain pairs, 2 queries per
  level, 1 seed, skip judge phase. Should complete in < 5 minutes.
- **No external API calls.** All generation and judging done locally on the GPU.
- **Deterministic query generation.** Same seed -> same queries for reproducibility.
- **Memory management:** Unload adapters between configurations. Do not hold all
  4 model configurations in memory simultaneously.

## Important Design Notes

1. **This is NOT a PPL-only experiment.** Previous experiments measured quality
   via perplexity on held-out text. This experiment generates free-form responses
   and evaluates them for cross-domain integration quality. The response content
   matters, not just the loss.

2. **The base model is a strong baseline here.** Qwen2.5-7B has already seen
   programming and medical texts in pretraining. The question is whether adding
   LoRA experts IMPROVES cross-domain responses, not whether the base model can
   answer at all. If the base model already answers well, experts need to
   demonstrably add value.

3. **Keyword-based scoring is a floor estimate.** It will miss sophisticated
   integration that uses paraphrases instead of exact keywords. If keyword
   scores are too noisy, the judge pairwise comparison provides a fallback
   quality signal.

4. **Equal-weight composition only.** We deliberately test the simplest
   composition (1:1 weighting). If this works, weighted composition (from
   the PPL-probe experiment) would only improve. If this fails, it motivates
   routing/weighting as essential for semantic transfer.
