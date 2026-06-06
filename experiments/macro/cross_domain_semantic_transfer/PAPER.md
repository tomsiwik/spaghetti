# Cross-Domain Semantic Transfer via LoRA Composition: Research Digest

## Hypothesis

Weight-space composition of two domain-specialized LoRA experts enables genuine
semantic transfer -- answering queries that require simultaneous understanding of
both domains -- not just sequential chaining of domain operations.

## What This Model Is

This experiment tests whether additive LoRA composition (W + B_i A_i + B_j A_j)
can produce responses that integrate knowledge from two distinct domains in a
single coherent answer. Unlike the micro predecessor (exp_cross_domain_composition),
which tested sequential queries ("compute then reverse"), this experiment uses
queries that require both domains active simultaneously:

- **Level 1 (Translation):** "Explain recursion using medical terminology"
- **Level 2 (Analogy):** "How is garbage collection similar to immune response?"
- **Level 3 (Synthesis):** "Design a diagnostic protocol inspired by binary search"

The experiment uses 8 domain pairs from the pilot-50 adapters, spanning code x
science, code x code, science x science, and code x humanities.

## Lineage in the Arena

```
exp_cross_domain_composition (proven, sequential chaining, micro)
    |
    +-- exp_cross_domain_dilution_vs_k (proven, PPL-probe weighting, micro)
    |
    +-- exp_pilot50_composition_quality (K1 FAIL, K3 PASS, macro)
    |
    +-- exp_cross_domain_semantic_transfer (THIS, macro)
```

## Key References

- Prabhakar et al., "LoRA Soups" (COLING 2025) -- CAT composition achieves
  super-linear improvement on compositional tasks
- Huang et al., "LoRAHub" (2023) -- gradient-free cross-task LoRA composition
- Wang et al., "LoRA-Flow" (2024) -- dynamic per-token per-layer fusion gates
- Tang et al., "MergeBench" (2024) -- benchmark for merged model evaluation
- "Task-Aware LoRA Composition" (2025) -- similarity-based LoRA retrieval

## Empirical Results

*[TO BE FILLED from results/cross_domain_semantic_transfer/results.json]*

### Domain Coverage (M1)

| Configuration | Mean DC | Std |
|---------------|---------|-----|
| Base | ... | ... |
| Expert A only | ... | ... |
| Expert B only | ... | ... |
| Composed (A+B) | ... | ... |

### Integration Score (M2)

| Configuration | Mean IS | Std |
|---------------|---------|-----|
| Base | ... | ... |
| Composed | ... | ... |

### Per-Level Breakdown

| Level | Base DC | Composed DC | Degradation | K2 Fail Rate |
|-------|---------|-------------|-------------|--------------|
| 1: Translation | ... | ... | ... | ... |
| 2: Analogy | ... | ... | ... | ... |
| 3: Synthesis | ... | ... | ... | ... |

### Per-Pair Results

| Domain Pair | Base DC | Composed DC | Degradation | Composed > Single? |
|-------------|---------|-------------|-------------|-------------------|
| python x medical | ... | ... | ... | ... |
| python x physics | ... | ... | ... | ... |
| python x rust | ... | ... | ... | ... |
| python x sql | ... | ... | ... | ... |
| medical x chemistry | ... | ... | ... | ... |
| biology x statistics | ... | ... | ... | ... |
| python x legal | ... | ... | ... | ... |
| math x ethics | ... | ... | ... | ... |

### Judge Win Rates

| Comparison | Win Rate | Ties |
|-----------|----------|------|
| Composed vs Base | ... | ... |
| Composed vs Best Single | ... | ... |

### Kill Criteria

| Criterion | Threshold | Measured | Status |
|-----------|-----------|----------|--------|
| K1: Coverage degradation vs base | < 20% | ... | ... |
| K2: Composed worse than best single | < 50% | ... | ... |

## Limitations

1. **Keyword-based scoring is coarse.** Domain coverage and integration scores
   rely on keyword presence, which misses paraphrasing and conceptual bridges
   expressed in novel vocabulary. The judge comparison partially compensates.

2. **Base model is a strong baseline.** Qwen2.5-7B already has cross-domain
   knowledge from pretraining. The experiment tests whether LoRA experts ADD
   value beyond the base, not whether the system can answer at all.

3. **Equal-weight composition only.** The PPL-probe weighting (r=0.990 oracle
   correlation at micro) is not tested here. If equal-weight composition works,
   weighted composition would only improve. If it fails, it motivates routing
   as essential for semantic transfer.

4. **Self-judging bias.** Using the base model as judge may favor the base
   model's responses. The keyword metrics provide an independent signal.

5. **Template-based queries.** The queries are generated from templates, not
   organically. Real user cross-domain queries may have different characteristics.

## What Would Kill This

**At macro scale (would falsify if observed):**
- K1: Composed model domain coverage degrades > 20% vs base (one domain
  vanishes from responses when two experts are merged)
- K2: Composed model loses to best single expert on > 50% of queries
  (experts interfere rather than complement)

**Interpretation of failure modes:**
- If K1 fails: composition disrupts the base model's cross-domain reasoning.
  This would mean LoRA perturbations are NOT localized enough to preserve
  general reasoning, and routing (selecting one expert) is strictly better
  than composition.
- If K2 fails: the second expert adds noise rather than signal for semantic
  transfer queries. This would motivate top-1 routing for cross-domain queries
  (route to the single most relevant expert) rather than multi-expert composition.
- If both fail: semantic transfer via weight-space composition is fundamentally
  limited to sequential chaining. Cross-domain reasoning must rely entirely on
  the base model's pretrained knowledge.

## Date
2026-03-15. Status: **SPEC READY** (awaiting experiment-programmer implementation).
