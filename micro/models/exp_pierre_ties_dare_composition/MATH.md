# MATH.md — TIES + DARE Composition for PoLAR Adapters

## Diagnosis (the missing essential)

Three KILLs of weight-space PoLAR composition under different conditions:
1. exp_beehive_polar_composition_mechanism: over-trained adapters + broken routing
2. exp_pierre_polar_composition_v2_routed: over-trained + proven routing (Gumbel top-2 + hidden-state router + z-score)
3. exp_polar_mild_adapters_compose: **mild adapters (F#73 conditions) + uniform 1/N**

Even mild adapters collapse: medical PPL 2.7 → 151.2 when composed (55× worse). HumanEval drops 87% → 20% on N=2 strategy×domain composition. Pattern is structural, not contingent on training intensity, routing, or N.

## What the merging literature says we're missing

After consulting NotebookLM (16 papers loaded) and inspecting source code of 4 production codebases (ties-merging, MergeLM, lorahub, mergekit), the diagnosis is unambiguous:

**Naive uniform 1/N averaging suffers from two well-documented failure modes:**

1. **Redundant parameter interference (TIES-Merging, arxiv:2306.01708)**
   Most adapter parameters change very little during fine-tuning — they're noise that doesn't carry task signal. Uniform averaging amplifies this noise across all N adapters, polluting every layer's effective ΔW with noise at scale × 1/N × N = scale × 1 — full strength.

2. **Sign conflicts across task vectors (TIES-Merging, arxiv:2306.01708)**
   Different adapters have opposite-sign deltas for the same parameters (medical may want +0.1, code may want -0.08). Uniform averaging cancels these to ~0, but neither task is well-served by 0 — the parameter is meaningful for both, just in different directions.

**The fixes (with concrete algorithms from the cited code):**

### TIES-Merging recipe (Yadav et al 2023, arxiv:2306.01708)

For each layer's task vectors {ΔW_1, ..., ΔW_N}:

```
Step 1 — TRIM:
  For each adapter's ΔW_i, keep only top-k% magnitudes per task vector
  (typical k=20%, drop 80% of smallest-magnitude params to zero).

Step 2 — ELECT SIGN:
  For each parameter position (i,j), compute the elected sign across N adapters:
    sign_elected[i,j] = sign(Σ_n ΔW_n[i,j])  (mass-based election)
  Or: norm-mass weighted: sign[i,j] = sign(Σ_n ΔW_n[i,j] × ||ΔW_n[:,j]||²)

Step 3 — DISJOINT MERGE:
  For each (i,j), aggregate ONLY the adapters whose sign matches sign_elected[i,j]:
    valid_adapters = {n : sign(ΔW_n[i,j]) == sign_elected[i,j]}
    ΔW_merged[i,j] = mean({ΔW_n[i,j] for n in valid_adapters})
  (Disjoint mean — averages only over non-zero, sign-aligned entries.)
```

Reference impl: `/tmp/composition_research/ties-merging/src/utils/merge_utils.py:345`
- `disjoint_merge(Tensor, merge_func, sign_to_mult)` is the load-bearing function

### DARE recipe (Yu et al 2023, arxiv:2311.03099)

For each adapter's ΔW_i:

```
Step 1 — DROP:
  Random Bernoulli mask with rate p (typical p=0.9, drop 90% of params)
  ΔW_i_dropped = ΔW_i × (1 - mask)  (zero out 90% randomly)

Step 2 — RESCALE:
  ΔW_i_rescaled = ΔW_i_dropped / (1 - p)  (divide by 0.1 → multiply by 10)
  (preserves expected value, like inverted dropout at training time)

Step 3 — MERGE:
  Standard linear or task-arithmetic on the dropped+rescaled adapters
  (Or: combine with TIES sign election → "DARE-TIES" per mergekit)
```

Reference impl: `/tmp/composition_research/MergeLM/model_merging_methods/mask_weights_utils.py:9`
- `mask_input_with_mask_rate(input_tensor, mask_rate, use_rescale, mask_strategy)`

### Combined: DARE-TIES (mergekit production default)

`/tmp/composition_research/mergekit/mergekit/merge_methods/generalized_task_arithmetic.py`

DARE drop+rescale → TIES sign election + disjoint merge. Empirically the strongest combination per mergekit's deployed configs.

## Hypothesis

Applying TIES-Merging (or DARE, or DARE-TIES) to the per-adapter task vectors ΔW_i = scale × A_i @ B_i on our existing 7 PoLAR adapters will preserve composed-model task accuracy within 5pp of best single-adapter on each benchmark, where uniform 1/N drops by 50-70pp.

## Theoretical grounding

**Why this works for our specific setup:**

PoLAR adapters have rank=6, but the implied ΔW = scale × A @ B is a full d_in × d_out matrix. At Gemma 4 q_proj (d_in=2560, d_out=2048), each ΔW has 5.2M entries per layer. After PoLAR training, most entries are small-magnitude noise — exactly what TIES/DARE prune.

**Why our adapters likely have especially aligned task vectors:**
All 4 strategy adapters trained on beehive prompts (uniformly dev-tooling). Their ΔW deltas push activations in a similar direction. Naive averaging amplifies this aligned direction (constructive on dev tasks but destructive on orthogonal capabilities like medical knowledge — explaining the 55× medical PPL inflation).

**TIES sign election directly resolves this**: when 6 of 7 adapters push +0.1 in one direction (the dev-leaning consensus) and the medical adapter pushes -0.05 in the orthogonal direction, sign election picks +. Medical adapter's contribution in that direction is dropped, but its contributions in the orthogonal-to-dev directions (where strategies don't conflict) survive.

**DARE works for a different reason**: random dropout + rescale is mathematically equivalent to noise injection on the merge process. This breaks the symmetric pile-up that uniform averaging produces. The resulting merged delta is sparser and less correlated with any single adapter's distribution.

## Predictions

Given prior literature's reproducible results on similar setups (TIES paper: T0-3B + 11 tasks, +6.4 to +24.7pp over uniform; DARE paper: WizardLM-13B + WizardMath + WizardCoder, recovers within 1pp of best single):

1. **K1**: TIES-merged composition preserves within 5pp on each benchmark (matches T0-3B paper's "no degradation" finding for sign-aligned merges)
2. **K2**: DARE-merged composition preserves within 5pp (matches WizardLM result)
3. **K3**: DARE-TIES (mergekit default) beats best single on ≥1 benchmark (matches mergekit's reported configs)
4. **K4**: Per-adapter PPL ratio composed/single ≤1.10× (the F#73 metric our experiments fail at 55×)
5. **K5**: Sparsity ≥70% per merged layer (TIES trim default 20%, DARE drop default 90% — both produce sparse merged deltas)

## Implementation plan

### Architecture
```
1. Load 7 existing PoLAR adapters (4 strategy + 3 domain)
2. For each layer, compute ΔW_i = scale × A_i @ B_i (full d_in × d_out per adapter)
3. Apply merge method:
   - TIES: trim → elect → disjoint merge
   - DARE: drop → rescale → average
   - DARE-TIES: drop+rescale → trim+elect+disjoint
4. Store merged ΔW per layer (sparse or dense fp16)
5. At inference: replace q_proj forward with `y = base(x) + x @ ΔW_fused`
6. Run benchmarks; compare to single-adapter baselines and uniform 1/N
```

### Files (planned, full implementation in run_experiment.py)
- `merge_methods.py` — TIES, DARE, DARE-TIES implementations (Python/numpy)
- `apply_fused_delta.py` — wrap q_proj with the merged delta, MLX
- `eval_methods.py` — run all 3 methods + baselines on GSM8K/HumanEval/MedQA
- `run_experiment.py` — orchestration

### Key numerical considerations

1. **fp32 for merge math**: TIES sign election is sensitive to tiny magnitudes; do all arithmetic in fp32 even if storage is fp16 / bf16.
2. **Stiefel constraint loss**: applying TIES to the IMPLIED full ΔW does NOT preserve the joint Stiefel constraint on A and B. This is acceptable — the constraint was a training-time guarantee for individual adapters; at inference the merged delta is a single matrix without rank structure.
3. **PoLAR scale factor**: include scale (4 or 6) in the task vector definition: ΔW_i = scale × A_i @ B_i. Otherwise TIES trim threshold is misscaled.
4. **Per-layer masking**: trim/drop applied INDEPENDENTLY per layer, not globally. Each layer has its own redundancy structure.

## Risks

1. **PoLAR + TIES interaction is novel.** TIES paper studied full-rank fine-tuned models. PoLAR's low-rank structure may interact with magnitude-pruning differently — perhaps the implied ΔW has DIFFERENT magnitude distribution than a full fine-tune's.

2. **Memory cost of fused delta.** 5.2M params × 42 layers × fp16 = ~440MB extra alongside base model. Fits on M5 Pro 48GB but not free.

3. **No low-rank inference path.** TIES output is full-rank dense. Loses the speed advantage of rank-6 PoLAR (rank-6 forward is ~425000× cheaper than dense). For inference, the fused delta replaces N rank-6 deltas with 1 dense delta — net cost depends on whether per-layer dense beats N×rank-6 (it does at small N like ours).

4. **DARE rescale at p=0.9 multiplies survivor magnitudes by 10×.** If our adapters' weights are already on the edge of stable range (due to PoLAR's Stiefel constraint forcing entries ≤1), 10× rescale may push some entries out of stable range. Mitigation: try lower p (0.5, 0.7) before 0.9.

## Pre-registered KCs

K2138: TIES preserves single-adapter best within 5pp per benchmark
K2139: DARE preserves within 5pp per benchmark
K2140: Best of {TIES, DARE, DARE-TIES} > best single on ≥1 benchmark
K2141: Per-adapter composed PPL ≤ 1.10× single PPL
K2142: Merged delta sparsity ≥70% per layer

## References

- Yadav et al 2023, arxiv:2306.01708 — TIES-Merging
- Yu et al 2023, arxiv:2311.03099 — DARE
- Ilharco et al 2022, arxiv:2212.04089 — Task Arithmetic (baseline)
- Buehler et al 2024, arxiv:2402.07148 — X-LoRA (token-level mixture, relevant)
- mergekit production library: github.com/arcee-ai/mergekit
- F#54 — N=24 composition supported (PRIOR WORKING result we couldn't replicate)
- F#73 — N=15 composition 0.53% degradation (PRIOR WORKING result we couldn't replicate)
- F#440 — N=100 interference-free Grassmannian (PRIOR result on different setup)

## Why this is the next critical experiment

If TIES/DARE rescues composition: Pierre's compositional product story is intact, just needs proper merging algorithm. The fix is a one-time precompute per adapter combination.

If TIES/DARE fails: PoLAR + parameter-interference fix doesn't work, suggesting something specific to the Stiefel constraint or our training data interacts pathologically with composition. At that point, single-adapter routing is the only viable path and Pierre v1 ships as router-only.

Either result is decisive.
