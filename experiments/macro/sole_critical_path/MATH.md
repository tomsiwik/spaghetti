# SOLE Critical Path: Mathematical Foundations

Three experiments in one GPU session. Each section is self-contained.

## Notation (Shared)

| Symbol | Shape / Type | Description |
|--------|-------------|-------------|
| W | (d_out, d_in) per layer | Frozen base model weight matrices |
| delta_i = (alpha/r) B_i A_i | (d_out, d_in) per layer | LoRA expert i's weight delta |
| N | scalar | Number of composed experts (N=5 for this batch) |
| r | scalar | LoRA rank (r=16) |
| alpha | scalar | LoRA scaling factor (alpha=16, so alpha/r = 1.0) |
| d | scalar | Model hidden dimension (d=3584 for Qwen2.5-7B) |
| L | scalar | Number of transformer layers (L=28) |
| S | set of texts | Calibration set {x_1, ..., x_T} |
| PPL(M, S) | scalar | Token-weighted perplexity of model M on set S |
| CE(M, x) | scalar | Cross-entropy loss of model M on text x |

**Adapter inventory:** {bash, math, medical, python, sql}. All rank-16, all-modules
(q/k/v/o/gate/up/down), trained via QLoRA on Qwen2.5-7B with NF4 quantization.

**Composition operator:** Naive addition with 1/N scaling (the macro default since
exp_ppl_probe_macro_composition_v2 showed this is near-lossless at -0.09pp):

```
W_composed^l = W^l + (1/N) * sum_{i=1}^{N} delta_i^l
```

**Note on unscaled vs scaled:** The script uses sequential `PeftModel.from_pretrained`
+ `merge_and_unload`, which applies each adapter's full alpha/r scaling (1.0 per
adapter). This is UNSCALED composition (each adapter contributes its full delta).
The PPL-probe experiment (Exp 2) explicitly tests weighted composition with 1/N
scaling as comparison. Exp 1 (LOO) and Exp 3 (monolithic) use unscaled to match
the prior composition_dropout_robustness results.

---

## Experiment 1: Poisoned Adapter Detection (Leave-One-Out PPL)

### 1.1 Problem Statement

Given N=5 composed experts where one (sql) is known to poison composition (PPL
31.6T with all 5, PPL 17,683 without sql -- 99.99% improvement from removal),
does a simple leave-one-out PPL screen correctly identify the harmful adapter?

### 1.2 Leave-One-Out Influence Score

For each expert i in {1, ..., N}:

```
W_{-i}^l = W^l + sum_{j != i} delta_j^l
```

The LOO PPL delta for expert i on calibration set S:

```
Delta_i(S) = (PPL_{-i}(S) - PPL_all(S)) / PPL_all(S) * 100%
```

**Interpretation:**
- Delta_i < 0: removing expert i DECREASES PPL -- expert i is harmful
- Delta_i > 0: removing expert i INCREASES PPL -- expert i is helpful
- Delta_i ~ 0: expert i has negligible impact

The most harmful adapter is:

```
i* = argmin_i Delta_i(S)
```

Equivalently, i* is the adapter whose removal produces the lowest PPL.

### 1.3 Why LOO Approximates Shapley at SOLE Cosines

The Shapley value phi_i averages the marginal contribution of expert i over all
possible subsets. LOO uses only the full set. The gap between LOO and Shapley
is bounded by pairwise interaction terms:

```
|LOO_i - phi_i| <= (1/2) sum_{j != i} |I_ij|
```

where I_ij = V({i,j}) - V({i}) - V({j}) + V({}) is the interaction index.

Under SOLE orthogonality (cos(delta_i, delta_j) ~ 0), expert effects on the
loss landscape are approximately additive, so I_ij ~ 0 and LOO_i ~ phi_i.

At Qwen2.5-7B production cosines (cos ~ 0.142 for trained adapters per
converged_adapter_orthogonality), interactions are non-negligible. However, we
only need the RANKING to be correct (which harmful adapter is worst), not the
exact influence magnitude. Ranking robustness requires only that the harmful
adapter's LOO delta is sufficiently separated from the others.

### 1.4 Expected Behavior

From prior evidence (composition_dropout_robustness):

| Condition | PPL |
|-----------|-----|
| All 5 adapters | 31.6T |
| Without sql | 17,683 |
| Without any other single adapter | ~order of trillions |

The sql adapter's LOO delta should be the most negative (most PPL reduction when
removed). This is the detection signal.

**Calibration set:** Mixed texts drawn from the tail of each adapter's training
data (CALIB_SAMPLES=30, MAX_SEQ_LEN=512). This introduces domain coverage across
all 5 adapter domains, ensuring the calibration set is sensitive to each adapter's
contribution.

### 1.5 Kill Criteria Formalization

**K1: Detection accuracy.**
```
i* = argmin_i PPL_{-i}(S) must equal "sql"
```
PASS if sql is ranked as most harmful (lowest PPL when removed).

**K2: Pruning effectiveness.**
```
(PPL_all(S) - PPL_{-sql}(S)) / PPL_all(S) > 0.50
```
Removing the top-1 harmful adapter must reduce composed PPL by >50%.

**K3: Runtime.**
```
T_total < 30 minutes
```
For N=5, we need 1 full composition + 5 LOO evaluations. Each requires loading
the base model + merging + evaluating 30 calibration texts at 512 tokens.

**Runtime estimate:**

| Operation | Time (est.) | Count | Total |
|-----------|------------|-------|-------|
| Load base model (fp16) | ~25s | 6 | 150s |
| Sequential merge (N-1 adapters) | ~5s | 6 | 30s |
| PPL eval (30 texts, 512 tokens) | ~15s | 6 | 90s |
| GPU cleanup | ~5s | 6 | 30s |
| **Total** | | | **~300s (5 min)** |

Note: the script reloads the base model for each LOO iteration (conservative but
correct approach). Still well under the 30-minute threshold.

### 1.6 Relationship to exp_leave_one_out_expert_ranking

This experiment doubles as the LOO ranking mechanism. The same procedure that
detects the poisoned adapter also ranks all experts by influence. The ranking
output (sorted by PPL_{-i}) provides the full expert quality ordering, which
exp_leave_one_out_expert_ranking requires at N=50 scale. At N=5 this is a proof
of concept; scaling to N=50 is straightforward (linear in N).

---

## Experiment 2: PPL-Probe Weighted Composition

### 2.1 Problem Statement

Does PPL-probe weighted composition outperform equal-weight composition at
macro scale? This extends the micro finding (r=0.990 oracle correlation,
+9.34pp over equal-weight) to Qwen2.5-7B with real adapters.

### 2.2 PPL-Probe Weighting

For each adapter i, compute its PPL on a set of probe texts P:

```
ppl_i = PPL(M_base + delta_i, P)
```

where M_base + delta_i is the base model with only adapter i applied.

Convert to weights via temperature-scaled softmax of inverse PPL:

```
w_i = exp((1/ppl_i) / tau) / sum_j exp((1/ppl_j) / tau)
```

where tau is the temperature parameter (tau=1.0 in the script).

**Note on temperature:** exp_ppl_probe_temperature_sensitivity showed tau=0.5
is locally optimal at micro, and exp_ppl_probe_macro_composition_v2 confirmed
tau=0.5 beats tau=1.0 at macro (+0.36pp vs +0.08pp). However, the script uses
tau=1.0. The PPL-probe experiment here serves as a DIRECTIONAL test (does
probe weighting help at all?), not a temperature optimization.

### 2.3 Weighted Composition

The weighted composed model at layer l:

```
W_composed^l = W^l + sum_{i=1}^{N} w_i * (alpha_i / r_i) * B_i^l A_i^l
```

Note: the script applies w_i as a multiplier on the full alpha/r scaling, so
the effective scaling per adapter is w_i * (alpha/r).

**Equal-weight baseline:** w_i = 1/N for all i. This gives each adapter a
fraction of its full delta.

**PPL-probe weights:** w_i from the softmax above. Lower-PPL (better)
adapters get higher weight.

### 2.4 Comparison Conditions

| Condition | Weights | Description |
|-----------|---------|-------------|
| Base | -- | No adapters |
| Equal-weight | w_i = 1/N | Uniform fraction of each delta |
| PPL-probe | softmax(1/ppl_i) | Quality-weighted composition |
| Top-1 | best single adapter | Upper bound (no composition) |

### 2.5 Metrics

Primary metric: average PPL across all 5 adapter domains' calibration texts
(10 probe examples drawn from the training data of all adapters).

```
PPL_condition = PPL(M_condition, P)
```

Improvement percentage:

```
improvement = (PPL_equal - PPL_probe) / PPL_equal * 100%
```

### 2.6 Connection to Prior Results

exp_ppl_probe_macro_composition_v2 (completed) already showed:
- equal_scaled (1/N): -0.09pp vs base (near-lossless)
- ppl_probe t=0.5: +0.36pp vs base (BEATS base)

This experiment measures the same phenomenon using PPL (not MMLU accuracy) and
with a different evaluation methodology (per-domain calibration texts vs MMLU
subjects). The results should be directionally consistent.

---

## Experiment 3: SOLE vs Monolithic LoRA

### 3.1 Problem Statement

Does SOLE composition of N=5 domain-specific experts match or exceed a single
monolithic LoRA trained on the UNION of all 5 domain datasets?

This is the fundamental value proposition test. If monolithic wins on quality,
SOLE's value is purely operational (updatability, per-expert cost). If SOLE
matches, modularity is free.

### 3.2 Formalization

**Monolithic LoRA:** A single LoRA adapter trained on D_union = D_1 + D_2 + ... + D_N:

```
W_mono^l = W^l + (alpha/r) * B_union^l A_union^l
```

where (A_union, B_union) are learned from the concatenation of all domain datasets.

**SOLE composition:** N adapters, each trained on its own domain dataset D_i, composed
via naive addition:

```
W_sole^l = W^l + sum_{i=1}^{N} (alpha/r) * B_i^l A_i^l
```

### 3.3 Theoretical Analysis

Both models use the same total data (D_union = D_1 + ... + D_N). The question is
whether decomposition hurts.

**Capacity argument:** The monolithic adapter has rank r. The SOLE composition has
effective rank up to N*r (if all deltas are orthogonal). With N=5, r=16, SOLE has
up to rank 80 vs monolithic rank 16. SOLE has MORE capacity.

However, each SOLE adapter is trained independently on 1/N of the total data,
so each B_i A_i may be lower quality than B_union A_union which sees all data.

**Interference argument:** SOLE composition adds N independent deltas. Even at
cos ~ 0.142, interference between N=5 adapters is bounded:

```
interference <= sum_{i<j} |cos(delta_i, delta_j)| * ||delta_i|| * ||delta_j||
             <= C(5,2) * 0.142 * ||delta||^2
             = 10 * 0.142 * ||delta||^2
             = 1.42 * ||delta||^2
```

This is non-negligible, which is why 1/N scaling helps.

The monolithic adapter has zero internal interference (it's a single rank-r matrix).

### 3.4 Training Configuration

The union LoRA is trained with the SAME hyperparameters as the pilot-50 adapters:

| Parameter | Value |
|-----------|-------|
| Rank | 16 |
| Alpha | 16 |
| Target modules | q/k/v/o/gate/up/down |
| Quantization | NF4 (4-bit) |
| Optimizer | AdamW 8-bit |
| Learning rate | 1e-4, cosine schedule |
| Batch size | 1 (grad accumulation 4) |
| Max steps | 300 (10 for smoke test) |
| Max sequence length | 512, packing enabled |
| Precision | bf16 |

**Data budget:** Union of all 5 training datasets (5 x ~1000 examples = ~5000
examples). At 300 steps with batch size 4, the model sees 1200 examples (~24%
of the union dataset in one pass). Each individual SOLE adapter saw 300 steps
of ~1000 examples (~30% in one pass), so training effort per example is comparable.

### 3.5 Evaluation

Both models are evaluated on the same calibration texts (5 examples per domain
= 25 total). Metrics:

**Aggregate PPL:**
```
PPL_sole = PPL(W_sole, S)
PPL_mono = PPL(W_mono, S)
```

**Per-domain win rate:** For each domain d in {bash, math, medical, python, sql}:
```
win_sole(d) = 1 if PPL(W_sole, S_d) < PPL(W_mono, S_d) else 0
```

### 3.6 Kill Criteria Formalization

**K1: Per-domain win rate.**
```
sum_d win_mono(d) / N > 0.70  =>  KILL
```
If the union LoRA beats SOLE on >70% of domains (>3.5 out of 5), SOLE's
modularity hurts quality. With N=5, this means monolithic must win on at
least 4 out of 5 domains to kill.

**K2: Aggregate PPL.**
```
(PPL_sole - PPL_mono) / PPL_mono > 0.10  =>  KILL
```
If union LoRA aggregate PPL is >10% better than SOLE.

### 3.7 Why This Might Fail (SOLE Loses)

1. **Rank bottleneck:** Each SOLE adapter captures domain knowledge in rank 16.
   If domains share common structure (e.g., instruction-following patterns),
   each adapter wastes rank on this shared structure. The monolithic adapter
   can allocate its rank 16 to the shared structure once and specialize the rest.

2. **Training efficiency:** The monolithic adapter sees all data shuffled together,
   enabling cross-domain transfer during training. SOLE adapters are trained in
   isolation and never see cross-domain patterns.

3. **Interference at composition:** Even with 1/N scaling, the 5 adapter deltas
   interact nonlinearly through the transformer's attention and activation
   functions.

### 3.8 Why This Might Succeed (SOLE Matches or Wins)

1. **Higher effective rank:** SOLE composition has up to rank 80 vs rank 16.
   Even with interference, the information capacity is much higher.

2. **Domain specialization:** Each SOLE adapter fully specializes on its domain
   (100% of training data from one domain). The monolithic adapter must
   compromise across 5 domains with only rank 16.

3. **1/N scaling empirics:** At N=5, 1/N scaling already achieves -0.09pp on
   MMLU (near-lossless). The composition mechanism works.

### 3.9 Worked Example

**Setup:** 5 adapters, 5000 union examples, 300 training steps.

**Scenario A (SOLE wins):**
- Monolithic PPL: 6.5 (limited by rank-16 on 5 domains)
- SOLE PPL: 5.8 (rank-80 effective, specialized per domain)
- K1: monolithic wins 1/5 domains (20% < 70%) -- PASS
- K2: PPL ratio = -10.8% (SOLE better) -- PASS

**Scenario B (SOLE loses):**
- Monolithic PPL: 5.5 (efficient shared structure)
- SOLE PPL: 6.2 (interference and uneven scaling)
- K1: monolithic wins 4/5 domains (80% > 70%) -- KILL
- K2: PPL ratio = +12.7% (SOLE worse) -- KILL

**Scenario C (Draw):**
- Monolithic PPL: 6.0
- SOLE PPL: 6.1 (1.7% worse)
- K1: monolithic wins 3/5 domains (60% < 70%) -- PASS
- K2: PPL ratio = +1.7% (< 10%) -- PASS
- Interpretation: SOLE matches. Modularity is free.

---

## Cross-Experiment Dependencies

```
Exp 1 (Poisoned Detection)
  |
  v
Exp 2 (PPL-Probe Weighting) -- uses same base model, same adapters
  |
  v
Exp 3 (SOLE vs Monolithic) -- uses SOLE composition from Exp 2 as comparison
```

Each experiment runs in a function scope. GPU memory is fully freed between
experiments. The script exits cleanly even if one experiment fails (try/except
per experiment).

## Assumptions

1. **Sequential merge produces the same result as manual delta addition.**
   The script uses PeftModel.from_pretrained + merge_and_unload sequentially,
   which is equivalent to W + sum delta_i when all adapters use the same alpha/r
   and there are no adapter-adapter interactions in the PEFT library.

2. **Training data tails are representative calibration texts.** The last
   CALIB_SAMPLES entries from each adapter's train.jsonl are used. These have
   not been seen during early training but are from the same distribution.

3. **300 training steps is sufficient for the union adapter.** The pilot-50
   adapters were trained for 300 steps each. The union adapter sees 5x more
   data variety but the same number of gradient steps.

4. **NF4 quantization is consistent across all conditions.** All models use
   the same quantization scheme, so absolute PPL values are comparable.

5. **PPL on in-domain calibration texts is a valid quality proxy.** We cannot
   run MMLU or HumanEval within this script's time budget, so PPL serves as
   the primary metric. This is a known limitation (PPL does not predict task
   accuracy, r=0.08 per ppl_vs_task_performance). However, for relative
   comparison between two models on the same texts, PPL is a reasonable proxy.
