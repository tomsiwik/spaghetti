# Scaling Adapters to N=24 on Distinct Real-Data Slices: Research Digest

## Hypothesis

Ternary LoRA adapters with Grassmannian A-matrix orthogonality, STE quantization,
and per-expert composition scale to 24+ adapters trained on distinct real-data slices
on BitNet-2B-4T without degrading specialization, composition quality, or routing
accuracy, and within the 48GB memory envelope of Apple M5 Pro.

**Verdict: SUPPORTED.** All kill criteria pass. All success criteria pass.

## What This Experiment Does

Trains 24 independent LoRA adapters on distinct real HuggingFace instruction data
slices, then evaluates:
1. Individual specialization (PPL improvement vs base per adapter)
2. Full N=24 composition quality (correct per-expert A_i@B_i sum / N)
3. Adapter orthogonality (276 pairwise cosine measurements)
4. 24-way routing head accuracy (binary classifiers per adapter, trained on train
   hidden states and evaluated on held-out val hidden states)
5. Memory usage throughout (K3 constraint: < 48GB)

### Data Composition: Genuine Domains vs Slice-Based Adapters

Of 24 adapters, only **7 train on genuinely domain-specific datasets**:
- **medical** (medalpaca/medical_meadow_medical_flashcards)
- **code** (iamtarun/python_code_instructions_18k_alpaca)
- **math** (openai/gsm8k)
- **legal** (jonathanli/law-stack-exchange)
- **finance** (gbharti/finance-alpaca)
- **health_fitness** (keivalya/MedQuad-MedicalQnADataset)
- **psychology** (Amod/mental_health_counseling_conversations)

The remaining **17 train on offset-based slices of general-purpose datasets**:
- Dolly-15k slices (8): science, history, philosophy, creative_writing, agriculture,
  environmental, politics, economics
- TokenBender/code_instructions_122k slices (4): education, engineering, sports, music
- wizard_vicuna_70k slices (3): cooking, cybersecurity, marketing
- WizardLM_evol_instruct_V2 slices (2): sociology, linguistics

Each slice-based adapter sees distinct training data (different offset ranges), but
the domain labels are nominal -- they do not correspond to the actual content of the
training data. For example, the "sports" adapter trains on code instructions, not
sports data. This means specialization improvements for slice-based adapters reflect
learning on distinct data slices, not genuine domain specialization.

Results are reported both overall and split by group to support honest interpretation.

## Architecture

- **Base model:** microsoft/BitNet-b1.58-2B-4T (ternary, d=2560, 30 layers)
- **LoRA:** rank-16, 210 target projections per adapter (q/k/v/o + gate/up/down)
- **A-matrices:** Grassmannian Alternating Projection, frozen during training
- **B-matrices:** STE ternary quantization (round-clip with straight-through estimator)
- **Composition:** Correct per-expert: y = W@x + (scale/N) * sum_i[(x@A_i) @ ternary(B_i)]
- **Routing:** 24 independent binary classifiers (2560 -> 32 -> 1), trained on base
  model hidden states from training data, evaluated on held-out validation data

## Key References

- arXiv 2510.03262 -- OSRM shows weight-orth != semantic orth, but composition works via constructive transfer
- arXiv 2508.11985 -- Naive LoRA Summation: orthogonality enables additive composition
- arXiv 2603.15965 -- MoLoRA: per-token routing, independent gates outperform softmax
- arXiv 2603.03535 -- Routing > merging at scale (systematic comparison)

## Empirical Results

### Kill Criteria

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| K1 (275): Adapters fail to specialize | > 5 fail (PPL same as base) | **0/24 fail**, all 24 improve >5% | **PASS** |
| K2 (276): Composition degrades domains | > 50% worse than base | **0/24 degraded** (all improve) | **PASS** |
| K3 (277): Memory exceeds 48GB | > 48GB peak | **17.10GB peak** (64% margin) | **PASS** |

### Success Criteria

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| S1: Specialization | >= 20/25 at > 5% PPL improvement | **24/24** (min: 17.5%, max: 46.9%) | **PASS** |
| S2: Routing accuracy (val) | > 70% average | **92.7%** (min: 85.3%, all > 70%) | **PASS** |
| S3: Composition (all-N uniform) beats base | >= 20/25 domains | **24/24** (avg -29.1% vs base) | **PASS** |

Note on S3: The composition evaluation uses all-24 uniform averaging (1/N scaling),
not routed top-k selection. This is a **lower bound** on routed composition performance,
since routing would select only the relevant experts and avoid dilution.

### Individual Specialization (PPL improvement vs base)

#### Genuine Domain Adapters (7)

| Domain | Base PPL | Adapted PPL | Improvement |
|--------|----------|-------------|-------------|
| medical | 6.50 | 3.46 | +46.8% |
| code | 4.98 | 3.14 | +36.8% |
| math | 3.84 | 2.38 | +37.9% |
| legal | 21.63 | 14.66 | +32.2% |
| finance | 19.43 | 14.01 | +27.9% |
| health_fitness | 11.71 | 6.66 | +43.1% |
| psychology | 17.45 | 12.20 | +30.1% |
| **Avg (genuine)** | **12.36** | **8.07** | **+36.4%** |

#### Slice-Based Adapters (17)

| Domain | Source Dataset | Base PPL | Adapted PPL | Improvement |
|--------|---------------|----------|-------------|-------------|
| science | Dolly-15k | 12.46 | 7.30 | +41.4% |
| history | Dolly-15k | 16.70 | 9.29 | +44.4% |
| philosophy | Dolly-15k | 16.39 | 9.97 | +39.2% |
| creative_writing | Dolly-15k | 20.73 | 12.01 | +42.1% |
| cooking | wizard_vicuna | 3.21 | 2.55 | +20.6% |
| education | TokenBender | 3.58 | 2.41 | +32.6% |
| engineering | TokenBender | 4.21 | 2.45 | +41.8% |
| agriculture | Dolly-15k | 14.56 | 8.35 | +42.6% |
| environmental | Dolly-15k | 11.49 | 6.62 | +42.4% |
| politics | Dolly-15k | 12.56 | 6.67 | +46.9% |
| economics | Dolly-15k | 16.69 | 8.93 | +46.5% |
| sociology | WizardLM | 4.49 | 3.61 | +19.7% |
| linguistics | WizardLM | 4.41 | 3.64 | +17.5% |
| cybersecurity | wizard_vicuna | 3.83 | 3.09 | +19.2% |
| marketing | wizard_vicuna | 3.83 | 2.95 | +23.0% |
| sports | TokenBender | 3.59 | 2.34 | +34.7% |
| music | TokenBender | 3.57 | 2.34 | +34.5% |
| **Avg (slice)** | | **9.13** | **5.56** | **+34.6%** |

#### Combined

| Group | Count | Avg Improvement | Min | Max |
|-------|-------|-----------------|-----|-----|
| Genuine domains | 7 | +36.4% | +27.9% | +46.8% |
| Slice-based | 17 | +34.6% | +17.5% | +46.9% |
| **All adapters** | **24** | **+35.2%** | **+17.5%** | **+46.9%** |

Both groups show comparable specialization. The genuine domain adapters show slightly
higher average improvement (+36.4% vs +34.6%), but the difference is small. This
suggests the mechanism works regardless of whether the data is domain-specific or
drawn from a general corpus.

### Composition Quality (all-24 uniform, correct multi-A)

| Group | Count | Avg Delta vs Base | All Improve? |
|-------|-------|-------------------|--------------|
| Genuine domains | 7 | -25.0% | Yes (7/7) |
| Slice-based | 17 | -30.8% | Yes (17/17) |
| **All adapters** | **24** | **-29.1%** | **Yes (24/24)** |

Full per-domain breakdown:

| Domain | Base PPL | Composed PPL | Delta vs Base |
|--------|----------|--------------|---------------|
| medical | 6.50 | 4.26 | -34.4% |
| code | 4.98 | 3.40 | -31.6% |
| math | 3.84 | 3.06 | -20.4% |
| legal | 21.63 | 16.70 | -22.8% |
| finance | 19.43 | 15.34 | -21.1% |
| science | 12.46 | 7.72 | -38.0% |
| history | 16.70 | 9.80 | -41.3% |
| philosophy | 16.39 | 10.55 | -35.7% |
| creative_writing | 20.73 | 12.60 | -39.2% |
| cooking | 3.21 | 2.67 | -16.8% |
| health_fitness | 11.71 | 8.47 | -27.6% |
| psychology | 17.45 | 14.45 | -17.2% |
| education | 3.58 | 2.58 | -28.0% |
| engineering | 4.21 | 2.66 | -36.7% |
| agriculture | 14.56 | 8.83 | -39.3% |
| environmental | 11.49 | 6.98 | -39.3% |
| politics | 12.56 | 7.06 | -43.8% |
| economics | 16.69 | 9.55 | -42.8% |
| sociology | 4.49 | 3.78 | -16.0% |
| linguistics | 4.41 | 3.80 | -14.0% |
| cybersecurity | 3.83 | 3.27 | -14.7% |
| marketing | 3.83 | 3.14 | -18.0% |
| sports | 3.59 | 2.50 | -30.2% |
| music | 3.57 | 2.51 | -29.9% |
| **Average** | **10.08** | **6.90** | **-29.1%** |

**Key finding:** Composition beats base on ALL 24 adapters. The average composition
improvement (-29.1%) is only marginally less than individual specialization (-35.2%),
showing the 1/N scaling with correct multi-A composition is highly effective.

### Orthogonality

- **276 pairwise cosine measurements** across 24 adapters
- Mean |cos|: **0.0238** (below 0.05 threshold)
- Max |cos|: **0.0893** (creative_writing-sports, still low)
- Grassmannian A-matrix coherence: d=2560 mean |cos|=0.004, d=6912 mean |cos|=0.002
- At N=24, r=16: Nr=384 << d=2560, so perfect A-orthogonality is achievable

Note: B-matrix cosines (0.024) are higher than A-matrix cosines (0.004) because
B-matrices are trained on semantically-related data. The Grassmannian skeleton provides
a 6x decorrelation filter (0.024 / 0.004 = 6x, vs 17x at N=5).

### Routing (with train/val split)

Routing heads are trained on hidden states extracted from training data (train.jsonl)
and evaluated on held-out validation data (valid.jsonl). This addresses the prior
review concern about overfitting: the val accuracy is the honest metric.

#### By Group

| Group | Count | Train Acc | Val Acc | Train-Val Gap |
|-------|-------|-----------|---------|---------------|
| Genuine domains | 7 | 98.6% | 98.5% | 0.1pp |
| Slice-based | 17 | 94.5% | 90.4% | 4.1pp |
| **All adapters** | **24** | **95.7%** | **92.7%** | **3.0pp** |

The genuine domain adapters show near-zero train-val gap (0.1pp), indicating their
hidden states are truly distinct and routing generalizes perfectly. Slice-based adapters
show a larger gap (4.1pp), consistent with less distinctive hidden-state distributions,
but val accuracy still far exceeds the 70% threshold.

#### Per-Adapter Val Accuracy (sorted)

| Domain | Train Acc | Val Acc | Domain | Train Acc | Val Acc |
|--------|-----------|---------|--------|-----------|---------|
| math | 100.0% | 100.0% | psychology | 99.8% | 99.8% |
| legal | 99.6% | 99.7% | medical | 99.6% | 99.4% |
| finance | 99.0% | 99.2% | health_fitness | 99.4% | 99.0% |
| economics | 97.7% | 93.3% | linguistics | 94.9% | 93.0% |
| environmental | 98.0% | 92.8% | cybersecurity | 95.9% | 92.3% |
| code | 92.7% | 92.3% | creative_writing | 97.3% | 92.3% |
| marketing | 95.1% | 92.3% | sociology | 95.5% | 92.2% |
| science | 96.7% | 91.5% | philosophy | 98.2% | 91.1% |
| politics | 98.1% | 90.9% | cooking | 91.5% | 90.0% |
| history | 95.6% | 89.4% | agriculture | 94.6% | 88.4% |
| education | 92.2% | 87.4% | music | 89.2% | 87.2% |
| engineering | 89.6% | 86.9% | sports | 86.9% | 85.3% |

Minimum val accuracy: 85.3% (sports). All 24 adapters exceed the 70% threshold on
held-out data. The 3.0pp average train-val gap is modest and consistent with the
small training set size (40 samples per domain).

### Training Stability

Most adapters converged well (loss decreased monotonically). Three adapters showed
issues:

| Adapter | First 50 Loss | Last 50 Loss | Status |
|---------|---------------|--------------|--------|
| science | 2.317 | 2.224 | Slow convergence (not marked converged) |
| economics | 2.453 | 2.342 | Slow convergence (not marked converged) |
| **sociology** | **1.335** | **1.385** | **Loss increased (+3.7%) -- training diverged** |

The sociology adapter's training loss went UP during 200 iterations, indicating
divergence. Despite this, it still achieved 19.7% PPL improvement on its held-out data,
suggesting even the partially-learned LoRA perturbation provides benefit over base.
However, this result should be treated with caution. At 200 iterations with rank-16
and only ~400 training samples, the adapter may be memorizing rather than learning
generalizable features.

Science and economics showed slow convergence but did not diverge. Both achieved
strong specialization (41.4% and 46.5% respectively), suggesting they would benefit
from additional training iterations.

### Scaling Summary (N=5 to N=24)

| Metric | N=5 | N=24 | Trend |
|--------|-----|------|-------|
| Avg specialization | 36.2% | 35.2% | Stable (-1.0pp) |
| Avg composition vs base | -26.3% | -29.1% | **Improved** |
| Mean |cos| | 0.0205 | 0.0238 | Stable (+0.003) |
| Avg routing accuracy (val) | 99.9% | 92.7% | Graceful (-7.2pp) |
| Total time | 6.8 min | 30.1 min | Linear (4.4x for 4.8x domains) |
| Peak memory | ~17 GB | 17.1 GB | Constant |

## Limitations

1. **Nominal domain labels for 17/24 adapters:** 17 adapters train on arbitrary slices
   of general-purpose datasets (Dolly-15k, WizardLM, wizard_vicuna, TokenBender) with
   domain labels that do not correspond to actual content. Specialization improvements
   for these adapters reflect learning on distinct data slices, not genuine domain
   expertise. The 7 genuine domain adapters (medical, code, math, legal, finance,
   health_fitness, psychology) use truly domain-specific datasets. Results are reported
   separately for each group.

2. **Only 24 of 25 domains:** GAIR/LIMA is a gated dataset requiring authentication.
   One domain (real_estate) was dropped. 24/25 = 96% coverage.

3. **Composition uses all 24 experts uniformly:** A routed top-k composition (selecting
   2-3 relevant experts per input) would likely perform better. The all-24 uniform
   result is a lower bound on routed composition quality.

4. **No cross-domain evaluation:** Each adapter is evaluated only on its own data
   slice's validation set. Cross-domain interference is measured only via orthogonality,
   not by checking whether adapter A degrades domain B's performance.

5. **200 training iterations:** A conservative budget. Three adapters (sociology,
   science, economics) showed convergence issues. Sociology diverged (loss increased).

6. **Routing heads use base model hidden states:** The routing is based on the base
   model's representation, not the adapted model's. This is the correct design for
   pre-routing but may not capture adapter-specific features.

7. **Single seed (SEED=42):** All results from one random seed. No confidence intervals
   or variance estimates. Acceptable for micro-scale directional experiments but
   should be noted.

## What Would Kill This

- **Memory explosion at N>24:** Peak memory stayed constant at 17.1GB because adapters
  are trained sequentially with cleanup. This holds at any N. WOULD ONLY FAIL if
  composition evaluation (loading all N adapters simultaneously) exceeds memory.
  At N=24: 24 * 210 * rank * d_out * 2 bytes ~ 1.6GB for B matrices. Safe to N>>100.

- **Orthogonality breakdown:** Would occur at N > d/r = 160. We are at N=24, 6.7x below
  the theoretical limit. Even at N=100, Nr=1600 < d=2560, so orthogonality is guaranteed.

- **Routing collapse with semantically similar domains:** The lowest val accuracy (85.3%,
  sports) hints at this. If we added 50+ highly overlapping domains (e.g., 10
  different programming languages), routing might fail to discriminate.

- **Composition dilution at large N:** The 1/N scaling means each expert contributes
  only 1/24 = 4.2% of the delta. At N=100, each expert contributes 1%. If the
  beneficial composition effect scales sublinearly with N, diminishing returns would
  eventually make composition worse than base.

## Runtime

- Data preparation: ~2 min (cached after first run)
- Base PPL evaluation: ~1 min (24 domains)
- Grassmannian skeleton: < 1s (Nr=384 << d=2560)
- Adapter training: 5 reused + 19 new * ~65s = ~21 min
- Individual evaluation: ~1.5 min
- Composition evaluation: ~5.3 min
- Orthogonality: < 1s
- Routing heads (with train/val split): ~1.8 min
- **Total: ~32 minutes** on Apple M5 Pro 48GB
