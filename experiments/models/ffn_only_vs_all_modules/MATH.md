# FFN-only vs All-Modules LoRA Composition: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Value |
|--------|-----------|-------------|
| d | Model embedding dimension | 3584 (Qwen2.5-7B) |
| d_ff | FFN intermediate dimension | 18944 (= 5.29d) |
| d_kv | KV head dimension total | 512 (= d_head * n_kv) |
| d_head | Per-head dimension | 128 |
| n_h | Number of attention heads | 28 |
| n_kv | Number of KV heads (GQA) | 4 |
| L | Number of layers | 28 |
| r | LoRA rank | 16 |
| alpha | LoRA scaling factor | 16 |
| N | Number of domain experts | variable |

## 2. LoRA Delta Structure

For a weight matrix W with LoRA adapter (A, B):

    W_adapted = W + (alpha / r) * B @ A

where A: (r, d_in), B: (d_out, r) in PEFT convention.

The weight delta is:

    dW = (alpha / r) * B @ A    shape: (d_out, d_in)

This delta has rank at most r.

## 3. Parameter Counting

### 3.1 FFN-only Configuration

Target modules: gate_proj, up_proj, down_proj per layer.

Per layer LoRA params:

    gate_proj: A(r, d) + B(d_ff, r) = r*d + r*d_ff
    up_proj:   A(r, d) + B(d_ff, r) = r*d + r*d_ff
    down_proj: A(r, d_ff) + B(d, r) = r*d_ff + r*d

    Total per layer: 3*r*d + 3*r*d_ff = 3*r*(d + d_ff)

For Qwen2.5-7B:

    Per layer: 3 * 16 * (3584 + 18944) = 48 * 22528 = 1,081,344
    Total (28 layers): 30,277,632

### 3.2 All-Modules Configuration

Additional attention target modules: q_proj, k_proj, v_proj, o_proj.

Per layer attention LoRA params:

    q_proj: A(r, d) + B(d, r)     = 2*r*d
    k_proj: A(r, d) + B(d_kv, r)  = r*d + r*d_kv
    v_proj: A(r, d) + B(d_kv, r)  = r*d + r*d_kv
    o_proj: A(r, d) + B(d, r)     = 2*r*d

    Total attn per layer: 6*r*d + 2*r*d_kv = r*(6*d + 2*d_kv)

For Qwen2.5-7B:

    Attn per layer: 16 * (6*3584 + 2*512) = 16 * 22528 = 360,448
    Attn total: 10,092,544
    All-modules total: 30,277,632 + 10,092,544 = 40,370,176

### 3.3 Parameter Ratio

    all_params / ffn_params = 40,370,176 / 30,277,632 = 1.33x

All-modules LoRA uses 33% more parameters than FFN-only at the same rank.

## 4. Delta Space Dimensionality

The flattened delta vector lives in a space of dimension equal to the total
number of weight elements perturbed (not the LoRA parameter count, but the
full weight matrix dimensions).

### 4.1 FFN Delta Dimensions

    gate_proj delta: d * d_ff    = 3584 * 18944 = 67,886,336
    up_proj delta:   d * d_ff    = 3584 * 18944 = 67,886,336
    down_proj delta: d_ff * d    = 18944 * 3584 = 67,886,336

    FFN delta dim per layer: 3 * d * d_ff = 203,659,008
    FFN delta dim total: 28 * 203,659,008 = 5,702,452,224

(Note: slight difference from code due to dimension ordering conventions.)

### 4.2 Attention Delta Dimensions

    q_proj delta: d * d     = 3584^2    = 12,845,056
    k_proj delta: d * d_kv  = 3584*512  = 1,835,008
    v_proj delta: d * d_kv  = 3584*512  = 1,835,008
    o_proj delta: d * d     = 3584^2    = 12,845,056

    Attn delta dim per layer: 2*d^2 + 2*d*d_kv = 29,360,128
    Attn delta dim total: 28 * 29,360,128 = 822,083,584

### 4.3 Ratio

    FFN delta dim / All delta dim = 5,702,452,224 / 6,524,535,808 = 87.4%

The FFN modules account for 87.4% of the total delta weight space. This is
because d_ff = 5.29d, making FFN weight matrices much larger than attention
projection matrices.

## 5. Orthogonality Analysis

### 5.1 Random Vector Expected Cosine

For two random unit vectors in D-dimensional space:

    E[|cos(u, v)|] = sqrt(2 / (pi * D))

This gives:

    FFN-only (D ~ 5.7B):    E[|cos|] ~ 1.06e-5
    All-modules (D ~ 6.5B): E[|cos|] ~ 9.88e-6

**Counterintuitively**, all-modules has LOWER expected cosine for random vectors
because its delta space is larger (more dimensions = more room for orthogonality).

### 5.2 Why Real Adapters Differ from Random

The Monte Carlo simulation confirms the random-vector prediction: in a larger
space, random deltas are more orthogonal. But real trained adapters show the
OPPOSITE pattern:

    FFN-only  mean |cos| = 0.0605 (lower = more orthogonal)
    All-mods  mean |cos| = 0.0711
    Attn-only mean |cos| = 0.0853 (highest = least orthogonal)

This reversal occurs because trained adapters are NOT random. Attention
parameters learn TASK-GENERAL patterns (how to route information between
token positions) that are similar across domains. FFN parameters learn
TASK-SPECIFIC patterns (which facts/knowledge to retrieve) that are more
domain-discriminative.

### 5.3 The Geva et al. Argument

Geva et al. (2021) showed that FFN layers operate as key-value memories:
- Each FFN neuron's first-layer weight is a "key" that activates on
  specific input patterns
- Each FFN neuron's second-layer weight is a "value" that promotes
  specific output tokens
- Different domains activate different key-value pairs

If domain A uses keys K_A and domain B uses keys K_B, and K_A and K_B
are largely disjoint (different knowledge), then:

    cos(dW_FFN^A, dW_FFN^B) ~ |K_A intersect K_B| / |K_A union K_B|

This is small when domains are distinct (e.g., bash vs medical).

Attention, by contrast, learns position-based routing patterns that are
more universal: "attend to the most recent noun" or "attend to the
beginning of the sentence" are useful for ALL domains. This creates
structural correlation:

    cos(dW_Attn^A, dW_Attn^B) > cos(dW_FFN^A, dW_FFN^B)

### 5.4 Composition Interference

When composing N experts via task arithmetic:

    W_composed = W_base + (1/N) * sum_{k=1}^{N} dW_k

The composition error depends on the cross terms:

    ||W_composed - W_optimal||^2 ~ sum_{i!=j} <dW_i, dW_j>

Lower pairwise cosine similarity means smaller cross terms, which means
better composition. Since attention has higher inter-expert cosine, it
contributes disproportionately to composition interference.

Removing attention LoRA from composition eliminates this interference
source, at the cost of not adapting attention patterns per domain.

## 6. The Math-Medical Outlier

One pair (math vs medical) shows extreme similarity:

    FFN cos = 0.590, Attn cos = 0.850, Full cos = 0.703

This is 100x larger than all other pairs (which are < 0.003). Two
hypotheses:

1. **Training data contamination**: the math and medical training data
   may share significant content (e.g., medical statistics, dosage
   calculations, scientific notation).

2. **Domain overlap in teacher outputs**: the 70B teacher may have
   generated similar reasoning patterns for both math and medical
   (step-by-step reasoning, numerical computation).

Note that even for this outlier, FFN cosine (0.590) is LOWER than
attention cosine (0.850), consistent with the hypothesis that attention
patterns are more shared than FFN knowledge.

## 7. Orthogonality Capacity

### 7.1 N_max Bound

The maximum number of nearly-orthogonal rank-r subspaces in
D-dimensional space is:

    N_max ~ D / r^2

For Qwen2.5-7B at rank 16:

    FFN-only:    N_max ~ 5.7B / 256 ~ 22.3M experts
    All-modules: N_max ~ 6.5B / 256 ~ 25.5M experts

Both are astronomical. The practical limit is not orthogonality
capacity but storage and routing.

**Note on D vs d formulas:** VISION.md and project memory use a
different formula: N_max ~ d^2/r^2 where d is the model embedding
dimension (e.g., d=896 gives ~122K at r=16). These measure different
things. The d^2/r^2 formula counts the number of rank-r subspaces
that can pack into a single d-by-d weight matrix (a per-layer,
per-module bound). The D/r^2 formula used here counts packing into
the FULL flattened delta vector across ALL layers and modules
(D = total number of weight elements perturbed). The per-layer
formula is more conservative and practically relevant since
interference is local to each weight matrix. The full-delta formula
gives an upper bound that is unlikely to be tight. For architectural
decisions, the per-layer d^2/r^2 bound from VISION.md is the
appropriate one to use.

### 7.2 Parameter Efficiency

At matched rank r=16:

    FFN-only:    30.3M params per expert, ~7.2 MB on disk
    All-modules: 40.4M params per expert, ~9.6 MB on disk

FFN-only uses 25% FEWER bytes per expert while capturing the
domain-specific knowledge (the "value" of the expert). Attention
parameters add 33% more bytes but primarily learn shared patterns
that are redundant across experts.

## 8. Assumptions

1. **FFN captures domain knowledge**: follows Geva et al. (2021).
   If some domains require domain-specific attention patterns (e.g.,
   code with deeply nested scope), FFN-only may underperform.

2. **Cosine similarity of raw parameters is an approximate proxy for
   delta cosine**: we compare flattened (A, B) parameter vectors, not
   the expanded deltas vec(B@A). The mapping from raw parameters to
   expanded deltas is nonlinear and NOT monotonic in cosine -- e.g.,
   if A1 = A2 but B1 is orthogonal to B2, the raw parameter cosine
   is ~0.5 but the delta cosine depends on the structure of A. The
   composition interference formula (Section 5.4) depends on expanded
   delta cosine <vec(B_i@A_i), vec(B_j@A_j)>, not raw parameter
   cosine. We use raw parameters because expanding all deltas (each
   ~200M elements for FFN per layer) is computationally prohibitive
   at this scale. The directional claim (FFN more orthogonal than
   attention) likely holds for expanded deltas given the Geva et al.
   structural argument, but this is an assumption, not a proven fact.
   This is a known limitation.

3. **The math-medical outlier is an artifact**: if it reflects genuine
   domain overlap, the pairwise cosine for that pair is inherent and
   not a problem with the methodology.

## 9. Worked Example (Micro Scale)

d=64, d_ff=256, L=4, r=8:

    FFN params per layer: 3 * 8 * (64 + 256) = 7,680
    FFN total: 4 * 7,680 = 30,720

    Attn params per layer: 8 * (6*64 + 2*64) = 8 * 512 = 4,096
    Attn total: 4 * 4,096 = 16,384

    All total: 30,720 + 16,384 = 47,104
    Ratio: 47,104 / 30,720 = 1.53x

    FFN delta dim: 4 * 3 * 64 * 256 = 196,608
    Attn delta dim: 4 * 4 * 64 * 64 = 65,536
    All delta dim: 262,144

    E[|cos|] FFN: sqrt(2/(pi*196608)) = 0.001798
    E[|cos|] All: sqrt(2/(pi*262144)) = 0.001557

    N_max FFN: 196,608 / 64 = 3,072
    N_max All: 262,144 / 64 = 4,096
