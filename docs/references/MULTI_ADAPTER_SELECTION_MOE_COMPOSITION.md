# Multi-Adapter Selection & MoE Composition: Deep Research

Date: 2026-03-28
NotebookLM ID: 7c644120-7191-43aa-bee6-f9f12f689da3

---

## 1. Multi-Adapter Selection: How Production MoE Models Handle Top-K Expert Selection

### 1.1 Core MoE Formulation

The standard MoE output for a token with hidden state `u_t`:

```
h_t = u_t + sum_{i=1}^{N} g_{i,t} * FFN_i(u_t)
```

Gating scores computed as:
```
s_{i,t} = G(u_t . e_i^T)
```

where `G` is the gating function (softmax for Mixtral/Qwen3, sigmoid for DeepSeek-V3), `e_i` is the expert embedding.

Top-K selection:
```
g_{i,t} = s_{i,t}  if s_{i,t} in TopK({s_{j,t} | 1<=j<=N}, K)
         = 0        otherwise
```

**Critical: experts compose in OUTPUT space.** Each selected expert independently processes `u_t`, and their outputs are weighted-summed. No parameter merging occurs.

### 1.2 DeepSeek-V3 (Top-2 of 256 routed + 1 shared)

- **Paper:** arxiv:2412.19437
- **Architecture:** 256 routed experts per MoE layer, top-2 selection, plus 1 always-active shared expert
- **Gating:** Sigmoid (NOT softmax) on `u_t . e_i^T`
- **Key innovation:** Auxiliary-loss-free load balancing via bias terms (see Section 3)
- **Complementary balance loss:** `L_Bal = alpha * sum_{i=1}^{N_r} f_i * P_i` with very small alpha
- **Sequence-level balance:** Balance computed per-sequence, not per-batch

### 1.3 Mixtral (Top-2 of 8)

- **Paper:** arxiv:2401.04088
- **Architecture:** 8 experts per MoE layer, top-2 selection
- **Gating:** Softmax-based router: `G(x) = Softmax(W_g * x)`, select top-2
- **Output:** `y = sum_{i in top2} g_i * FFN_i(x)` where g_i are the softmax weights of selected experts
- **No explicit auxiliary loss reported** - relies on natural balance from top-2 selection with 8 experts

### 1.4 Qwen3 (Top-8 of 128)

- **Paper:** arxiv:2505.09388
- **Architecture:** 128 experts per MoE layer, top-8 activated
- **Gating:** Top-k learned gates with gating noise
- **Balance:** Global-batch load balancing loss + router z-loss for stability
- **No shared experts** (unlike Qwen2.5-MoE)
- **Qwen3-30B-A3B:** 30.5B total params, 3.3B active per token

### 1.5 How This Differs from LoRA Merging

**MoE (output-space composition):**
```
y = W*x + sum_{i in topK} g_i * (B_i * A_i * x)
```
Each expert processes input independently. Outputs are linearly combined.

**LoRA merging (parameter-space composition):**
```
Delta_W = sum_i alpha_i * B_i * A_i
y = (W + Delta_W) * x
```
Parameters are combined BEFORE the forward pass. Creates cross-terms.

**The cross-term problem (from LoRA Soups, arxiv:2410.13025):**

Linear merging expands to:
```
Delta_W = (sum_i alpha_i * B_i)(sum_j alpha_j * A_j)^T
        = sum_i alpha_i^2 * B_i*A_i^T  +  sum_{i!=j} alpha_i*alpha_j * B_i*A_j^T
                ^-- desired terms              ^-- cross-terms (interference!)
```

Concatenation (output-space) eliminates cross-terms:
```
Delta_W = sum_i alpha_i * B_i * A_i^T   (no cross-products)
```

---

## 2. Adapter Retrieval and Library-Based Selection WITHOUT Training a Router

### 2.1 LoraHub (arxiv:2307.13269) - Gradient-Free Composition

**Method:** Given N candidate LoRA modules, find scalar weights w_1...w_N using gradient-free optimization on few-shot examples.

**Composition formula:**
```
m_hat = (w_1*A_1 + w_2*A_2 + ... + w_N*A_N)(w_1*B_1 + w_2*B_2 + ... + w_N*B_N)
```

**Optimization:** CMA-ES (Covariance Matrix Adaptive Evolution Strategy) via Nevergrad library.

**Objective:** `min L + 0.05 * sum_i |w_i|`
- L = cross-entropy loss on K few-shot examples (typically K=5)
- L1 regularization prevents extreme coefficients
- Weight bounds: |w_i| <= 1.5
- Max iterations: 40 CMA-ES steps
- Initial weights: all zero

**Scaling behavior:**
- Standard config: N=20 randomly selected from ~200 candidates
- Tested N=5 to N=100: performance variance increases with N, but max achievable performance also improves
- Single LoRA retrieval baseline: 31.7% avg
- LoraHub N=20: 34.7% avg (+3%)
- LoraHub best: 41.2%

**Limitation:** This is PARAMETER-SPACE composition (creates cross-terms). The `(sum w_i A_i)(sum w_j B_j)` formulation mixes A and B matrices across tasks.

### 2.2 LoraRetriever (arxiv:2402.09997) - Input-Aware Retrieval + Composition

**Method:** Retrieve-then-compose framework with three phases:
1. **Encode:** Each LoRA adapter represented by averaging instruction-guided sentence embeddings of a few training examples
2. **Retrieve:** Top-k most similar adapters via cosine similarity in embedding space
3. **Compose:** Either linear interpolation (parameter-space) or on-the-fly output composition

**Retrieval:** Contrastive learning on (same-task, different-task) pairs to learn adapter embeddings. Uses instruction-tuned sentence encoders.

**Key advantage over LoraHub:** No per-query optimization needed. Retrieval is a single forward pass through the sentence encoder + nearest neighbor search.

### 2.3 Arrow Routing (Ostapenko et al., arxiv:2402.05859) - Zero-Shot SVD-Based

**Method:** Construct a prototype vector per expert using SVD of LoRA weight matrices.

**Mathematical formulation:**
```
U_t, S_t, V_t = SVD(B_t * A_t)
v_t = V_t[:, 0]   # first right singular vector (maximally affected direction)
```

The prototype `v_t` is the input direction that the LoRA adapter modifies most.

**Routing score:** `score_t = |<v_t, u_t>|` (absolute dot product between prototype and input activation)

**Selection:** `experts = argmax_topK(|P_l * x|)` where P_l[t] = v_t is the routing matrix.

**Weakness:** Only uses rank-1 approximation. For rank-r adapters, captures only 1/r of total variation. Performance degrades as adapter rank increases.

**Results (from PHATGOOSE paper):**
- Arrow underperforms PHATGOOSE on all benchmarks
- T0 Held-In: Arrow 55.1% vs PHATGOOSE higher
- The trained gating step in PHATGOOSE is essential

### 2.4 SpectR (arxiv:2504.03454) - Spectral Routing (Improves Arrow)

**Key insight:** Arrow only uses top eigenvector. SpectR uses the ENTIRE spectrum.

**Reparameterization:**
```
U_t, S_t, V_t = SVD(B_t * A_t)
B_t* = U_t        # reparameterized
A_t* = S_t * V_t^T  # captures full spectrum
```

**Routing score:** `s_t = ||A_t* * x||_2` (L2 norm of projected input)

**When A_t* is rank-1, this reduces to Arrow.**

**Results:** ~4pp average routing accuracy improvement over Arrow, with up to 24pp gains on individual tasks (dbpedia, qqp).

### 2.5 PHATGOOSE (arxiv:2402.05859) - Per-Expert Trained Gates

**Method:** After training each LoRA adapter, freeze it and train a sigmoid gate per expert:
```
gate_score = sigmoid(g_i . h_t)
```
where g_i is a learned gate vector for expert i, h_t is the token representation.

**Routing:** Stack and normalize all gates, compute dot-product scores, select top-k.

**Advantage:** Trained gates capture what each expert is good at.
**Disadvantage:** Requires per-expert gate training (not zero-shot).

### 2.6 AdapterSoup (arxiv:2302.07027) - Domain Clustering + Weight Averaging

**Method:** Train domain-specific adapters, then for a novel domain:
1. Cluster training domains by text similarity
2. Select adapters from the most similar cluster
3. Simple weight averaging of selected adapters

**Key finding:** Clustering-based selection outperforms random selection or using all adapters.

**Limitation:** Weight averaging is PARAMETER-SPACE composition (cross-terms apply).

### 2.7 Scaling to N>100 Without Routing Collapse

| Method | N tested | Routing type | Scales? |
|--------|----------|--------------|---------|
| LoraHub | N=100 | Gradient-free optimization | Slow (40 CMA-ES steps per query) |
| LoraRetriever | N~50 | Embedding retrieval | Yes (nearest neighbor is O(N)) |
| Arrow | N~20 | SVD prototype | Degrades with rank |
| SpectR | N~20 | Full spectral | Better than Arrow |
| PHATGOOSE | N~20 | Trained gates | Requires per-expert training |
| AdapterSoup | N~15 | Clustering | Limited by cluster quality |

---

## 3. Routing Collapse: Why It Happens and How Production Systems Prevent It

### 3.1 What Is Routing Collapse?

Routing collapse occurs when the router consistently sends most tokens to a small subset of experts:
```
exists e: f_e ~ 1 and forall e' != e: f_e' ~ 0
```

**Detection metrics:**
- Load imbalance ratio: `LIR = max(f_e) / min(f_e)` (safe: < 2.0, collapse: > 5.0)
- Routing entropy: `H(g) = -sum f_e/N * log(f_e/N)` (collapse when H < 0.5 * log(E))
- Expert utilization variance: `Var(f_e)` should be minimized

### 3.2 Why It Happens at N>10

**Root cause:** Self-reinforcing feedback loop.
1. Expert i gets slightly more tokens early in training
2. Expert i trains faster (more gradient updates)
3. Expert i becomes better
4. Router sends MORE tokens to expert i
5. Goto 2

**Scaling makes it worse:** With N=8 (Mixtral), random initialization gives each expert ~12.5% of tokens. With N=256 (DeepSeek), each gets ~0.4%. Small perturbations at 0.4% create much larger relative imbalances than at 12.5%.

**The router can single-handedly destroy the model.** Perfect expert architecture, tuned hyperparameters, unlimited compute - if the router collapses, you get dense model performance regardless of expert count.

### 3.3 Traditional Auxiliary Loss Approach

**GShard/Switch Transformer loss:**
```
L_aux = alpha * sum_{e=1}^{E} f_e * P_e
```
where f_e = fraction of tokens routed to expert e, P_e = average gating probability for expert e.

**Problem:** alpha creates a Goldilocks dilemma:
- Too small: insufficient balancing, collapse still occurs
- Too large: impairs model performance (forces uniform routing, destroying specialization)

**Expert capacity factor (Switch Transformer):**
```
C = CF * (tokens_per_batch / num_experts)
```
Typical CF = 1.25. Tokens exceeding expert capacity are DROPPED. Trade-off: too high wastes compute, too low drops tokens.

### 3.4 Router Z-Loss (ST-MoE, OpenMoE)

Prevents routing instability by penalizing large logits:
```
L_z = (1/B) * sum_{i=1}^{B} (log sum_{j=1}^{N} exp(x_{ij}))^2
```
This prevents the router from becoming overconfident (peaky distributions that lead to collapse).

### 3.5 DeepSeek Auxiliary-Loss-Free Bias (arxiv:2408.15664)

**Core innovation:** Replace auxiliary loss with a bias term updated outside backpropagation.

**Selection with bias (for routing only):**
```
g_{i,t} = s_{i,t}  if (s_{i,t} + b_i) in TopK({s_{j,t} + b_j}, K)
         = 0        otherwise
```

**Output weighting (NO bias):**
```
h_t = u_t + sum_i g_{i,t} * FFN_i(u_t)
```

**CRITICAL: b_i is used for SELECTION but NOT for output weighting.** This means the bias only influences which experts are chosen, not how much they contribute.

**Bias update rule:**
```
e_i = c_bar_i - c_i     (expected load minus actual load)
b_i = b_i + u * sign(e_i)
```
where u is the update rate (hyperparameter). Overloaded experts get negative bias (less likely selected), underloaded get positive bias (more likely selected).

**Why it works:** No gradient interference with the main training objective. The bias acts as a simple feedback controller.

**Results:** Better model performance AND better load balance than traditional auxiliary loss.

### 3.6 Expert Choice Routing (Google, arxiv:2202.09368)

Inverts the routing: instead of tokens choosing experts, EXPERTS choose tokens.
```
S = Softmax(W_g * X^T)   # [num_experts x num_tokens]
For each expert i: select top-C tokens by score
```

**Advantage:** Perfect load balance by construction (each expert gets exactly C tokens).
**Disadvantage:** Token may be processed by 0 or many experts (no guarantee).

### 3.7 Random Routing as Baseline

Random uniform routing (each token sent to K random experts) provides:
- Perfect load balance
- Zero routing cost
- But no specialization

**Key insight from literature:** Random routing often performs surprisingly well, especially at small N. The gap between random and learned routing narrows as N decreases because with few experts, every expert must be somewhat general.

### 3.8 Minimum Routing Accuracy for Composition to Help

From the "Pause Recycling LoRAs" paper (arxiv:2506.13479):

**Devastating finding:** LoRA composition fundamentally fails for tasks NOT in the training data.
- Two-adapter composition: ~10% accuracy on unseen compositional tasks
- Routing accuracy doesn't matter if the composed adapters can't do the task
- Arrow routing HURTS on math: Uniform 10%, Arrow 9% on zero-shot GSM-P2
- In-context examples + routing actually WORSENED results: 27% -> 6%

**What matters is not routing accuracy but whether the selected experts contain the needed capability.** If no adapter covers the target task, no routing strategy helps.

**Empirical threshold from LoRA Soups:** Composition helps for k=2 skill tasks (43% improvement over data-mix). At k>=3 skills, data-mix outperforms ALL merging methods.

---

## 4. Output-Space vs Parameter-Space Composition: Mathematical Comparison

### 4.1 Parameter-Space Composition (LoRA Merge)

Given T tasks with LoRA adapters (A_t, B_t):
```
Delta_W_merge = sum_t alpha_t * B_t * A_t^T
y = (W + Delta_W_merge) * x
```

**With shared low-rank structure, linear merging creates cross-terms (LoRI analysis, arxiv:2504.07448):**
```
Delta_merge = (sum_t alpha_t * A_t)(sum_t alpha_t * (B_t . M_t))
            = sum_s sum_t alpha_s * alpha_t * A_s * (B_t . M_t)
```

For s != t, the terms `A_s * (B_t . M_t)` are cross-task interference.

### 4.2 Output-Space Composition (MoE / Concatenation)

```
y = W*x + sum_t alpha_t * (B_t * A_t * x)
```

Each adapter processes input independently. **NO cross-terms.**

**Equivalently (LoRI formulation):**
```
A' = [alpha_1*A_1 ; alpha_2*A_2 ; ... ; alpha_T*A_T]   (concatenated)
B' = [(B_1.M_1)^T, (B_2.M_2)^T, ..., (B_T.M_T)^T]^T  (stacked)
Delta_merge = A' * B' = sum_t alpha_t * Delta_t          (no cross-terms)
```

### 4.3 When Does Each Strategy Win?

**Parameter-space wins when:**
- Few adapters (k=2) on overlapping tasks
- Super-linear improvement possible: LoRA Soups reports combining math (14.18%) and code (5.91%) yields 21.11% accuracy (exceeding sum of individual gains)
- Adapters occupy similar subspaces (constructive interference from cross-terms)

**Output-space wins when:**
- Many adapters (k>=3)
- Diverse/conflicting tasks (cross-terms become destructive)
- LoRI shows: Linear LoRA merging drops to 22.3% HumanEval vs 63.2% single-task. Concatenated LoRI: 62.2% (near single-task)
- Preserves nonlinearity: each expert's B_i * A_i applies independently, maintaining each task's learned nonlinear structure

### 4.4 Nonlinearity Preservation

Parameter-space merging happens BEFORE the nonlinearity:
```
y = activation(W*x + sum alpha_i B_i A_i x)
  = activation(W*x + merged_Delta * x)       # single merged perturbation
```

Output-space composition CAN preserve per-expert nonlinearity if composition happens after expert FFN:
```
y = sum_i g_i * FFN_i(x)    # each FFN_i has its own internal nonlinearities
```

This is why full MoE (with separate expert FFNs) is strictly more expressive than merged LoRA.

### 4.5 LoRI's Orthogonal Solution (The Hybrid)

**arxiv:2504.07448**

Freeze A_t as random Gaussian projections. Then for independent A_s, A_t:
```
A_s^T * A_t ~ 0_{r x r}   (with high probability when r << d_in)
```

Therefore cross-terms vanish:
```
<Delta_s, Delta_t> = Tr[(B_s.M_s)^T * A_s^T * A_t * (B_t.M_t)] ~ 0
```

**Result:** Parameter-space merging that behaves like output-space composition (no interference) while maintaining the efficiency of merged weights. The trick is ensuring adapters occupy orthogonal subspaces.

### 4.6 CoMoL: Core Space Merging (arxiv:2603.00573)

A true hybrid: decompose each expert via SVD, then merge in a reduced "core space."

```
B = U_B * Sigma_B * V_B^T
A = U_A * Sigma_A * V_A^T
M = Sigma_B * V_B^T * U_A * Sigma_A   (core matrix, r x r)
Delta_W = U_B * M * V_A^T
```

**Token-level MoE in core space:**
```
h = W*x + U_B * (sum_i G(x)_i * M_i) * V_A^T * x
```

**Efficiency:** Only the small r x r core matrices M_i are routed/merged per token. The expensive d x r matrices U_B, V_A are shared.

FLOPs comparison:
- Output-level MoE: `2*L*r*(2n+r)*N`
- Core-space merging: `2*L*r*(2n+r) + L*r^2*(N-1)`

Reduces per-expert overhead by factor of 1/N.

---

## 5. Key Paper Index

| Paper | ArXiv ID | Key Contribution |
|-------|----------|------------------|
| DeepSeek-V3 | 2412.19437 | Aux-loss-free bias, top-2/256, sigmoid gating |
| Mixtral | 2401.04088 | Top-2/8 softmax MoE, practical deployment |
| Qwen3 | 2505.09388 | Top-8/128, global-batch balance loss, no shared experts |
| LoraHub | 2307.13269 | CMA-ES gradient-free composition, N=20 |
| LoraRetriever | 2402.09997 | Embedding-based retrieval + composition |
| PHATGOOSE/Arrow | 2402.05859 | Trained gates vs SVD zero-shot routing |
| SpectR | 2504.03454 | Full-spectral routing (improves Arrow) |
| AdapterSoup | 2302.07027 | Domain clustering + weight averaging |
| LoRA Soups | 2410.13025 | Skill composition analysis, cross-terms |
| LoRI | 2504.07448 | Orthogonal projections eliminate cross-terms |
| Pause Recycling LoRAs | 2506.13479 | Fundamental limits of LoRA composition |
| Aux-Loss-Free Balance | 2408.15664 | Bias-based load balancing (DeepSeek method) |
| CoMoL | 2603.00573 | Core-space MoE merging hybrid |
| Switch Transformer | 2101.03961 | Expert capacity factor, aux loss, top-1 routing |
| Expert Choice Routing | 2202.09368 | Experts choose tokens (perfect balance) |
| MoE Balance Review | HF blog | Comprehensive survey of balance strategies |

---

## 6. Actionable Implications for SOLE Architecture

1. **Output-space composition (MoE-style) is strictly superior to parameter merging at N>2.** Cross-terms from linear merging are destructive for diverse tasks.

2. **DeepSeek's bias trick is the state of the art for load balancing.** Simple, no gradient interference, works at N=256. Directly applicable to adapter selection.

3. **Zero-shot routing (Arrow/SpectR) works but has limits.** SpectR's spectral approach is better than Arrow's rank-1 approximation. Both degrade for high-rank adapters.

4. **LoRI's orthogonal projection trick** could make parameter-space merging safe (cross-terms vanish) while keeping inference efficient (single merged weight matrix). Worth investigating.

5. **Routing accuracy is secondary to adapter coverage.** If no adapter covers the target task, no routing strategy helps (Pause Recycling LoRAs finding). The composition of two adapters only works if the target task is representable by their combined capabilities.

6. **k=2 is the sweet spot for composition.** LoRA Soups shows super-linear gains at k=2, but k>=3 loses to data mixing. MoE systems use k=2 (DeepSeek, Mixtral) or k=8 (Qwen3, but from 128 experts).

7. **CoMoL's core-space merging** is a promising hybrid: MoE routing with 1/N the per-expert cost by operating in the r x r core space rather than full d x d parameter space.
