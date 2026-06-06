# Teacher Size vs Expert Quality: Mathematical Framework

## 1. Problem Statement

Given a student model S (Qwen2.5-7B, d=3584) and two teacher models:
- T_8B: Llama 3.1 8B (barely larger than student)
- T_70B: Llama 3.3 70B (10x larger than student)

We distill domain-specific LoRA experts from each teacher and compare quality.
The question: does teacher capacity affect LoRA adapter quality?

## 2. Notation

| Symbol | Definition | Shape/Value |
|--------|-----------|-------------|
| S | Student model (Qwen2.5-7B) | 7.6B params |
| T_s | Small teacher (Llama 3.1 8B) | 8B params |
| T_L | Large teacher (Llama 3.3 70B) | 70.6B params |
| d | Student hidden dimension | 3584 |
| r | LoRA rank | 16 |
| D | Number of domains | 10 |
| N | Training examples per domain | 1000 |
| A_i, B_i | LoRA factors for expert i | R^{d x r}, R^{r x d} |
| PPL_a(e, x) | Answer-conditioned PPL of expert e on test x | scalar |
| Q(T, d) | Quality of expert trained on teacher T's data for domain d | scalar |

## 3. Quality Metric: Answer-Conditioned PPL

Following the proven metric from exp_answer_conditioned_scoring (r=0.811):

For a test example (x_prompt, x_answer):

PPL_a(e) = exp( - (1/T_a) * sum_{t in answer_tokens} log p_e(x_t | x_{<t}) )

where T_a = |answer_tokens| and p_e is the probability under model S + expert e.

Key property: PPL_a isolates the quality of the response, not the prompt modeling.

## 4. Distillation Quality Model

### 4.1 Information Bottleneck

The LoRA adapter has capacity C = r * d * L_target parameters (where L_target
is the number of target modules per layer times number of layers). For
Qwen2.5-7B with 7 target modules and 28 layers:

C = 16 * 3584 * 7 * 28 * 2 = 22,478,848 params (~22.5M)

This capacity is fixed regardless of teacher. The adapter can only capture
~22.5M parameters worth of the teacher's knowledge.

### 4.2 Teacher Quality Ceiling

Let q_T(d) be the quality of teacher T's responses for domain d, measured
as the expected quality of a randomly sampled response.

For any domain d:
- q_{T_L}(d) >= q_{T_s}(d)  (larger teacher generally better)
- The gap varies by domain complexity

### 4.3 Quality Decomposition

The quality of an expert trained on teacher T's data for domain d:

Q(T, d) = min(C_adapter, q_T(d)) - epsilon_noise

where:
- C_adapter: adapter capacity (fixed for both teachers)
- q_T(d): teacher response quality for domain d
- epsilon_noise: training noise (optimizer, data sampling)

**Key insight:** When q_T(d) < C_adapter (teacher quality is the bottleneck),
a better teacher helps. When q_T(d) > C_adapter (adapter capacity is the
bottleneck), teacher quality is wasted -- the adapter can't absorb more.

### 4.4 Domain Complexity Hypothesis

Domains can be characterized by a complexity c(d):
- Low complexity (c(d) << C_adapter): Both 8B and 70B teachers saturate the
  adapter. Expected gap ~ 0. Examples: SQL, accounting, bash.
- High complexity (c(d) ~ C_adapter): 70B teacher provides richer training
  signal that the adapter can absorb. Expected gap > 0. Examples: ethics,
  creative fiction, legal reasoning.

This motivates the mixed strategy: use 8B (cheap) where it's sufficient,
70B (expensive) where it matters.

## 5. Mixed Strategy Formalization

### 5.1 Cost Model

| Teacher | Data cost/expert | Training cost/expert | Total/expert |
|---------|-----------------|---------------------|-------------|
| T_8B | $0.02 | $0.09 | $0.11 |
| T_70B | $0.19 | $0.09 | $0.28 |

### 5.2 Mixed Strategy

Partition domains D = D_simple U D_complex. The mixed strategy:
- Use T_8B for d in D_simple (cost: $0.11/expert)
- Use T_70B for d in D_complex (cost: $0.28/expert)

Cost savings vs uniform T_70B:
savings = |D_simple| / |D| * (0.19 - 0.02) / 0.28 * 100%

For |D_simple| = |D_complex| = |D|/2:
savings = 0.5 * 0.17/0.28 * 100 = 30.4%

### 5.3 Quality Constraint

Mixed strategy is viable iff:
mean(Q(T_8B, d) for d in D_simple) >= mean(Q(T_70B, d) for d in D_simple) - tau

where tau = 5% (kill threshold).

## 6. Kill Criteria (Formalized)

### K1: Teacher Size Irrelevant
KILLED if: mean_d |Q(T_70B, d) - Q(T_8B, d)| / Q(T_70B, d) < 0.05

This would mean the adapter capacity, not the teacher, is the bottleneck
for ALL domain types.

### K2: Mixed Strategy Fails
KILLED if: mean_d PPL_mixed(d) > mean_d PPL_70B(d)

where PPL_mixed(d) = PPL_8B(d) if d in D_simple, else PPL_70B(d).

## 7. Worked Example

At micro scale (for intuition):
- Student: d=3584, r=16, capacity ~22.5M params
- 8B teacher generates Python code with ~95% accuracy (code is structured)
- 70B teacher generates Python code with ~98% accuracy
- Both produce good training data; adapter bottleneck means difference is ~3%

For ethics:
- 8B teacher produces shallow ethical analysis (~70% depth)
- 70B teacher produces nuanced multi-framework analysis (~92% depth)
- Adapter CAN capture the richer signal: gap is ~22%

Expected result: Domain complexity mediates the teacher size effect.

## 8. Assumptions

1. **Same student model:** Both adapters are trained on identical Qwen2.5-7B base
2. **Same hyperparameters:** rank=16, steps=300, lr=2e-4, all-modules
3. **Same training data format:** messages [{user, assistant}] with chat template
4. **Same test data:** 70B-generated held-out examples as gold standard
5. **Answer-conditioned PPL is valid metric:** Proven at r=0.811 (micro scale)
6. **Groq API delivers consistent quality:** No significant variation in API quality
   between models beyond their inherent capabilities
