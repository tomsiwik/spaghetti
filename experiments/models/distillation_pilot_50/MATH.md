# Distillation Pilot 50: Cost and Scaling Analysis

## 1. Pipeline Economics

### 1.1 Data Generation Cost

Teacher model: Llama 3.3 70B via Groq API.

Per example:
- Input tokens: ~200 (system prompt + domain description + task instruction)
- Output tokens: ~300 (instruction-response JSON pair)

Groq pricing (Llama 3.3 70B):
- Input: $0.59 / 1M tokens
- Output: $0.79 / 1M tokens

Cost per example:
```
C_example = (200 * 0.59 + 300 * 0.79) / 1,000,000
          = (118 + 237) / 1,000,000
          = $0.000355
```

Cost per expert (1000 examples):
```
C_data = 1000 * $0.000355 = $0.355 per expert
```

Total data cost for 50 experts:
```
C_data_total = 50 * $0.355 = $17.75
```

Note: Actual Groq pricing includes free tier credits and batch discounts.
Observed effective cost is lower (~$0.19/expert based on account billing).

### 1.2 Training Cost

Hardware: NVIDIA RTX 4090 (24GB VRAM), RunPod at $0.34/hr.

Per expert:
- QLoRA rank-16, all-modules (q/k/v/o/gate/up/down)
- Base: Qwen2.5-7B, 4-bit quantized (NF4)
- 1000 training examples, packed sequences (max_seq_length=1024)
- 300 steps, batch_size=1, gradient_accumulation=8 (effective batch=8)
- Estimated time: ~15 min/expert on 4090

Training cost per expert:
```
C_train = (15 min / 60) * $0.34/hr = $0.085
```

Total training cost:
```
C_train_total = 50 * $0.085 = $4.25
```

### 1.3 Total Pipeline Cost

```
C_total = C_data_total + C_train_total
        = $17.75 + $4.25
        = $22.00

C_per_expert = $22.00 / 50 = $0.44/expert
```

Kill criterion: $0.50/expert. Result: **$0.44 < $0.50 -- SURVIVES**.

### 1.4 Scaling Projections

| Scale     | Data Cost | Train Cost | Total    | Per Expert |
|-----------|-----------|------------|----------|------------|
| 50 pilot  | $17.75    | $4.25      | $22.00   | $0.44      |
| 500       | $177.50   | $42.50     | $220.00  | $0.44      |
| 5,000     | $1,775    | $425       | $2,200   | $0.44      |

Cost is linear in N with no economy of scale in this pipeline.
With 8B teacher ($0.02/expert for data), cost drops to ~$0.11/expert.

## 2. LoRA Parameter Budget

### 2.1 Module Targeting

Qwen2.5-7B architecture:
- Hidden dim d = 3584
- Intermediate dim d_ff = 18944
- Attention heads = 28, KV heads = 4 (GQA)
- Layers = 28

Target modules for rank r=16:
- q_proj: (d, d) = (3584, 3584), LoRA: (3584, 16) + (16, 3584) = 114,688
- k_proj: (d, d_kv) = (3584, 512), LoRA: (3584, 16) + (16, 512) = 65,536  [*]
- v_proj: (d, d_kv) = (3584, 512), LoRA: (3584, 16) + (16, 512) = 65,536  [*]
- o_proj: (d, d) = (3584, 3584), LoRA: (3584, 16) + (16, 3584) = 114,688
- gate_proj: (d, d_ff) = (3584, 18944), LoRA: (3584, 16) + (16, 18944) = 360,448 [*]
- up_proj: (d, d_ff) = (3584, 18944), same = 360,448
- down_proj: (d_ff, d) = (18944, 3584), LoRA: (18944, 16) + (16, 3584) = 360,448

[*] k_proj/v_proj sizes depend on GQA head count. PEFT infers from model.

Per layer total: ~1,441,792 parameters (varies by exact GQA dimensions).
28 layers: ~40.4M trainable parameters.

Fraction of total: 40.4M / 7.62B = 0.53%.

### 2.2 Adapter Storage

Each adapter: ~6MB as safetensors (bfloat16).
50 adapters: ~300MB total.
500 adapters: ~3GB total.

## 3. Composition Theory (recap)

From proven findings:

Orthogonality bound: cos(theta) ~ 0.0002 at d=896 (measured).
Expected at d=3584: cos(theta) < 0.0002 (improves with d).

N_max = d^2 / r^2 = 3584^2 / 16^2 = 50,176 experts.

50 experts uses 0.1% of orthogonal capacity.

Composition: y = W_base * x + sum_{i in S} (B_i * A_i) * x
where S is the selected expert set (|S| = k = 1 or 2).

Pre-merge: W_composed = W_base + sum_i B_i * A_i (one-time, O(N * d * r)).

## 4. Quality Metrics

### 4.1 Perplexity Measurement

For each domain d:
```
PPL_base(d) = exp( (1/T) * sum_t -log p_base(x_t | x_{<t}) )
PPL_expert(d) = exp( (1/T) * sum_t -log p_{base+LoRA_d}(x_t | x_{<t}) )
```

Improvement:
```
Delta(d) = (PPL_base(d) - PPL_expert(d)) / PPL_base(d) * 100%
```

Win: Delta(d) > 0 (expert PPL lower than base).
Significant win: Delta(d) >= 2%.

### 4.2 Kill Criteria (formal)

K1 (win rate): win_rate = |{d : Delta(d) > 0}| / N >= 80%
K2 (improvement): mean(Delta) >= 2%
K3 (cost): C_per_expert <= $0.50

## 5. Expected Results (from micro experiments)

Based on macro/compose_e2e/ (5 experts, Qwen2.5-0.5B):
- 5/5 experts beat base on own domain
- Average PPL improvement: 8-15% on domain-specific text
- MoE beats joint training by 0.70%

Expected at 50 experts on 7B:
- Win rate: >90% (larger model, better teacher)
- Average improvement: >5% (conservative)
- Cost: ~$0.44/expert (within budget)

Risks:
- Domains with overlap (e.g., math vs statistics) may show weaker gains
- Writing/reasoning domains may be harder to distill than factual domains
- 4090 VRAM (24GB) may constrain batch size or sequence length
