# Pierre Adapter Registry

## Naming Convention

```
<domain>-<source>-<type>-v<version>

domain:   math, code, medical, legal, finance, reasoning, thinking, personal, ...
source:   gsm8k, numimath, s1k, limo, synthetic, openthoughts, ...
type:     knowledge  (domain facts, NTP-trained)
          reasoning  (problem-solving traces, SFT on thinking chains)
          thinking   (meta-cognitive, trained WITH enable_thinking=True)
          format     (output structure: SOAP, legal citation, JSON)
          mcq        (multiple choice classification loss)
          grpo       (RL-trained via GRPO)
          personal   (per-user, online-trained)
version:  v0, v1, v2, ...

Examples:
  math-gsm8k-knowledge-v0       ← our first math adapter (T2.1, +82pp)
  math-s1k-reasoning-v0         ← trained on s1K reasoning traces
  math-s1k-thinking-v0          ← same but with thinking enabled
  math-limo-grpo-v0             ← GRPO on top of LIMO SFT
  code-openthoughts-reasoning-v0
  medical-pubmedqa-knowledge-v0
  thinking-meta-r1-v0           ← domain-agnostic thinking improvement
  format-soap-v0                ← SOAP note structure
```

## Directory Structure

```
adapters/
├── registry.json              ← adapter index with metadata + eval scores
├── math-gsm8k-knowledge-v0/
│   ├── adapters.safetensors   ← the weights
│   ├── adapter_config.json    ← LoRA config (rank, target modules, scale)
│   ├── metadata.json          ← training details, dataset, findings
│   └── evals.json             ← benchmark scores (MMLU-Pro, GSM8K, etc.)
├── math-s1k-reasoning-v0/
│   ├── adapters.safetensors
│   ├── adapter_config.json
│   ├── metadata.json
│   └── evals.json
└── ...
```

## registry.json Schema

```json
{
  "adapters": [
    {
      "name": "math-gsm8k-knowledge-v0",
      "domain": "math",
      "source": "gsm8k",
      "type": "knowledge",
      "version": 0,
      "path": "adapters/math-gsm8k-knowledge-v0/",
      "base_model": "mlx-community/gemma-4-e4b-it-4bit",
      "training": {
        "method": "sft",
        "dataset": "openai/gsm8k",
        "n_examples": 2000,
        "steps": 1000,
        "rank": 6,
        "target_modules": ["self_attn.q_proj"],
        "thinking_enabled": false,
        "experiment_id": "exp_p1_t2_single_domain_training",
        "finding_id": 421
      },
      "evals": {
        "gsm8k": {"score": 82.0, "n": 50, "date": "2026-04-10"},
        "mmlu_pro": {"score": 36.1, "n": 1400, "date": "2026-04-12"},
        "mmlu_pro_thinking": null
      },
      "size_mb": 5.0,
      "created": "2026-04-10",
      "status": "baseline",
      "notes": "First math adapter. NTP-trained. Degrades MCQ (-6.2pp, Finding #517)."
    }
  ]
}
```

## metadata.json Per Adapter

```json
{
  "name": "math-s1k-reasoning-v0",
  "parent": null,
  "derived_from": null,
  "training": {
    "method": "sft",
    "polar": true,
    "dataset": "simplescaling/s1K-1.1",
    "dataset_size": 1000,
    "steps": 1000,
    "learning_rate": 1e-4,
    "batch_size": 2,
    "rank": 8,
    "scale": 6.0,
    "target_modules": ["self_attn.v_proj", "self_attn.o_proj"],
    "thinking_enabled": true,
    "thinking_format": "gemma4_channel",
    "max_seq_length": 4096,
    "seed": 42
  },
  "provenance": {
    "experiment_id": "exp_p11_reasoning_sft_s1k",
    "finding_id": null,
    "training_log": "training_log.jsonl",
    "commit": null
  }
}
```

## evals.json Per Adapter

```json
{
  "benchmarks": {
    "gsm8k": {
      "score": 85.0,
      "n": 100,
      "thinking": true,
      "date": "2026-04-14",
      "experiment_id": "exp_p11_reasoning_sft_s1k"
    },
    "mmlu_pro": {
      "score": 67.0,
      "n": 1400,
      "thinking": true,
      "date": "2026-04-14"
    },
    "mmlu_pro_no_thinking": {
      "score": 44.0,
      "n": 1400,
      "thinking": false,
      "date": "2026-04-14"
    }
  },
  "comparison": {
    "vs_base": {"gsm8k": "+68pp", "mmlu_pro": "+4.9pp"},
    "vs_previous": {"adapter": "math-gsm8k-knowledge-v0", "gsm8k": "+3pp", "mmlu_pro": "+30.9pp"}
  }
}
```

## Adapter Types and When to Use

| Type | What It Teaches | Training Data | Thinking? | Use Case |
|------|----------------|---------------|-----------|----------|
| `knowledge` | Domain facts | Domain Q&A (GSM8K, MedMCQA) | No | Domain-specific answers |
| `reasoning` | Problem-solving traces | s1K, LIMO, OpenThoughts | Yes | Better thinking on hard problems |
| `thinking` | Meta-cognitive skills | Meta-R1, self-generated | Yes | Planning, self-checking, backtracking |
| `format` | Output structure | SOAP, legal, JSON examples | No | Structured output compliance |
| `mcq` | Multiple choice selection | MCQ with classification loss | Optional | Benchmark performance |
| `grpo` | RL-refined reasoning | GRPO on top of SFT adapter | Yes | Best benchmark scores |
| `personal` | User style/preferences | Conversation history | Optional | Per-user personalization |

## Composition Rules

```
COMPOSABLE (exclusive routing, one at a time):
  knowledge + knowledge  ← different domains, routed
  reasoning + knowledge  ← reasoning is universal, knowledge is domain-specific
  
NOT COMPOSABLE (pre-merge fails, Finding #527):
  Any two adapters merged into same weights

THINKING COMPATIBILITY:
  thinking adapters   ← MUST be trained with enable_thinking=True
  knowledge adapters  ← work with or without thinking
  format adapters     ← applied at decode time, not weight level
  mcq adapters        ← SUPPRESS thinking if trained without it (Finding #530)
```

## Ralph Loop Integration

```
Ralph claims experiment → trains adapter → evaluates → registers in registry

After training:
  1. Save adapter to adapters/<name>/
  2. Write metadata.json with training config
  3. Run eval suite (GSM8K, MMLU-Pro with+without thinking)
  4. Write evals.json
  5. Update registry.json
  6. Compare to previous best in same domain
  7. If improved: promote to "current best"
  8. Record in experiment DB: experiment evidence <id> --claim "adapter:<name> GSM8K=85%"
```

## Current Adapters (to migrate)

| Current Location | Registry Name | Status |
|---|---|---|
| exp_p1_t2_single_domain_training/adapters/math | math-gsm8k-knowledge-v0 | baseline |
| exp_p1_t2_single_domain_training/adapters/code | code-codealpaca-knowledge-v0 | baseline |
| exp_p1_t2_single_domain_training/adapters/medical | medical-medmcqa-knowledge-v0 | baseline |
| exp_p1_t2_multi_domain_5/adapters/legal | legal-mmlu-knowledge-v0 | baseline |
| exp_p1_t2_multi_domain_5/adapters/finance | finance-mmlu-knowledge-v0 | baseline |
| (to be trained) | math-s1k-reasoning-v0 | P11.A0 |
| (to be trained) | math-limo-reasoning-v0 | P11.A1 |
| (to be trained) | math-s1k-grpo-v0 | P11.B0 |
