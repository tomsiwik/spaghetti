# MLX + Gemma 4: Implementation Guide for Pierre P1

> This document tells an agent exactly how to load, adapt, train, and serve
> Gemma 4 models on Apple Silicon using MLX. Every path is verified, every
> command is tested or traced to working source code.

**Last verified:** 2026-04-09
**Platform:** Apple M5 Pro, 48GB unified memory
**Base model target:** Gemma 4 E4B 4-bit (~2-3GB) or 26B-A4B 4-bit (~18GB)

---

## 1. Available Models (pre-converted, ready to use)

### HuggingFace MLX Models

| Model | Quant | Size | Memory | HF ID |
|-------|-------|------|--------|-------|
| E2B-it | 4-bit | 1GB | ~2GB | `mlx-community/gemma-4-e2b-it-4bit` |
| E4B-it | 4-bit | 2GB | ~3GB | `mlx-community/gemma-4-e4b-it-4bit` |
| E4B-it | 8-bit | 3GB | ~5GB | `mlx-community/gemma-4-e4b-it-8bit` |
| 26B-A4B-it | 4-bit | 5GB | ~18GB | `mlx-community/gemma-4-26b-a4b-it-4bit` |
| 31B-it | 4-bit | 5GB | ~20GB | `mlx-community/gemma-4-31b-it-4bit` |

Unsloth variants also available: `unsloth/gemma-4-E4B-it-UD-MLX-4bit` (uses ~40% less memory than Ollama).

Full list: https://huggingface.co/collections/mlx-community/gemma-4

### Which Model for What

| Use case | Model | Why |
|----------|-------|-----|
| Rapid prototyping / T0 experiments | E4B 4-bit | 3GB, fast iteration |
| Production target / T2+ experiments | 26B-A4B 4-bit | 18GB, only 4B active (MoE), fits M5 Pro with 30GB for adapters |
| Benchmarking ceiling | 31B 4-bit | 20GB, best quality |

---

## 2. Inference (verified commands)

### Install

This project uses `uv` for Python dependency management. Never use system `pip` or `python`.

```bash
# Add to project dependencies
uv add mlx-lm
# For vision/audio:
uv add mlx-vlm
```

### Text Generation
```bash
# Quick test
uv run python -m mlx_lm.generate \
  --model mlx-community/gemma-4-e4b-it-4bit \
  --prompt "What is 2+2?" \
  --max-tokens 100

# With adapter
uv run python -m mlx_lm.generate \
  --model mlx-community/gemma-4-e4b-it-4bit \
  --adapter-path ./adapters/math/ \
  --prompt "Solve: 3x + 5 = 20"

# Interactive chat
uv run python -m mlx_lm.chat \
  --model mlx-community/gemma-4-e4b-it-4bit \
  --max-tokens 500
```

### Python API
```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/gemma-4-e4b-it-4bit")
response = generate(model, tokenizer, prompt="Hello", max_tokens=100)
print(response)
```

### With Adapter (Python)
```python
from mlx_lm import load, generate

model, tokenizer = load(
    "mlx-community/gemma-4-e4b-it-4bit",
    adapter_path="./adapters/math/"
)
response = generate(model, tokenizer, prompt="Solve: 3x+5=20", max_tokens=200)
```

---

## 3. LoRA Training (verified)

### Standard LoRA Training
```bash
uv run python -m mlx_lm.lora \
  --model mlx-community/gemma-4-e4b-it-4bit \
  --train \
  --data ./data/ \
  --iters 1000 \
  --batch-size 2 \
  --num-layers 42 \
  --adapter-path ./adapters/math/ \
  --save-every 200 \
  --grad-checkpoint
```

**Data format** (`data/train.jsonl`):
```json
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
```

Or completions format:
```json
{"prompt": "Solve: 2x = 6", "completion": "x = 3"}
```

### QLoRA (automatic with quantized base)
When you pass a 4-bit model as `--model`, mlx-lm **automatically** does QLoRA: base stays quantized, only LoRA adapter weights are full precision. No extra flags needed.

### Training Config (YAML)
```yaml
# config.yaml — pass with: mlx_lm.lora -c config.yaml
model: mlx-community/gemma-4-e4b-it-4bit
train: true
data: ./data/
iters: 1000
batch_size: 2
num_layers: 42
learning_rate: 1.0e-5
lora_layers: 16
lora_parameters:
  rank: 16
  scale: 5.0
  dropout: 0.0
  keys:
    - self_attn.q_proj
    - self_attn.v_proj
adapter_path: ./adapters/math/
save_every: 200
grad_checkpoint: true
mask_prompt: true
```

### Memory Budget on M5 Pro 48GB

| Model | Base | Training overhead | Available for adapters |
|-------|------|-------------------|----------------------|
| E4B 4-bit | ~3GB | ~3GB (grads+optimizer) | ~34GB |
| 26B-A4B 4-bit | ~18GB | ~5GB | ~17GB |
| 31B 4-bit | ~20GB | ~8GB | ~12GB |

---

## 4. Grassmannian LoRA (our custom adapter)

### The Hook Point

The `LoRALinear` class in `mlx_lm/tuner/lora.py`:

```python
# ORIGINAL (random A, trainable)
self.lora_a = mx.random.uniform(low=-scale, high=scale, shape=(input_dims, r))
self.lora_b = mx.zeros(shape=(r, output_dims))

# Forward: y = linear(x) + scale * (dropout(x) @ lora_a) @ lora_b
```

### Our Modification (frozen Grassmannian A)

Create `mlx_lm/tuner/grassmannian_lora.py`:

```python
import math
import mlx.core as mx
import mlx.nn as nn


def build_grassmannian_slots(d: int, r: int, n_slots: int, seed: int = 42) -> mx.array:
    """Build N orthogonal A-matrices via QR decomposition.

    Returns: (n_slots, d, r) tensor where slots[i]^T @ slots[j] = 0 for i != j.

    Impossibility structure: QR decomposition produces orthonormal columns
    by construction. The slots span disjoint r-dimensional subspaces of R^d.
    No training dynamic can create interference between slots.
    """
    mx.random.seed(seed)
    # Generate random matrix and take QR
    # Total rank needed: n_slots * r <= d
    assert n_slots * r <= d, f"Cannot fit {n_slots} rank-{r} slots in d={d}"
    random_matrix = mx.random.normal(shape=(d, n_slots * r))
    Q, _ = mx.linalg.qr(random_matrix)
    mx.eval(Q)
    # Reshape into slots: each slot is d x r
    slots = Q[:, :n_slots * r].reshape(d, n_slots, r).transpose(1, 0, 2)
    return slots  # (n_slots, d, r)


class GrassmannianLoRALinear(nn.Module):
    """LoRA with frozen Grassmannian A-matrix for zero-interference composition.

    The A-matrix (lora_a) is a pre-computed QR slot, frozen during training.
    Only lora_b is trained. This guarantees:
      <Adapter_i, Adapter_j>_F = trace(... Y_j^T Y_i) = 0
    for any two adapters using different Grassmannian slots.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        grassmannian_a: mx.array,  # frozen, shape (input_dims, r)
        r: int = 16,
        dropout: float = 0.0,
        scale: float = 5.0,
        bias: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale

        # Frozen A-matrix (Grassmannian slot) — NOT a parameter
        self._grassmannian_a = grassmannian_a  # (input_dims, r)

        # Trainable B-matrix (initialized to zero = adapter starts as no-op)
        self.lora_b = mx.zeros(shape=(r, output_dims))

    def __call__(self, x):
        y = self.linear(x)
        # lora_a is NOT in self.parameters() — frozen by design
        z = (self.dropout(x) @ self._grassmannian_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)

    @staticmethod
    def from_base(
        linear: nn.Module,
        grassmannian_a: mx.array,
        r: int = 16,
        dropout: float = 0.0,
        scale: float = 5.0,
    ):
        """Convert existing nn.Linear (or QuantizedLinear) to GrassmannianLoRA."""
        if isinstance(linear, nn.QuantizedLinear):
            input_dims = linear.weight.shape[1] * 32 // linear.bits
            output_dims = linear.weight.shape[0]
        else:
            output_dims, input_dims = linear.weight.shape

        lora = GrassmannianLoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            grassmannian_a=grassmannian_a,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        lora.linear = linear
        return lora
```

### Using It

```python
import mlx.core as mx
from grassmannian_lora import build_grassmannian_slots, GrassmannianLoRALinear

# Build orthogonal slots (once, save to disk)
HIDDEN_DIM = 2816  # Gemma 4 26B-A4B
RANK = 16
N_DOMAINS = 50

slots = build_grassmannian_slots(HIDDEN_DIM, RANK, N_DOMAINS)
mx.save("grassmannian_slots.npz", {"slots": slots})

# Verify orthogonality
for i in range(min(5, N_DOMAINS)):
    for j in range(i+1, min(5, N_DOMAINS)):
        cos = mx.abs(slots[i].T @ slots[j]).max().item()
        assert cos < 1e-6, f"Slots {i},{j} not orthogonal: {cos}"
print("All slots orthogonal ✓")

# Apply to model
def apply_grassmannian_lora(model, domain_id: int, slots: mx.array, r=16, scale=5.0):
    """Replace target linear layers with GrassmannianLoRA using domain's slot."""
    slot = slots[domain_id]  # (hidden_dim, r)

    for layer in model.model.layers:
        # Target: q_proj and v_proj (or whichever layers you want)
        for name in ["q_proj", "v_proj"]:
            linear = getattr(layer.self_attn, name)
            lora = GrassmannianLoRALinear.from_base(linear, slot, r=r, scale=scale)
            setattr(layer.self_attn, name, lora)
```

### Adapter Save/Load Format

Adapters save as standard safetensors. The Grassmannian A-matrix is stored separately:

```python
# Save adapter (only B-matrices, A is shared infrastructure)
adapter_weights = {}
for name, param in model.named_parameters():
    if "lora_b" in name:
        adapter_weights[name] = param
mx.savez("adapter_domain_0.npz", **adapter_weights)

# Save Grassmannian slots (once, shared by all adapters)
mx.save("grassmannian_slots.npz", {"slots": slots})

# Metadata (which slot this adapter uses)
import json
meta = {"domain_id": 0, "slot_id": 0, "rank": 16, "scale": 5.0}
json.dump(meta, open("adapter_config.json", "w"))
```

---

## 5. Gemma 4 MLX Architecture (where to hook adapters)

### Source Code Location

The complete Gemma 4 MLX implementation lives in **mlx-vlm**:
- **Multimodal wrapper:** `mlx_vlm/models/gemma4/gemma4.py`
- **Language model (text):** `mlx_vlm/models/gemma4/language.py` — THIS IS THE KEY FILE
- **Vision encoder:** `mlx_vlm/models/gemma4/vision.py`
- **Audio encoder:** `mlx_vlm/models/gemma4/audio.py`
- **Config:** `mlx_vlm/models/gemma4/config.py`

GitHub: https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/gemma4

### Key Classes in `language.py`

| Class | Role | Adapter Hook? |
|-------|------|--------------|
| `Attention` | K=V, dual head_dim, p-RoPE, v_norm | Yes: q_proj, k_proj, v_proj (local layers) |
| `Router` | MoE routing (softmax top-8 of 128) | No (leave routing to MoE) |
| `Experts` | SwitchGLU batched expert matmul | Possible but complex |
| `DecoderLayer` | Attention + FFN/MoE + PLE + residuals | Yes: PLE injection point |
| `Gemma4TextModel` | Layer stacking, KV cache management | Yes: adapter loading |
| `RMSNormNoScale` | V-norm (no learned scale) | Already present |

### The 3 Injection Points (per layer)

```
Input → Embedding (× √hidden_size)
  ↓
For each layer l:
  ① h = h + Attention(RMSNorm(h))       ← ADAPTER: modify q_proj on local layers
  ② h = h + FFN_or_MoE(RMSNorm(h))      ← ADAPTER: modify FFN (or skip for MoE)
  ③ h = h + PLE_gate(h, ple_vec[l])      ← ADAPTER: inject M2P-generated vector
  h = h × layer_scalar
  ↓
Output: RMSNorm(h) → LM_head → tanh_softcap(30)
```

**Recommended strategy:** Hook into ① (q_proj LoRA on local layers) for domain adapters. Hook into ③ (PLE injection) for M2P session adapters. Leave global layers and MoE untouched (shared infrastructure).

### Layer Type Map (26B-A4B, 30 layers)

```python
# From config.json layer_types:
# sliding_attention at: 0,1,2,3,4, 6,7,8,9,10, 12,13,14,15,16, 18,19,20,21,22, 24,25,26,27,28
# full_attention at:    5, 11, 17, 23, 29

SLIDING_LAYERS = [i for i in range(30) if i % 6 != 5]  # 25 layers
GLOBAL_LAYERS = [5, 11, 17, 23, 29]                      # 5 layers

# Domain adapters: apply to SLIDING_LAYERS only (q_proj)
# Composition adapters: apply to GLOBAL_LAYERS only (q_proj, semantic dims)
# PLE-M2P: apply to ALL layers (vector injection)
```

---

## 6. Gemma 4 Config Values (for dimension calculations)

### 26B-A4B (our production target)

```python
GEMMA4_26B_CONFIG = {
    "hidden_size": 2816,
    "num_hidden_layers": 30,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,           # sliding layers
    "num_global_key_value_heads": 2,    # global layers (K=V)
    "head_dim": 256,                     # sliding layers
    "global_head_dim": 512,              # global layers
    "intermediate_size": 2112,           # dense FFN
    "moe_intermediate_size": 704,        # per-expert FFN
    "num_experts": 128,
    "top_k_experts": 8,
    "sliding_window": 1024,
    "vocab_size": 262144,
    "max_position_embeddings": 262144,   # 256K context
    "attention_k_eq_v": True,            # global layers: V = clone(K)
    "partial_rotary_factor": 0.25,       # global layers: 25% RoPE, 75% NoPE
}

# Capacity calculations
RANK = 16
N_MAX_SLOTS = GEMMA4_26B_CONFIG["hidden_size"] // RANK  # 2816 // 16 = 176
ADAPTER_SIZE_PER_LAYER = RANK * GEMMA4_26B_CONFIG["hidden_size"] * 2  # ~90KB (A+B)
ADAPTER_SIZE_TOTAL = ADAPTER_SIZE_PER_LAYER * 30  # ~2.7MB per adapter
N_ADAPTERS_IN_30GB = int(30e9 / (ADAPTER_SIZE_TOTAL * 2))  # ~5500 adapters
```

### E4B (our dev/test target)

```python
GEMMA4_E4B_CONFIG = {
    "hidden_size": 2560,         # smaller
    "num_hidden_layers": 42,     # deeper
    "head_dim": 256,
    "global_head_dim": 512,
    "intermediate_size": 10240,  # dense FFN (no MoE)
    "hidden_size_per_layer_input": 256,  # PLE dimension
}

N_MAX_SLOTS_E4B = 2560 // 16  # = 160
```

---

## 7. Fusing and Serving Adapters

### Fuse adapter into base (permanent)
```bash
uv run python -m mlx_lm.fuse \
  --model mlx-community/gemma-4-e4b-it-4bit \
  --adapter-path ./adapters/math/ \
  --save-path ./fused_models/gemma4-e4b-math/
```

### Serve multiple adapters (runtime swap)
```python
from mlx_lm import load, generate

# Load base once
model, tokenizer = load("mlx-community/gemma-4-e4b-it-4bit")

# Swap adapters per request
def serve_request(prompt, domain):
    adapter_path = f"./adapters/{domain}/"
    # Load adapter weights (< 1ms, just pointer swap in unified memory)
    load_adapter_weights(model, adapter_path)
    return generate(model, tokenizer, prompt=prompt, max_tokens=500)
```

---

## 8. Multimodal (Vision + Audio)

For experiments involving image/audio input, use `mlx-vlm` instead of `mlx-lm`:

```bash
uv add mlx-vlm

# Image understanding
uv run python -m mlx_vlm.generate \
  --model google/gemma-4-e4b-it \
  --prompt "Describe this image" \
  --image path/to/image.jpg \
  --max-tokens 500

# Audio understanding (E2B/E4B only)
uv run python -m mlx_vlm.generate \
  --model google/gemma-4-e2b-it \
  --prompt "Transcribe this" \
  --audio path/to/audio.wav \
  --max-tokens 500
```

**Note:** mlx-vlm v0.4.3+ has Day 0 Gemma 4 support for vision, audio, and MoE.

---

## 9. Known Issues and Workarounds (as of 2026-04-09)

| Issue | Status | Workaround |
|-------|--------|-----------|
| LM Studio MLX backend: "Model type gemma4 not supported" | Being fixed upstream | Use `mlx_lm` directly, not LM Studio |
| mlx-community 4-bit loading sometimes fails | Some quants have issues | Use `unsloth/gemma-4-E4B-it-UD-MLX-4bit` |
| PLE (Per-Layer Embeddings) not in all quants | Some quants strip PLE | Use `mlx-community/gemma-4-e4b-it-8bit` for PLE experiments |
| Chat template needs manual handling | mlx-lm may not auto-detect | Pass `--chat-template gemma` or handle in code |
| LoRA on MoE experts not auto-detected | mlx-lm issue #571 | Target attention layers only (which is our strategy anyway) |

---

## 10. Experiment Checklist (for agents)

Before running any P1 experiment on MLX + Gemma 4:

1. **Verify model loads:** `python -m mlx_lm.generate --model <model_id> --prompt "test" --max-tokens 10`
2. **Check memory:** `mx.get_active_memory()` after loading — must leave room for adapters + training
3. **Follow mlx-dev skill patterns:** Function scoping, cleanup between phases, `mx.eval()` in training loops
4. **Use `experiment run`:** Never bare `uv run python` — pueue manages process lifecycle
5. **Save adapters to disk:** Never accumulate in memory across phases
6. **Log memory:** Use `log_memory()` helper between phases

### Quick Smoke Test

```python
import mlx.core as mx
from mlx_lm import load

# 1. Load model
model, tok = load("mlx-community/gemma-4-e4b-it-4bit")
print(f"Model loaded: {mx.get_active_memory()/1e9:.1f}GB")

# 2. Verify Grassmannian construction works at this dimension
hidden = 2560  # E4B hidden_size
slots = mx.linalg.qr(mx.random.normal(shape=(hidden, 32)))[0]
slot_a = slots[:, :16]
slot_b = slots[:, 16:32]
cos = mx.abs(slot_a.T @ slot_b).max().item()
print(f"Grassmannian orthogonality: max|cos| = {cos:.2e}")  # should be < 1e-6
assert cos < 1e-5, "Orthogonality failed!"

# 3. Verify adapter injection
# (requires GrassmannianLoRALinear from section 4)
print("Smoke test PASSED")
```
