# MATH.md — T4.3: MLX-Native Adapter Serving with Runtime Hot-Swap

## V2 Audit Section (2026-04-18)

Status: **KILLED** (precondition-probe + four operationalisation flaws).

V1 KC thresholds below are preserved byte-for-byte. V2 does not modify K1081–
K1084 or any theorem statement; it identifies four distinct mis-operationalisations
of the theorems plus a missing-artefact precondition.

### V2 kill causes

- **C1 — precondition failure.** `EXPECTED_ADAPTERS` (math, code, medical, legal,
  finance) maps to 5 dirs; 0/5 hold `adapters.safetensors` as of 2026-04-18.
  T2.1 upstream `status=killed` (2026-04-18, MCQ metric-swap + format-artefact).
  T2.6 adapter weights lost. `load_adapters(model, path)` raises FileNotFoundError
  before any KC can be measured. V1 `results.json` also absent on disk, so the
  "supported" verdict is unverifiable even on provenance.

- **C2 — Theorem 1 omits MLX graph-recompile (mem-antipattern-011 specialisation).**
  V1 operationalises `T_swap` as
  `{t0; model.load_weights(...); mx.eval(model.parameters()); t1}`.
  MLX recompiles the forward compute graph on the **first forward pass** after
  a parameter mutation; `mx.eval(parameters)` only materialises the new tensors
  on-device, not the traced graph. Theorem 1 `T_swap ≤ S_adapter / B_mem + T_eval`
  omits the recompile term. A correct operationalisation defines swap latency as
  incremental time-to-first-token after swap minus baseline prefill time of the
  same prompt. V1's Phase 2 loops 20 back-to-back swaps with zero forward passes
  between — the recompile cost never enters the clock.

- **C3 — Theorem 2 uses FLOPs model for memory-bound decode + OOD prompt bias.**
  Theorem 2's `α = 2r/d_model ≈ 0.47%` is a FLOPs ratio. Apple Silicon decode is
  memory-bandwidth-bound (weight matrix reads dominate per-token cost), not
  compute-bound, so the arithmetic predicts 99.5% but the measurement is governed
  by bytes moved per token (re-reading X through base path and LoRA path). Beyond
  the FLOPs/bandwidth model mismatch, V1's `generate_tokens` conflates prefill
  (compute-bound) and decode (memory-bound) into one tok/s number, and evaluates
  the math adapter on `"Explain the concept of machine learning in simple terms."` —
  a non-math prompt. Out-of-domain generation commonly early-EOSes or shifts
  the token distribution (V1's own Phase 1 flags the 3.7 tok/s medical adapter
  outlier as "model answered briefly, denominator small" — same failure mode
  applied to K1083 by construction).

- **C4 — Theorem 3 operationalised as identity-dict (mem-antipattern-002).**
  V1 code:
  `routing_registry = {d: p for d, p in ADAPTER_PATHS.items()}; selected = routing_registry[domain]`
  where `domain` is the iteration key. `selected == adapter_path` is `dict[k] == dict[k]` —
  a set-theoretic identity, not routing. K1084 requires a router that takes raw
  prompt text and predicts a domain label; T4.1's pipeline (tokenise → sparse
  matmul → argmax) was never invoked. V1's <1µs latency is Python dict-hash
  microbench cost, not TF-IDF routing cost.

### V2 prediction vs measurement

| Criterion | V1 Prediction | V2 Measurement (probe) | V2 Result |
|-----------|--------------|------------------------|-----------|
| K1081: loads+generates | 5/5 valid | 0/5 upstream `.safetensors` on disk | **FAIL** |
| K1082: swap p99 < 50ms | ~1.5ms (I/O bound) | Wrong object: omits MLX graph-recompile | **FAIL** |
| K1083: throughput ≥ 80% | ~99.5% (FLOPs) | Prefill/decode conflation + OOD prompt | **FAIL** |
| K1084: routing correct | 5/5, <1µs | Identity-dict: `dict[k] == dict[k]` by construction | **FAIL** |

### V2 probe characteristics

- No model load. Pure `os.path` inspection + upstream-verdict parse + trivial
  dict-lookup microbench (to reproduce and expose V1's K1084 measurement as
  domain-agnostic).
- Runtime < 1s. Side-effect-free.
- Preserves V1 numbers in `results._v1_numbers_for_reference` for provenance.

### V3 blockers (do not auto-spawn)

- T2.1 rebuild with MedQA USMLE 5-choice (DB KC #1030), `max_tokens >= 512`,
  persisted `.safetensors`, `adapters/code/` created.
- T2.6 adapter weights recovered or retrained on disk.
- `swap_adapter` rewrite: measure `t_first_token(after_swap) - t_first_token(baseline)`,
  not `load_weights + mx.eval` alone.
- `generate_tokens` rewrite: return `(text, decode_tok_s)` where
  `decode_tok_s = N_decoded / (t_end - t_first_token)`.
- K1083 must use in-domain prompts per adapter (math prompt on math adapter, etc.).
- K1084 must invoke T4.1's actual TF-IDF router on raw prompt text — not
  `ADAPTER_PATHS[domain]`. Latency budget should time tokenise + sparse matmul
  + argmax, per T4.1 math.

---

## V1 Setting (2026-04-17, preserved — thresholds unchanged)

## Setup

We have:
- Base model: Gemma 4 E4B (4-bit quantized, `mlx-community/gemma-4-e4b-it-4bit`)
- N = 5 LoRA adapters: math, code, medical, legal, finance
- Each adapter: rank r = 6, applied to `self_attn.q_proj` in all L = 35+ layers
- Adapter format: mlx_lm standard (adapters.safetensors + adapter_config.json)
- Platform: Apple M5 Pro, 48GB, MLX (not vLLM/CUDA)

Kill criteria:
- K1081: MLX loads Gemma 4 E4B + 5 LoRA adapters, generates valid output
- K1082: Adapter swap between requests < 50ms overhead
- K1083: Throughput with active adapter >= 80% of base throughput
- K1084: Correct adapter selected per request via routing registry

---

## Theorem 1: Adapter Swap Cost Is Bounded by Adapter File I/O

**Theorem:** In an MLX-native serving setup where LoRA structure is initialized once,
swapping adapter k → adapter j requires only reloading A, B matrices — not the base model.
The swap time is bounded by: T_swap ≤ S_adapter / B_mem + T_eval

where S_adapter is the adapter file size, B_mem is memory bandwidth, T_eval is evaluation time.

**Proof:**

After the first adapter load (via `load_adapters(model, adapter_path)`), the model's
linear layers are replaced with LoRALinear objects. Each LoRALinear stores:
- self.linear: the frozen base weights (4-bit quantized)
- self.lora_a: the A matrix (d_in × rank, BF16)
- self.lora_b: the B matrix (rank × d_out, BF16)

For Gemma 4 E4B with q_proj adaptation on all L layers:
```
L = 35 attention layers (Gemma 4 E4B has 34 layers + 1 embedding — checked empirically)
d_in = d_out = 2560 (Gemma 4 E4B hidden dim)
r = 6 (rank)
```

Adapter file size per domain:
```
S = L × 2 × (d_in × r + r × d_out) × sizeof(BF16)
  = 35 × 2 × (2560 × 6 + 6 × 2560) × 2
  = 35 × 2 × 30720 × 2
  = 4,300,800 bytes ≈ 4.1 MB
```

M5 Pro unified memory bandwidth: B_mem ≈ 273 GB/s

Theoretical minimum swap time:
```
T_io = 4.1 MB / 273 GB/s ≈ 0.015 ms
```

MLX additional overhead (Python call + mx.eval):
- model.load_weights() parses safetensors header + schedules copies: ~1 ms
- mx.eval() to materialize on GPU: ~0.5 ms
- Total predicted: T_swap ≈ 1.5 ms << 50 ms

**Prediction:** K1082 will PASS with ≥33× margin (50ms / 1.5ms).

**QED**

---

## Theorem 2: LoRA Forward Pass Overhead Is Negligible (< 1%)

**Theorem:** The computational overhead of rank-r LoRA on q_proj relative to base model
is α = 2r / d_model ≈ 0.47%, yielding throughput ratio T_lora / T_base ≥ 99.5% >> 80%.

**Proof:**

Base forward pass FLOPs for q_proj (one token):
```
F_base = 2 × d_in × d_out = 2 × 2560² = 13,107,200 FLOPs/layer
```

LoRA additional FLOPs (x @ A @ B, rank decomposition):
```
F_lora = 2 × d_in × r + 2 × r × d_out = 4 × d_in × r
       = 4 × 2560 × 6 = 61,440 FLOPs/layer
```

Overhead fraction per layer:
```
α = F_lora / F_base = 61,440 / 13,107,200 = 0.469%
```

Since LoRA is applied to q_proj only (L layers), and the full model has many more
operations (v_proj, o_proj, MLP, etc.), the actual overhead is:
```
α_model = α × L × F_qproj / F_model_total << 0.469%
```

Even using the upper bound α = 0.47%, predicted throughput ratio:
```
T_ratio = 1 / (1 + α) ≈ 99.53% >> 80%
```

**Prediction:** K1083 will PASS with measured throughput ratio ≥ 98%.

**QED**

---

## Theorem 3: Routing via Registry Is O(1)

**Theorem:** Domain-to-adapter routing via Python dict lookup is O(1) in expectation
(hash table), with latency < 1 μs — negligible relative to generation time.

**Proof:**

Python dict uses open-addressing hash table with O(1) average lookup.
For N = 5 domains with string keys (domain names), expected comparisons ≤ 2.
Empirical overhead: < 0.001 ms (1 μs).

This satisfies K1084 trivially: domain label → adapter path lookup is deterministic
and zero-LLM-parameter by construction.

**QED**

---

## Prediction-vs-Measurement Summary (to be filled after experiment)

| Criterion | Prediction | Measurement | Pass? |
|-----------|-----------|-------------|-------|
| K1081: Loads + generates | Valid output × 5 adapters | TBD | TBD |
| K1082: Swap < 50ms | ~1.5ms (33× margin) | TBD | TBD |
| K1083: Throughput ≥ 80% | ~99.5% | TBD | TBD |
| K1084: Routing correct | 5/5 domain correct | TBD | TBD |

---

## Connection to Pierre Architecture

This experiment validates the serving layer of the Pierre stack:
1. **T3.4** proved adapter composition interference is zero under exclusive routing
2. **T4.1** proved TF-IDF routing achieves 96.6% accuracy at N=5
3. **T4.3 (this)** proves adapter hot-swap overhead is < 50ms — the serving loop is viable

Together: route → swap → generate is a complete, low-overhead serving pipeline.
The next step (T4.6) will measure end-to-end latency: route + swap + first token.

## References
- mlx_lm.tuner.utils.load_adapters — confirms swap = load_weights() call on LoRALinear
- Finding #431 (T4.1): TF-IDF routing 96.6% N=5, 86.1% N=25
- Finding #428 (T3.4): N=25 Grassmannian composition verified
- Gemma 4 architecture: d_model=2560, 34 transformer layers, q_proj adapted
