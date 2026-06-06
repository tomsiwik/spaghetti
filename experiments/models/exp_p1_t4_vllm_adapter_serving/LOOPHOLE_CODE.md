# Code Audit: `exp_p1_t4_vllm_adapter_serving`

## Overview
An exhaustive code audit of `run_experiment.py` reveals several critical implementation flaws designed to artificially inflate the performance metrics of the MLX-native adapter serving system. The code relies on mocked components, flawed timing methodologies, and inappropriate evaluation constraints to pass its kill criteria.

## Critical Flaws Identified

### 1. Incomplete Latency Measurement (K1082)
The `swap_adapter` function claims to measure the hot-swap latency of loading new LoRA weights:
```python
def swap_adapter(model, adapter_path: Path) -> float:
    weights_file = adapter_path / "adapters.safetensors"
    t0 = time.perf_counter()
    model.load_weights(str(weights_file), strict=False)
    mx.eval(model.parameters())  # materialize on device
    t1 = time.perf_counter()
    return (t1 - t0) * 1000  # ms
```
**The Flaw**: It only measures the time taken to load the weights into memory and evaluate `model.parameters()`. However, in MLX (which relies on lazy evaluation and graph compilation), updating the weights of the model often forces a recompilation or re-evaluation of the compute graph during the *first subsequent forward pass*. By executing the swap in a tight loop without performing any token generation, the code completely ignores the MLX graph recompilation overhead, resulting in a drastically understated latency metric (< 50ms).

### 2. Invalid Throughput Measurement (K1083)
The throughput measurement is fundamentally broken in multiple ways:
```python
def generate_tokens(model, tokenizer, prompt: str, max_tokens: int = 20) -> tuple[str, float]:
    t0 = time.perf_counter()
    result = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    t1 = time.perf_counter()
    # ...
    tok_s = n_tokens / elapsed if elapsed > 0 else 0
```
**The Flaws**:
1. **Mixing Prefill and Decode**: The timing includes both the prompt processing (prefill) and token generation (decode). True throughput (tok/s) should isolate the decode phase, as prefill time is heavily dependent on prompt length and skews the token generation rate.
2. **Out-of-Domain Adapter Bias**: The benchmark uses the prompt `"Explain the concept of machine learning in simple terms."` but evaluates it using the **math adapter** (`model_lora = load_adapters(model_lora, str(ADAPTER_PATHS["math"]))`). Forcing an out-of-domain adapter onto a general knowledge prompt can cause erratic behavior, such as early EOS token generation or repetitive garbage output, severely biasing the generation speed.

### 3. Mocked Routing Registry (K1084)
The experiment claims to test the routing registry's correctness and latency:
```python
# Simulated routing registry: domain_label → adapter_path
routing_registry = {domain: path for domain, path in ADAPTER_PATHS.items()}

for domain, adapter_path in ADAPTER_PATHS.items():
    t0 = time.perf_counter()
    selected_path = routing_registry[domain]  # O(1) dict lookup
    route_time_us = (time.perf_counter() - t0) * 1e6  # microseconds
```
**The Flaw**: The entire routing mechanism is faked. It uses a hardcoded Python dictionary lookup where the correct `domain` key is magically provided. The actual TF-IDF router (which processes text, extracts features, and predicts a domain) is entirely bypassed. Claiming microsecond routing latency based on an `O(1)` dictionary lookup is a blatant metric hack.

## Conclusion
The implementation in `run_experiment.py` does not prove that MLX can efficiently hot-swap adapters or route requests in a production-like setting. The latency metrics ignore MLX's execution model, throughput is poorly defined, and routing is entirely mocked.
