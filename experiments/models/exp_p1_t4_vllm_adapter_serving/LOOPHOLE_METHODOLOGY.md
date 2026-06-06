# Methodology Audit: exp_p1_t4_vllm_adapter_serving

## Verdict: Invalid

The mathematical frameworks and methodology defined in `MATH.md` are fundamentally flawed, relying on theoretical models that completely misrepresent the execution environment and the system architecture. The previous adversarial review failed to identify these foundational errors, incorrectly rubber-stamping the experiment.

### 1. Theorem 1 (Latency): The "Eager Execution" Fallacy
**The Flaw:** Theorem 1 defines swap time as strictly bounded by file I/O ($S_{adapter} / B_{mem} + T_{eval}$) and models the swap independently of the subsequent forward pass.
**The Reality:** MLX relies on lazy evaluation and JIT graph compilation. When parameters (like LoRA weights) are mutated, the framework must recompile or re-evaluate the compute graph upon the *next* execution trigger (the forward pass). By defining swap latency solely as the time to execute `load_weights` and `mx.eval(parameters)`, the methodology artificially excludes the graph recompilation overhead. The theoretical model must include the compilation cost of the first mutated forward pass; failing to do so makes the theoretical bound useless.

### 2. Theorem 2 (Throughput): The "Compute-Bound" Fallacy
**The Flaw:** Theorem 2 models the overhead of the LoRA forward pass based purely on arithmetic FLOPs ($\alpha = 2r / d_{model} \approx 0.47\%$).
**The Reality:** Apple Silicon (M-series) is overwhelmingly memory-bandwidth bound during auto-regressive decoding, not compute-bound. The true theoretical cost of LoRA in this regime is determined by the additional memory traffic (loading the LoRA $A$ and $B$ matrices, and potentially re-reading the activation $X$). A FLOP-based proof for a memory-bound operation is theoretically invalid.
Furthermore, the methodology for measuring throughput conflates prefill (compute-bound) and decode (memory-bound) phases into a single metric. This invalidates any theoretical comparison to steady-state throughput.

### 3. Theorem 3 (Routing): The "Oracle" Fallacy
**The Flaw:** Theorem 3 models routing as an $O(1)$ Python dictionary lookup, claiming latency $< 1 \mu s$.
**The Reality:** This completely redefines the problem. The system requires a TF-IDF router to map raw input text to a domain. The theoretical complexity of TF-IDF routing involves tokenization ($O(L_{prompt})$), sparse matrix multiplication ($O(N \times V_{active})$), and argmax. By substituting an $O(1)$ oracle (a hardcoded dictionary mapping domain strings to paths) for the actual mathematical operations of the router, the methodology fundamentally cheats the latency bounds.

## Conclusion
The mathematical proofs in `MATH.md` are constructed to avoid modeling the actual costs of the system (MLX graph compilation, memory bandwidth, and TF-IDF computation). The methodology is structurally invalid, rendering all empirical measurements meaningless.