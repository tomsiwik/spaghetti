# Final Synthesis and Follow-Up: exp_p1_t4_vllm_adapter_serving

## Final Verdict: Invalid

The original experiment for `exp_p1_t4_vllm_adapter_serving` is completely invalid. It systematically misrepresents the performance of MLX-native adapter serving through flawed methodology, artificial constraints, and direct metric hacking.

1. **Hidden Graph Compilation Overhead:** The swap latency (claimed 4.77ms) evaluates `load_weights` in a tight loop without a subsequent forward pass, entirely failing to capture MLX's graph compilation and cache invalidation overhead, which strictly occurs on the first forward pass following a parameter mutation.
2. **Invalid Throughput Metric:** The throughput metric conflates compute-bound prefill with memory-bound decode latency, heavily penalizing concise responses. Furthermore, throughput was tested using an out-of-domain prompt with a math adapter, leading to biased and erratic generation dynamics.
3. **Faked Routing:** The routing latency (claimed <1μs) was achieved by entirely bypassing the TF-IDF router and replacing it with an $O(1)$ hardcoded Python dictionary lookup.

The mathematical methodology in `MATH.md` is equally flawed, incorrectly modeling MLX execution as eager, using FLOPs instead of memory bandwidth to model Apple Silicon LoRA overhead, and assuming $O(1)$ oracle routing.

## Follow-Up Experiment Design

To rigorously test the viability of MLX-native adapter serving, we must design an experiment that strictly isolates decode throughput, properly captures graph compilation overhead, and executes the actual TF-IDF routing logic.

### 1. New Hypothesis
MLX can serve dynamically routed LoRA adapters on Apple Silicon with a true end-to-end hot-swap latency (including the first forward pass graph compilation) of < 100ms, and a decode-only throughput degradation of < 15% compared to the base model, while using a fully functional TF-IDF text router.

### 2. Math Sketch
*   **True Swap Latency ($T_{swap}$):** $T_{swap} = T_{load\_weights} + T_{recompile\_and\_forward}$. This must be measured by timing the entire operation from weight loading through the completion of the first generated token, subtracting the baseline prefill time of the base model.
*   **Decode Throughput ($T_{decode}$):** $T_{decode} = \frac{N_{generated}}{T_{end} - T_{first\_token}}$. This strictly isolates the memory-bound auto-regressive decoding phase, eliminating prompt length bias.
*   **Routing Latency ($T_{route}$):** $T_{route} = T_{tokenize}(L) + T_{sparse\_matmul}(V) + T_{argmax}$. Measured using the actual `sklearn` or MLX-native TF-IDF pipeline on the raw input text.

### 3. Kill Criteria
*   **Kill if** true end-to-end swap latency (including the first forward pass) exceeds 100ms.
*   **Kill if** decode-only throughput of the adapter drops by more than 15% compared to the base model under identical in-domain prompts.
*   **Kill if** the actual TF-IDF routing latency exceeds 5ms per request.
*   **Kill if** prefill and decode times are conflated in any throughput metric.