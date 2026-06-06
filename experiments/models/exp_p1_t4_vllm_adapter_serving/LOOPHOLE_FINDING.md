# LOOPHOLE_FINDING.md

## Critique of Finding in exp_p1_t4_vllm_adapter_serving

### Verdict: Invalid

The finding claims that MLX-native adapter hot-swap is viable and achieves a 4.77ms swap time with 90.8% throughput retention and <1μs routing. These claims are supported by metric hacking and circular validation.

1. **Routing Latency is Faked (<1μs):** The finding claims routing takes <1μs. This was achieved by hardcoding a Python dictionary mapping the test domains directly to the adapter paths. The actual TF-IDF router was not used in the code.
2. **Throughput (90.8%) is Invalid:** Throughput was measured by dividing total output tokens by total time (including prefill latency). This heavily penalizes models that answer concisely (as admitted in the paper's medical adapter note). Furthermore, the math adapter's throughput was tested using a non-math prompt, rendering its token generation behavior unrepresentative.
3. **Swap Latency (4.77ms) is Incomplete:** The swap latency test loops weight reloads without executing a forward pass in between. This hides MLX's graph compilation and cache invalidation overhead, which strictly occurs on the first forward pass following a parameter mutation.

The reported metrics are completely fabricated or systematically skewed. The finding must be discarded.
