# LOOPHOLE_FOLLOWUP.md — Synthesizer Final Verdict

**Target:** `exp_p1_t4_adapter_format_compat`
**Final Verdict:** INVALID

## Synthesis of Flaws

The `exp_p1_t4_adapter_format_compat` experiment relies on severe metric hacking, logical fallacies, and empirical falsification to claim format compatibility and structural guarantees:

1.  **Fake Validation (Metric Hacking):** The code explicitly skips the `peft` library validation if it isn't installed, silently passing the test based on a naive key-subset check.
2.  **Logical Fallacy in Proofs:** Theorem 3 claims that PEFT compatibility implies vLLM and Unsloth compatibility because vLLM format is a subset of PEFT format. This is an elementary set theory error ($A \subseteq B \not\implies x \in B \implies x \in A$). In reality, vLLM requires fused projections (e.g., `qkv_proj`) not guaranteed by standard PEFT.
3.  **Superficial Compatibility Checks:** The checks for vLLM and Unsloth are superficial string-matching tests for keys like `.lora_A.weight`, completely ignoring runtime mechanics, fused QKV projection requirements, and actual runtime loadability.
4.  **Falsified Grassmannian Guarantees:** Theorem 2 claims Grassmannian property preservation, but the empirical deviation is 0.579. The script writes `"property": "orthonormal_rows"` to the adapter metadata regardless of the actual mathematical state of the tensor, outright lying to downstream systems.

The previous adversarial review (`REVIEW-adversarial.md`) was catastrophically lenient and completely missed these fatal flaws, specifically the set theory error and the implications of the Grassmannian drift.

## Follow-up Experiment Design

To rigorously test format compatibility and structural integrity, a new experiment must be designed that empirically tests runtime loading and mathematically enforces (or correctly measures) structural constraints.

### Hypothesis
An MLX-trained LoRA adapter can be natively loaded by CPU-based `peft`/`transformers` and vLLM without structural mismatches, provided that target-specific structural transformations (e.g., QKV fusion) are explicitly mapped and applied during export.

### Math Sketch
Let $F_{mlx}$ be the MLX adapter format and $F_{target}$ be the target runtime format.
Compatibility is not an identity map $F_{mlx} \approx F_{target}$, but a transformation $T: F_{mlx} \to F_{target}$.
For vLLM, $T$ must include $f_{fuse}(q\_proj, k\_proj, v\_proj) \to qkv\_proj$.
For Grassmannian integrity, let $A$ be the projection matrix. We define a strict threshold $\epsilon < 1e-4$. The metadata property `"orthonormal_rows"` is valid if and only if $\|A A^T - I\|_\infty < \epsilon$.

### Methodology
1.  **Empirical Runtime Validation:** The test must instantiate a CPU-based `transformers` model and load the adapter using `PeftModel.from_pretrained()`. If it crashes, it fails.
2.  **vLLM Fused Projection Test:** The test must explicitly map MLX individual Q, K, V projections into a fused `qkv_proj` tensor and validate the expected byte shapes against a mock vLLM state dictionary.
3.  **Strict Error Bounds:** The test must explicitly compute $\|A A^T - I\|_\infty$ and fail if the deviation exceeds $\epsilon$. The metadata tag must only be written if this assertion passes.
4.  **No Bypasses:** If `peft` or `transformers` is missing, the script must hard-fail (`sys.exit(1)`), not silently pass.

### Kill Criteria
-   **K1:** The exported adapter fails to initialize via `PeftModel.from_pretrained()` on CPU.
-   **K2:** The exported adapter fails to map its attention projections to a mock vLLM fused `qkv_proj` target shape.
-   **K3:** The measured Grassmannian deviation exceeds $1e-4$ post-training, invalidating the orthonormality claim.