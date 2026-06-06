# LEARNINGS: M2P v2 — MLX Functional Autodiff Requires Functional LoRA Forward

**Status:** KILLED — Finding #375 (killed)
**Experiment:** exp_m2p_qwen06b_gsm8k_v2
**Critique:** #3 (no real NLP) STILL UNRESOLVED

---

## Core Finding

MLX's functional autodiff (`nn.value_and_grad`) cannot trace gradients through module
attribute mutation (`layer.lora_b = m2p_output`). This is a structural invariant of
functional autodiff frameworks — the gradient path from loss → model → lora_b → M2P
parameters requires B-matrices to be tensor *arguments* in the differentiable function,
not mutable module state. M2P loss flatlined at 11.93 (≈ ln(vocab_size) = pure noise)
for all 1000 steps despite SFT converging normally (loss 1.05 → 0.96).

**The positive result:** K909 PASS (base=20%) and K910 PASS (SFT +6pp) confirm the
evaluation pipeline, weight-space LoRA, causal masking, and GQA dimension fixes all
work correctly. These baselines transfer directly to v3.

---

## Why This Happened

### Primary: MLX functional autodiff paradigm

MLX, like JAX, uses functional autodiff: computation graphs are built through function
application chains, not mutable object state. When `nn.value_and_grad(m2p, fn)` traces
`fn`, it records tensor operations on `m2p.parameters()`. A statement like
`model.layer.lora_b = m2p_output` inside `fn` creates a new object attribute — it is
**not** recorded as a differentiable operation in the computation graph. The gradient
path is severed.

This is not an MLX bug — it is the correct behavior for a functional autodiff system.
The same pattern would fail in JAX with the same symptom (zero gradients). PyTorch
would succeed here because it builds graphs dynamically through `Tensor.grad_fn`
references regardless of how tensors are passed or stored.

**Definitive evidence:** The toy experiments that achieved 99.6% quality (Findings
#361-#365) used a fully functional forward:
```python
def lora_forward_with_B(x, W_base, A, B, scale):
    return W_base(x) + scale * (x @ A) @ B
```
Here B is a tensor argument — it IS in the computation graph. The toy experiments
worked precisely because the researcher controlled the entire forward pass.

Attempting to inject B into an existing model's LoRALinear module via attribute
assignment creates a seam that breaks the gradient chain. This was not identified in
MATH.md (which proved SFT correctness and capacity but never analyzed gradient flow).

### Secondary: Architectural divergence from SHINE (d_M2P bottleneck)

Even if gradients flowed, d_M2P=128 vs d_model=1024 creates an 8x compression
bottleneck and 5376:1 total compression ratio. SHINE uses d_M2P = d_model (~1:2
expanding compression). The reviewer correctly noted this is downstream of the gradient
flow issue — fixing it matters for v3 but not for the diagnosis of v2.

### Procedural gap: No gradient norm monitoring

Running 1000 steps without checking `max(|grad|) > 0` wasted compute. A 5-step smoke
test would have revealed zero gradients in seconds. The fail-fast principle (Fix #6 for
base accuracy in v1) must be extended to gradient health.

---

## Why It Happened vs Prior Literature

### Confirming evidence

**Ha et al. (arXiv:1609.09106) — HyperNetworks:** Original hypernetwork paper explicitly
passes generated weights as function arguments to a primary network. No module mutation.
This is the canonical pattern: `y = primary_net(x, W_generated)` where W_generated flows
through the graph as a tensor.

**SHINE (arXiv:2602.06358):** Uses the frozen LLM as its own encoder and generates
B-matrices via a transformer that outputs tensors — never mutations. The SHINE training
loop computes `loss = NTP(model(x, B=m2p(h)))` where B is a tensor arg.

**JAX documentation on `jit`/`grad`:** Explicitly states that in-place mutations to
arrays inside `jit`-compiled functions produce undefined behavior or zero gradients,
because JAX's XLA lowering treats function arguments as immutable. MLX has the same
constraint.

**HyperTuning (arXiv:2210.03726, ref #545):** Generates LoRA parameters as explicit
tensors injected via functional forward. No module state mutation. Achieves 90%+ of
full fine-tuning quality on 20 NLP tasks.

### Contradicting evidence

No papers found that successfully train hypernetworks via module attribute mutation
inside functional autodiff frameworks (MLX, JAX). All successful approaches use tensor
argument passing.

**PyTorch exception:** PyTorch's dynamic computation graph tracks tensor operations
regardless of how tensors are stored. PyTorch hypernetworks commonly use
`functional.linear(x, W_generated, b)` — but even this is functional (weight as arg),
not attribute mutation. Module mutation in PyTorch (`module.weight = new_w`) bypasses
autograd unless the new tensor was itself computed in the graph.

### Alternative approaches with paper evidence

1. **Functional LoRA forward** (our toy experiments, all SHINE variants):
   Pass B as a tensor argument to a custom attention forward function.
   ```python
   def attn_forward_with_lora(x, q_proj, k_proj, v_proj, B_q, B_v, scale):
       q = q_proj(x) + scale * (x @ A_q) @ B_q  # B_q in graph
       ...
   ```
   Evidence: All toy M2P experiments (Findings #361-#366), SHINE (arXiv:2602.06358).

2. **Parameter substitution via `model.update()`** (MLX idiomatic pattern):
   Build a new parameter dict with M2P-generated B values, then call
   `model.update(new_params)` before each forward. This creates a new differentiable
   forward without mutation.
   Evidence: MLX documentation; used in test-time adaptation literature.

3. **Functional.linear wrapper** (PyTorch LoRA papers):
   Replace module forward with functional call: `F.linear(x, W + BA/r)`.
   Generalizes to MLX as `mx.linear(x, W_base + lora_delta)` where `lora_delta`
   is computed from M2P tensors.
   Evidence: LoRA (arXiv:2106.09685, ref #543), GaLore (arXiv:2403.03507).

---

## Implications for Next Experiments

### v3 design requirements (non-negotiable)

1. **[BLOCKING] Functional LoRA forward** — B as tensor arg, never module mutation.
   Pattern: `y = W_base(x) + scale * (x @ A) @ B` where B flows through graph.
   Implementation: write custom Qwen3 attention forward that accepts B_q, B_v.

2. **[BLOCKING] Gradient health check at step 1** — `assert max(|grad|) > 0` before
   committing to full training run. 5 seconds → prevents 15 minutes wasted compute.

3. **[IMPORTANT] Output scaling ×0.032** (SHINE sqrt(0.001)) — M2P output starts near-
   zero, matching LoRA convention (B=0 at init). Without this, initial B-matrices are
   random perturbations at wrong scale.

4. **[IMPORTANT] d_M2P = d_model** — Remove 8x bottleneck. Aghajanyan d_int is unknown
   for real NLP; conservative choice is d_M2P = d_model = 1024 for Qwen3-0.6B.

5. **[ADVISORY] Per-layer positional embeddings** — Replace mean-pool across 28 layers
   with per-layer identity tokens. Finding #336 (SHINE port) showed this is essential
   for SHINE's accuracy.

6. **[ADVISORY] LR warmup 100-200 steps, LR = 5e-5** — Consistent with SHINE config
   for Qwen3-0.6B. Avoids early instability when B-matrices first go non-zero.

### Eval pipeline confirmed reusable

K909 (base=20%) and K910 (SFT=26%) are valid for Qwen3-0.6B-4bit on GSM8K with
rank=4 LoRA (1000 steps, 2000 training examples). v3 can skip re-measuring these
and use them as reference targets directly.

### The 1.15ms generation latency claim needs revision

SHINE-aligned v3 uses the frozen base LLM as its own encoder: generating adapter
B-matrices costs one full LLM forward pass (~10ms for 0.6B, ~50ms for 4B). The
"1.15ms generation" result from toy SHINE port (Finding exp_shine_session_adapter)
was on a 4L/128d toy model where the forward pass is trivially fast. This does not
transfer to production models.

---

## Recommended Follow-Up

**exp_m2p_qwen06b_gsm8k_v3** (P0 — resolves Critique #3):
- MOTIVATION: This is the direct fix of the gradient flow issue proven by Finding #375.
  K909/K910 baselines transfer directly; only M2P training needs redesign.
- LITERATURE: SHINE (arXiv:2602.06358) proves functional forward with d_M2P=d_model
  achieves 90%+ SFT quality on real NLP. HyperTuning (arXiv:2210.03726) confirms
  hypernetworks work for LoRA-style weight generation with proper tensor arg passing.
- FIX: Rewrite Qwen3 attention forward to accept B_q, B_v as tensor args. Add gradient
  smoke test at step 1. Set d_M2P=d_model=1024, output_scale=0.032.
- SUCCESS CRITERION: M2P loss decreases from ln(vocab) within first 50 steps (smoke
  test). quality_ratio >= 70% at 1000 steps on GSM8K.

Do NOT proceed to v3 without implementing and passing the gradient smoke test first.
