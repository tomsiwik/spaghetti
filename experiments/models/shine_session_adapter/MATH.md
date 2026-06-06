# MATH.md: SHINE-Style Session Adapter

**Experiment type:** Frontier Extension (Type 3)
**Based on:** SHINE arXiv:2602.06358 + Finding #336 (M2P ports to MLX, outputs non-random)

---

## A. Proven Result Being Extended

**SHINE (arXiv:2602.06358)** proves that a Memory-to-Parameter (M2P) Transformer can
map a grid of LLM hidden states (L layers, M memory tokens, H hidden dim) to adapter
weights via a single forward pass. The training signal is next-token loss: given a
context, M2P generates an adapter; the adapter is applied to the base LM; the combined
model is trained to predict the next token. Gradient flows through M2P only — the base
LM is frozen. This is proven to work in the original paper on transformer-based LMs.

**Finding #336 (SUPPORTED):** The M2P Transformer ports to MLX. Its outputs are
statistically distinguishable from random noise (K827 PASS: t=3.33, p=0.0023,
|diff|=0.0815 > 0.05). Architecture validated: 197K params, 4.1ms forward.

---

## B. The Gap (What New Math Is Needed)

SHINE was validated with:
1. Real LLM hidden states as memory inputs (not random)
2. End-to-end task supervision (NTP loss through M2P-generated adapter)
3. The full training loop

Finding #336 only validated that the M2P architecture computes distinguishable outputs
from **random inputs** — it did not test real LLM hidden states or the end-to-end
training objective.

**The gap:** Does the full SHINE training loop (frozen base + M2P adapts to context)
converge, and does the resulting generated adapter capture any of the SFT adapter's
domain knowledge?

This gap requires:
1. A learning signal connecting M2P weights to domain task performance
2. Evidence that one-pass inference-time adapter generation transfers knowledge
   (not just that gradients flow through M2P during training)

The gap cannot be closed by a formal proof — it requires experimental evidence. This
is why the experiment type is frontier extension.

---

## C. Mathematical Framework

### C.1 The SHINE Training Objective

Let:
- **B** = frozen base language model (Qwen3-4B-4bit in macro; toy model here)
- **M2P** = Memory-to-Parameter Transformer (197K params from Finding #336)
- **c** = context sequence (1-2 pages of domain text)
- **t** = task sequence (held-out text from same domain)

The SHINE forward pass for one training example:

```
h_{1..L,1..T} = B(c)        # extract hidden states from all layers (frozen)
M = sample(h, M=8 tokens)  # sample M memory tokens from sequence dimension
W_A, W_B = M2P(M)           # generate LoRA weights: A ∈ R^{d×r}, B ∈ R^{r×d}
ΔW = B_mat @ A_mat          # LoRA adapter (r=4, much smaller than d)
logits = B(t, adapter=ΔW)   # apply adapter to base model, predict t
L = NTP(logits, t)           # next-token prediction loss
∂L/∂θ_M2P only              # gradient flows through M2P, NOT through B
```

This is standard LoRA fine-tuning with the adapter weights replaced by M2P outputs.

### C.2 Why Gradient Flows Through M2P

The LoRA adapter computation is:
```
h_adapted = h_base + (h_base @ A^T) @ B^T  * (alpha/r)
```

Since A and B are direct outputs of M2P (differentiable), the chain rule gives:
```
∂L/∂θ_M2P = ∂L/∂(A,B) · ∂(A,B)/∂θ_M2P
```

Finding #336 verified that gradients reach all M2P parameters (min norm > 0). So the
training signal propagates correctly. This is a known property of the architecture —
it is not the gap we are investigating.

### C.3 What "Capturing 50% of SFT Quality" Means

Let:
- PPL_base = 6.058 (Finding #333: Qwen3-4B-4bit on medical text)
- PPL_SFT = 5.249 (Finding #333: Qwen3-4B-4bit + trained medical LoRA adapter)
- delta_SFT = PPL_base - PPL_SFT = 0.809

The K832 threshold of 50% means:
```
PPL_M2P_generated < PPL_base - 0.5 * delta_SFT
                  = 6.058 - 0.5 * 0.809
                  = 6.058 - 0.4045
                  = 5.654
```

This is a regression from the base PPL by at most 0.404 units (6.7% of base PPL).

**Why PPL is the right metric here:** Finding #333 showed that the medical adapter
improves PPL by 13.4% on medical held-out text. Unlike the cross-domain PPL
comparisons that this project showed to be uncorrelated (r=0.08), same-domain PPL
improvement IS a valid signal for domain adaptation quality. The SFT adapter was
trained on medical text and improves medical PPL — a session adapter trained on the
same distribution should show the same directional effect.

### C.4 The Toy Model Architecture (Scope Decision)

To meet the micro budget (<2 hours), we use a **toy language model** approach
rather than loading Qwen3-4B-4bit. Reasons:

1. Loading Qwen3-4B-4bit (~6GB) for training M2P would leave <34GB for M2P
   gradients and optimizer state — tight but feasible. However, full model
   backprop through the toy task is unnecessary since we only need to verify
   the mechanism, not the scale.

2. The toy model can be made to match the structural properties we care about:
   - Same number of layers L as the base model (we use L=4 for micro speed)
   - Same mechanism: hidden states → M memory tokens → M2P → adapter weights

3. The kill criterion is about **relative quality** (50% of SFT), which can be
   measured equivalently on a toy model:
   - Train SFT adapter on toy LM for domain D
   - Train M2P to generate domain adapter from context
   - Compare PPL improvement: M2P-generated vs SFT-trained

**Toy model spec (actual values used in experiment):**
- Transformer LM: 4 layers, d=128 hidden, 4 heads, vocab=65 (character-level), 804K params
- LoRA rank r=4 applied to Q, V projections in all layers (8 adapter matrices total)
- Domain = "medical" = synthetic text with specific patterns (simulated)
- SFT adapter: 200 gradient steps on domain data
- M2P: generates all 8 adapter matrices from 8 memory tokens across 4 layers

---

## D. Quantitative Predictions

This is a frontier extension — we cannot derive exact numerical predictions from
the proven SHINE result (which was tested at a different scale and with real LLMs).
However, we can reason about what the math requires:

### Prediction D.1: M2P can learn (gradient signal exists)

The SHINE paper proves that M2P training converges when:
1. The base LM provides informative hidden states (not degenerate)
2. The adapter dimensionality matches the task (r < d, which holds at r=4, d=128)
3. The training data provides consistent domain signal

At our toy scale, all three hold by construction. **Prediction: M2P training loss
decreases monotonically over 300 steps (ratio final/initial < 0.5).** (Note: originally
stated as 200 steps; implementation used 300 steps — see Self-Test Q5.)

### Prediction D.2: K832 threshold (50% of SFT quality)

The SHINE paper claims M2P-generated adapters approach SFT adapter quality. At our
toy scale, if the mechanism works:

**Prediction D.2:** PPL_M2P_generated / PPL_base < (PPL_SFT / PPL_base + 1) / 2

Equivalently, the M2P adapter captures at least 50% of the PPL improvement of the
SFT adapter.

**What would falsify D.2:** If PPL_M2P_generated is within 5% of PPL_base (i.e., the
M2P adapter has essentially no effect), the mechanism does not transfer knowledge.

### Prediction D.3: K833 threshold (< 5s generation)

M2P has 9.18M parameters. A single forward pass through the M2P Transformer on an
input of shape (L=4, M=8, H=128) at toy-model hidden dim d=128, the M2P forward is
O(L*M*H) = fast.

**Prediction: K833 easily passes.** M2P inference (context encoding + adapter
generation) should take < 100ms on M5 Pro.

---

## E. Assumptions and Breaking Conditions

**Assumption E.1:** The toy model produces informative hidden states.
*If violated:* M2P gets uniform inputs, cannot learn domain-specific patterns.
*Detection:* Check that hidden state cosine similarity across domains differs
(inter-domain cos < intra-domain cos by > 10%).

**Assumption E.2:** 100 training steps on synthetic domain data are sufficient
for SFT adapter to show measurable PPL improvement.
*If violated:* The SFT baseline is at base PPL, making the 50% criterion vacuous.
*Detection:* Verify PPL_SFT < 0.95 * PPL_base before running M2P evaluation.

**Assumption E.3:** M2P can compress domain information from 8 memory tokens
into r=4 rank-4 adapter matrices.
*This is the core open question.* The rank constraint means M2P must solve a
low-rank approximation to the SFT adapter.
*If violated:* M2P learns to copy some structure but loses the key domain signal.

**Assumption E.4:** The training signal (NTP loss) provides sufficient gradient
to train M2P in 200 steps.
*If violated:* M2P parameters do not update meaningfully.
*Detection:* Check M2P parameter change norm between step 0 and step 200.

---

## F. Worked Example (Toy Scale, d=16, L=2, r=2)

Let the toy LM have d=16 hidden, L=2 layers, and generate a domain adapter of rank r=2.

**M2P output needed:**
- Q_proj in layer 0: A=(16,2), B=(2,16) → 64 params
- V_proj in layer 0: A=(16,2), B=(2,16) → 64 params
- Q_proj in layer 1: A=(16,2), B=(2,16) → 64 params
- V_proj in layer 1: A=(16,2), B=(2,16) → 64 params
- Total: 256 adapter params

**M2P input:**
- L=2 layers, M=8 memory tokens, H=16 hidden dim
- M2P output: (L=2, M=8, H=16) = 256 values
- Each layer's M*H = 8*16 = 128 values → enough for one Q and one V adapter matrix

**Parameter extraction:**
For layer i:
```
flat = M2P_output[i].reshape(-1)  # 128 values
A_Q = flat[:16*2].reshape(16, 2)  # 32 values
B_Q = flat[32:64].reshape(2, 16)  # 32 values
A_V = flat[64:96].reshape(16, 2)  # 32 values
B_V = flat[96:128].reshape(2, 16) # 32 values
```

**Forward pass with adapter:**
```
# Layer i attention Q-projection
h_in: (T, 16)
q_base = h_in @ W_Q    # base projection
q_delta = (h_in @ A_Q) @ B_Q  # LoRA delta: (T, 2) @ (2, 16) = (T, 16)
q = q_base + (alpha/r) * q_delta
```

This is the standard LoRA forward — M2P generates A_Q and B_Q instead of training them.

---

## G. Complexity and Architecture Connection

**M2P overhead:**
- Context encoding: O(T * d * L) for base LM hidden state extraction (frozen)
- M2P forward: O(L * M * H * N_layers_M2P) = O(4 * 8 * 64 * 4) = O(8192) — negligible
- Total adapter generation: dominated by context encoding
- At H=64, M=8, L=4: 2048 flops per M2P step → << 1ms

**Adapter application overhead:**
- LoRA forward: O(T * d * r) per layer = O(T * 128 * 4) per token per layer
- At T=1 (autoregressive): 512 flops per layer → < 0.1ms

**Memory:**
- M2P weights: 9.18M params @ float32 = 36.7 MB
- LoRA adapter (d=128, r=4, 8 matrices): 8 * 2 * 128 * 4 = 8K params = 32 KB
- Toy LM (d=128, L=4, vocab=65): 804K params = 3.2 MB
- Total peak: < 100 MB — well within M5 Pro budget

---

## Self-Test (MANDATORY)

**1. What is the ONE mathematical property that makes the failure mode impossible?**

This is a frontier extension — there is no impossibility structure. The failure mode
(M2P generates useless adapters) is possible. K832 measures whether it occurs.

**2. Which existing theorem(s) does the proof build on?**

SHINE arXiv:2602.06358 demonstrates empirically that M2P training with NTP loss converges
when the base LM provides informative hidden states and the adapter rank is sufficient.
SHINE does not contain a formal convergence theorem — this is an empirical result from
the original paper, not a proof. Finding #336 confirms M2P ports to MLX with non-random
outputs.

**3. What specific numbers does the proof predict?**

- D.1: M2P training loss ratio final/initial < 0.5 (convergence check)
- D.2: PPL_M2P_generated < PPL_base * (1 - 0.5 * (1 - PPL_SFT/PPL_base))
  = toy_base_PPL - 0.5 * (toy_base_PPL - toy_SFT_PPL)
- D.3: Adapter generation time < 5s (K833) — expected < 100ms

**4. What would FALSIFY the proof?**

The SHINE mechanism is falsified if:
- M2P training loss does not decrease (no gradient signal through NTP)
- Generated adapter PPL = base PPL (M2P outputs have no effect when applied)
- Generated adapter PPL increases (M2P outputs are harmful noise)

**5. How many hyperparameters does this approach add?**

- M2P learning rate: 1 (set to 1e-3, from Finding #336 convergence test)
- Number of training steps: 1 (implementation used 300 steps, not 200 as stated in D.1.
  This was changed to give M2P extra optimization time. The D.1 prediction of <0.5 loss
  ratio was written for 200 steps; the actual experiment ran 300 steps. Even with 50%
  more steps, D.1 failed — making the failure mode more interesting, not less.)
- Number of memory tokens M: 1 (set to 8, matching Finding #336)
Total: 3 hyperparameters; the step count was quietly increased from 200 to 300 in
implementation, which should be acknowledged as an uncontrolled change.

**6. Hack check: Am I adding fix #N to an existing stack?**

No. This is a direct application of the SHINE training objective. The M2P architecture
is proven. We are testing whether the training loop converges on our toy setup, which
is the minimal viable test of the frontier extension.
