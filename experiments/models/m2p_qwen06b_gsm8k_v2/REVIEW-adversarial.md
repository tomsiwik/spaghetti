# Peer Review: m2p_qwen06b_gsm8k_v2

## Experiment Type
frontier-extension (Type 3) -- Correctly identified. Extends proven M2P quality scaling
(toy synthetic domains, d=256-1024, Findings #359-#362) to a real model (Qwen3-0.6B-4bit)
on a real NLP benchmark (GSM8K). Finding status correctly capped at provisional.

## Hack Detector
- Fix count: 6 corrections to v1's implementation bugs (all correctness fixes, not new mechanisms). The M2P architecture itself (encoder + memory bank + per-layer heads) is unchanged from toy experiments. This is a retry, not a mechanism pile-up. **NO FLAG.**
- Is MATH.md a proof or a description? **Mixed.** Theorem 1 (SFT correctness via LoRALinear) is a genuine proof with QED. Theorem 2 (GQA shapes) is trivially correct. Theorem 3 (M2P capacity) is a proof *sketch* -- it invokes Aghajanyan's intrinsic dimensionality but does not prove convergence, only capacity. For Type 3 frontier-extension, this is acceptable.
- Metric used as evidence: quality_ratio = M2P_gain / SFT_gain. This is the right metric for the question being asked.
- Kill criteria source: K909 and K910 derived from literature baselines. K911 and K912 derived from Aghajanyan's d_int bound and prior toy-scale results. **Reasonably grounded, not arbitrary.**

## Self-Test Audit
MATH.md Section G contains a completed Self-Test. Checking each item:

1. **One-sentence impossibility property:** "Using mlx_lm's standard model forward path with create_attention_mask makes the bidirectional-attention training failure impossible." -- Genuine single property targeting Bug #3 from v1. PASS.

2. **Cited theorems:** Hu et al. (LoRA), Aghajanyan et al. (intrinsic dim), JL-Lemma/FlyLoRA (random A), Cobbe et al. (GSM8K eval). All real papers with correct arxiv IDs. Conditions are applied reasonably. PASS.

3. **Predicted numbers:** base_acc > 5%, sft_gain >= 5pp, quality_ratio >= 70%, B-matrix shape (4, 2048). Specific and falsifiable. PASS.

4. **Falsification condition:** Three specific conditions stated. These target the proof's predictions, not just arbitrary thresholds. PASS.

5. **Hyperparameter count:** States 4 (lora_rank, lora_scale, d_M2P, train_steps) with justifications for each. Accurate for the M2P-specific additions; the broader experiment has more (lr, max_seq_len, max_gen_tokens, L_M2P, N_MEMORY) but these are acknowledged in the architecture section. MARGINAL -- undercounts by ~5 but most are standard choices.

6. **Hack check:** Correctly identifies this as a retry fixing implementation bugs, not adding new mechanisms. Constraint count unchanged from v1. PASS.

## Mathematical Soundness

**Theorem 1 (SFT correctness):** Sound. The proof correctly establishes that using
LoRALinear.from_base + standard model forward path guarantees correct weight-space LoRA
with causal masking. All three steps (a-c) are verified by the implementation. The SFT
results (loss 1.05->0.96, accuracy 20%->26%) confirm the theorem's prediction empirically.
**HOLDS.**

**Theorem 2 (GQA shapes):** Trivially correct. The approach of reading shapes from
model.args and verifying against weight.shape is robust. The runtime verification
confirmed q_proj_out=2048, v_proj_out=1024 as predicted. **HOLDS.**

**Theorem 3 (M2P capacity):** This is a proof sketch, not a full proof. It correctly
invokes Aghajanyan's intrinsic dimensionality result but only establishes *capacity*
(d_M2P >= d_int implies sufficient expressive power), not *convergence* (that training
will find the right B-matrices). For a Type 3 experiment, this is acceptable -- the
question of whether M2P training converges on real NLP is the frontier question.
**HOLDS as capacity argument; convergence is the empirical question.**

**Critical gap MATH.md missed:** The MATH.md contains an elaborate proof that the SFT
pipeline is correct and that M2P has sufficient capacity, but it contains NO analysis
of the gradient flow path for M2P training. The comment at lines 628-638 of
run_experiment.py claims "In MLX, the gradient through the mutation works because MLX
traces lazy computations." This is an architectural *assumption* about MLX's autodiff
behavior, not a proven property. It turned out to be wrong. MATH.md should have
contained a Theorem 4 about gradient flow through module attribute mutation, which
would have been falsified before code was written.

## Prediction vs Measurement

PAPER.md contains a clean prediction-vs-measurement table:

| Criterion | Predicted | Measured | Match? |
|-----------|-----------|----------|--------|
| K909 base_acc > 0% | > 5% | 20.0% | YES |
| K910 sft_gain >= 5pp | +5-20pp | +6.0pp | YES |
| K911 quality_ratio >= 70% | 70-90% | -333% | NO (catastrophic) |
| K912 KILL < 30% | should not trigger | TRIGGERED | FAIL |

The table is well-structured and honest. K909 and K910 are clean passes confirming that
the evaluation and SFT pipelines work correctly. K911's failure is catastrophic (not
marginal), with M2P loss flat at ln(vocab)=11.93 for all 1000 steps, indicating zero
gradient signal rather than insufficient capacity.

## Root Cause Analysis Assessment

### Is the kill correct?
**YES.** The kill is unambiguous. M2P loss is flat at 11.93 (ln(vocab_size)) for all
1000 training steps. This is the loss of random token prediction -- the M2P network
produces effectively random B-matrices throughout training. 0% accuracy with degenerate
generation ("strapstrapstrap...") confirms total failure. K912 (quality_ratio < 30%)
is cleanly triggered at -333%.

### Is the root cause correctly identified?
**PARTIALLY.** PAPER.md identifies two root causes:

1. **Gradient flow broken through module attribute mutation:** CORRECT. This is the
   primary root cause. `nn.value_and_grad(m2p, fn)` traces gradients w.r.t.
   `m2p.parameters()`. When `fn` sets `model.layer.lora_b = m2p_output` via attribute
   mutation, MLX's computation graph cannot trace the dependency from the NTP loss back
   through the LoRALinear forward, through the mutated lora_b attribute, to the M2P
   parameters that produced it. The flat loss curve (11.93 for 1000 steps) is definitive
   evidence of zero gradient.

   **Confirmation:** The toy experiments that achieved 99.6% quality (m2p_macro_quality)
   use a completely different approach: a **functional forward** (`lora_forward_with_B`)
   where B-matrices are tensor arguments, not module attribute mutations. The gradient
   flows through `base_out + LORA_SCALE * (x_in @ A) @ B` directly because B is a
   normal tensor in the computation graph.

2. **Scale mismatch:** SECONDARY, but would be relevant even if gradients flowed. M2P
   generates B-matrices from random initialization with non-zero values, while LoRA
   convention initializes B=0. Without output scaling (SHINE uses sqrt(0.001)~=0.032),
   M2P immediately injects a large random perturbation into every layer.

### Additional issues the researcher missed

**Issue 1: d_M2P=128 vs d_model=1024 bottleneck (partially identified).**
LEARNINGS.md correctly identifies this gap vs SHINE (which uses d_M2P=d_model), but
this is downstream of the gradient flow issue. Even with d_M2P=1024, the attribute
mutation approach would still produce zero gradients. However, this is important for v3
planning: the 5376:1 compression ratio (344K SFT params compressed into 128-dim latent
via 44.7M head parameters) is architecturally backwards -- the heads are 130x larger
than what they generate.

**Issue 2: Mean-pooling across layers destroys layer identity.**
The M2P encoder does `h = mx.mean(layer_hs, axis=0)` which averages all 28 layers'
hidden states into a single vector. This destroys per-layer information that SHINE
preserves through layer positional embeddings (per the SHINE architecture study,
Finding #336). At 28 layers, averaging produces a centroid that carries minimal
layer-specific signal. This was identified in LEARNINGS.md but not in PAPER.md.

**Issue 3: No gradient norm monitoring.**
The experiment ran 1000 M2P training steps without monitoring gradient norms. A simple
`max(|grad|)` check at step 1 would have immediately revealed zero gradients, saving
~800 steps of wasted compute. This is a procedural gap -- the fail-fast principle
(Fix #6 for base accuracy) should have been extended to gradient health.

### What carries forward for v3?

PAPER.md's recommendation is correct: use a **functional LoRA forward** where B-matrices
are tensor arguments, not module attribute mutations:

```python
def functional_lora_forward(x, W_base, A, B, scale):
    return W_base(x) + scale * (x @ A) @ B
```

LEARNINGS.md adds critical corrections from SHINE source code:
1. Output scaling (x0.032) to prevent random perturbation
2. d_M2P = d_model (not bottleneck)
3. Per-layer positional embeddings (not mean-pool)
4. LR warmup + lower learning rate

These are all well-grounded in SHINE's published architecture and the project's own
SHINE port experiment (Finding #336).

**Additional recommendation for v3:** Before committing to a full 2.5-hour run,
implement a 5-step gradient flow smoke test: run 5 M2P training steps, check that
`max(|grad|) > 0` for at least one M2P parameter. This costs seconds and prevents
the category of failure seen in v2.

### Is the impossibility structure correctly derived?

PAPER.md Section "Mathematical structure that makes this failure inevitable" correctly
identifies the structural issue: `nn.value_and_grad(m2p, fn)` cannot trace through
mutable module attribute assignment. The explanation of why this breaks the gradient
chain is sound.

The impossibility statement should be sharpened: the issue is not that MLX "may not"
track the dependency -- it definitively does not, as proven by the flat loss curve.
The toy experiments' functional forward approach is the correct pattern. Module
mutation is a dead end for hypernetwork training in MLX's functional autodiff paradigm.

## Novelty Assessment
This experiment's novel contribution is not positive (M2P did not work on real NLP) but
is informatively negative: it conclusively demonstrates that the gradient flow pattern
that works at toy scale (where the researcher controls the full forward pass) breaks
when attempting to inject M2P-generated parameters into an existing model's LoRA modules
via attribute mutation. This is a genuine architectural finding about MLX's autodiff.

The K909/K910 results (base=20%, SFT=26%) also provide useful baseline measurements
for Qwen3-0.6B-4bit on GSM8K with rank=4 LoRA, which can be reused in v3.

## Macro-Scale Risks (advisory)
1. The functional forward approach for v3 requires rewriting Qwen3's attention mechanism
   to accept B-matrices as explicit tensor arguments. This is model-specific and does
   not generalize across architectures without per-model adaptation.
2. SHINE's approach (using the frozen LLM as its own encoder) adds approximately one
   full forward pass of latency for M2P generation. At 0.6B this is ~10ms; at 4B it
   would be ~50ms. The "1.15ms generation" claim from toy experiments needs revision.
3. At 44.7M M2P parameters (for a 0.6B base), the M2P network is 7.5% the size of the
   base model. This ratio should decrease at larger base model sizes if d_M2P tracks
   d_model rather than growing with n_layers x n_modules x output_dims.

## Verdict

**KILL** -- confirmed.

The kill is clean, correctly motivated, and the root cause analysis is substantially
correct. The experiment successfully demonstrated that:

1. The evaluation pipeline works (K909 PASS: base=20%)
2. SFT LoRA training works (K910 PASS: +6pp gain)
3. M2P via module attribute mutation does NOT work (K911 FAIL: flat loss at ln(vocab))

The root cause (gradient flow broken through module attribute mutation in MLX's
functional autodiff) is correctly identified and supported by comparison with the toy
experiments' functional forward pattern.

**For v3, the following specific changes are required:**

1. **[BLOCKING] Functional LoRA forward:** Rewrite attention computation to accept
   B-matrices as tensor arguments, not module attribute mutations. Pattern:
   `y = W_base(x) + scale * (x @ A) @ B` where B flows through the computation graph.
2. **[BLOCKING] Gradient health check:** Add `assert max(|grad|) > 0` at step 1.
   Five seconds of compute prevents 15 minutes of wasted training.
3. **[IMPORTANT] Output scaling:** Initialize M2P output with scaling factor ~0.032
   (SHINE's sqrt(0.001)) so initial B-matrices are near-zero, matching LoRA convention.
4. **[IMPORTANT] d_M2P = d_model:** Remove the 8x bottleneck. At d_model=1024, use
   d_M2P=1024 (or at minimum d_M2P=512 with explicit acknowledgment of compression).
5. **[ADVISORY] Per-layer structure:** Replace mean-pooling with layer positional
   embeddings per SHINE architecture.
6. **[ADVISORY] LR warmup:** Add 100-200 step linear warmup; reduce LR to 5e-5.

---

## V2 Finalize Review (2026-04-18, reconstruction-only)

Researcher re-finalized this experiment today to attach a results.json. No re-run.
Original 876s run executed 2026-04-07; all 5 MD artifacts were committed in 7b1cd80
but the results.json was never committed (`micro/**/results.json` is in `.gitignore`).
Researcher reconstructed it from PAPER.md's prediction-vs-measurement table and
attached an explicit `_reconstruction_note` for audit transparency.

Adversarial checklist for the reconstruction:

- (a) results.json `verdict=KILLED` matches DB `status=killed` and `--status killed` claim → clean
- (b) `all_pass=false`, K911/K912 both fail, consistent with KILLED → clean
- (c) PAPER.md verdict context contains no PROVISIONAL / PARTIALLY / INCONCLUSIVE
  (word "degenerate" appears only in root-cause prose, not in verdict) → clean
- (d) `is_smoke=false`, full N=200 eval, 1000 training steps → clean
- (e) `git diff HEAD -- MATH.md` empty → no KC drift since reg → clean
- (f) No tautology — K911 is M2P acc vs SFT acc (real measurement); K912 is
  a threshold trigger on the same ratio → clean
- (g) Code at run_experiment.py:715-716 implements the `layer.self_attn.{q,v}_proj.lora_b
  = b_by_key[...]` mutation that PAPER.md names as the root cause; K909-K912 measure
  the quantities MATH.md §D and DB rows describe → clean
- (h)–(m) same as original review → clean (single-domain GSM8K, no composition
  math; LORA_SCALE=5.0; no shutil.copy; no hardcoded pass dict; model matches)
- (r) PAPER.md prediction-vs-measurement table present → clean

Numerical cross-check (results.json vs PAPER.md):
- base_accuracy: 0.20 ↔ 20.0% (40/200) ✓
- sft_accuracy: 0.26 ↔ 26.0% (52/200) ✓
- m2p_accuracy: 0.00 ↔ 0.0% (0/200) ✓
- sft_final_loss: 0.9607 ↔ 0.9607 ✓
- m2p_final_loss: 11.93 ↔ ~11.93 ✓
- quality_ratio: -3.333 ↔ -3.33 ✓
- total_time_s: 876 ↔ 876s ✓

All numbers reconcile. Verdict **KILL** stands, unchanged from original review.

**Finding recommendation:** add a finding capturing the MLX autodiff gradient-flow
failure mode (module attribute mutation inside `nn.value_and_grad`-traced function).
This is a novel architectural lesson, not a simple metric miss, and should
propagate to any future hypernetwork-in-MLX experiment design.
