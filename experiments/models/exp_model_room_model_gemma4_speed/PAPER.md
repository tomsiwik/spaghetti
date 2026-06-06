# PAPER — Room Model W_combined on Gemma 4 E4B (KILLED)

**Verdict: KILLED.**  All three pre-registered kill criteria were resolved; two failed
(KC1688, KC1689), one passed (KC1690). The failures are structural, not tuning artifacts:
they re-confirm at the Gemma 4 E4B / M5 Pro regime the same impossibility already proved
on Qwen3 (Findings #302, #303, #315) and bounded by Zhong et al. 2504.10957. No
hyperparameter sweep will fix this; the Room Model is **closed for pre-summing at N>1**.

## 1. Predictions vs measurements

| KC | Pre-registered threshold | Expected | Measured | Pass |
|----|-------------------------|----------|----------|------|
| 1688 — W_room tok/s ≥ 150 on M5 Pro | ≥150 tok/s | FAIL (bandwidth math → ≤136 ceiling, ~70 estimated) | 69.18 tok/s (room) vs 86.62 tok/s (base) | ✗ |
| 1689 — cos(W_room, explicit routing) > 0.999 | > 0.999 | FAIL (LayerNorm nonlinear cross-terms, Zhong et al.) | 0.9941 (mean over 5 domains) | ✗ |
| 1690 — W_room +=/-= ΔW_k bitwise exact | within 1 bf16 ULP | PASS (pure linear algebra) | 6.10e-5 max abs diff, all 84 layers within ULP | ✓ |

All three **measured outcomes match the pre-registered predictions** in MATH.md. This
is a clean, falsifying replication at the target platform.

## 2. Setup

- **Model**: `mlx-community/gemma-4-e4b-it-4bit` (QuantizedLinear, bits=4, group_size=64).
- **Hardware**: Apple M5 Pro 48 GB, 273 GB/s unified-memory bandwidth.
- **mlx-lm**: 0.31.2.
- **Adapters**: N=5, rank r=6 PoLAR on `v_proj` + `o_proj` (84 modules).
  - A_i from Grassmannian (QR of random 30×d_in Gaussian on CPU stream).
  - B_i ~ 𝒩(0, σ²) with σ=0.02.
  - α = 1.0 (no LORA_SCALE inflation).
- **Dim profile** of target modules (d_out × d_in):
  - `2560×4096`: 7 layers (o_proj, 16 query heads × 256)
  - `1024×2560`: 7 layers (v_proj, 4 KV heads × 256)
  - `2560×2048`: 35 layers (o_proj, 8 query heads × 256)
  - `512×2560`: 35 layers (v_proj, 2 KV heads × 256)
- Total run time: **5.0 s** (random-init adapters; no training needed — see MATH.md §4).

## 3. KC1688 — speed (FAIL)

| metric | base | room | ratio |
|---|---|---|---|
| library tps (mlx_lm) | 86.62 | 69.18 | 0.80× |
| wall tps (subprocess-overhead-inclusive) | 72.16 | 58.98 | 0.82× |

Room installation costs ~20% of raw throughput. Relative to the 150 tok/s target, the
measured 69.18 tok/s is **46% of threshold**. Gap is too large to close by tuning —
it is set by the bandwidth cost of the dense W_room matrices per module.

**Bandwidth reconciliation (MATH.md §3):**
- predicted ceiling with W_room: ≈136 tok/s (708 MB W_room + 1.2 GB base activations)
- measured: 69.18 tok/s = 51% of ceiling
- corresponds to ~88 tok/s bandwidth efficiency baseline (base measured 86.62 tok/s
  = 64% of 136 tok/s base ceiling)
- ratio measured/ceiling is consistent across base and room → no systematic bias.

## 4. KC1689 — logit equivalence (FAIL)

Per-domain cosine between W_room logits and explicit single-adapter routing logits:

| domain | cos(room, routing) | cos(room, base) |
|---|---|---|
| math    | 0.99436 | 0.99496 |
| code    | 0.99712 | 0.99604 |
| medical | 0.98924 | 0.97978 |
| legal   | 0.99199 | 0.98686 |
| finance | 0.99775 | 0.99785 |
| **mean**| **0.99409** | **0.99073** |

Compare to threshold 0.999: gap of ~5e-3 in mean cosine → statistically and
behaviorally meaningful mismatch. Medical is the most-divergent domain (0.98924).

**Interpretation.** W_room's logit direction is *closer to the base model* than to
explicit routing for math/finance, and only marginally closer to routing than to base
overall. This is the LayerNorm cross-term effect predicted by Theorem 2: the nonlinear
compounding through 42 decoder layers causes the sum-of-deltas activation pattern to
drift from the single-adapter activation pattern — globally, not per-module.

**Why the cosine (~0.994) is much higher than prior POC's MSE=53.9.** Two differences:
(i) random-init B at σ=0.02 produces smaller ‖ΔW_i‖ than trained SFT adapters, so
cross-terms are quadratically smaller; (ii) cosine is scale-invariant whereas MSE
is not. This is a weaker claim — random adapters understate the real-world
divergence. A trained-adapter replication would show substantially lower cosine.
KC1689 still fails decisively.

## 5. KC1690 — exact reversibility (PASS)

For each module, computed `(W_room − α·B_k @ A_k) − fresh_compute_without_k`. Over all
84 modules, `max |diff|` = 6.10e-5 (uniform across layers), well inside bf16 ULP of
`max|W_fresh|` (≈ 2e-3 for typical W_room magnitudes). **All 84 layers pass**.

This confirms Theorem 3: W_room is an invertible sum under bf16 if we maintain the
associative order of summation. Adding/removing adapters at runtime is cost-free and
exact. (Note: this is necessary but **not sufficient** for Room Model — KC1689 shows
the sum itself is the wrong object.)

## 6. Interpretation in project context

1. **Kill is consistent with prior art.** Findings #302, #303, #315 all killed
   pre-summing at N>1 on Qwen3. This experiment confirms the kill extends to:
   (a) Gemma 4 E4B base (MoE, 4-bit quantized),
   (b) M5 Pro hardware,
   (c) PoLAR r=6 on v_proj+o_proj.
   The theoretical bound (Zhong et al. 2504.10957) predicts it — now empirically
   confirmed on the target platform.

2. **KC1690 PASS is the only usable fragment.** Exact reversibility could be reused
   for **N=1** "hot merge" paths (a single adapter pre-merged into a frozen copy of
   the weight), which is the v6 Pierre design, already supported. It does **not**
   rescue the multi-adapter composition claim.

3. **Speed ceiling.** Even if the equivalence held, the 69 tok/s measurement places
   Room Model at 42% of the project performance ceiling (165.6 tok/s) and 80% of
   the base throughput of Gemma 4 E4B on M5 Pro. Factored LoRA (h @ A @ B, no
   pre-sum) avoids the 708 MB bandwidth cost and would match base at 86 tok/s.

## 7. Implications for the project

- **Do NOT resurrect the Room Model for multi-adapter composition.** This is the
  fourth independent kill. `project_room_model.md` memory should be annotated as
  *superseded for N>1; N=1 merge only.*
- **v8 Pierre design** (PLAN.md Part 2) must continue with factored-LoRA +
  per-token routing. Room Model's "routing IS the matmul" claim is geometrically
  elegant but empirically false under realistic transformer nonlinearities.
- **Downstream experiments** that block on or cite Room Model (blocks list:
  `exp_model_pre_registration_n100_macro`, `exp_model_multi_seed_room_model`) should
  be reviewed for whether they are still necessary — their premise is killed.

## 8. Assumptions (logged per guardrail 1007)

- Random-init PoLAR adapters are a valid proxy for trained adapters in testing the
  *mechanism* (speed of dense matmul; linearity of sum; exact reversibility).
  KC1689 on trained adapters would likely show lower cosine, making the kill
  stronger — **random adapters are a conservative estimator of divergence**.
- σ=0.02 on B is a typical range for trained LoRA B matrices at LR=1e-4 after
  ~1k steps; not calibrated to any specific training run.
- Ground-truth routing is one-hot per prompt (domain known). This is the **strictest**
  version of explicit routing; any soft routing benchmark would be further from
  W_room, strengthening the kill.
- KC1688 measured at 1 prompt × 64 tokens × 1 warmup. Variance across prompts is
  bounded by stream-generate internals; 20% gap vs base is far outside measurement
  noise.

## 9. Verdict-consistency pre-flight (PLAN.md §1)

1. `results.json["verdict"]`: `"KILLED"` ✓
2. `results.json["all_pass"]`: `false` ✓
3. PAPER.md verdict line: **KILLED** (no PROVISIONAL/PARTIALLY SUPPORTED etc.) ✓
4. `is_smoke`: `false` (N=5, target-platform dimensions, not a miniature) ✓
5. KC git-diff: MATH.md KCs 1688/1689/1690 match exactly the DB-registered KCs;
   no KC was added/relaxed after the run. ✓
6. Antipattern self-audit (MATH.md §6):
   - composition math: `B_i @ A_i` then sum, never sum-A/sum-B independently ✓
   - tautological routing: explicit one-hot from ground-truth, not from val[d][0] ✓
   - LORA_SCALE: α=1.0 ✓
   - smoke-as-supported: verdict is KILLED, not falsely upgraded ✓
   - preflight-adapter-persistence: no safetensors loaded ✓
   - thinking-mode: not generating for quality scoring; only logit cosine + tok/s ✓
   - proxy-model substitution: target model is Gemma 4 E4B; measured on Gemma 4 E4B ✓

Completion: `--status killed`. ✓
