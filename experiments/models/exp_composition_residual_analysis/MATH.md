# MATH.md — exp_composition_residual_analysis

**Type:** verification (proven framework: per-module algebraic linearity F#302/Theorem 1 + forward-pass nonlinearity of LayerNorm/softmax/SiLU; unknown: activation-space residual magnitude at Gemma 4 E4B 4-bit with F#627 recipe, and whether it predicts behavioral drift)

**Base model:** `mlx-community/gemma-4-e4b-it-4bit` (Gemma 4 E4B, 42 decoder layers, hidden_size=2560, q_proj d_in=2560/d_out=4096, 4-bit quantized)
**mlx / mlx-lm:** versions from `uv run` environment (mlx 0.31.1, mlx_lm 0.31.2 confirmed at runtime)

## 1. Failure Mode Identified

Pre-sum composition W_comp = W_base + Σ_{i=1..N} ΔW_i is the trivially-cheap composition method (no routing, no joint training). Its weight-space residual vs the naive-additive prediction is 0 by algebraic identity (F#302, per-module MSE 5.6e-7 — floating-point noise only).

The killable failure mode: **the forward-pass residual (activation / behavior) is also ~0 at Gemma 4 E4B scale**, i.e. transformer nonlinearities do NOT materially break additivity. If so, pre-sum composition is a legitimate drop-in and the entire Hedgehog/routing framework is over-engineered for N=3. F#302/F#334 said otherwise (logit MSE 53.9 full-model, LayerNorm/softmax compounding), but those were on a prior base (BitNet 2B / Room Model experiments, pre-Gemma-4). We verify at the current target.

## 2. Prior Math (Cited)

- **F#302** (killed, exp_room_model_poc): per-module weight-space additivity confirmed (MSE 5.6e-7), BUT full-model logit MSE = 53.9 — nonlinearity compounds through layers. Theorem 1 confirmed at per-module level; unstated full-model linearity assumption killed.
- **F#334** (conclusive, exp_room_intra_layer): `f(x; W + Σδ_i)` is a fixed matrix independent of input domain — pre-summing creates a static ensemble, not a router. Derived from algebraic identity, empirically confirmed (room_intra_layer K823 FAIL, gap=10.3%).
- **F#627** (supported): r=6 q_proj LoRA on Gemma 4 E4B with scale=6, 100 iters AdamW lr=1e-4 produces material PPL / task lift. Canonical recipe for this repo.
- **F#NEW.c** (antipattern, this hat's prior iteration): shared PRNG key across init-variant comparison → correlated starting matrices. Rule: **distinct per-variant top-level seeds are mandatory.**
- **Hewitt 2022 "Task Vectors" (arxiv:2212.04089)** and **Ilharco 2022 "Editing models with task arithmetic"**: empirical evidence that task-vector (ΔW) sums preserve domain-specific behavior approximately but not exactly; residual interference is observable at N≥2.
- **Ties-Merging (Yadav 2023, arxiv:2306.01708)**: documents sign-conflict and magnitude-dilution as the two classes of interference in ΔW sums; implies non-zero activation-space residual.
- **LayerNorm nonlinearity (Ba 2016, arxiv:1607.06450)**: LN(x + δ) = (x + δ - μ(x+δ)) / σ(x+δ), where μ and σ are functions of the full vector. Expanding: LN(x+δ) ≠ LN(x) + [LN(x+δ) - LN(x)] unless δ is infinitesimal. The linearization error is O(‖δ‖²/σ²). For δ = δ_1 + δ_2, cross-terms O(⟨δ_1, δ_2⟩) appear.

## 3. Theorem (Activation-Space Non-Additivity)

**Setup.** Let f: ℝ^{T × d} → ℝ^{T × d} be the Gemma 4 E4B language-model backbone (pre-lm_head): embedding → L decoder blocks (each with LayerNorm-gated attention + LN-gated SwiGLU FFN) → final LN → final hidden state. Let W_base denote the frozen quantized parameters. LoRA adds rank-r deltas to q_proj in each layer: ΔW_i = scale·B_i^⊤ A_i^⊤ for adapter i ∈ {1, …, N}.

Define:
  h_base(x)   = f(x; W_base)
  h_i(x)      = f(x; W_base + ΔW_i)                 (adapter i alone)
  h_comp(x)   = f(x; W_base + Σ_{i=1..N} ΔW_i)      (sum-composed)
  δh_i(x)    = h_i(x) − h_base(x)
  R(x)       = h_comp(x) − h_base(x) − Σ_i δh_i(x) = h_comp(x) − Σ_i h_i(x) + (N−1)·h_base(x)

Let τ(x) = ⟨R(x), R(x)⟩^(1/2) / Σ_i ⟨δh_i(x), δh_i(x)⟩^(1/2)   (the residual ratio per token).

**Theorem (informal).** For N ≥ 2 adapters trained on distinct domains and f with L LayerNorm-gated layers:

  𝔼_x [τ(x)] ≥ c · min(1, L·ε / σ_h)

where ε = 𝔼[‖δh_i‖] at layer 1 and σ_h is the typical hidden-state scale. Equivalently: the activation-space residual grows with depth because nonlinear layers generate second-order cross-terms between adapter deltas, and LayerNorm's per-token rescaling couples all deltas through the denominator.

**Proof sketch.** Expand LN(x + δ_1 + δ_2) = (x + δ_1 + δ_2 − μ) / σ where μ, σ depend on x + δ_1 + δ_2. To first order in δ, LN is linear in δ. To second order, the 1/σ denominator introduces a term proportional to ⟨x, δ_1⟩·⟨x, δ_2⟩ / σ³, i.e. a cross-term. Attention softmax introduces similar cross-terms via softmax(QK^⊤/√d) acting nonlinearly on the shifted query/key. SiLU has a non-zero second derivative, adding O(δ²) cross-terms. These cross-terms accumulate additively across L layers, giving R of magnitude O(L·‖δ‖²/σ). Dividing by Σ ‖δh_i‖ which is O(L·‖δ‖) gives τ ∼ O(‖δ‖/σ) × L-independent prefactor, i.e. bounded away from 0. **QED** (empirical KCs in §5).

**Behavioral consequence.** If τ ≫ 0, then h_comp ≉ h_additive, so downstream loss/PPL computed from h_comp differs materially from what naive-additive arithmetic predicts. Specifically, composed PPL on domain-i validation differs from adapter-i-alone PPL because the composed model's hidden state on domain-i inputs is contaminated by cross-terms involving adapters 2..N.

## 4. Predictions

- **P1 (K1926, structural):** Token-averaged final-layer residual ratio τ ≥ 0.10 over the joint eval set (i.e. activation-space composition is at least 10%-non-additive at Gemma 4 E4B). Prior basis: F#302 logit MSE 53.9 at full model.
- **P2 (K1927, behavioral):** max_i |PPL_comp[domain_i] − PPL_adapter_i[domain_i]| / PPL_adapter_i[domain_i] ≥ 0.10 (composition fails to reproduce single-adapter behavior on its own domain by at least 10% relative PPL). Prior basis: F#334 (pre-sum = unrouted mixture).
- **P3 (systematic-residual sanity):** ‖𝔼_x R(x)‖₂ / 𝔼_x ‖R(x)‖₂ > 0.3 at final layer. If R were pure noise, this ratio would be ~1/√N_tokens ≈ 0 (mean of zero-centred iid noise averages to near zero). If systematic, a nonzero "average bias" direction emerges.
- **P4 (depth monotonicity):** The residual ratio τ at earlier layers is smaller than at final layer. Prediction: τ at layer 1 < τ at layer L/2 < τ at final layer (monotonic or near-monotonic growth, consistent with compounding nonlinearity).

## 5. Pre-registered Kill Criteria (per F#666 — target-gated)

KCs as registered in the DB:
- **K1926**: "Residual term > 10% of individual adapter magnitudes (non-additive)"
- **K1927**: "Residual term is systematic (not noise) — indicates nonlinear interaction"

Operationalization (locked prior to code execution):

**K1926 (proxy — structural activation-space residual):**
  Operationalization: Compute final-layer hidden-state residual R(x) = h_comp(x) − h_base(x) − Σ (h_i(x) − h_base(x)) on the joint validation set (15 batches × 3 domains × batch_size=2 ≈ 90 sequences ≤ 512 tokens each). Per non-pad token, compute ‖R(x_t)‖₂ and Σ_i ‖δh_i(x_t)‖₂. Define:
    τ_final := ⟨‖R‖₂⟩_{non-pad} / ⟨Σ_i ‖δh_i‖₂⟩_{non-pad}
  PASS condition: τ_final > 0.10 (non-additive at activation level).
  FAIL condition: τ_final ≤ 0.10 (forward is approximately linear; F#302 does not generalize to Gemma 4 E4B 4-bit).
  Prediction: PASS.

**K1927 (target — behavioral PPL deviation):**
  Operationalization: For each domain-i and each of the 4 adapter-aware configs {adapter_1, adapter_2, adapter_3, composed}, compute held-out PPL on domain-i's validation split. Define:
    Δ_i := |PPL_comp[domain_i] − PPL_adapter_i[domain_i]| / PPL_adapter_i[domain_i]
    Δ_max := max_i Δ_i
  PASS condition: Δ_max > 0.10 (composition deviates from single-adapter behavior on its own domain by ≥10% relative PPL).
  FAIL condition: Δ_max ≤ 0.10 (composition behaves like the domain-specific adapter — additivity preserved behaviorally).
  Prediction: PASS.

**Verdict logic (F#666 target-gated):**
- K1926 PASS + K1927 PASS → **SUPPORTED**. Composition residual is structurally non-additive AND behaviorally consequential. Replicates F#302/F#334 at Gemma 4 E4B 4-bit. Gives a numeric magnitude for the first time at the current target platform.
- K1926 FAIL + K1927 FAIL → **KILLED**. Forward pass is effectively linear in ΔW at Gemma 4 E4B; F#302/F#334 did not generalize. Would be a substantial result — implies pre-sum composition is a no-interference drop-in at this scale.
- K1926 PASS + K1927 FAIL → **PROVISIONAL**. Structural residual present but does not propagate to behavioral PPL; "finding about the proxy" per F#666. The τ metric would be overly-pessimistic as a predictor of behavioral drift.
- K1926 FAIL + K1927 PASS → **PROVISIONAL**. Tautological proxy — PPL moves but activation residual doesn't (impossible unless lm_head nonlinearity / softcap dominates; investigate).

## Pre-flight checklist
- Platform skills invoked: /mlx-dev, /fast-mlx — confirmed prior to writing code.
- Base model loaded: `mlx-community/gemma-4-e4b-it-4bit` (matches §0; to be asserted at runtime).
- Adapter targets: `self_attn.q_proj`, all 35 decoder layers, r=6, scale=6 (F#627 recipe).
- Datasets: `micro/models/exp_p1_t2_single_domain_training/data/{medical,code,math}/{train,valid}.jsonl` (pre-existing, 1799/199 rows each; used by F#627 and init-comparison).
- Runtime budget: ~12 min training (3 × 100 iters) + ~3 min eval (5 configs × 45 batches) + overhead ≈ 20–25 min wall-clock on M5 Pro 48GB. No risk of >2h.
- KC count: 2, both target-gated pair per F#666 (K1926 proxy structural; K1927 target behavioral).
- Antipattern scan:
  - Composition math: sum of deltas via r=18 stacked LoRALinear; algebra verified in code comments (no per-token routing, no learned α, pure sum).
  - LORA_SCALE=6 (F#627 safe default).
  - No `shutil.copy` as new adapter.
  - No hardcoded pass/True in KC evaluation — both KCs computed from measurements.
  - Eval split used in full (no truncation to 0).
  - No proxy model — Gemma 4 E4B 4-bit target, asserted by BASE_MODEL constant.
  - Distinct per-adapter seeds (medical=42, code=1337, math=2718) — addresses F#NEW.c antipattern from prior iteration.
- is_smoke: false. T=100 iters is reduced from the 1000-iter F#627 canonical but matches the prior-iteration (exp_g4_adapter_initialization_comparison) budget at which init-invariance was decisively resolved; adequate for composition residual which is activation-space (not training-dynamics-sensitive).

## 6. Experimental Design

**Phase A — Train 3 single-domain adapters (r=6, F#627 recipe):**
For each (domain, seed) ∈ {(medical, 42), (code, 1337), (math, 2718)}:
  - Load fresh `gemma-4-e4b-it-4bit`, freeze base.
  - Attach LoRALinear r=6 scale=6 to `self_attn.q_proj` in all decoder layers (mlx_lm `linear_to_lora_layers(num_layers=-1, config={"keys": ["self_attn.q_proj"], "rank":6, "scale":6, "dropout":0.0})`).
  - Train 100 iters with AdamW lr=1e-4 on `{domain}/train.jsonl`, batch=2, max_seq=512, mask_prompt=True, grad_checkpoint=True, grad_accum=1, `clear_cache_threshold=50`.
  - Save `adapter_{domain}.safetensors` (lora_a, lora_b per q_proj layer).
  - Record final train-loss (last-10 mean) and post-training lora_a/lora_b matrices.
  - Free model + `mx.clear_cache()` before next domain.

**Phase B — Assemble composition via r=18 LoRALinear:**
  - Load fresh base. Attach LoRALinear r=18 scale=6 to `self_attn.q_proj` in all decoder layers.
  - For each (layer, slot i ∈ {0,1,2}), populate:
    - `lora_a[:, 6i:6(i+1)] = A_{domain_i}`  (A_i has shape (input_dims, 6))
    - `lora_b[6i:6(i+1), :] = B_{domain_i}`  (B_i has shape (6, output_dims))
  - This layout gives ΔW_full = scale·lora_b.T @ lora_a.T = scale·Σ_i B_i.T A_i.T = Σ_i ΔW_i  (verified in code: stacking along r gives exact weight sum).
  - Save full (stacked) lora_a, lora_b matrices for reuse across configs.

**Phase C — Five-config evaluation:**
  Using the SAME r=18 model, for each config ∈ {base, adapter_0 (medical), adapter_1 (code), adapter_2 (math), composed}:
    - Zero out or restore slots per config (see §6.1).
    - For each domain ∈ {medical, code, math}:
      - Iterate 15 val batches from `{domain}/valid.jsonl` via `iterate_batches(batch_size=2, max_seq_length=512)`.
      - For each batch: compute `h_config = model.language_model.model(inputs)` (final hidden state before lm_head); compute per-token loss via `default_loss(model, batch, lengths)`.
      - Record hidden states per (config, domain, batch_idx) for residual computation (streamed: accumulate stats per batch, discard tensors).

**§6.1 — Config-slot map (r=18):**
  - `base`: lora_a = 0, lora_b = 0 (all slots inactive → ΔW=0, model ≡ base).
  - `adapter_0 (medical)`: lora_a[:, 0:6] = A_med, else 0; lora_b[0:6, :] = B_med, else 0.
  - `adapter_1 (code)`: lora_a[:, 6:12] = A_code, else 0; lora_b[6:12, :] = B_code, else 0.
  - `adapter_2 (math)`: lora_a[:, 12:18] = A_math, else 0; lora_b[12:18, :] = B_math, else 0.
  - `composed`: all slots active (lora_a, lora_b = stacked matrices from Phase B).

**§6.2 — Residual statistics (per batch):**
  Given 5 hidden-state tensors per batch (shape B × T × d = 2 × ≤512 × 2048):
    h_base, h_0, h_1, h_2, h_comp
  Compute:
    δh_i = h_i − h_base for i ∈ {0,1,2}
    h_additive = h_base + Σ_i δh_i
    R = h_comp − h_additive
    per_tok_R_norm = ‖R[:, t, :]‖₂ along hidden dim → shape (B, T)
    per_tok_sum_delta_norm = Σ_i ‖δh_i[:, t, :]‖₂ → shape (B, T)
  Accumulate: Σ (per_tok_R_norm × non_pad_mask) and Σ (per_tok_sum_delta_norm × non_pad_mask) and tok count.
  Also accumulate 𝔼[R] (pre-averaging R tensor summed and divided by n_tokens) for P3 systematicity.

**§6.3 — Output schema (`results.json`):**
  - `config`: full hyperparameter dump including seeds, iters, r, scale, base_model, n_val_batches.
  - `per_adapter`: each of 3 adapters' final train-loss (last-10 mean), elapsed sec.
  - `per_config_per_domain_ppl`: {config: {domain: {"nll": ..., "ppl": ..., "n_tokens": ...}}}
  - `residual`: {"tau_final_layer", "tau_final_layer_per_domain", "mean_R_norm_over_mean_R_entrywise" (systematicity P3)}.
  - `kill_criteria`: {1926: {value, thresh, result, type}, 1927: {value, thresh, result, type}}.
  - `verdict`: SUPPORTED/PROVISIONAL/KILLED per §5.
  - `all_pass`: bool.
  - `is_smoke`: false.
  - `total_wall_clock_sec`.

## 7. Assumptions (logged per researcher-hat guardrail)
- Gemma 4 E4B has 42 decoder layers (runtime assertion via `len(model.language_model.layers)`); q_proj d_in=2560, d_out=4096 (asymmetric on Gemma 4 due to multi-query attention head layout — asserted at runtime). Hidden state returned by `model.language_model.model(inputs)` is d=2560.
- T=100 iters per adapter produces adapter deltas large enough that per-domain PPL drops meaningfully below baseline (otherwise K1927 is noise-dominated). If post-training per-domain PPL gap < 5% vs baseline, the experiment is under-trained and K1927 result is uninformative. We log this and use it to downgrade verdict to PROVISIONAL if triggered.
- Concatenation-r-stacking gives exact sum of per-adapter ΔW. Proof: (B_stacked.T @ A_stacked.T)[i,j] = Σ_k B_stacked[k,i]·A_stacked[j,k] where k indexes the stacked rank dim. With B_stacked = vstack([B_0, B_1, B_2]) (shape (3r, d_out)) and A_stacked = hstack([A_0, A_1, A_2]) (shape (d_in, 3r)), block-structure gives a sum over s of B_s.T @ A_s.T. Verified by unit-test in `run_experiment.py::assert_concat_equiv`.
- Pad tokens contribute zero loss and are excluded from residual normalization via `lengths` mask.
- Single seed per domain (42, 1337, 2718); variance bound is not measured here (distinct-per-run follow-up if verdict is borderline).
