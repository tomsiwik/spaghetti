# MATH.md — exp_g4_layerwise_adapter_sensitivity

**Type:** guided-exploration (proven framework: residual-stream perturbation propagation + ShortGPT block-influence; unknown: per-layer sensitivity profile of Gemma 4 E4B)

**Base model:** `mlx-community/gemma-4-e4b-it-4bit` (42 decoder layers, hidden_size=2560)
**mlx version:** 0.31.1 / mlx-lm 0.31.2

## 1. Failure Mode Identified

Adapter placement in Pierre defaults to `v_proj+o_proj` per F#627 across **all layers** (uniform). This is operationally convenient but assumes either (a) all layers are equally informative for adapter learning, or (b) the few important layers are guaranteed to be in the trained set.

If neither holds — i.e. there is a small contiguous band of high-sensitivity layers and the rest are near-redundant — then:
- Adapter parameter budget is wasted on low-sensitivity layers.
- Future adapter-placement experiments (e.g. layer-selective LoRA) are designed without an empirical sensitivity map.

The killable failure mode is the null hypothesis: **all 42 layers are equally sensitive to perturbation**, in which case targeted-band adapter placement provides no advantage and uniform placement is optimal.

## 2. Prior Math (Cited)

- **ShortGPT (arxiv:2403.03853)** — Block Influence: BI_l = E_x [1 − cos(x_l, x_{l+1})] measures how much layer l alters the residual stream. Layers with low BI can be removed with minimal PPL impact. Empirically, BI is highly non-uniform across depth in Llama/Mistral.
- **LLaMA layer importance (arxiv:2308.04949)** — empirical: removing a single layer from middle of LLaMA-2 produces ΔPPL spanning ~3 orders of magnitude across layer index.
- **Operator perturbation theory (Higham, "Accuracy and Stability of Numerical Algorithms" Ch. 7)** — for a residual transformer x_{l+1} = x_l + f_l(x_l), perturbing the layer's contribution by δ_l propagates linearly through the residual stream. The output perturbation magnitude is bounded by ‖δ_N‖ ≤ ‖δ_l‖ · ∏_{m=l+1}^{N} (1 + L_m), where L_m is the local Lipschitz constant of f_m. For Gemma-4-style transformers with bounded RMSNorm and softmax-stable attention, ∏(1+L_m) is empirically bounded by a small constant (≤ ~5).

## 3. Theorem (Layer-Sensitivity Differentiation)

**Setup.** Let f_l : ℝ^d → ℝ^d denote the contribution of decoder layer l in Gemma 4 E4B (i.e. x_{l+1} − x_l after the layer's full block has been applied; this is the sum of attention-residual + MLP-residual contributions). Let M(η) denote the model with η ~ N(0, I/d) injected as additive perturbation at the output of layer l, scaled to relative magnitude ε:

  out'_l = x_{l+1} + ε · ‖x_{l+1} − x_l‖₂ · η / √d

Let PPL(M(η)) denote the perplexity of M(η) on a fixed evaluation distribution D.

Define the **per-layer sensitivity** s_l := E_η [ PPL(M_l(η)) ] − PPL(M_baseline).

**Theorem (informal).** If layers in Gemma 4 E4B carry differential semantic content — i.e. there exist layers whose f_l adds high-information-density signal (e.g. disambiguation, copying, induction) and layers whose f_l adds redundant or refining signal — then s_l is non-uniform across l ∈ {0, ..., 41}.

**Proof sketch.** The output perturbation magnitude ‖δ_N‖ at the logits is approximately constant across choice of l (bounded by the operator-perturbation product, which is order-1 for stable transformers). Hence the *direction* of the perturbation reaching the logits is roughly random with similar magnitude regardless of l. The PPL change therefore reflects how much *information content* of f_l is destroyed by the random perturbation — a layer carrying a precise feature (e.g. an induction head's copying contribution) loses that feature, while a layer carrying redundant content suffers a smaller PPL change because subsequent layers can recover.

Formally: let P(x | layer-l output) denote the predictive distribution given a clean residual stream up through layer l. Then ΔKL(P‖P') ∝ Var_η[P'(η)] which is monotone in I(f_l(x); next-token | x_l) — the conditional mutual information of the layer's contribution. This information varies across layers in trained transformers (well-established empirically in mechanistic-interpretability literature: induction heads, copying heads, attention heads with specific roles).

**QED** (modulo empirical predictions in §4).

## 4. Predictions

P1. **Coefficient of variation** CV(s_l) := σ(s)/μ(s) > 0.30 across the 42 layers.
P2. **Range:** max(s_l) / min(s_l) > 3.0 (top layer at least 3× more sensitive than bottom layer).
P3. **Contiguity:** the top-7 most-sensitive layers form ≤3 contiguous bands (informational hubs are clustered, not scattered).
P4. **Most-sensitive band intersects [16, 31]** — middle-late layers, consistent with the F#627 / Todd 2310.15213 pattern that semantic content peaks in middle-late.

## 5. Pre-registered Kill Criteria (per F#666 — both target-paired)

We pair a structural/proxy KC with a target/behavioral KC. The original DB KC #1919 is the proxy.

**K1919 (proxy — structural, refined to the operational measurable):**
  Statement: "All layers equally sensitive to adapter perturbation (no layer specialization)"
  Operationalization: CV(s_l) across 42 layers, where s_l := PPL(perturbed at layer l) − PPL(baseline).
  PASS condition: CV(s_l) > 0.30.
  FAIL condition: CV(s_l) ≤ 0.30 (sensitivity uniform).

**K_NEW (target — actionable layer-band, F#666 paired):**
  Statement: "Sensitivity profile reveals an actionable contiguous layer band for targeted adapter placement"
  Operationalization: Take the set L_top = top-7 layers by s_l. Compute connected components of L_top in the layer-index graph (consecutive integers form one band).
  PASS condition: number of bands ≤ 3 AND largest band has ≥ 3 contiguous layers.
  FAIL condition: top-7 are scattered (≥4 bands) OR no contiguous block of ≥3.

**Verdict logic (F#666):**
- Both PASS → SUPPORTED. Sensitivity is non-uniform AND clustered into actionable bands.
- Both FAIL → KILLED. Sensitivity is uniform OR scattered, no actionable adapter-placement signal.
- Proxy PASS + target FAIL → PROVISIONAL. Layers differ in sensitivity, but top-sensitive layers are scattered (no clean band). Finding: "no contiguous-band structure" — flag for richer placement strategies (e.g. importance-weighted across all layers).
- Proxy FAIL + target PASS → PROVISIONAL. Tautological proxy: CV is below threshold but a contiguous band still emerges from rank order. Means CV threshold of 0.30 was too strict for this measurement.

## 6. Experimental Design

### Dataset
- 30 rows from `micro/models/exp_p1_t2_single_domain_training/data/medical/valid.jsonl` (medical 4-option MCQ format, chat-templated, MAX_SEQ_LEN=512). Same source as the prior pruning experiment.
- Reason for size: 42 layers × (1 baseline + 1 perturbed) ≈ 43 forward passes per row × 30 rows = 1290 forward passes. At ~1.5 s/forward on M5 Pro 4-bit, total ≈ 32 min. Within 2h budget.

### Perturbation procedure (per-layer)
For each l in 0..41:
1. Hold baseline model.
2. Monkey-patch layer l's `__call__` so that after computing `(h, shared_kv, offset)` it returns `(h + ε · δ_norm · noise, shared_kv, offset)`, where:
   - `δ_norm = ||h - x_input||₂ / √d` — per-token norm of the layer's contribution
   - `noise ~ N(0, I)` of shape h, fresh seeded per row (deterministic via mx.random.seed)
   - `ε = 0.10` (10% relative perturbation; chosen so single-layer perturbation produces measurable but not catastrophic ΔPPL)
3. Compute PPL on 30 rows, teacher-forcing.
4. Restore the original `__call__`.

### Random control (sanity check)
At end, also measure PPL with **no perturbation** (baseline) to compute s_l := PPL_l − PPL_base, and PPL with perturbation **outside** the network (input embeddings) for a noise-floor control.

### Determinism
`mx.random.seed(42)` per layer; same noise tensor reproducible. The same eval batch is used for all 42 layers.

## 7. Antipattern Scan (pre-flight)

- composition math bug — N/A (no LoRA composition in this experiment)
- LORA_SCALE=20 — N/A
- shutil.copy as new adapter — N/A
- hardcoded "pass": True — KCs computed from CV/contiguity, not hardcoded
- eval template truncation — chat template applied; MAX_SEQ_LEN 512 chosen to fit medical prompts
- proxy model — uses the actual `mlx-community/gemma-4-e4b-it-4bit` (matches base model claim)
- KC measures wrong object — s_l is computed on PPL change, which is the operationalization in §5
- composition math — N/A
- KC swap after run — KCs locked in this MATH.md before run_experiment.py is invoked

## 8. Assumptions logged

- Operator-perturbation product ∏(1+L_m) ≈ O(1) across Gemma 4 layers (cited Higham; not re-derived here).
- 30 rows is sufficient for CV estimation with 42 samples (CV-of-mean has SE ≈ CV/√30 ≈ 0.05, giving us margin to discriminate CV>0.30 vs <0.30).
- ε=0.10 lies in the linear-perturbation regime (PPL change roughly proportional to ε; not so large that the model diverges to garbage).
- Medical-MCQ teacher-forcing PPL is a usable scalar response variable — the experiment is about layer ranking, not absolute behavioral quality. PPL r=0.08 task-correlation caveat applies to absolute claims, not relative ordering across layers.
