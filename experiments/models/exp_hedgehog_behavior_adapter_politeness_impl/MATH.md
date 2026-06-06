# MATH.md — exp_hedgehog_behavior_adapter_politeness

**Claim:** Per-layer cosine-similarity distillation between (a) base Gemma 4 E4B under a polite-teacher system prompt and (b) base + rank-8 LoRA under a neutral prompt trains an adapter that encodes politeness as attention-routing perturbation, preserving base generation capacity on unrelated tasks.

---

## 0. Platform skills + versions (PLAN.md §1011/1012)

- **Skills required before coding:** `/mlx-dev` (array/nn/training patterns, `mx.eval` discipline, `mx.clear_cache` between phases, `nn.value_and_grad` functional gradients) + `/fast-mlx` (compile, lazy eval, bandwidth-aware kernels). Both MUST be invoked before any MLX training-loop code lands in `run_experiment.py`. This is a hard gate per Finding #673 and the 2026-04-17 audit.
- **mlx-lm version pin:** target `mlx-lm 0.31.x` (current pueue venv); API breakage risk between 0.21 and 0.31 has silently broken prior experiments — cite the exact installed version in `results.json["mlx_lm_version"]`.
- **Base model:** `mlx-community/gemma-4-e4b-it-4bit` (exact HF repo id). No proxy substitution — the reviewer's (m) check blocks on any model-downgrade.
- **Adapter targets:** `v_proj + o_proj` (Pierre F#627 — proven Gemma 4 E4B adapter target).
- **Scope-preservation (antipattern-t).** If the Hedgehog training loop (Phase B) cannot land in a single iteration, file PROVISIONAL; do NOT silently substitute a cross-entropy SFT objective. Doing so would change what K1–K4 measure.

## 1. Failure mode

The degenerate behavior to rule out: "Per-layer cos-sim distillation captures prompt-artifact routing, not politeness-relevant routing. The adapter learns to route toward the teacher's *system-prompt token positions* rather than toward polite *semantic directions*. Under this failure, politeness improves on prompts structurally similar to the teacher distribution but degrades or is null on novel prompts — and base tasks drop because spurious routing shifts leak into unrelated contexts." K2 ≤ +5pp and K3 ≥ 5pp drop under this failure.

## 2. Cited prior math / findings

- **Moudgil et al. 2604.14191 §3.1, eq. 6 (Hedgehog φ_MLP):** learnable feature map trained by matching teacher attention output vs. Hedgehog-linearized version via cosine similarity. Here we reuse the *loss form*, not the linearization target — we match teacher-attention vs. adapter-modified-attention, both softmax.
- **Zhang et al. 2402.04347:** cosine loss recovers 99% of softmax attention behavior with MLP feature maps. Evidence that per-layer cos-sim is a dense training signal sufficient to fit attention-space behavior.
- **Pierre F#627:** rank-6 LoRA on `v_proj+o_proj` of Gemma 4 E4B captures domain specialization end-to-end — parameterization is sufficient for behavior encoding at this rank.
- **Pierre F#666 (target-gated kill):** a cosine structural KC is a proxy; behavior change (auto-judge politeness score) is the target.

## 3. Theorem (informal)

Let `A_l(x; θ)` denote the output of attention block `l` on input `x` under parameters `θ`. Let `θ_base` be frozen base Gemma 4 E4B weights and `Δθ` the rank-8 LoRA perturbation on `v_proj, o_proj`. Let `π_polite` be the polite system-prompt prefix and `π_null` the empty prefix.

**Theorem.** There exists `Δθ` with ‖Δθ‖ bounded (r=8) such that:

$$
\mathcal{L}(\Delta\theta) \;=\; \mathbb{E}_{x \sim D_{\text{neutral}}} \sum_{l=1}^{L} \bigl(1 - \cos\bigl(A_l(\pi_{\text{polite}} \oplus x;\, \theta_{\text{base}}),\, A_l(\pi_{\text{null}} \oplus x;\, \theta_{\text{base}} + \Delta\theta)\bigr)\bigr)
$$

is minimized to `L̄ < 0.15 · L` (K1 target: mean per-layer cos > 0.85), AND the induced policy `p(y | x; θ_base + Δθ)` produces outputs judged more polite than base on unrelated-to-teacher prompts (K2: judge-score Δ ≥ +20%), AND base-task accuracy on MMLU/HumanEval drops ≤ 3pp (K3).

**Proof sketch.**
1. *Expressivity (existence of Δθ).* Zhang 2024 showed a single-MLP feature map suffices to match softmax attention output with cos > 0.99. A rank-8 LoRA on `v_proj+o_proj` has strictly more degrees of freedom than a per-head MLP feature map at the same dimension, so a solution with K1 PASS exists.
2. *Behavior transfer (K1 ⇒ K2 on related prompts).* If per-layer attention outputs match the polite-teacher within cos > 0.85, downstream logits match within bounded KL (attention is Lipschitz in its output through the residual stream; rescaling factors are bounded by the LayerNorm chain). High attention-output similarity → high output-distribution similarity → similar polite-token preference.
3. *Non-interference (K3).* LoRA on `v_proj+o_proj` at rank 8 is ≪ base rank. Per Pierre F#627 at r=6, domain adapters did not cause cross-task collapse on unrelated subjects. Same rank-budget argument: perturbation magnitude is bounded and mostly orthogonal to unrelated tasks' principal directions.
4. *Behavior NOT information (K4 ablation).* If the adapter encoded token-level information, removing the polite system prompt from teacher forward passes would barely change the learned routing (information is in the corpus, not the prompt). If the adapter encodes *routing behavior induced by the polite prompt*, ablating the prompt collapses the teacher signal and K2 regresses by ≥10pp. K4 PASS is evidence for the "behavior" interpretation.

QED sketch.

## 4. Kill-criterion map

| KC | Measured quantity | Threshold | Type |
|---|---|---|---|
| K1 | mean over layers of cos(A_l_teacher, A_l_student) on 100 held-out neutral prompts | > 0.85 | structural proxy |
| K2 | auto-judge politeness Δ = score(student) − score(base), neutral prompts n=100 | ≥ +20 (0-100 scale) | target (pair K1) |
| K3a | MMLU subset accuracy under adapter | drop < 3pp vs base | target non-interference |
| K3b | HumanEval pass@1 under adapter | drop < 3pp vs base | target non-interference |
| K4 | K2 regression under teacher-without-polite-prompt training | ≥ 10pp regression | target ablation |

K1, K3a, K3b use Gemma 4 E4B 4-bit MLX. K2 auto-judge: GPT-4 or Claude 3.7 rubric (0=rude, 100=perfectly polite); pair-compare student-vs-base outputs; score = % prompts where judge prefers student.

## 5. Predicted measurements

- K1: cos ∈ [0.88, 0.94] per layer, mean ≈ 0.91
- K2: ΔJudge ∈ [+22, +35] pp on neutral prompts
- K3a: MMLU drop ≤ 1.5pp (per F#627 non-interference pattern)
- K3b: HumanEval drop ≤ 2pp
- K4: under teacher-prompt-ablation run, K2 ∈ [+0, +8] pp (collapses — confirms behavior-not-info)

If K1 PASS but K2 FAIL, the finding is: **attention-output cos similarity is not sufficient for behavior transfer at this scale** — sharpens the proxy/target distinction from F#666.
