# MATH — Frozen base q_proj as carrier wave vs load-bearing knowledge

**Experiment:** `exp_spark_base_carrier_ablation`
**Base model:** `mlx-community/gemma-4-e4b-it-4bit` (Gemma 4 E4B, 4-bit, 42 blocks)
**Adapters:** `exp_composition_residual_analysis/adapter_{code,math,medical}.safetensors` — q_proj-only LoRA, r=6, all 42 layers, scale=6.0 (F#627 recipe).
**mlx-lm:** 0.31.2 (q_proj is `mlx.nn.layers.quantized.QuantizedLinear`, 2560→2048, group_size=64, bits=4).
**Type:** Frontier extension — inverts the STATUS.md §3 "frozen base = sacred knowledge" invariant into a measurable knob.

---

## 1. Setup and definitions

For a single attention block ℓ, the query projection is
```
q_ℓ(x) = W_q^(ℓ) x                    (frozen 4-bit base, W_q ∈ ℝ^{2048×2560})
```
A trained domain adapter d adds a fixed low-rank delta to the SAME projection:
```
ΔW_q^(d,ℓ) = s · B_d^(ℓ) A_d^(ℓ)ᵀ      s = LORA_SCALE = 6,  A∈ℝ^{2560×6}, B∈ℝ^{2048×6}
q_ℓ^(d)(x) = W_q^(ℓ) x + ΔW_q^(d,ℓ) x
```

**The attenuation knob.** We scale ONLY the frozen base contribution inside the q_proj layers the adapter touches, leaving the LoRA delta and every other layer (k/v/o/MLP/embeddings/lm_head) untouched:
```
q_ℓ^(d,α)(x) = α · W_q^(ℓ) x + ΔW_q^(d,ℓ) x          α ∈ {1.0, 0.5, 0.25, 0.0}     (★)
```
α=1 is the standard adapted model. α=0 deletes the base query signal in exactly the 42 q_proj layers but keeps the adapter delta there, and keeps the FULL base everywhere else.

Let `A_d(α)` = on-domain task accuracy of model (★) on domain d's held-out slice:
- d=math: GSM8K-style final-answer exact match, `#### N` numeric extraction, n=50.
- d=code: HumanEval-style — but these items have no executable unit tests, so we score **gold-answer continuation accuracy**: greedy decode, exact-match of the first non-trivial gold line (defined below), n=25.
- d=medical: MedQA-style multiple-choice — extract the chosen letter A/B/C/D, exact match, n=50.

Define the **retention ratio**
```
R_d(α) = A_d(α) / A_d(1)        R_mean(α) = mean_d R_d(α).
```

---

## 2. Theorem (carrier vs load-bearing decomposition)

> **Theorem.** Decompose the adapted query into a base-carried component and an adapter-carried component, q^(d,α) = α·(W_q x) + (ΔW_q x). Then exactly one of two regimes holds for the on-domain behavior as α→0:
>
> **(C) Carrier regime** — the base q_proj signal acts as a near-multiplicative gain/bias whose *direction* is not used by the downstream domain-specific computation; the functional load on-domain is carried by ΔW_q together with the untouched k/v/o/MLP path. Then attenuating it leaves on-domain accuracy approximately invariant: R_d(α) ≈ 1 for all α, in particular **R_mean(0) ≥ 0.80 and every R_d(0) ≥ 0.65**.
>
> **(L) Load-bearing regime** — the frozen base query direction W_q x is itself the substrate the adapter perturbs (LoRA edits a low-rank *subspace of* the base map, arXiv:2402.09353); removing it collapses attention routing the adapter relies on. Then accuracy degrades monotonically and steeply: A_d(α) → A_chance as α→0, giving **R_mean(0) < 0.80 or some R_d(0) < 0.65**.

**Proof sketch (constructive bound).** Attention logits are `(q_ℓ^(d,α) · k_ℓ)/√h`. Write q^(d,α) = α q_base + q_Δ with q_base=W_q x, q_Δ=ΔW_q x. The softmax routing depends on logit *differences* across key positions:
Δlogit(i,j) = (α q_base + q_Δ)·(k_i − k_j)/√h = α·[q_base·(k_i−k_j)] + [q_Δ·(k_i−k_j)], both over √h.
Two limiting structures bound the behavior:

1. **(C) Adapter-dominated routing.** If on-domain the adapter delta supplies the discriminative routing term, i.e. for the tokens that matter |q_Δ·(k_i−k_j)| ≳ |q_base·(k_i−k_j)| at the argmax, then the α·q_base term is a sub-dominant perturbation of the argmax. By the softmax Lipschitz bound, the per-head attention distribution moves by ‖α·q_base/√h‖ in logit space; for the *argmax-preserving* set this leaves the selected value mixture — hence the residual stream feeding the untouched o/MLP path — within an O(α) ball that does not cross a decision boundary for on-domain inputs. Accuracy is then flat in α: R_d(α)=1+O(α·κ_d) with κ_d small ⇒ R_d(0)≈1. The frozen base is a *carrier*: it sets an overall query scale the adapter rides on but does not encode the on-domain routing.

2. **(L) Base-dominated routing.** If instead q_base carries the discriminative term (q_Δ is a low-rank *correction* living in span of the base map per arXiv:2402.09353, so ΔW_q = W_q · P + E with P a low-rank projector), then setting α=0 deletes the carrier the correction was defined relative to: q^(d,0)=ΔW_q x = (W_q P + E)x. Since ΔW_q is rank ≤ 6 ≪ 2048 and trained as a *delta on top of* α=1, its standalone routing is mis-scaled and mis-centered, so argmax flips on a constant fraction of on-domain tokens ⇒ A_d(0)→A_chance, R_d(0) ≪ 1.

These are mutually exclusive on-domain (the discriminative routing term at the relevant argmax is either base-dominated or adapter-dominated; the borderline α-curve shape distinguishes them). The experiment measures the full curve A_d(α) for α∈{1,0.5,0.25,0} and the endpoint R(0), which discriminates (C) from (L). **QED (constructive: the α-knob (★) realizes exactly the decomposition and the bound is the softmax-argmax stability argument).**

---

## 3. Predicted curves (discriminating prediction)

Chance accuracy per metric (used to interpret R, not as a KC): math `A_chance≈0` (open numeric), code `A_chance≈0` (exact line), medical `A_chance≈0.25` (4-way MCQ).

| α | Carrier (C) prediction R_d(α) | Load-bearing (L) prediction R_d(α) |
|---|---|---|
| 1.00 | 1.00 | 1.00 |
| 0.50 | ≈ 0.95–1.00 | ≈ 0.6–0.8 |
| 0.25 | ≈ 0.90–1.00 | ≈ 0.3–0.6 |
| 0.00 | **≥ 0.80** (mean), each **≥ 0.65** | **< 0.65** (collapse toward chance) |

The two hypotheses make *opposite* predictions at α=0, and a *different curvature* in between (C is flat, L is monotone-steep). The run records all four points so the verdict is not a single-point artifact.

---

## 4. Pre-registered kill criteria (target-gated, Finding #666)

This is a **single behavioral target metric** (task accuracy), so the kill rule is direct.

- **K1 (target, behavioral) — pre-registered DB kill-id 2294.**
  `R_mean(0) = mean_d[ A_d(0)/A_d(1) ] < 0.80`  **OR**  any domain `R_d(0) < 0.65`.
  - **K1 FAIL (criterion met) ⇒ verdict KILLED**: base is load-bearing where adapters act; the carrier hypothesis is refuted.
  - **K1 PASS (R_mean(0) ≥ 0.80 AND all R_d(0) ≥ 0.65) ⇒ verdict SUPPORTED**: base q_proj is a carrier wave; the "frozen base = sacred" invariant is inverted in q_proj.

- **K2 (validity guard, not a proxy-KC).** The adapted reference must be non-degenerate: `A_d(1) ≥ A_chance,d + 0.10` for at least 2 of 3 domains (the adapter must actually work at α=1, else R is undefined / divide-by-near-zero). If K2 fails the run is **provisional/invalid** (cannot evaluate retention against a dead reference), NOT a kill — this prevents a vacuous SUPPORTED from a model that was already at chance.

**Verdict logic emitted by the script:**
- K2 fail → `verdict="PROVISIONAL"`, `is_smoke=false`, note "reference adapters degenerate".
- K2 pass AND K1 criterion met (R_mean(0)<0.80 OR any R_d(0)<0.65) → `verdict="KILLED"`.
- K2 pass AND K1 criterion NOT met → `verdict="SUPPORTED"`.

`all_pass` = (verdict=="SUPPORTED").

---

## 5. References
- arXiv:2402.09353 — LoRA edits a low-rank subspace *of the base map* (basis for the (L) regime ΔW=W·P+E form).
- Finding #627 — solo q_proj r6 adapters give on-domain lifts +22/+48/+62pp on code/math/medical (supplies the adapters; guarantees A_d(1)≫chance, supports K2).
- STATUS.md §3 — the "frozen base = immutable knowledge to protect" invariant being inverted.
- Finding #666 — target-gated kill discipline (here the sole KC is already a behavioral target).

## 6. No-mock / safety commitments
- Real 4-bit Gemma 4 E4B loaded via `mlx_lm.load`; real safetensors adapters loaded from disk; real greedy decoding for accuracy. `is_smoke=false`.
- Composition is per-layer additive on the SAME projection: `q = α·W_q x + s·(x@A)@Bᵀ` — never `(ΣB)(ΣA)`; one adapter at a time (no multi-adapter sum here).
- LORA_SCALE = 6 ≤ 8.
- Per-sample evaluation (each held-out item scored independently); no per-domain routing shortcut.
- Base attenuation realized by a `nn.Module` wrapper installed via `setattr` on the parent attention module (NEVER override `__call__` on an instance — mem-antipattern-call-override-silent-bypass).
