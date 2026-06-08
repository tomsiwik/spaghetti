# MATH.md — Temporal localization of off-domain LoRA interference

## 0. Environment / provenance
- Base model (frozen): `mlx-community/gemma-4-e4b-it-4bit` (Gemma-4 E4B, 4-bit).
- `mlx-lm` version: **0.31.2** (`mlx_lm.__version__`).
- Adapters (REAL, pre-trained, verified to load and to differ):
  - math (on-domain): `data/adapters/math/adapters.safetensors` — r=6, scale=6.0, target `self_attn.q_proj`, all 42 layers (84 tensors).
  - code (off-domain): `experiments/models/exp_composition_residual_analysis/adapter_code.safetensors` — same shape/recipe, distinct weights (mean |Δ| layer0 lora_a ≈ 0.015, norms 1.68 vs 1.58).
- Eval data: real GSM8K test split (`experiments/models/exp_p9_ttlora_polar_hybrid/data/gsm8k_test.jsonl`, 1319 items with `question` / `answer` where answer ends `#### N`). We use the first N_EVAL items, deterministic order.
- LoRA composition: `ΔW = Σ_i scale·(B_i @ A_i)` applied as `h += scale·(x @ A_i) @ B_i` per adapter i — never `(ΣB)(ΣA)`. scale = 6.0 ≤ 8. ✓

## 1. Failure mode being tested
Prior work (Finding #827, #837, #844; NRE / Fisher-Rao ceiling) treated off-domain
LoRA interference as a **static weight-pair property**: composing a code adapter onto a
frozen Gemma-4 that also carries a math adapter degrades GSM8K accuracy, and every fix
searched weight space (TIES `arXiv:2306.01708`, orthogonality, routing) and hit a wall.

The frame-break hypothesis: interference is **temporally localized**. It is not uniform
across decode steps; it is concentrated on a small set of **high-base-entropy "choice-point"
decode steps** (reasoning pivots and the answer token), where the next-token distribution
of the *frozen base* is flat and a small off-domain perturbation flips the trajectory.
At low-entropy steps (the base is already near-deterministic — copying, formatting,
within-number digits) an off-domain perturbation is absorbed and does not change the argmax.

## 2. Theorem (why entropy-localized interference is plausible)

**Setup.** At decode step t with prefix context, let `z_t ∈ R^V` be the base (no-adapter)
logits and `p_t = softmax(z_t)`. Composition adds a perturbation `δ_t` to the logits
(the net effect of the code adapter's `B_c@A_c` propagated to the head). The next token is
`argmax(z_t + δ_t)` for the composed model vs `argmax(z_t)` for the on-domain-only model.

**Claim.** The probability that perturbation `δ_t` flips the argmax is a monotonically
increasing function of the base entropy `H(p_t)`, and is ≈0 when `H(p_t)` is small.

**Argument.** A flip at step t requires some token j to overtake the base-argmax token `a`:
`z_t[j] + δ_t[j] > z_t[a] + δ_t[a]`, i.e. the base **margin** `m_t = z_t[a] − max_{j≠a} z_t[j]`
must satisfy `m_t < δ_t[a] − δ_t[j] ≤ 2‖δ_t‖_∞`. The margin `m_t` and the entropy `H(p_t)`
are inversely related: low entropy ⇒ one logit dominates ⇒ large margin ⇒ flip-proof against
bounded `‖δ_t‖_∞`; high entropy ⇒ flat distribution ⇒ small margin ⇒ flippable. With
`‖δ_t‖_∞` roughly position-independent (same adapter rank/scale everywhere), flips
concentrate where `m_t` is small ⇔ `H(p_t)` is large. ∎ (heuristic bound; behavioral test below.)

**Corollary (the intervention).** If interference flips concentrate at high-base-entropy
steps, then **zeroing the off-domain (code) adapter only at the top-q% highest-base-entropy
decode steps** (q small) removes most flips and recovers most lost on-domain accuracy, while
the math adapter stays on at every step. If instead interference is uniform, gating q% of
steps recovers only ~q% of the loss, and gating *random* q% does just as well (it's dropout).

## 3. Gate-location definition (pre-registered)
- **Base entropy** at step t = Shannon entropy (nats) of the **frozen base** model's
  next-token distribution at the SAME prefix the composed model has decoded so far, computed
  from base logits with no adapter active. This makes the gate location a property of the base,
  independent of which adapter is on.
- **top-5% per-sequence**: for each evaluated question we generate the entropy-gate arm's
  trajectory while recording base entropy at every step; the gate fires at a step iff that
  step's base entropy is in the **top 5% of that sequence's own per-step base-entropy values**
  (rank-based, per-sequence threshold). This is the recommended per-sequence rule (§ spec).
  Because the threshold is per-sequence and rank-based, the gated fraction is ≤5% by construction
  for that arm (ties resolved to stay ≤5%); K3 is still measured empirically as the realized fraction.
- **random-gate (K2 control)**: gate the SAME number of steps as the entropy-gate arm chose
  for that sequence, but at uniformly random step positions (fixed seed 42), NOT the
  high-entropy ones. Equal count ⇒ isolates "choice-point-specific" from "just dropout."

### Two-pass realization (honest, non-circular)
Per question, deterministic greedy decoding (temp 0, argmax), `enable_thinking=True`,
max_tokens fixed, same prompt/seed across all arms:
- **Pass 1 (compose, all steps)** = arm C. At each step record the base entropy at that prefix
  (extra base forward over the composed prefix, no adapters) → the per-step base-entropy profile
  along C's trajectory, and the per-sequence top-5% step set S_hi and its size k.
- Arm **D (entropy-gate)**: re-decode the same question; the code adapter is gated to 0 at the
  steps whose realized base entropy lands in the per-sequence top-5% (computed online with the
  same per-sequence threshold derived from C's profile, applied by step index). Math stays on.
- Arm **E (random-gate)**: re-decode; gate code at k random step indices (seed 42), math on.
- Arms **A (base)** and **B (math-only)** decode independently (no code adapter present).

All five arms share: identical prompts, greedy decode, max_tokens, tokenizer template
(`enable_thinking=True`), and the same GSM8K item ordering. Accuracy = exact match on the
extracted final numeric answer (`#### N`, else last number in the response), the behavioral
target metric (NOT perplexity; PPL↔accuracy r≈0.08, Finding #666).

## 4. Quantitative predictions
- B − C drop (interference): predicted ≈ 12–14 pp on-domain GSM8K accuracy lost by composition.
  (If |B−C| is small, the premise is not reproduced — report honestly, §spec.)
- Recovery fraction of arm X: `recov(X) = (acc_X − acc_C) / (acc_B − acc_C)`.
- Predicted: `recov(D) ≥ 0.50` (entropy-gate recovers ≥half the loss).
- Predicted: `acc_D − acc_E ≥ 2 pp` (choice-point-specific beats equal-count random dropout).
- Predicted gated fraction ≤ 5% of decode steps.

## 5. Pre-registered kill criteria (do NOT edit)
- **K1 (id 2288, target/behavioral):** entropy-gating the code adapter to scale=0 at the
  top-5% high-base-entropy decode steps recovers **< 50%** of the lost on-domain accuracy,
  i.e. `recov(D) < 0.50` → hypothesis FALSE (interference not temporally concentrated).
  PASS (hypothesis survives) iff `recov(D) ≥ 0.50`.
- **K2 (id 2289, target/behavioral, LOAD-BEARING CONTROL):** entropy-gated suppression recovers
  no more than equal-count random gating, `acc_D − acc_E < 2 pp` → it's just dropout → FALSE.
  PASS iff `acc_D − acc_E ≥ 2 pp`.
- **K3 (id 2290, structural):** the gate must touch **≤ 5%** of decode steps; if recovery needs
  **> 15%** of tokens, interference is NOT concentrated → reframe rejected.
  PASS iff realized gated fraction ≤ 0.05 (and certainly fails if it would need >0.15).

**Verdict rule.** SUPPORTED iff K1 PASS ∧ K2 PASS ∧ K3 PASS (all_pass). Otherwise KILLED.
A clean kill is a valid outcome. `is_smoke=false` for any non-provisional verdict.

## 6. References
- `arXiv:2306.01708` (TIES-Merging) — the static weight-space merge ceiling this inverts.
- Finding #827 / #837 / #844 — interference is real, behavioral, non-uniform (license for temporal structure).
- Finding #666 — target-gated kill rule: behavioral accuracy is the target metric, not PPL.
