# MATH — exp_rdt_loop_lora_gemma4_full

Follow-up to `exp_rdt_loop_lora_gemma4` (PROVISIONAL, F#667/673).
Parent built the architecture in smoke mode (synthetic surrogate, `is_smoke=true`)
and deferred the behavioral and dynamical claims. This experiment tests those claims
against the *real* Gemma 4 E4B forward pass with per-layer monkey-patched LoRA.

`is_smoke: false` — no surrogate forward. Every activation in the K-FULL-* KCs
goes through the real `mlx-community/gemma-4-e4b-it-4bit` weights.

Skills invoked before coding: `/mlx-dev` (lazy eval, `mx.eval` discipline, NHWC
format awareness, idiomatic `nn.value_and_grad` pattern, unified-memory phased
cleanup). Gemma 4 source inspected at
`.venv/lib/python3.12/site-packages/mlx_lm/models/gemma4_text.py` to confirm
`self_attn.v_proj` / `self_attn.o_proj` are separate `nn.Linear` modules
(quantised to 4-bit; the wrapper adds a full-precision LoRA delta additively).

## Architecture (real integration)

Frozen base: `mlx-community/gemma-4-e4b-it-4bit`, 42 layers, hidden=2560,
num_heads=8, num_kv_heads=2, head_dim=256. Recurrent block = layers 12..20
(9 consecutive `DecoderLayer`s). Per-loop LoRA rank r=16 on `v_proj` and
`o_proj` per layer, per loop t ∈ {1..T}. Six loop banks. LTI-injection
(F#667 primitive) between successive loop iterations at the block entry.

**Monkey-patch.** For each ℓ ∈ [12,21) and p ∈ {v_proj, o_proj}, the
`nn.Linear` at `model.model.layers[ℓ].self_attn.p` is wrapped in
`LoopLoRALinear(base, [LoRADelta_t for t in 1..N_LOOPS], loop_idx_ref)`
where `loop_idx_ref` is a shared mutable `[int]`. On `__call__`:

    y = base_linear(x) + loop_deltas[loop_idx_ref[0]](x)

and the LoRA delta is `alpha/r · (x @ A.T) @ B.T` with fixed Grassmannian
`A ∈ ℝ^{r × in_dim}` (partition-QR init per F#562) and trainable
`B ∈ ℝ^{out_dim × r}` initialised to zero.

**Loop forward.** `Gemma4TextModel.__call__` is replaced on the loaded
model instance with a version that:
1. Processes layers 0..11 once (standard path).
2. Runs layers 12..20 T times: each iteration sets `loop_idx_ref[0]=t`,
   applies LTI-injection `h ← A_d⊙h + B⊙h0 + block_out` with `h0` = the
   block-entry hidden state, and forwards through all 9 layers.
3. Processes layers 21..41 once.

Only LoRA B matrices (v and o, per layer, per loop) and LTI params
(log_A, log_dt, B) train. Base weights frozen.

## Kill criteria

Pre-registered (locked — no post-hoc modification per PLAN §1 rule (e)).

### Target KCs (behavioural; inherited verbatim from parent)

- **K1740** — Looped-T=3 variant beats base Gemma 4 E4B by ≥ +5pp on
  GSM8K-Hard, n ≥ 200 problems, full eval, greedy decoding, `is_smoke=false`.
- **K1741** — MMLU within 1pp of base Gemma 4 E4B (target-gated pair
  for K1740 per F#666).
- **K1742** — Quality follows saturating exponential
  y(T) = y∞ − (y∞ − y0)·exp(−T/τ) on T ∈ {1..6}, R² > 0.90 on GSM8K-Hard.

### Structural + dynamical KCs (new, from REVIEW-adversarial.md of smoke run)

- **K-FULL-A** — Real block integration: loop-LoRA is wired into the live
  Gemma 4 `DecoderLayer.self_attn.v_proj` and `.o_proj` paths via wrapper,
  not a surrogate `zero-pad` slice. Pass iff every forward through layers
  12..20 at every loop iteration uses the corresponding `B_t` matrix and
  the final block output matches the shape `(batch, seqlen, 2560)`.
- **K-FULL-B** — Both v_proj and o_proj LoRA B-matrices receive non-zero
  gradients on a real Gemma 4 loss. Pass iff `max |∂L/∂B_{v,ℓ,t}| > 1e-6`
  AND `max |∂L/∂B_{o,ℓ,t}| > 1e-6` over the first training batch for all
  ℓ ∈ [12,21), t ∈ [1,N_LOOPS]. (Smoke only exercised v_proj.)
- **K-FULL-C** — K1744 dynamics verified: over ≥ 200 steps of real GSM8K
  token-prediction loss, `max_t ρ(A_d,t) < 1` at every step AND
  `|log_A_final − log_A_init|_max > 1e-4` AND `|log_dt_final − log_dt_init|_max > 1e-4`
  (rules out the gradient-underflow artifact where smoke kept ρ constant
  at exp(-exp(0))=0.368).

## Theorem 1 (integration correctness by construction)

**Claim.** K-FULL-A passes whenever the monkey-patch is installed before
the first forward and `loop_idx_ref` is mutated at every iteration.

**Proof.** `LoopLoRALinear.__call__` routes `x` through `base(x) + deltas[idx](x)`.
`base` is the original quantised `nn.Linear`; its output dims are
preserved. The delta adds a full-precision `(out_dim,)`-shaped tensor
broadcast-compatible with `base(x)`. Therefore shapes are preserved
exactly, and the selected `B_t` is exercised iff `idx` is set before the
call. The replacement of `Gemma4TextModel.__call__` loops layers 12..20
with monotonic `loop_idx_ref[0] = t` before each block pass, so every
(ℓ, t) pair's `B_t` is exercised for every prompt. QED.

## Theorem 2 (v_proj and o_proj co-gradient under CE loss)

**Claim.** K-FULL-B passes under next-token cross-entropy on any prompt
whose final logits depend on `output = self.o_proj(attention_output)`.

**Proof.** In Gemma 4's `Attention.__call__`, attention output
`O = softmax(QK^T/√d) · V`, with `V = v_proj(x)` and final projection
`Z = o_proj(O)`. Next-token logits are `W_lm Z + …`. CE loss L is
differentiable in Z, hence in O (via o_proj), hence in V (via attention),
hence in B_v via the LoRA path `Z_v = (x A_v^T) B_v^T · scale`. Similarly
∂L/∂B_o is non-zero because the o_proj additive delta directly produces
Z. Both paths have non-degenerate Jacobians at `B_v = B_o = 0` because
the fixed Grassmannian A matrices are orthonormal (Q from partition-QR),
so `x A_v^T ≠ 0` in expectation over natural text. QED.

## Theorem 3 (LTI stability preserved under real loss)

**Claim.** K-FULL-C's ρ<1 clause holds by F#667 Theorem 1 for any Adam
trajectory that stays within the clamp `s ∈ [-20, 20]`; movement clause
holds iff the LoRA/LTI subgraph carries non-trivial gradient mass.

**Proof sketch.** (a) `ρ(A_d) = exp(-exp(clamp(log_dt+log_A, -20, 20)))`
∈ (exp(-e^{20}), exp(-e^{-20})) ⊂ (0, 1) in exact arithmetic; the clamp
is differentiable-through-the-middle so Adam doesn't push s out of
(-20, 20) unless loss structure demands it (empirically verified in
F#667 over 1000 steps, max |s| = 0.64 ≪ 20). (b) Movement: LTI params
enter the loss via `h' = A_d⊙h + B⊙h0 + tfm_out`, so ∂L/∂log_A ∝
∂L/∂A_d · (-A_d) · exp(s) ≠ 0 as long as ∂L/∂h' is non-zero, which
holds whenever the CE loss is non-degenerate. QED modulo Adam step-size
pathologies.

## Theorem 4 (target KCs require full-scale compute; scope judgement)

**Claim.** K1740/K1741/K1742 are not reliably testable under the
researcher-hat single-iteration budget; this experiment verifies
K-FULL-{A,B,C} at full scale and reports K1740/K1741/K1742 with the
largest eval budget feasible; if the largest feasible budget is below
the pre-registered `n ≥ 200` threshold, those KCs report
`not_measured` and the verdict is PROVISIONAL (per `REVIEW-adversarial.md`
§(t) / F#666 / F#673 — `not_measured` is not `FAIL`, KILL unjustified).

**Reasoning.** On M5 Pro 48GB, Gemma 4 E4B generation at `max_tokens=512`
runs at ~25 tok/s (from F#652 speed-ceiling). T=3 multiplies the inner
loop factor for layers 12..20 by 3, so effective tok/s on the looped
variant is ~25/1.27 ≈ 20 (layers 12..20 are ~21% of total forward
cost; 3× on that segment = 1.27× total cost). GSM8K-Hard at n=200
with 512 tokens/problem is ~200·512/20 ≈ 5100s ≈ 85 min per T value;
T ∈ {1..6} is ~8.5 h. MMLU-57 at 5-shot × thinking is
~1000 problems at ~256 tokens each ≈ 3.4 h. Total ≥ 12 h — exceeds
one researcher-hat cycle. Pragmatic scope:
- K-FULL-A/B/C: run at full spec.
- K1740 / K1742: fall back to max-feasible n and mark PROVISIONAL
  (document actual n measured in PAPER.md).
- K1741: not measured; flagged as follow-up (`exp_rdt_loop_mmlu_eval`).

## Antipattern self-audit (auto-injected antipattern checklist per PLAN §1)

| Code | Antipattern | Status |
|---|---|---|
| composition-bug | summing safetensor A and B independently | not applicable — monkey-patch exercises `B_t @ A_t` per loop at forward time; no safetensor aggregation |
| tautological-routing | `route(val[d][0])` | not applicable — loop index is scheduled, not routed |
| lora-scale-20 | `LORA_SCALE ≥ 12` | α=2 → scale=0.125 (safe) |
| shutil-copy-adapter | `shutil.copy` of sibling adapter | not applicable — LoRA tensors built fresh |
| hardcoded-pass | `{"pass": True}` literal | not applicable — all KC results derived from measurements |
| thinking-truncation | base eval with `avg_thinking_chars=0` | uses `apply_chat_template(add_generation_prompt=True)` with `max_tokens=1024`, preserves thinking |
| proxy-model | MATH says X, code loads Y | MATH = `mlx-community/gemma-4-e4b-it-4bit`, code loads same |
| smoke-as-full | `is_smoke=true` in full-run claim | `is_smoke=false`; if KC un-measurable, verdict PROVISIONAL not supported |
| kc-tautological | K passes by algebraic identity | K-FULL-A is binary (shape match) and structural; paired with K-FULL-B (gradient) and target K1740 per F#666 |
| kc-swap | KC modified after first result | no — KCs locked at MATH.md write time, pre-registered in DB |

## Prediction vs measurement (for PAPER.md)

| KC | Prediction (proof) | Measurement path |
|---|---|---|
| K-FULL-A | replacement succeeds; all 9 layers × 2 projs exercise LoRA | `assert isinstance(layer.self_attn.v_proj, LoopLoRALinear)` + a forward-pass counter |
| K-FULL-B | `max|∂L/∂B_v| > 1e-6` and `max|∂L/∂B_o| > 1e-6` on first batch | inspect grads dict after `nn.value_and_grad` |
| K-FULL-C | max ρ < 1 across N_STEPS steps; `\|Δlog_A\| > 1e-4` and `\|Δlog_dt\| > 1e-4` | record per-step `measure_rho_all`; record log_A/log_dt pre/post |
| K1740 | +5pp GSM8K-Hard at T=3, n=200 | if n < 200 feasible → `not_measured`; if measured → report pct diff |
| K1741 | MMLU ±1pp at T=3 | deferred to follow-up; `not_measured` |
| K1742 | R² > 0.90 saturating-exp fit on T∈{1..6} | if n < threshold → `not_measured`; else scipy.optimize.curve_fit |

## Libraries

- `mlx==0.31.1`, `mlx-lm==0.31.2` (pinned per F#652 compat audit).
- `datasets==4.3.0`, `dill==0.4.0`.
- Seed 42, stream `mx.default_stream()` except linalg.qr forced to `mx.cpu`.
