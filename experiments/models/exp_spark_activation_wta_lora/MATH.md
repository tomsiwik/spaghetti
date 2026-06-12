# MATH — Per-token activation-L2 winner-takes-all vs weight-merge composition (is loudness = correctness?)

Experiment: `exp_spark_activation_wta_lora`
Base: `mlx-community/gemma-4-e4b-it-4bit` (frozen, 4-bit), `mlx-lm == 0.31.2`.
Adapters: r=6 q_proj LoRA, scale=6.0, 42 layers, same base — from
`data/adapters/{math,python,medical}/adapters.safetensors`
(each 84 keys = 42 layers × {lora_a (d_in,6), lora_b (6,d_out)}, shape (2560,6)/(6,d_out)). The
on-domain experts are **math** (for GSM8K) and **python** (for HumanEval); **medical** is the
off-domain distractor for both. **N = 3** adapters in the pool.

> Realizability note (real-not-mock): the pool was pre-registered as N=4 with a 4th distractor, but
> the only adapters in the repo that are structurally compatible (target `q_proj`, all 42 layers,
> r=6) are math / python / medical. `data/adapters/sql/` ships no `adapters.safetensors`, and
> `thinking-openthoughts-universal-v0` targets `o_proj`/`v_proj` on layers 26-41 only (64 keys, no
> layer-0 q_proj) — it cannot be injected at the q_proj site. N drops 4→3; the hypothesis, both
> magnitude-matched controls, and kill criterion K2303 are unchanged (s/N dilution and the argmax
> both adapt to N=3). On each benchmark there are still ≥2 off-domain distractors in the pool
> (e.g. on GSM8K: python + medical), so the dilution/interference failure mode is preserved.

References:
- F#827 / F#837: real off-domain interference (−12 to −14pp) when naively summing two LoRA deltas.
- F#863 (strobe_multiplex KILLED): a per-token routing "win" can be a **pure injection-magnitude
  confound** — winner-takes-all injects one full-scale delta while uniform-1/N injects scale/N per
  adapter, so total injected magnitude differs across arms. Any naive WTA>uniform result must be
  isolated from total-injection-magnitude before it can be read as "loudness=correctness". This
  experiment pre-registers two magnitude-matched controls (below) so that a WTA win cannot be
  explained by injection magnitude alone.
- 12 prior composition findings (Fisher-Rao / TIES / DARE / Pico / learned-scalars) all live in
  **weight space**; all routing was **per-prompt / per-domain**. Nobody removed the merge operator
  and routed at the residual-injection site by raw per-token activation loudness with zero training.

---

## 1. Setup

For one q_proj at one layer, adapter x emits a per-token **delta output** on activation `h ∈ ℝ^{d_in}`:
```
δ_x(h) := s · B_x A_x h ∈ ℝ^{d_out},   A_x ∈ ℝ^{r×d_in}, B_x ∈ ℝ^{d_out×r}, r=6, s=6.0≤8.
```
(In code the stored convention is `lora_a:(d_in,r)`, `lora_b:(r,d_out)`, so δ_x = s·(h@A_x)@B_x.)
`d_out` is the model's true q_proj output width, read from the adapter tensor — never hardcoded.
The per-token **loudness** of adapter x at this layer is `ℓ_x(h) := ‖B_x A_x h‖₂` (scale-free; the
common `s` cancels in any argmax over x).

Composition is always `Σ_i B_i A_i` (each adapter an independent delta added to the residual), never
`(ΣB)(ΣA)`.

## 2. The two composition operators

**Uniform-1/N sum (weight-merge analogue at the injection site).**
```
y_sum(h) = W h + (s/N) Σ_{i=1}^{N} B_i A_i h.
```
Each adapter contributes scale/N. Off-domain experts (medical, thinking on a code/math prompt) emit
low-but-nonzero coherent deltas that **dilute** the on-domain signal: the on-domain delta is scaled
down to 1/N and summed with N−1 distractor biases. This is the interference failure mode.

**Winner-takes-all by activation L2 (WTA-full).** Per token, per layer, pick the loudest adapter and
inject **only** its delta at full scale:
```
i*(h) = argmax_i ℓ_i(h) = argmax_i ‖B_i A_i h‖₂ ,   y_wta(h) = W h + s · B_{i*} A_{i*} h.
```
The hypothesis ("loudness = correctness"): the on-domain expert is, per token, the **loudest** because
the prompt activations lie in its trained input subspace; selecting it removes all distractor bias AND
restores full on-domain magnitude. Zero training, zero learned router — magnitude is asserted to be a
free relevance signal.

## 3. The confound (F#863) and the magnitude-matched controls — PRE-REGISTERED

WTA-full changes **two** things at once relative to uniform-sum:
  (a) **routing** — only one adapter's direction is injected (no distractor bias), and
  (b) **magnitude** — that adapter is injected at `s` instead of `s/N` (≈N× larger per-token norm).

A naive `wta_full > sum_uniform` win therefore cannot distinguish "the model needs the *correct*
single expert (routing/loudness=correctness)" from "the model just needs *more total injected
magnitude* than the diluted 1/N sum provides". To break this, we pre-register two controls that each
hold one factor fixed:

| Arm | inject | isolates |
|---|---|---|
| `base` | nothing | floor |
| `sum_uniform` | (s/N) Σ_i δ̂_i | diluted merge (interference baseline) |
| `wta_full` | s · δ̂_{i*} | routing **+** full magnitude (the hypothesis) |
| `wta_scaled` | (s/N) · δ̂_{i*} | **routing at the SAME per-adapter magnitude as sum_uniform** |
| `rand_full` | s · δ̂_{r}, r∼Uniform(adapters) per token | **full magnitude with WRONG/random routing** |

(δ̂_i := B_i A_i h, the unscaled delta; the s and s/N factors are applied explicitly.)

- `wta_scaled` injects the L2-winner but at scale/N — **identical injected magnitude to one term of
  sum_uniform**. If WTA's benefit is routing (picking the right single expert and dropping distractor
  bias), `wta_scaled` keeps most of the WTA gain over `sum_uniform`. If WTA's benefit were pure
  magnitude, `wta_scaled` collapses back toward `sum_uniform`.
- `rand_full` injects a single delta at full scale `s`, like `wta_full`, but the picked adapter is
  **random per token** — same total injection magnitude as `wta_full`, wrong routing. If "loudness =
  correctness", `wta_full` must beat `rand_full` by a real margin; if any-single-full-scale-delta is
  what matters, `wta_full ≈ rand_full`.

**These two controls are the scientific core.** The kill criterion (id 2303) compares `wta_full` to
`sum_uniform`, but the *interpretation* of a 2303 pass is gated on the controls: a WTA win is only
"loudness=correctness" if **both** `wta_full > rand_full` (routing beats random at matched magnitude)
**and** `wta_scaled > sum_uniform` (routing helps even at matched magnitude). We report all five arms
and the two control deltas alongside the kill verdict so the confound is auditable, not assumed away.

## 4. Why per-token L2 routing is plausibly correct (theorem sketch)

Adapter x was trained so that on **in-distribution** tokens h (its domain), the input projection A_x h
is large (the rank-6 row space of A_x is aligned with the domain's activation statistics), while on
**out-of-distribution** tokens A_x h is small (those tokens lie mostly in ker(A_x) ⊕ low-energy
directions). Since `‖B_x A_x h‖ ≤ σ_max(B_x) ‖A_x h‖`, an on-domain expert's delta loudness scales
with the matched input energy. **Claim:** for a code token, ℓ_python(h) > ℓ_{medical,math,thinking}(h)
on a majority of tokens, so argmax recovers python; symmetrically math wins on GSM8K tokens. This is an
empirical, falsifiable claim about trained low-rank geometry — the run measures both the behavioral
accuracy AND (logged) the per-token argmax selection rate of the on-domain expert.

## 5. Predicted accuracy ordering (avg over GSM8K n=40 + HumanEval n=40, greedy)

| Hypothesis (loudness=correctness) | Null (magnitude-only) | Null (routing inert) |
|---|---|---|
| wta_full > wta_scaled > sum_uniform ≳ base ; wta_full ≫ rand_full | wta_full ≈ rand_full > sum_uniform ; wta_scaled ≈ sum_uniform | wta_full ≈ sum_uniform |

acc(arm) := mean(GSM8K_exact_match, HumanEval_pass@1) over the two n=40 sets.

## 6. Pre-registered kill criterion (DB id 2303, target-behavioral)

**K2303:** KILL if `acc(wta_full) − acc(sum_uniform) < +0.05` (avg accuracy, GSM8K n40 + HumanEval n40).
- `Δ_wta_vs_sum := acc(wta_full) − acc(sum_uniform)`.
- `verdict = "killed" if Δ_wta_vs_sum < 0.05 else "supported"`; `all_pass = not killed`.

**Confound-isolation riders (reported, do NOT relax 2303 — they qualify a pass):**
- `Δ_routing_matched := acc(wta_scaled) − acc(sum_uniform)`  (routing benefit at matched magnitude).
- `Δ_routing_vs_random := acc(wta_full) − acc(rand_full)`     (loudness beats random at matched magnitude).

A 2303 pass with `Δ_routing_vs_random ≤ 0` would mean the win is the magnitude confound F#863 warned
of, not loudness=correctness; we flag this explicitly in results as `confound_magnitude_only=true` so
the reviewer sees it even though the bare 2303 clause passed.

## 7. Implementation invariants (enforced in code)
- Composition `Σ_i B_i A_i` per q_proj as `s·(h@A_i)@B_i`; never `(ΣB)(ΣA)`.
- Loudness `ℓ_i = ‖(h@A_i)@B_i‖₂` per token (last-axis L2), argmax over the N adapters, per layer,
  per token. The common scale `s` cancels in argmax (computed on unscaled δ̂, then scaled).
- Wrapper attaches via **subclass nn.Module + setattr** on `layer.self_attn.q_proj`; never override
  `__call__` on an instance (F#831).
- `LORA_SCALE = 6.0 ≤ 8`.
- Per-sample/per-token routing by construction; the same N-adapter pool is applied to every prompt of
  both benchmarks — no per-domain hand-routing.
- `rand_full` uses a fixed seed so the random pick is reproducible.
- `enable_thinking=True`, MAX_NEW_TOKENS headroom (no thinking truncation).
- Phased execution; `del`+`gc.collect()`+`mx.clear_cache()` between arms.
- Real model, real adapters, real GSM8K exact-match + real HumanEval unit-test execution.
- `is_smoke: false`.
