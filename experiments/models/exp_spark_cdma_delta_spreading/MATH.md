# MATH — CDMA delta-spreading: a fixed orthogonal rotation of the off-domain LoRA delta-output decoheres interference

Experiment: `exp_spark_cdma_delta_spreading`
Base: `mlx-community/gemma-4-e4b-it-4bit` (frozen, 4-bit), `mlx-lm == 0.31.2`.
Adapters: r=6 q_proj LoRA, code(HumanEval) and math(GSM8K), from F#627 recipe
(`experiments/models/exp_composition_residual_analysis/adapter_{code,math}.safetensors`,
84 keys = 42 layers × {lora_a (2560,6), lora_b (6,2048)}).

References:
- F#827 / F#837: measured real off-domain interference (−12 to −14pp) when summing two LoRA deltas.
- F#822 / F#823 / F#815: rotating the **A-input basis** is dead — it breaks the adapter's read of the residual stream. We therefore rotate the **post-B delta output**, never A.
- arXiv:2108.11811 (Walsh–Hadamard spreading) — orthogonal spreading codes decorrelate superimposed signals; the algebraic mechanism reused here.
- F#666: every proxy KC must be paired with a target-metric KC.

---

## 1. Setup and the failure mode

For a single q_proj at one layer, the LoRA-augmented output of token activation `h ∈ ℝ^{d_in}` (d_in=2560) is
```
y(h) = W h + s · B A h ,           B ∈ ℝ^{d_out×r}, A ∈ ℝ^{r×d_in}, r=6, s=LORA_SCALE≤8.
```
Here `d_out` is the q_proj delta-output width (n_heads × head_dim) of the **actual** base model,
read dynamically from the model at run time — **never hardcoded**. The first run crashed because a
hardcoded d_out=2048 disagreed with the deployed q_proj output width (4096); the rotation P is now
built at exactly whatever width the model emits (see §3/code RotBox), and every d_out-dependent claim
below holds at that true width.
Write the per-adapter **delta output** δ_x(h) := s · B_x A_x h ∈ ℝ^{d_out}, x ∈ {code, math}.

Naive sum (condition C) produces
```
y_C(h) = W h + δ_code(h) + δ_math(h).
```
**Failure mode (interference).** On a code prompt, δ_math(h) is *not* zero: the math adapter was
trained on GSM8K and its B_math has a dominant low-rank column space U_math = colspan(B_math) ⊂ ℝ^{d_out}
(r=6, so dim ≤ 6). Across code tokens, A_math h is sign/magnitude-correlated, so δ_math(h) is a
**coherent** vector — it points, token after token, into the *same* few directions U_math with a
consistent sign. This is a systematic additive bias on the shared q_proj output dimensions, exactly
the residual measured non-zero in F#827/837. Coherent bias survives the subsequent attention/RMSNorm
pipeline and corrupts code generation (−12 to −14pp pass@1).

## 2. What RMSNorm does to a vector

q_proj output feeds attention; downstream every block re-normalizes through RMSNorm:
```
RMSNorm(v) = v / sqrt( (1/d) Σ_j v_j² ) · g     (g = learned gain).
```
RMSNorm is **scale-invariant** and divides by the RMS energy of `v`. The key fact: RMSNorm does **not**
remove a coherent additive bias b that lies along high-gain directions — b shifts the normalized output
in a consistent, learnable-to-be-harmful way. But if the same energy ‖b‖ is **redistributed
incoherently** across all d_out coordinates as a near-isotropic perturbation, two things happen:
1. its projection onto any *single* downstream-relevant low-dim readout direction shrinks by ~√(k/d_out)
   (energy spread over all d_out coords instead of concentrated in ≤6), and
2. it adds to the RMS denominator as variance, so RMSNorm *attenuates* the relative contribution of the
   off-domain term while leaving the on-domain coherent term (δ_code) comparatively intact.

## 3. The construction: rotate only the math delta-output by a fixed orthogonal P

Let P ∈ ℝ^{d_out×d_out} be a **fixed, seeded** orthogonal matrix (random orthogonal via QR of a
Gaussian, seed=1337), PᵀP = I. Condition D:
```
y_D(h) = W h + δ_code(h) + Pᵀ δ_math(h)
       = W h + s·B_code A_code h + s·(Pᵀ B_math) A_math h.
```
Crucially A_math is untouched (so the adapter still *reads* h correctly — avoids F#822/823/815), and the
delta magnitude is preserved (‖Pᵀ δ_math‖ = ‖δ_math‖ since P orthogonal). Only the **output direction**
of the math delta is scrambled.

### Theorem (decoherence of off-domain bias).
Let δ_math(h_t) for code tokens t=1..T be coherent: they lie (to leading order) in a fixed subspace
U_math of dim ρ ≤ r = 6, with mean direction ū. Let P be drawn Haar-uniform on O(d_out). Then for the
rotated deltas v_t := Pᵀ δ_math(h_t):
1. **Magnitude preserved:** ‖v_t‖ = ‖δ_math(h_t)‖  (P orthogonal). ∎-part-1
2. **Direction decohered:** for any *fixed* downstream readout direction w (‖w‖=1) chosen independently
   of P,  E_P[(wᵀ v_t)²] = ‖δ_math(h_t)‖² / d_out, and the coherent overlap across tokens
   E_P[(wᵀ v_s)(wᵀ v_t)] = (δ_math(h_s)ᵀ δ_math(h_t)) / d_out. Hence the *coherent* projected energy
   onto any low-dim readout drops by the factor 1/d_out relative to the unrotated, U_math-aligned case
   where it was Θ(1)·‖δ‖². At the model's true d_out (=4096 for gemma-4-e4b q_proj) this is a
   ≈d_out× (≈4096×) suppression of the per-direction coherent
   leakage.

**Proof.** (1) ‖Pᵀδ‖² = δᵀ P Pᵀ δ = δᵀδ. (2) For Haar P and fixed unit w, Pw is uniform on the sphere,
so E[(Pw)(Pw)ᵀ] = (1/d_out) I. Then E[(wᵀ Pᵀ δ_s)(wᵀ Pᵀ δ_t)] = δ_sᵀ E[(Pw)(Pw)ᵀ] δ_t =
δ_sᵀ δ_t / d_out. Setting s=t gives the variance; the cross term is the coherence. The unrotated
U_math-aligned overlap is δ_sᵀ δ_t · (overlap of U_math with w)² = Θ(‖δ‖²) when w probes U_math. ∎

### Corollary (RMSNorm suppression).
After q_proj, the attention/MLP readouts act as (approximately) fixed low-dim directions w independent
of our experiment-chosen P. By the Theorem the coherent code-corrupting leakage from the math adapter is
attenuated ~d_out-fold, while ‖Pᵀδ_math‖ adds isotropic variance to the RMS denominator, further
shrinking its *relative* weight. δ_code is untouched and stays coherent in its on-domain readouts.
Therefore D should recover the code accuracy lost in C, approaching the code-solo ceiling B. ∎

**Caveat (honest gap).** P is independent of the *true* downstream readouts only to the extent those
readouts are not adversarially aligned with Pᵀ U_math; with a single fixed seed there is a measure-zero
chance of accidental re-alignment. The behavioral KC is what adjudicates — the theorem predicts a
direction and a large effect, the run measures it.

## 4. Predicted pass@1 ordering

n=50 HumanEval, greedy, real unit-test execution. Conditions:
- A base only, B code-solo (ceiling), C naive sum (code+math, both unrotated), D delta-spread (code unrotated + math Pᵀ-rotated).

| Hypothesis (decoherence true) | Null (rotation inert) |
|---|---|
| D ≈ B  >  C  >  A (or A≈C) | D ≈ C  <  B |

The hypothesis predicts D recovers most of the B−C interference gap. The null predicts the rotation does
nothing (D stays at the interfered level C) — which would mean either interference is not the coherent,
RMSNorm-survivable bias claimed, or P also scrambles task-relevant geometry.

## 5. Pre-registered kill criterion (matches DB id 2295, target-metric, Finding #666 style)

**K2295 (target, behavioral — HumanEval pass@1 via real test execution):**
KILL if `pass@1(D) < pass@1(C) + 8pp`  **OR**  `pass@1(D) < pass@1(B) − 6pp`.

Interpretation:
- `pass@1(D) ≥ pass@1(C) + 8pp` is required: the rotation must recover a real, ≥8pp chunk of the
  interference gap, not noise.
- `pass@1(D) ≥ pass@1(B) − 6pp` is required: D must land within 6pp of the code-solo ceiling — i.e. the
  decoherence must *actually* restore code competence, not merely beat the broken C baseline by a sliver.
- **SUPPORTED** = neither clause fires (D both clears C+8pp and lands within 6pp of B).
- **KILLED** = either clause fires.

This is a single behavioral target metric (pass@1 from executing the model's generated code against
HumanEval unit tests) — no proxy stands in for it.

## 6. Implementation invariants (enforced in code)
- Composition is `Σ_i B_i A_i` applied per q_proj as `(h@A_iᵀ... )` — never `(ΣB)(ΣA)`. C and D each sum
  two independent deltas; D rotates only the math delta's output by Pᵀ.
- Wrapper attaches via **subclass nn.Module + setattr** on `layer.self_attn.q_proj`; never override
  `__call__` on an instance (mem-antipattern-call-override-silent-bypass / F#831).
- `LORA_SCALE = 6.0 ≤ 8`.
- Routing is per-sample by construction (the same fixed adapter set is applied to every prompt; the
  experiment is the off-domain code benchmark — no per-domain cheating).
- P built once, seeded (1337), verified `‖PᵀP − I‖ < 1e-4`, FIXED (never learned), shared across all 42 layers.
- `enable_thinking=True` in chat template with MAX_NEW_TOKENS headroom (mem-antipattern-thinking-truncation).
- Phased execution, `del`+`gc.collect()`+`mx.clear_cache()` between the 4 conditions.
- `is_smoke: false`.
