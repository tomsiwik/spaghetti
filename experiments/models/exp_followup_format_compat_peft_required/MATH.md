# MATH.md — exp_followup_format_compat_peft_required

## 0. Context & audit lineage

Parent: `exp_p1_t4_adapter_format_compat` — verdict `KILLED`, Finding #585.
Loophole audit (`LOOPHOLE_{FINDING,CODE,METHODOLOGY}.md` in parent dir) flagged
two non-ignorable failures in the v1 run:

1. **Silent bypass** — `except ImportError: pass` in K1088 let the KC auto-pass
   when `peft` was not installed. The claim "PEFT-compatible" was made without
   `peft` ever touching the adapter.
2. **Subset fallacy (Theorem 3)** — PEFT-compat was claimed to imply vLLM-compat
   because `F_vLLM ⊆ F_PEFT`. This is a *stricter-than* relation, so the direction
   is reversed: being loadable by PEFT does **not** imply loadable by vLLM
   (fused `qkv_proj` requirement).

The parent's audit-rerun (kept on Apple Silicon) closed K1088 + K1091 via schema
checks on a synthetic Grassmannian substrate; K1089 (vLLM) and K1090 (Unsloth)
were correctly returned as `SKIP` (CUDA-unreachable on MLX target). This
follow-up closes the remaining honest gap on Apple Silicon: **actual PEFT
runtime load + forward pass + QKV-fusion verification**, all CPU-reachable with
PyTorch.

## 1. Failure mode being prevented

The v1 code PASSED K1088 with a dict-subset check. A real PEFT runtime has three
distinct validation steps that the subset check never reaches:

- **(F1) Adapter key parsing** — PEFT strips `base_model.model.` prefix and
  expects `.lora_A.weight` / `.lora_B.weight` exactly. A typo or missing prefix
  makes `PeftModel.from_pretrained()` raise.
- **(F2) Shape matching** — PEFT checks `lora_A.weight: [r, d_in]` and
  `lora_B.weight: [d_out, r]` against the base module's `in_features` /
  `out_features`. Shape mismatch raises at load-time.
- **(F3) Module targeting** — `target_modules` must name modules present on the
  base. Missing modules raise `ValueError("Target modules ... not found")`.

A format is "PEFT-compatible" iff all three gates pass under an actual
`PeftModel.from_pretrained(base, adapter_dir)` call, followed by a successful
forward pass on a real input tensor.

Additionally, **fused-QKV compatibility** (F4) is a distinct concern: some
runtimes (vLLM, some HF architectures like GPT-NeoX) use a single
`query_key_value` / `qkv_proj` module instead of separate `q_proj`/`k_proj`/
`v_proj`. Our MLX adapters target separate projections. The transformation
`fuse(A_q, A_k, A_v) → A_qkv` with output dim `d_q + d_k + d_v` is a necessary
step for fused-runtime interop. F4 is testable structurally without actually
running vLLM.

## 2. Theorem (deterministic stand-or-fall)

**Theorem 1 (Real-PEFT bijection).**  Let `A_mlx` denote an adapter with MLX
layout `{lora_a: [d_in, r], lora_b: [r, d_out]}` for each targeted module, and
let `A_peft = T(A_mlx)` where
    T(lora_a)     = transpose(lora_a) = [r, d_in]   with PEFT key `.lora_A.weight`
    T(lora_b)     = transpose(lora_b) = [d_out, r]  with PEFT key `.lora_B.weight`
    key_prefix    = "base_model.model." + module_path.
Let `M_base` be an HF `transformers` PreTrainedModel exposing
`M_base.config.hidden_size = d_in = d_out` (square-projection regime) and
attention modules named `...self_attn.q_proj`. Then:

    peft.PeftModel.from_pretrained(M_base, adapter_dir)                 (L1)
    peft_model(input_ids).logits.shape == (B, T, |V|)                    (L2)

both succeed iff F1 (keys), F2 (shapes), F3 (targets) all hold.

*Proof sketch:* `PeftModel.from_pretrained` iterates the adapter state_dict,
strips `base_model.model.`, splits the remainder on the target module suffix,
and calls `setattr` on the `LoraLayer` for each `lora_A` / `lora_B`. Key parse
failure ⇒ KeyError or silent unused weight (depending on peft version, which
we pin ≥0.18.1 — `LoraConfig.target_modules` unmatched raises `ValueError`).
Shape mismatch ⇒ `RuntimeError` at `setattr` time (peft asserts against
parent module dims). Forward pass composes the base linear with `α/r · B @ A`,
which is well-defined iff shapes are consistent. No hidden bypass exists in
`peft 0.18.1` for these three checks. ∎

**Theorem 2 (QKV-fusion bijection).**  For a Gemma/Llama-style base with
separate `q_proj` (out dim `d_q`), `k_proj` (`d_k`), `v_proj` (`d_v`), the
fused-QKV adapter factors are
    A_qkv = A_q    ∈ ℝ^{r, d_in}        (shared input projection — NOT A's can be fused only if A_q = A_k = A_v, otherwise fused adapter must keep rank-per-head)
    B_qkv = [B_q ; B_k ; B_v]  ∈ ℝ^{d_q + d_k + d_v, r}   (row-concat).
This mapping is well-defined iff `A_q == A_k == A_v` (shared-A regime) — a
degenerate case. In the general case, fusion requires `r_qkv = 3r` with
block-diagonal A:
    A_qkv_block = diag(A_q, A_k, A_v) ∈ ℝ^{3r, 3·d_in}    (if inputs are tiled) OR
    A_qkv_stack = [A_q ; A_k ; A_v]   ∈ ℝ^{3r, d_in}      (if input shared),
    B_qkv_stack = blkdiag(B_q, B_k, B_v) ∈ ℝ^{d_q+d_k+d_v, 3r}.
The mathematical content of F4 is that **separate-QKV MLX adapters are not
transparently fused-QKV compatible without a rank expansion**; downstream
claims of "vLLM-ready" that omit this transformation are subset-direction
fallacies.

*Proof:* the fused projection computes `[Q;K;V] = W_qkv @ x` where
`W_qkv = [W_q;W_k;W_v]`. The additive low-rank perturbation is
`[ΔQ;ΔK;ΔV] = (α/r) [B_q@A_q; B_k@A_k; B_v@A_v] @ x`. Writing this as a single
`(α/r) · B_qkv @ A_qkv @ x` requires A_qkv to produce three independent latent
vectors, which demands rank ≥ 3r (block-diagonal A) unless `A_q = A_k = A_v`. ∎

## 3. Predictions (numeric)

On the synthetic Grassmannian substrate built here (square 64-dim base,
4 layers, r=6, separate q/k/v adapters):

| # | Prediction | Formula |
|---|---|---|
| P1 | `import peft` succeeds (hard-required) | raise if `ModuleNotFoundError` |
| P2 | `PeftModel.from_pretrained(base, dir)` returns without exception | L1 |
| P3 | `model(input_ids).logits` is produced; shape `[B=1, T=4, V]` | L2 |
| P4 | Fused-QKV naive stack `[A_q;A_k;A_v]` has shape `[3r, d_in]` = `[18, 64]` | Theorem 2 |
| P5 | Fused-QKV naive stack `blkdiag(B_q,B_k,B_v)` has shape `[3d_q, 3r]` = `[192, 18]` | Theorem 2 |
| P6 | Grassmannian max-deviation of synthetic A < 1e-6 | QR property |

## 4. Kill criteria (pre-registered — DO NOT EDIT AFTER RUN)

**K1576** (DB ID 1576 — deterministic stand-or-fall). Defined by **all three**
of the following gates; any failure → KC fails → verdict KILLED:

- **K1576.a** `import peft` succeeds at module-load time (hard-required, no
  `try/except ImportError: pass`). FAIL if peft not installed.
- **K1576.b** `peft.PeftModel.from_pretrained(M_base, adapter_dir)` returns
  without exception, AND a forward pass `pm(input_ids).logits` completes
  without shape errors.
- **K1576.c** Fused-QKV stack reshape is structurally valid (shapes match
  Theorem 2), AND the adapter keys explicitly enumerate separate
  `q_proj` / `k_proj` / `v_proj` (not pre-fused `qkv_proj`), demonstrating the
  non-trivial transformation required for fused-runtime interop.

Verdict logic (no ambiguity):

- All three pass → **SUPPORTED** (K1576 PASS).
- Any fails → **KILLED** with the failure mode recorded verbatim in
  `results.json["k1576_failure"]`.

Pre-registration protocol: this MATH.md is git-committed before
`run_experiment.py`. The KC threshold is binary (pass/fail per gate). No
post-hoc threshold tuning possible.

## 5. Assumptions & scope

- **Base model**: we use a from-scratch tiny LlamaConfig (hidden=64, 4 layers,
  4 heads, vocab=256) to avoid HF-Hub network dependency and keep CPU runtime
  <10s. This isolates the FORMAT/LOAD question from any specific base model;
  the theorem is base-agnostic (Llama/Gemma/Qwen all expose separate q/k/v on
  HF).
- **peft pin**: `peft == 0.18.1` (from project venv). `transformers == 5.5.0`.
- **Grassmannian construction**: QR decomposition on Gaussian Rs (standard,
  matches parent's synthetic substrate).
- **We do NOT claim vLLM runtime compatibility** — that remains CUDA-only and
  out of scope. F4/K1576.c tests the structural *transformation*, not runtime.
- **We do NOT claim Unsloth compatibility** — CUDA-only.
- **Apple Silicon reachability**: every gate executes on CPU via PyTorch.
  MLX-target constraints (PLAN.md Part 2) do not apply; this is a *format
  interop* test, not a training/inference target.

## 6. SIGREG chain check

- *Disease vs symptom*: the symptom was "K1088 passes". The disease was
  *silent-bypass + subset-fallacy* that made the KC insensitive to actual load
  failures. Cure: hard-required import (kills bypass) + real
  `PeftModel.from_pretrained` (kills subset fallacy).
- *Structural impossibility*: with K1576.a as a module-load assert, the bypass
  path literally cannot exist in the executed code — the script would crash
  on import, not "silently pass".
- *Math-not-analogy*: the bijection in Theorem 1 and the block-diagonal fusion
  in Theorem 2 are verbatim from PEFT's state-dict loader implementation
  (`peft/tuners/lora/model.py`) and standard block-matrix algebra
  (any LA textbook). No analogy.
- *Eliminated hyperparameters*: rank r, α scale, and base dim d are set once
  and not swept — the KC is structural, not quantitative, so sweeps would add
  zero signal.

## 7. References

- Parent experiment: `micro/models/exp_p1_t4_adapter_format_compat/` (KILLED,
  Finding #585).
- Parent loophole audit: `LOOPHOLE_FINDING.md`, `LOOPHOLE_FOLLOWUP.md` in
  parent dir (subset fallacy + silent bypass).
- Parent post-rerun: `exp_p1_t4_format_compat_v2` (KILLED with honest SKIP on
  K1089/K1090).
- PEFT reference: `peft/src/peft/tuners/lora/model.py` —
  `_create_and_replace`, `_replace_module` (loader logic).
- arxiv:2106.09685 (Hu et al., LoRA) — original low-rank adapter formulation.
- Finding #433 (parent-of-parent) — transpose bijection; now subsumed by
  Theorem 1 which adds the runtime-load clause.
