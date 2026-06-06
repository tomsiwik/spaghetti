# LEARNINGS.md — T4.5: Pierre Adapter Format Compatibility (AUDIT-RERUN, KILL)

## Core Finding
Format-compat on the MLX/Apple-Silicon target is **structurally unfinishable**. The
MLX↔PEFT double-transpose bijection (Theorem 1) and the Grassmannian-metadata
round-trip (Theorem 2, synthetic QR substrate, max_dev=2.38e-7) both PASS cleanly
(K1088, K1091). But the vLLM (K1089) and Unsloth (K1090) runtime KCs require a
live CUDA load — unreachable here. Verdict: **KILLED**, because PLAN.md §1 forbids
`supported` with unreached KCs. Finding #585 records the closure.

## Why
Theorem 3 (prior run) was a subset-direction fallacy: `F_vllm ⊆ F_peft` means
vLLM is *stricter* than PEFT (e.g. fused `qkv_proj` required), so PEFT-validity
does **not** imply vLLM-loadability. The prior run papered over this with a
string-suffix structural check and a `"property": "orthonormal_rows"` metadata
tag written regardless of a measured 0.579 deviation. The audit-rerun retracts
Theorem 3 in MATH.md, conditions the property tag on measured deviation, and
honestly returns `pass=False, skip_reason="cuda_unavailable_on_platform"` for
K1089/K1090. These are the exact antipatterns the rerun was commissioned to
eliminate — faking PASS would satisfy `all_pass` but reintroduce the lie.

## Implications for Next Experiment
- **`exp_followup_format_compat_peft_required`** (CUDA host) is the only valid
  path to close K1089/K1090. Its KCs are a real `PeftModel.from_pretrained` +
  forward pass and a fused-QKV vLLM probe. Do NOT retry on Apple Silicon.
- **T5 (user-local training)** may proceed using the proven MLX→PEFT bijection
  for export. It must not rely on trained Grassmannian structure being
  preserved post-SGD; prior evidence showed ~0.579 deviation after training, so
  interference bounds that assume `A A^T ≈ I_r` apply only to fresh/synthetic
  adapters or post-QR-reorthogonalized ones. Either re-orthogonalize after
  training or derive a drifted-case bound.
- **Design-time rule to propagate:** a platform-unreachable KC is SKIP, never
  PASS via substitute. If more than ~30% of KCs are platform-unreachable,
  re-target the experiment to a host that can satisfy them before running.
- **Substrate swap justification:** synthetic QR Grassmannian is valid *only*
  for FORMAT tests (schema + bijection); do not reuse it to make training-drift
  claims.
