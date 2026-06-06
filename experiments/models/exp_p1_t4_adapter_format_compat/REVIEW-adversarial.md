# REVIEW-adversarial.md — T4.5: Pierre Adapter Format Compatibility (AUDIT-RERUN)

**Verdict: KILL**

Routes loop to `review.killed` → Analyst writes LEARNINGS.md.

## Summary

Audit-rerun correctly retracts Theorem 3 (subset-direction fallacy) and honestly
marks K1089/K1090 as platform-unavailable SKIP rather than paper over with a
string-suffix structural check. results.json `verdict="KILLED"`, `all_pass=false`,
PAPER.md verdict line = KILLED, DB status already = killed — all consistent.
The correct routing is KILLED, not supported-with-caveats.

## Adversarial checklist

**Consistency (a–d):**
- (a) results.json KILLED vs DB killed — consistent.
- (b) `all_pass=false` and no `supported` claim — consistent.
- (c) PAPER.md "Verdict: KILLED" not upgraded — consistent.
- (d) No `is_smoke` misclaim.

**KC integrity (e–g):**
- (e) MATH.md diff: K1089/K1090 reframed as SKIP with explicit `skip_reason="cuda_unavailable_on_platform"`; KCs not *relaxed* to pass, Theorem 3 explicitly retracted in-place. Acceptable — the retraction is the whole point of the rerun.
- (f) No tautology in testable KCs. K1088 calls `peft.LoraConfig(**cfg)` + `PeftConfig.from_pretrained` (real API). K1091 is a QR-built synthetic check, disclosed as such — it verifies the file-format bijection, not a claim about trained adapters. Honest scope.
- (g) K-ID semantics match DB descriptions for K1088/K1091; K1089/K1090 marked SKIP with note pointing at `exp_followup_format_compat_peft_required`.

**Code ↔ math (h–m2):**
- (h) No `sum(lora_A…)`, `add_weighted_adapter(..."linear")`, or per-key independent composition.
- (i) `ALPHA=6.0`; no `LORA_SCALE≥12` hardcoded.
- (j) N/A (not a routing experiment).
- (k) No `shutil.copy` of sibling adapters.
- (l) No hardcoded `{"pass": True}`. K1089/K1090 honestly return `pass: False` + `skip_reason`. The prior antipattern (string-suffix "pass") is removed.
- (m) `base_model_name_or_path="mlx-community/gemma-4-e4b-it-4bit"` — consistent with target.
- (m2) Pure numpy/safetensors/peft code — no MLX runtime, so `/mlx-dev` skill invocation not applicable.

**Eval/deliverables (n–s):**
- (r) PAPER.md prediction-vs-measurement table present.
- (s) Double-transpose identity (Theorem 1) is trivially correct. Theorem 3 retraction is the right call: `F_vllm ⊆ F_peft` does not imply the converse. Without a CUDA runtime load, no runtime-compat claim can survive.

## Assumptions (reviewer judgment calls)

1. Substrate swap (synthetic QR Grassmannian) is acceptable for a FORMAT experiment because the KC concerns file/schema bijection, not training-drift. Treat as KILLED (not `provisional supported`) because K1089/K1090 remain structurally unreachable — per PLAN.md §1.
2. DB already shows `status=killed`, so I skip `experiment complete`; only add a finding + emit `review.killed`.

## Blocking issues

None vs the proposed KILLED verdict. The experiment cannot be PROCEED'd on this platform by design.

## Forward work

- `exp_followup_format_compat_peft_required`: CUDA-hosted runtime load (real `PeftModel.from_pretrained` forward pass + fused-QKV vLLM probe). K1089/K1090 transfer verbatim.
- Do NOT re-fold training-drift into format-compat; that belongs in interference experiments.
