# REVIEW-adversarial.md — exp_prod_llama_cpp_bridge

## Verdict
**KILL** (preempt-structural, ap-017(s) hardware-topology-unavailable, **2nd instance**, 1st = F#650).

## One-line
llama.cpp runtime is structurally outside parent `exp_prod_adapter_format_spec_v1` Apple-only-MLX scope; defense-in-depth across 5 independent theorems. F#763 registered. Promotion candidate at 3rd instance.

## Adversarial checklist

| # | Item | Result |
|---|---|---|
| (a) | results.json `verdict=KILLED` ↔ DB `status=killed` | PASS — consistent |
| (b) | `all_pass=false` ↔ status=killed | PASS — consistent |
| (c) | PAPER.md "KILLED_PREEMPTIVE" verdict line | PASS — consistent |
| (d) | `is_smoke=false`, full preempt — no smoke-vs-full mismatch | PASS |
| (e) | KC mutation post-claim | PASS — K1654/K1655 verbatim from DB pre-reg |
| (f) | Tautology sniff | INTENTIONAL — K1655 non-falsifiability IS the T4 finding, surfaced not concealed |
| (g) | KC ID semantics | PASS — both K-IDs map to MATH text |
| (h–l) | Composition / scale / routing / shutil / hardcoded-pass | N/A — graceful-failure runner emits no MLX/torch code |
| (m) | Target model substitution | N/A — no model loaded |
| (m2) | Platform-skill invocation | CARVE-OUT — preempt-structural with no MLX path executed; runner is pure stdlib (`json`+`pathlib`+`platform`+`shutil`+`subprocess`+`time`). Same carve-out as F#669-family / F#666-pure preempt clauses |
| (n–q) | Eval integrity | N/A — no measurement performed |
| (r) | Prediction-vs-measurement table | PASS — 5×3 table present in PAPER.md |
| (s) | Math errors | PASS — 5-theorem stack sound; F#650/F#60/F#61/F#627 all real and correctly cited; PoLAR-not-in-GGML-LoRA claim is technically correct (GGML LoRA expresses standard A/B only, not Stiefel/orthogonal factorisation) |
| (t) | Target-gated kill (F#666) | CARVE-OUT — preempt-structural verdict; **no KC was measured** (proxy or target). F#666 gates on proxy-FAIL with measurement; preempt-KILL on hardware-topology-unavailability is parallel to F#669-family carve-out per reviewer hat §KILL-preempt-structural |
| (u) | Scope-changing fix | PASS — graceful-failure stub is the canonical preempt-structural artifact, not silent SFT→LoRA / max_length-shrink / monitoring-disable. The "fix" here is to *acknowledge* the parent-scope breach, not to silently downgrade the claim |

## Distinctions confirmed

- **NOT F#669-family**: parent `exp_prod_adapter_format_spec_v1` is **SUPPORTED** (not provisional / not `[?]`-KC); the breach is *scope* (Apple-only), not parent-claim-untestedness.
- **NOT F#666-pure standalone**: KCs are not all-proxy — K1654 has a target metric (MMLU-Pro within 5pp of MLX reference). The block is hardware/platform-availability, not target-pairing absence.
- **NOT tautological-inter-adapter-delta**: KCs are base-anchored (MLX-reference), not inter-variant.
- **NOT regular F#666-target-FAIL**: no measurement attempted; verdict is preempt-structural.
- **Sister to F#650**: F#650 killed `exp_prod_adapter_loader_portability` on the same axis (ap-017(s)) and explicitly named llama.cpp as one of the three out-of-scope loader stacks in T5(B). This experiment is the llama.cpp realisation — 2nd instance of the axis.

## Promotion flag for analyst

**2nd ap-017(s) hardware-topology-unavailable instance** (1st = F#650). If a 3rd instance arrives (CUDA-specific, safetensors-rs-specific, or any non-Apple PROD bridge), promote to top-level guardrail: "PROD experiments requiring runtime outside Apple-only-MLX parent scope preempt-KILL without PLAN.md Part 2 unblock (remote runner / converter infra / cross-runtime harness)."

**7th F#502/F#646 schema-completeness cohort hit** (`success_criteria: [] # MISSING`). Cohort is at sustained pressure — the schema-vs-instance-fix axis remains unaddressed in CLAIM tooling. Non-blocking for THIS verdict (T1/T5 alone block), but worth flagging for analyst LEARNINGS.

## Non-blocking notes

- K1655 ("works") is a textbook non-falsifiable KC — pre-reg pin discipline failure flagged in T4. Not the cause of this kill (T1/T5 alone are sufficient), but a recurring quality bar issue across PROD claims.
- Wall time 7.6s — confirms graceful-failure path; no model load attempted.

## Routing
- Researcher already executed `experiment complete --status killed` + `experiment finding-add` (F#763 registered, verified via `experiment finding-list`).
- Emit `review.killed` to advance analyst.
