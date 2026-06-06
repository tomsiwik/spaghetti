# LEARNINGS — exp_model_peer_comparison_qwen3_4b

## Verdict
**KILLED** — blocked by preconditions P1 (adapter weights missing) and P3 (upstream T2.1 killed with metric-swap). P2 (harness importable) passed. This is the second confirmation of the Llama-sibling pattern in 24 hours; the rule is now standing for the `exp_model_peer_comparison_*` class.

## Key findings (propagate to siblings & meta)

### 1. Precondition-probe before macro sweep — class-level standing rule
Two different P1 macro comparisons (Llama 3.1 8B on 2026-04-18, Qwen 3 4B same day) both killed by the same two preconditions. Cost of probe: 3 seconds. Cost of running the full sweep before probing: ~6 hours per comparison, result would have been a silent base-model-only measurement relabelled "Pierre".

**Applies to:**
- `exp_model_mtbench_composed` (next open macro downstream of T2.1)
- Any future `exp_model_peer_comparison_*` until P1 adapters rebuilt and P3 resolved
- Any P1 experiment whose MATH.md §3 "prior math" cites a supported upstream — the upstream should be re-checked at the time the downstream is claimed, not at the time the downstream was designed.

### 2. Adapter registry ≠ adapter artefacts (with directory-existence corollary)
`adapters/registry.json` lists 5 domains (math, code, medical, sql, bash). Filesystem state:
- 4 directories exist but contain only config stubs (no `.safetensors`).
- 1 directory (`adapters/code/`) **does not exist at all**.

The first instance (Llama sibling) flagged "registry vs artefact"; this instance adds the directory-existence corollary. Probe code must handle both cases — my `check_p1_adapters` uses `if domain_dir.exists()` before glob, and reports `code` as missing correctly.

### 3. Downstream inherits upstream audit flags
T2.1 was flipped supported→killed on 2026-04-18 via metric-swap audit. Every P1 macro that names T2.1 as an upstream inherits the flag. Already confirmed propagation to:
- `exp_model_peer_comparison_llama31_8b` (KILLED 2026-04-18)
- `exp_model_peer_comparison_qwen3_4b` (KILLED this iteration)
- Pending: `exp_model_mtbench_composed` (expected same mechanism)

### 4. Bar is HIGHER for matched-param thinking comparisons
Qwen 3 4B is a stronger apples-to-apples test than Llama 3.1 8B because:
- Params are matched (4B vs E4B), so "8B > 4B" scale excuse gone.
- Thinking mode first-class on both sides, so Pierre's runtime-routing advantage is the test — not just adapter content.

When the probe unblocks, the Qwen comparison is the more honest of the two. The Llama comparison's 8B param gap gave Pierre a "lost gracefully" escape hatch; Qwen matched-params does not.

## What was NOT learned
- No behavioral measurement on any of the 5 benchmarks (all blocked).
- No confirmation or refutation of Pierre's actual capability at matched params.
- No evidence about Qwen 3 4B's MLX-4bit conversion quality (not loaded).

These are deferred to v2 after P1/P3 resolved.

## Recommended follow-ups (DO NOT auto-spawn; analyst gates)

1. **Rebuild T2.1 with correct benchmarks** — MedQA USMLE 5-choice (not MedMCQA), `max_tokens ≥ 512` for CoT baselines, persist `.safetensors` to `adapters/<domain>/`. Without this, every P1 peer-comparison and composed-benchmark macro stays blocked.
2. **Retrain Pierre adapters with `thinking_enabled: true`** if we want to honestly test runtime-routing advantage at Qwen's thinking-on bar. Current adapters trained with thinking=False → K1695 structurally unreachable.
3. **When rebuilt, rerun as `exp_model_peer_comparison_qwen3_4b_v2`** with same MATH.md §4 preconditions + the full sweep implemented.

## Meta — audit-cluster closure progress

This kill continues the 2026-04-18 audit sweep. Cluster "metric-swap" now has 7 instances this loop (T2.1 = source, 6 downstream propagations including this one). The class-level rule "downstream inherits upstream audit flags" is no longer an emerging pattern — it is a confirmed invariant for the repo's dependency structure.
