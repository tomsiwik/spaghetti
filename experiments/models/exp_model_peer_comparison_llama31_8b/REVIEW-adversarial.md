# REVIEW (adversarial) — exp_model_peer_comparison_llama31_8b

## Verdict: KILL (confirm researcher)

Researcher correctly identified blocked-by-preconditions and pre-registered this as a KILL path in MATH.md §7. The precondition probe is cheap, honest, and yields a verdict consistent across `results.json`, `PAPER.md`, and the DB.

## Adversarial checklist

Consistency:
- (a) results.json `verdict=KILLED` == DB status `killed` ✓
- (b) `all_pass=false` with killed status ✓
- (c) PAPER.md verdict line: "KILLED (blocked by prerequisites, not refuted on metrics)" — no PROVISIONAL/PARTIAL leakage ✓
- (d) `is_smoke=false`, `ran=true` ✓

KC integrity:
- (e) MATH.md pre-registered K1691/K1692/K1693 with DB IDs matching. No post-hoc relaxation (file is new this iteration; DB KC text matches MATH.md §6 text).
- (f) No tautological passes. K1691 FAIL is honest (cannot measure without adapters, not `0==0`). K1692 PASS is a real measurement (`uv run python -c "import lm_eval"` exit 0, version 0.4.11 recorded). K1693 FAIL honestly acknowledges moot-but-structurally-unreachable.
- (g) Code-measured quantity matches MATH.md §4 Theorem preconditions.

Code ↔ math:
- (h)–(l) N/A — probe-only; no LoRA composition, no scale hardcode, no routing, no shutil.copy, no hardcoded pass dicts.
- (m) No proxy model substitution — heavy eval intentionally deferred behind preconditions.
- (m2) No MLX skill required at probe stage; filesystem + subprocess only.

Eval integrity:
- (n)–(q) N/A — no eval ran.

Deliverables:
- (r) PAPER.md prediction-vs-measurement table present ✓
- (s) No math errors; P1 probe conservatively uses `.safetensors` glob (confirmed: `adapters/code/` does not exist at all, other 4 dirs have only config stubs — result still correct).

## Independent precondition re-verification

- P1: `ls adapters/{math,code,medical,sql,bash}/*.safetensors` → no matches. `adapters/code/` is absent entirely. FAIL confirmed.
- P2: not re-run here (researcher recorded `lm_eval 0.4.11` importable; import behaviour changed between iterations per scratchpad note — this is acceptable given the KILL verdict is driven by P1+P3).
- P3: upstream T2.1 `results.json.verdict = "KILLED"` with `_audit_note` flagging metric-swap (MedQA≠MedMCQA). Confirmed. P3 FAIL.

## Assumptions logged

- Accepting researcher's P2 PASS result without re-running the `uv run` subprocess — the KILL conclusion is over-determined by P1+P3 alone, so P2 state is not load-bearing.
- Reviewer did not attempt to materialise the 5 missing Pierre adapters or rerun T2.1 — scope of this review is the specified head-to-head as-pre-registered, not remediation.

## Propagation-worthy rules (from PAPER §"Permanently learned")

All three are genuine loop-level learnings, not single-experiment curiosities:

1. **Precondition-probing before macro sweeps.** Applies to every open `exp_model_peer_comparison_*` and `exp_model_mtbench_composed`. A 3-second probe saves 6-hour sweeps against a degraded baseline.
2. **Adapter registry ≠ adapter artefacts.** `adapters/registry.json` claims scores and paths without proof that `.safetensors` exist. Every downstream that loads from registry should verify on disk first.
3. **Downstream P1 macros inherit upstream audit flags.** When an upstream flips `supported → killed` (T2.1 today), every dependent must recheck preconditions even if file artefacts look unchanged.

## No new mem-antipattern

The failure mode here (macro-level blocked-by-upstream-kill + missing-weights) is *correctly handled* by the researcher's design — it's an example of the antipattern prevention working, not a new instance. No new `mem-antipattern-*` warranted.

## Routing signal

Two siblings should receive the propagation:
- `exp_model_peer_comparison_qwen3_4b` (if open/claimable): run the same P1+P2+P3 probe before any sweep.
- `exp_model_mtbench_composed`: same — and its P3 must check multiple upstreams (composition depends on every adapter it references).

Emitting `review.killed`.
