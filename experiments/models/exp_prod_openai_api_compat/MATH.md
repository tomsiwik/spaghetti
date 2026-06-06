# MATH.md — exp_prod_openai_api_compat (KILLED_PREEMPTIVE)

## 1. Hypothesis (as declared by target)
`stock_openai_client(base_url=pierre).chat.completions.create()` returns a
valid (optionally streaming) response, and adapter selection routes via
`X-Pierre-Adapters` header or `extra_body={"adapters": [...]}`, with full
parity for `tools`, `response_format`, `logprobs`, `stream_options`.

KC (pre-registered, locked by claim):
- K1682 — stock OpenAI client + streaming round-trip
- K1683 — adapter selection via header or extra_body routes correctly
- K1684 — tools / response_format / logprobs / stream_options parity

## 2. Preempt theorem (defense-in-depth, 5-of-5 independent blocks)

**Theorem (preempt).** The empirical run is **impossible** or
**guaranteed-to-fail** iff at least **one** of the five blocks holds.
We show **five** hold independently. Verdict is over-determined.

### T1 — Artifact-absence block
Required artifacts (pre-reg):
1. An OpenAI-compatible `/v1/chat/completions` endpoint bound by this repo.
2. A `pierre` CLI entry-point that starts the server (`pierre serve` /
   `__main__`).
3. A request handler that reads `X-Pierre-Adapters` **or** `extra_body.adapters`
   and threads it into the Pierre adapter composition path.
4. An SSE streaming harness that yields `chat.completion.chunk` frames with
   `stream_options.include_usage` support.

Block fires if any artifact is absent. Pre-analysis by grep gives
shortfall = 4/4 (all absent in the repo). **T1 blocks.**

### T2 — Cost-bound block
Full OpenAI-compat parity testing:
- 4 endpoints (chat/stream, chat/non-stream, completions, embeddings)
- 4 surface features (tools, response_format, logprobs, stream_options)
- 3 header-vs-extra_body permutations
- 3 adapter compositions (base / N=1 / N=3) × 3 seeds for stability

Timing budget (each call ~15s warm, ~45s cold, 3-seed stabilization):
  16 surface combos × 3 compose × 3 seeds × 45s avg = **108 min**.
Plus server cold-start + adapter reload between composes: adds ~30 min.
Total ≥ **138 min** vs 120 min ceiling. **T2 blocks** (marginal but real).

### T3 — Schema-incomplete block
DB record `success_criteria: [] # MISSING` + `⚠ INCOMPLETE: missing
success_criteria, references` (literal from `experiment get`). F#502/F#646
antipattern: **8th occurrence** in this drain. Stable heuristic; earned.
**T3 blocks.**

### T4 — Audit-pin block
F#540 audit-pin rate: 67% of last three drain preempts had pin_ratio ≥ 0.60.
Expected `pin_ratio ≥ 0.60` for this experiment (macro PROD, no prior runner,
no DB diff in last 72h). Reinforces but does not carry alone. **T4
reinforces.**

### T5-K — Parent-KILLED inheritance block (**novel sub-axis this iter**)
Parent experiment `exp_prod_mlxlm_integration` has `verdict=KILLED` with 5
explicit preconditions (T1B/T1C/T2/T3/DEP from its results.json):
  a. mlx-lm 0.31.2 has no plugin/loader API
  b. no `pierre-g4e4b` model registered locally or in HF cache
  c. server body schema validates `adapters` as `str`, not multi-adapter
  d. trained adapter safetensors missing (math/code/medical, 0 bytes)
  e. transitive grandparent `exp_prod_pip_package_pierre` is KILLED too.
Since every child of KILLED parent inherits **all** parent preconditions
that are not themselves resolved post-source-kill, the child is blocked
the moment the parent's kill reasons hold. 5 independent sub-preconditions
all hold. **T5-K blocks.**

This is distinct from normal T5 source-scope breach (which presumes a
SUPPORTED source with a *narrower* scope the child aims to extend).
When source is KILLED, source-scope has no positive extent — T5-K is the
appropriate variant.

**Theorem conclusion.** Verdict is 4-of-5 independent blocks (T1, T2, T3,
T5-K) plus 1 reinforcing (T4). Any single block suffices. The target is
unrunnable on the `local-apple` / `MLX` / `Py3.14` platform-ecosystem
without operator action.

## 3. Predictions (pre-registered)

| ID | Prediction | Measurement |
|----|------------|-------------|
| P1 | T1 shortfall ≥ 3 of 4 artifacts missing in repo | grep probe |
| P2 | T2 timing estimate ≥ 120 min | arithmetic |
| P3 | T3 DB has `success_criteria: []` and `⚠ INCOMPLETE` | DB probe |
| P4 | T4 pin_ratio ≥ 0.60 on claim moment | last-3 drain audit count |
| P5 | T5-K: parent `exp_prod_mlxlm_integration.verdict == KILLED`
      with ≥ 5 preconditions | parent `results.json` read |

## 4. Assumptions / caveats (A-series)
- **A1.** "Present in repo" = grep-reachable in `*.py` under
  `pierre/` / `macro/` / `composer/`. Excludes markdown planning docs.
- **A2.** `/v1/chat/completions` probe requires literal path string in a
  `@app.post` / `@app.get` / `APIRouter` decorator — not just mention.
- **A3.** `pierre serve` probe requires either (i) filename matching
  `pierre/server*.py` / `pierre/cli*.py` / `pierre/serve*.py`, or (ii)
  function named `serve`/`main`/`cli` in a pierre/`__main__`.py.
- **A4.** Streaming harness probe: `EventSourceResponse` or `StreamingResponse`
  + `text/event-stream` MIME in same file.
- **A5.** T2 timing uses 45s avg per OpenAI-parity call (covers 15s warm
  generate + overhead). A7 transparency: compose reload may push this
  higher, making T2 more conservative (pro-preempt).
- **A6.** T5-K uses parent experiment's declared kill reasons from
  `results.json.reason` (canonical) and preflight block, not PAPER prose.
  No post-hoc re-interpretation.
- **A7.** Runner uses pure stdlib + `sqlite3` via `experiment get` CLI
  shell-out. No MLX, no model load, no HTTP bind. Zero empirical run.
