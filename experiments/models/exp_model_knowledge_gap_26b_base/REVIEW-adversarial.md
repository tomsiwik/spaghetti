# REVIEW-adversarial — `exp_model_knowledge_gap_26b_base`

> Independent reviewer-hat pass. Overwrites the prior self-review by the filing
> researcher. Verdict: **PROVISIONAL** (BLOCKED-on-resource, macro-scope
> design-only sub-case per reviewer.md).

## Verdict

**PROVISIONAL** — accept the filing.

DB action: `experiment update --status provisional` + evidence (verdict
inconclusive) + finding-add (status=provisional). Route `review.proceed` with
`PROVISIONAL:` prefix to analyst.

## Adversarial checklist

Consistency
- (a) `results.json["verdict"]="PROVISIONAL"` matches proposed DB status
  `provisional` ✓
- (b) `all_pass=false` consistent with PROVISIONAL (every KC `untested`) ✓
- (c) PAPER.md L3 verdict line `PROVISIONAL (BLOCKED on base model not cached;
  F#478 monotonic prior strongly predicts kill)` ✓
- (d) `is_smoke=false`; this is design-only-blocked, not a smoke run ✓

KC integrity
- (e) MATH.md K1702/K1703 + added K1816 match DB-registered text byte-for-byte
  per `experiment get` output. K1816 was added pre-run as a numeric
  threshold-supplier for the vague K1703 (per F#666); not a post-failure
  swap. ✓
- (f) No tautology: K1702 measures `δ_d ≥ 5pp` on MMLU-Pro; K1816 measures
  `win_rate ≥ 60%` on N=30. F#666 paired proxy/target. ✓
- (g) Code KC IDs in `run_experiment.py` L75-L79 (`K1702_structural_proxy`,
  `K1703_target_behavioral`, `K1816_target_win_rate`) match MATH.md §5 ✓

Code↔math
- (h-l) Graceful-failure scaffold imports only `json`, `os`, `sys`, `time`,
  `pathlib`. No composition math, no LORA_SCALE, no routing, no `shutil.copy`,
  no hardcoded `pass: True` in KCs (all `"untested"`). ✓
- (m) Target model in MATH.md §6 (`mlx-community/gemma-4-26b-a4b-it-4bit`)
  matches `BASE_MODEL_ID` constant in `run_experiment.py` L35. The cache-check
  branch refuses to proxy to 4B (researcher antipattern 'm'). ✓
- (m2) MATH.md §6 cites `/mlx-dev` (phased-execution memory pattern) and
  `/fast-mlx` (`mx.compile`). Code is design-only — no MLX training-loop
  landed, so the (m2) antipattern about unidiomatic MLX does not apply. ✓

Eval integrity (N/A — no measurement performed)
- (n-q) ✓

Target-gated kill (F#666)
- (t) **N/A** — verdict is PROVISIONAL, not KILL. F#666 paired
  proxy (K1702) + target (K1816) preserved. ✓

Scope-changing fix
- (u) **N/A** — researcher explicitly refused to swap to a smaller variant
  (4B / E4B / 8B). Scaffold raises `NotImplementedError` on the live path
  guard, never falls back to a different base. ✓

Deliverables
- (r) PAPER.md L11-L17 prediction-vs-measurement table present (3 rows:
  P1/P2/P3, all "NOT MEASURED — untested"). ✓
- (s) Math: §3.1 monotonic claim is acknowledged as a "motivated prior, not
  a strict proof" (PAPER.md L37, MATH.md A1). §3.2 MoE-niche escape is
  paper-grounded (Fedus 2022, Zhou 2022). Honest framing. ✓

## Why PROVISIONAL (and not KILLED via §3.1)

The dense-capacity monotonic extension (§3.1) is *strongly* predictive but
not a strict proof — Gemma 4 26B-A4B is **MoE**, and the §3.2 niche
mechanism creates a non-monotonic capacity curve `M_eff(d)`. KILLED would
discard the MoE-niche hypothesis prematurely; the proper response is to
park as PROVISIONAL with an explicit unblock path (routing-distribution
measurement first, then targeted single-domain run). This matches the
"macro-scope design-only" PROVISIONAL sub-case in reviewer.md (Pattern 1:
§0 skill citations ✓; Pattern 2: graceful-failure `main()` writing
`results.json` ✓; Pattern 4: prediction-vs-measurement table all-rows
"not measured" ✓). Pattern 3 (`_impl` companion at P3) is **not** required
here because the unblock is external (14GB model cache + a separate routing
experiment), not new code.

## Doom-loop break verification

Prior researcher iteration RELEASED-TO-OPEN (2026-04-25, scratchpad
mid-stream). Same experiment was re-claimed this iteration. Per researcher
hat §0 doom-loop guidance, the structurally-different action is to
**escalate to PROVISIONAL via reviewer hat** rather than file a 2nd
RELEASE-TO-OPEN. The current researcher pass took that action by emitting
`experiment.done` with payload requesting reviewer escalation; this pass
performs it.

## Drain-scope note

P=3 (already downgraded from P=1). `experiment update --status provisional`
clears the experiment from `active` (where it is currently stuck), which
satisfies success criterion 2 of the drain objective (`active` queue
empty). It does **not** satisfy criterion 1 directly (P≤2 open queue still
has 14 entries) — but does prevent the loop from re-claiming this exp.

## Routing decision

`review.proceed` with `PROVISIONAL:` prefix to analyst hat for LEARNINGS.md
write (the only deliverable still missing per `ls -la`).

## Attack surfaces I considered

1. **"Could you upgrade to KILLED on the F#478 monotonic prior?"** — No.
   The MoE-niche mechanism is non-trivial and paper-grounded. KILL would
   need either (a) empirical 26B-A4B data or (b) a tighter proof
   accounting for the routing-niche case.
2. **"Could the experiment proceed at the platform's compute budget?"**
   — No. 14GB download + ~2.5h training × 3 domains exceeds the
   single-iteration 30-min/40-tool-call cap (guardrail 1009). Researcher
   correctly refused to deferred-train silently.
3. **"Is the MoE-niche §3.2 mechanism real or an excuse?"** — Real.
   Fedus 2022 (Switch Transformer) and Zhou 2022 (MoE scaling) explicitly
   discuss expert-routing variance creating per-domain effective-capacity
   gaps. Whether Gemma 4 26B-A4B exhibits narrow routing on any of
   {code, math, medical, legal, finance} is an empirical question
   answered by the routing-distribution experiment named in PAPER.md §3.

## Assumptions logged

- A1. `mem-antipattern-blocked-on-resource-provisional` (if it exists) or
  the macro-scope design-only sub-case is the correct routing precedent
  for "BLOCKED on 14GB cache + multi-hour compute" experiments. The prior
  researcher hedged on whether F#1629 is a finding-precedent; my check
  shows F#1629 references K#1629 (kill ID), not a finding. So this
  filing is the **first** finding-ledger record of the
  "BLOCKED-on-resource + proof-first kill prior + non-trivial escape
  mechanism" pattern combination — worth registering as a new
  finding-status `provisional` to anchor future BLOCKED filings.
