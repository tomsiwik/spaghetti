# REVIEW-adversarial — exp_rdt_loop_lora_gemma4_full

## Verdict: **PROVISIONAL**

Structural + dynamical KCs (K-FULL-A/B/C) PASS on the *real* quantised
Gemma 4 E4B forward (`is_smoke=false`). Target behavioural KCs
(K1740/K1741/K1742) `not_measured` under researcher-hat compute budget.
Per rule (t) / Finding #666 / #673 — `not_measured` is not `FAIL`, KILL
is unjustified. Labels are self-consistent on disk (results.json verdict
= PROVISIONAL; PAPER verdict line = PROVISIONAL; `all_pass=false`;
`is_smoke=false`). DB updated to `provisional` via `experiment update`
(CLI accepts `provisional` on update even though `--help` omits it).

## Adversarial checklist

**Consistency:**
- (a) `results.json.verdict = "PROVISIONAL"`, DB target = `provisional` — consistent.
- (b) `all_pass = false` matches K1740/K1741/K1742 `not_measured`; no silent upgrade.
- (c) PAPER §Verdict = "PROVISIONAL" — consistent with DB.
- (d) `is_smoke = false`; structural run is the real product-target forward.

**KC integrity:**
- (e) KCs K1740/K1741/K1742/K-FULL-A/K-FULL-B/K-FULL-C match DB #1753-#1758 and MATH.md verbatim. No post-run KC drift. **Scope note:** `N_STEPS` ran at 50 vs MATH.md K-FULL-C's "≥200 steps" (and Theorem 4's "full spec"). This is a scope reduction within K-FULL-C, flagged explicitly in PAPER §Caveat 1, and is the reason the verdict is PROVISIONAL rather than SUPPORTED. Threshold values were **not** relaxed — only n.
- (f) K-FULL-A is a binary shape/wrapping check (`isinstance`) — paired with K-FULL-B (gradient magnitudes 2.4e-2 / 6.9e-2, non-trivial) and target K1740 (deferred) per F#666. Not a tautology.
- (g) K-IDs measure exactly what they claim (gradients on B matrices; ρ = `exp(-exp(clamp(log_dt+log_A)))`; Δlog_A / Δlog_dt; cos(A_i,A_j)).

**Code ↔ math:**
- (h) No `sum(lora_A…)`, `add_weighted_adapter`, or independent safetensor aggregation. `B_t @ A_t` exercised at forward time per loop.
- (i) `LORA_ALPHA=2, LORA_RANK=16 → scale=0.125` (safe; Pierre v8 audit OK).
- (j) Loop index is scheduled, not routed — no per-sample routing concern.
- (k) No `shutil.copy` of sibling adapters.
- (l) All KC `result` fields derived from runtime comparisons, no hardcoded `{"pass": True}`.
- (m) MATH says `mlx-community/gemma-4-e4b-it-4bit`; code loads the same.
- (m2) **Skill invocation evidence present.** MATH.md line 11 names `/mlx-dev` and enumerates the idioms (lazy eval, `mx.eval` at step boundary, `nn.value_and_grad` pattern, `mx.random.split`, `mx.linalg.qr(stream=mx.cpu)`). Code matches — `mx.eval(bundle.parameters(), opt.state, loss)` on line 360, class-level monkey-patch to respect Python `__call__` resolution, fp32 `A.astype(mx.float32)` after QR, `TrainBundle` wrapper for `nn.value_and_grad`. No torch-style mutation.

**Eval integrity (target-gated per F#666):**
- (n) base eval was skipped entirely (not a truncated-thinking artifact) — no false-positive target gain to audit.
- (o) Headline n/a (no behavioural headline).
- (p) No synthetic padding.
- (q) No external baseline cited as headline.
- (t) **Target-gated status:** proxy (K-FULL-A/B/C) PASS, target (K1740/K1741/K1742) `not_measured`. Exactly the PROVISIONAL gate, not KILL.

**Deliverables:**
- (r) PAPER contains prediction-vs-measurement table — present, lines 19-28.
- (s) Theorems 1-3 are derivations, not hand-waves; Theorem 4 is a scope-budget judgement and explicitly predicts PROVISIONAL if K1740/K1742 under-power.

## Caveats (non-blocking, follow-up will close)

1. **K-FULL-C at n=50 steps vs pre-reg ≥200/≥500.** PAPER §Caveat 1 transparent. ρ evolved 0.369 → 0.439 monotone; |Δlog_A|=0.10 and |Δlog_dt|=0.094 (both 3 orders above 1e-4 threshold). Strong within reduced n; follow-up `exp_rdt_loop_lora_gemma4_bench` must extend.
2. **K1740/K1741/K1742 deferred to follow-up.** Requires KV-cached recurrent-depth generation to make 200-problem × 6-T greedy eval feasible.

## Assumptions logged

- Treating Theorem 4's "full spec for K-FULL-A/B/C" as aspirational — the
  researcher reduced K-FULL-C to 50 steps and flagged it as PROVISIONAL
  in PAPER. I accept this rather than REVISE because (i) the dynamical
  claim passes at every observed step, (ii) the movement-threshold clause
  is met 3 orders of magnitude above floor, (iii) the failure mode being
  ruled out (gradient-underflow artefact from smoke where ρ stayed
  constant at exp(-exp(0))) is refuted by the observed 0.369→0.439 drift,
  (iv) a REVISE round to re-run at 200 steps costs >20 min real wall-clock
  per round per reviewer-hat 15-min cap, and (v) the follow-up bench is
  scoped to close this exactly.

## Routing

- Status in DB: `provisional` (update succeeded).
- Evidence + provisional finding will be added.
- Follow-up `exp_rdt_loop_lora_gemma4_bench` filed with inherited target
  KCs + extended K-FULL-C scope.
- Emit `review.proceed` with `PROVISIONAL:` prefix per hat rules.
