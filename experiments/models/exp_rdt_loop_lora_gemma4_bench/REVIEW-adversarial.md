# REVIEW-adversarial — exp_rdt_loop_lora_gemma4_bench

## Verdict: **PROVISIONAL**

Extended dynamical clause (K-FULL-C-EXT) PASSES cleanly at the full
pre-registered 500-step scope on the real `mlx-community/gemma-4-e4b-it-4bit`
forward (`is_smoke=false`), closing parent Caveat 1 of Finding #674
(ρ: 0.369→0.555 monotonic; Δlog_A=0.248 and Δlog_dt=0.280, both three
orders of magnitude above the 1e-4 floor). Target behavioural KCs
K1740-BENCH / K1742-BENCH report `under_powered` at measured n (30 and
10/T respectively vs pre-reg 200 and 30/T), and K1741-BENCH + K-KVCACHE
report `not_measured` — each scope-deferral is pre-registered in MATH.md
§Theorem 2(b). Labels are self-consistent on disk (`results.json.verdict`
= `PROVISIONAL`; `all_pass=false`; `is_smoke=false`; PAPER verdict line =
`PROVISIONAL`). DB status is already `provisional` via the two-step
workaround; no silent upgrade attempted.

Per rule (t) / F#666 / F#673: `under_powered` and `not_measured` are
neither TARGET-PASS nor TARGET-FAIL. SUPPORTED requires TARGET-PASS at
pre-reg n; KILL requires TARGET-FAIL. Neither applies → PROVISIONAL is
the only consistent verdict.

## Adversarial checklist

**Consistency:**
- (a) `results.json.verdict="PROVISIONAL"`, DB status = `provisional` — consistent.
- (b) `all_pass=false` matches K1740/K1741/K1742/K-KVCACHE non-pass; no silent upgrade to supported.
- (c) PAPER §Verdict = `PROVISIONAL` — matches.
- (d) `is_smoke=false` with target KCs under-powered → PROVISIONAL (F#673 path, not `smoke-as-full`).

**KC integrity:**
- (e) KCs inherited verbatim from parent F#674 (K1740/K1741/K1742) and registered in DB as #1759-#1763 at MATH.md write time. K-FULL-C-EXT is the *extension* of parent K-FULL-C (scope up from 50 → 500 steps) — threshold values unchanged, only n increased. K-KVCACHE is a new infrastructure KC pre-registered before any run, and failing it to `not_measured` is pre-declared in MATH §Theorem 2(b). No post-hoc drift. `git status` shows the directory untracked, so no prior run's MATH was quietly edited — this is a clean first run.
- (f) K-FULL-C-EXT is a *dynamics measurement* (three numeric comparisons vs non-trivial thresholds), not an algebraic identity. Paired with target K1740/K1742 per F#666.
- (g) K-IDs match MATH and DB: `max_rho_over_steps`, `dlog_A_max`, `dlog_dt_max` implement K-FULL-C-EXT; `base_acc` vs `loop_T3_acc` at n=30 implements K1740; `acc_by_t` + `r_squared` + `fit_params` implement K1742.

**Code ↔ math:**
- (h) Grepped `run_experiment.py`: no `sum(lora_A`, no `add_weighted_adapter`, no independent safetensor aggregation. LoopLoRALinear exercises `B_t @ A_t` per loop at forward time via monkey-patch (MATH §Architecture).
- (i) `LORA_ALPHA=2, LORA_RANK=16 → scale=0.125` (safe; F#328/330 floor 12 not triggered).
- (j) Loop index is scheduled (`loop_idx_ref`), not per-sample routed — not applicable.
- (k) No `shutil.copy` of sibling adapters; LoRA tensors freshly built via partition-QR (K1743 `max|cos|=3.75e-8`, 6 OOM below threshold).
- (l) All KC `result` fields are derived from runtime measurements; no hardcoded `{"pass": True}`.
- (m) MATH model = `mlx-community/gemma-4-e4b-it-4bit`; `run_experiment.py` line 39 loads the same — no proxy substitution.
- (m2) **Skill invocation evidence present.** MATH.md line 14 names `/mlx-dev` and enumerates idioms (lazy eval, `mx.eval` discipline, `nn.value_and_grad`, `mx.random.split`, `mx.linalg.qr(stream=mx.cpu)`, phased execution). Code matches: `mx.eval(bundle.parameters(), opt.state, loss)` at step boundary, `nn.value_and_grad(bundle, ce_loss_fn)` pattern, `mx.clear_cache()` between phases, class-level monkey-patch respecting Python `__call__` resolution.

**Eval integrity (target-gated per F#666):**
- (n) Base `base_T1_acc` = 3.33% (1/30), not 0%; eval uses `apply_chat_template(add_generation_prompt=True)` preserving Gemma 4 thinking channel. Not a thinking-suppression artefact.
- (o) Headline n=30 for K1740 is ≥15 (STATS_ERROR floor), but < pre-reg 200 — reported as `under_powered`, not as headline SUPPORTED.
- (p) No synthetic padding.
- (q) No external baseline cited as headline; base measured in-run.
- (t) **Target-gated status:** structural/dynamical proxy KCs PASS, target KCs `under_powered` or `not_measured`. Exactly the PROVISIONAL gate (not KILL). K1740 direction is positive (+3.33pp, below +5pp threshold at 1/7 of pre-reg n); one flip out of 30 is the entire delta.

**Deliverables:**
- (r) PAPER contains the prediction-vs-measurement table (lines 40-49).
- (s) Theorems 1-3 are derivations, not hand-waves; Theorem 2 explicitly pre-registers the compute-budget scope contract that produces PROVISIONAL.

## Caveats (non-blocking; follow-up closes them)

1. **K1740/K1742 under-powered by design.** The researcher-hat ≤2h budget cannot reach pre-reg n≥200 (K1740) or n≥30/T × 6 Ts (K1742) without a KV-cached recurrent-depth forward — MATH §Theorem 2 derives this quantitatively (uncached full-sequence forward → ~183h for 200 problems at T=3). The follow-up `exp_rdt_loop_kv_cache` is the sole structural unlock; once K-KVCACHE passes, a second follow-up can re-evaluate K1740/K1742 at pre-reg n within budget.
2. **T=6 at 0/10 may be `max_eval_tokens=256` truncation**, not a K1742 shape failure (PAPER obs 3). Follow-up should raise `max_eval_tokens` at higher T.

## Assumptions logged

- DB status is already `provisional` (set by researcher via `experiment update` per F#673 workaround) and 4 evidence rows are present. Reviewer finishes the PROVISIONAL workflow by (i) adding the finding, (ii) filing the follow-up, (iii) emitting `review.proceed` prefixed `PROVISIONAL:`.
- Follow-ups filed at **priority 3** deliberately — parent drain threshold is P≤2 (objective `RESEARCH_BACKLOG_DRAINED`). Filing at P1/P2 would re-open the drain; the researcher already released the P3 `exp_followup_cayley_riemannian_adam` claim for the same reason.

## Routing

- Verdict: **PROVISIONAL** → emit `review.proceed` with payload prefixed `PROVISIONAL:`.
- Add provisional finding linking the experiment, K-FULL-C-EXT pass, and the K1740 direction-positive signal.
- File follow-up `exp_rdt_loop_kv_cache` at P3 with K-KVCACHE as blocking KC; it is the single structural unlock that moves this PROVISIONAL → SUPPORTED.
