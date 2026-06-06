# REVIEW-adversarial — exp_followup_grassmannian_native_macro

## Verdict: KILL (ratifies existing DB status=killed)

16th consecutive audit-2026-04-17 cohort precondition-probe KILL. Runner is
a pure file-probe + DB-check tripwire; no MLX, no composition, no cosine
computation. K1550 correctly marked UNMEASURABLE per MATH.md tripwire.

## Adversarial checklist (all (a)–(s))

**Consistency:**
- (a) results.json verdict=killed ↔ DB=killed ↔ PAPER="Verdict: KILLED" — PASS
- (b) all_pass=false, no supported claim — PASS
- (c) PAPER verdict line = "Verdict: KILLED" (no PROVISIONAL/DEGENERATE) — PASS
- (d) is_smoke=false, probe is real — PASS

**KC integrity:**
- (e) MATH.md authored for this probe; no post-hoc KC relaxation. K1550 pre-
  registered as tripwire before runner executed — PASS
- (f) No tautology — probe evaluates three independent file-system + DB
  conditions, none reduce to an identity. K1550 = All(P1,P2,P3) does NOT
  test the orthogonality claim; it tests measurement feasibility, which
  MATH.md §Kill Criterion states explicitly — PASS
- (g) K1550 in code matches MATH.md §26 text and DB kill-criteria text — PASS

**Code ↔ math:**
- (h) No `sum(lora_A`, no `add_weighted_adapter`, no safetensor combination
  logic in runner (pure glob + subprocess) — PASS
- (i) No LORA_SCALE hard-coded (no training) — N/A
- (j) No routing (no eval) — N/A
- (k) No `shutil.copy` — PASS
- (l) Kill-criteria dict uses dynamic `"pass" if all_pass else "fail"`, not
  hardcoded `True` — PASS
- (m) Target model Gemma 4 E4B / Qwen3-4B in MATH.md; runner never loads
  a proxy; probes look for exactly these target patterns — PASS
- (m2) No platform code exercised (no mlx-lm import, no `mx.eval`, no
  forward pass). Skills requirement (PLAN.md Part 2 `/mlx-dev`, `/fast-mlx`)
  does not apply to pure file-probe runners — N/A

**Eval integrity:**
- (n) No base eval — N/A
- (o) Headline n: no eval, no N — N/A
- (p) No adapters claimed as "N=25 domains" — N/A
- (q) No cited baseline in this run — N/A

**Deliverables:**
- (r) PAPER.md §Prediction vs. measurement table present with all 6 gates — PASS
- (s) Math claims honest. PAPER openly documents P1 probe-bias (loose glob
  matched 15 non-target pre-Gemma-4 adapters) and explicitly states verdict
  is unaffected because P2=FAIL and P3=FAIL independently force KILL. This
  is scientific honesty, not a claim error — PASS

## Independent verification

- P1 loose-glob false-positive confirmed: 15 hits are all non-target
  pre-Gemma-4 experiments (`exp_knowledge_disentanglement_control`,
  `exp_score_kl_constrained_mcq`, `exp_method_vs_domain_adapter`,
  `exp_p11_cloq_calibrated_init`, etc.). A stricter glob scoped to
  `gemma*4*` / `qwen3*4b*` would return 0 — correctly flagged in PAPER.
- P2 verified: no `grassmannian*/**/*.safetensors`, `ap_init*/**/*.safetensors`,
  or `*grassmannian*init*.json` on disk.
- P3 verified: `experiment get exp_p1_t2_single_domain_training` returns
  Status: killed (K1030 metric-swap, K1028 format-artifact).
- DB status already `killed` — no silent upgrade risk, no double-complete.

## Cohort escalation — ORCHESTRATOR-ACTION-REQUIRED

This is the 16th consecutive audit-2026-04-17 cohort KILL. The event
payload flagged an upstream-claim obstacle this iter: **killed experiments
cannot be re-claimed via `experiment claim`** (CLI refuses "Cannot claim —
status is killed, not open"). This means the highest-leverage single
action — rerunning `exp_p1_t2_single_domain_training` — is mechanically
blocked for researcher hats.

Orchestrator options (neither is a reviewer-hat action):
1. Design a **v2 experiment** `exp_p1_t2_single_domain_training_v2` that
   preserves KCs + pre-registers code-bug fixes (K1028/K1030 root causes).
2. Allow researcher to run `experiment update --status open` on a killed
   upstream when the fix is scoped to runner-code bugs only. Guardrail
   1009 must still forbid silent KC changes.

Until one is applied, the claim queue will keep surfacing cohort members
and researchers will keep probe-KILLing them — the work is bounded but
continues to burn iterations.

## Non-blocking notes

- PAPER §Probe-bias is a good template for future probe-runner
  documentation. Loose P1 is acceptable when P2 & P3 independently force
  the verdict; flag openly so a resurrector can tighten.
- LEARNINGS.md owned by next hat should capture the 16-instance cohort
  saturation pattern and the orchestrator-action-required escalation.

## Assumptions logged

- I treated the event payload's "orchestrator must design v2 OR permit
  open-reset" as a real operational constraint worth propagating into
  the finding, not just a researcher-hat side note.
- I did not re-run the probe; results.json wall_seconds=1.054 + my
  `experiment get` cross-check is sufficient for verification.
