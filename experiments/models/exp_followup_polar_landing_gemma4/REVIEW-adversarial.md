# REVIEW-adversarial — exp_followup_polar_landing_gemma4

**Verdict: KILL (preempt-structural — F#666-pure standalone clause)**
**Reviewer pass:** 2026-04-25, drain-window iteration ~32
**Triggering event:** experiment.done (KILLED preempt-structural payload from researcher)

---

## Verdict basis (compound, all four required for confidence)

1. **F#666-pure standalone (governing clause)** — single KC #1561 measures `sr(V)` collapse, a structural-manifold proxy. No paired target KC (no behavioral / task-accuracy / oracle-gap KC). `success_criteria` empty. `depends_on=[]`. Pre-reg fails F#666-discipline pre-measurement; the F#666-pure standalone preempt-KILL clause governs. Antipattern memory match: `mem-antipattern-f666-pure-standalone-preempt-kill` (canonical).
2. **Parent-supersession (3 directly relevant prior findings)** — F#419 (killed, Qwen) + F#442 (supported, joint Stiefel verified on **actual Gemma 4 E4B**, sr=r exactly) + F#444 (supported, joint Stiefel scale stability on Gemma 4). Replicating the known-broken variant on a target where the known-correct variant is verified adds zero behavioral information.
3. **Architecture-irrelevance** — Pierre/P1 deployment surface uses joint Stiefel PoLAR per F#442/F#444 + project memories. Landing-on-U-only is not in the deployment surface. Even a clean PASS or FAIL of K#1561 transfers no actionable signal.
4. **Disease-vs-symptoms compound** — F#419's `Impossibility Structure` field identifies disease as **rank-1 single-domain SFT gradient** (universal, not Qwen-specific). The followup's `Fix: replicate on actual target model` treats the Qwen-proxy choice as the disease — symptom-level fix to a structural finding.

---

## Adversarial checklist (a-u)

- (a) Consistency: results.json `verdict=KILLED`, claim status `killed` — **PASS**.
- (b) `all_pass=false`, status `killed` — **PASS**.
- (c) PAPER.md verdict line "KILLED (preempt-structural)" — **PASS**.
- (d) `is_smoke=false`; no full-run claim — **PASS**.
- (e) KC #1561 unchanged in DB; no post-claim KC mutation — **PASS**.
- (f) Tautology sniff test: 4-cell truth table is degenerate **by design** (this is the *finding*, not a code-level tautology). KC outcomes do not pass by algebraic identity — they fail to identify any behavioral conclusion under either branch. **PASS** (the degenerate truth table is the preempt argument, not a defect).
- (g) K#1561 in code/results matches MATH.md description (sr(V) on landing-on-U-only) — **PASS** (graceful-failure stub; KC text exact match).
- (h)-(l) No MLX composition, LoRA scale, routing, shutil.copy, or hardcoded `{pass:True}` — graceful-failure stub imports only `json`+`pathlib`. **PASS**.
- (m) No model loaded; no proxy substitution. **PASS**.
- (m2) MATH.md §0 / Pre-flight notes "Platform skills invoked: N/A (no MLX code emitted; preempt-structural)" — matches the F#666-pure standalone preempt-KILL clause's no-MLX carve-out (no training-loop code to validate). **PASS**.
- (n)-(q) Eval integrity: not applicable — no measurement performed. **N/A**.
- (t) Target-gated kill (F#666): **DOES NOT APPLY** per F#666-pure standalone clause carve-out — F#666 is the *reason* for the preempt-KILL, not a blocker on it. No KC was measured, so neither proxy-FAIL+target-PASS nor proxy-PASS+target-FAIL ambiguity exists.
- (u) Scope-changing fixes: honest preempt-KILL with graceful-failure stub. NOT a silent SFT→LoRA swap, max_length reduction, monitoring disablement, or KC drop. **PASS**.
- (r) PAPER.md prediction-vs-measurement table present (single row, K#1561 untested with predicted measurement). **PASS**.
- (s) Math errors: theorem chain (T1 KC structural insufficiency → T2 parent-supersession via universal gradient mechanism → T3 architecture-irrelevance → T4 disease-vs-symptoms) is internally consistent. F#419's gradient-rank universality argument correctly cites parent's `Impossibility Structure`. F#442/F#444 correctly cited as Gemma-4-target-verified joint-Stiefel fix. **PASS**.

---

## Distinctions confirmed (not other KILL clauses)

- NOT F#669-family preempt-structural — `depends_on=[]` (parent-orthogonal); F#666-pure standalone is the correct clause.
- NOT tautological-inter-adapter-delta clause — KC is not of the form `op(f(variant_i), f(variant_j))`; single-variant structural property.
- NOT F#702 hygiene-patch PROVISIONAL — no target-metric KC exists; F#666-pure standalone fires regardless of hygiene count.
- NOT regular F#666 KILL — no KC measured (target or proxy); preempt-structural verdict.

## Promotion candidate (informational, not blocking)

LEARNINGS / scratchpad note F#762 as the **2nd audit-2026-04-17+followup-without-rerun super-family instance** (after F#761 spectral-surgery-followup). Per analyst hat §6 promotion threshold (3rd instance), no super-family-level guardrail promoted yet. **If a 3rd audit-2026-04-17+followup-without-rerun arrives**, promote tag-combination to top-level guardrail (preempt-KILL on tag combination alone, before KC inspection). Analyst should flag this in LEARNINGS.

## Routing

DB state already reconciled by researcher: `experiment complete --status killed` and `experiment finding-add` (F#762) already executed (verified via `experiment query` — F#762 present in DB findings index, experiment absent from active/open lists). No further DB writes required from reviewer.

Emit `review.killed` to advance to analyst.
