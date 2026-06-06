# Peer Review: m2p_hard_moe (Hard Top-1 Gumbel MoE)

**Verdict: KILL** — K861 literal FAIL (median 0.239 < 0.25) and pre-registered D1 re-label gate robust across re-runs.

## Summary

Researcher claimed the P2 audit-rerun. Directory had predecessor content (m2p_domain_conditioned / Finding #342); `run_experiment.py` already upgraded to hard top-1 Gumbel MoE + STE but emitted the wrong K-IDs (K855/K856/K857) and had no `results.json`. Pre-run fixes (all documented in PAPER.md §Audit-Rerun Context):

1. MATH.md rewritten for hard MoE — Theorem 1 (STE gradient isolation), Lemma 1 (router-collapse as stable equilibrium without aux load).
2. §G re-label gate pre-registered: K861 PASS under D1 collapse → mechanical KILLED.
3. `phase_router_check` diagnostic added; D1 wired into verdict (`relabel_killed_by_d1_router_collapse`).
4. Eval-time Gumbel disabled via `training_mode=False` flag (deterministic K861).
5. Internal K-IDs remapped so only the DB-tracked K861 is reported.

## Adversarial Checklist (a)–(s)

**Consistency (a–d):** PASS. `results.json["verdict"]="KILLED"`, `all_pass=false`, `is_smoke=false`, PAPER.md verdict line = KILLED, DB status = killed, evidence row added for 2026-04-18.

**KC integrity (e–g):** PASS. `git diff MATH.md` shows the file was fully rewritten from the stale predecessor; no prior `results.json` existed, and the DB KC #861 threshold (≥0.25) was unchanged from 2026-04-07. Not a post-hoc KC relaxation. K861 in code (line 670, `median_quality >= 0.25`) matches MATH.md §C.1 and DB KC text. No tautology — K861 is a real median over 5 loss-ratio computations; D1 is a real router argmax uniqueness count.

**Code ↔ math (h–m2):**
- (h) Composition is per-expert then weighted sum: `sum(route_weights[i] * expert_outputs[i] for i in range(n_experts))` at run_experiment.py:240 — NOT `(ΣB)(ΣA)`. STE forward is one-hot, backward soft: `mx.stop_gradient(hard - soft) + soft` (line 230), matching MATH.md Theorem 1.
- (i) `LORA_SCALE=2.0` (line 295, 419). Safe.
- (j) Routing per-sample via `domain_id` argument to `__call__`.
- (k) No `shutil.copy`. (l) No hardcoded `"pass": True`. (m) Toy-GPT by design; no proxy substitution.
- (m2) MLX skill evidence: code uses `mx.eval`, `mx.clear_cache`, `nn.value_and_grad`, `mx.stop_gradient`, MLX-idiomatic modules with `__call__`, `device_info()` memory budget, QR on numpy side (see MATH.md §E crosswalk). Skill invocation evident.

**Deliverables (r–s):** PASS. Prediction-vs-measurement table in PAPER.md §"Prediction vs. Measurement" shows C.1 FAIL (0.239 vs 0.15–0.40), C.2 confirmed (3/5 unique, in predicted {1,2,3}), C.3 surprising (0.3354, below predicted band), C.4 PASS (0.0 ≤ 1e-5), C.5 confirmed (-3.306 on repeat, matching MATH.md §C.5's "repeat is expected worst").

## Mechanistic Note (Pre-reg validated, NOT introduced after-run)

B-matrix diversity: 0.9956 (#341) → 0.3354 (this). STE gradient isolation DOES prevent centroid collapse at the hypernetwork layer (Theorem 1 mechanism). But Lemma 1 predicts router collapse at N_e = N_domains without aux load loss; observed 3/5 unique experts (arithmetic/reverse/repeat → expert 0, expert 0 drifts toward highest loss-gap → "repeat" starves at -3.306). The D1 re-label gate was pre-registered (MATH.md §G) before the run, so this is not a moving-goalpost.

## Run Stability

Two back-to-back runs observed (23.9% and 34.2% medians; both with D1=3/5). Re-label gate makes verdict robust across Gumbel stochasticity: literal-fail path in run-2 and re-label-kill path in run-1 both terminate at KILLED. Final recorded verdict is the literal-fail path (23.9%); `relabel_killed_by_d1_router_collapse=false` because the literal check already fired.

## Assumptions

- **A1.** The MATH.md rewrite is treated as pre-registration because no prior `results.json` existed and the stale predecessor content was for a different experiment (Finding #342). Reviewer accepts this per PAPER.md §Audit-Rerun Context and audit-2026-04-17-rerun tag.
- **A2.** Ground-truth routing input (`domain_id` → router) is a best-case; real routing from hidden states would fail at least as hard. Noted in PAPER.md §Assumptions. Does not rescue the verdict.

## Non-Blocking Observations

1. Docstring at top of run_experiment.py (lines 1–16) still references K855/K856/K857 and "domain conditioning". Internal-only; does not affect verdict.
2. `phase_evaluate_m2p` function name references the predecessor (evaluates M2P quality by domain); no behavioral issue.

## Routing Signal for Analyst

Three permanently-learned rules from PAPER.md §"Three permanently-learned rules" should be encoded as pattern memories if not already captured:

1. Hard top-1 MoE without aux load-balance loss collapses at N_e ≥ N_domains with heterogeneous losses (same root structure as Finding #574, different DOF).
2. K861-only PASS under router collapse is a metric-swap false-positive — always gate verdict on D1 when MoE is present.
3. STE gradient isolation fixes centroid collapse at the B-matrix layer (0.9956 → 0.3354) but the failure moves to the routing layer — future M2P variants must carry aux load-balance terms, not bare STE.

Closes 3-way M2P B-matrix-collapse mitigation sweep: additive (#342) / soft MoE (#574) / hard top-1 MoE (this) — all KILLED. Unifying constraint: gradient competition across domains without explicit load-balancing is a stable failure attractor.

Sibling `exp_m2p_hard_moe_v2_aux_loss` (Switch Transformer-style aux loss) is the directly-addressed follow-up; do NOT auto-spawn — gate via analyst/planner.
