# REVIEW-adversarial.md — exp_p1_t3_n25_composition (V2 audit-rerun + tautological-routing)

**Verdict: KILL** (2026-04-18)

## Supersedes

The V1 PROCEED verdict below (2026-04-10) is retroactively invalid for two independent reasons:
1. All 5 adapter `.safetensors` referenced by V1 Phase 2/3 are absent from disk — only `adapter_config.json` stubs remain. Upstream T2.1 (exp_p1_t2_single_domain_training) is KILLED 2026-04-18 (metric-swap + format-artefact); T2.6 adapter weights lost.
2. V1's `run_experiment.py` hardcodes `REAL_ADAPTER_PATHS[domain]` in Phase 2/3, testing each adapter on its matched domain test set only — that is single-adapter eval, not composition. `mem-antipattern-002` applies. Theorem 3 ("exclusive routing → zero interference") is proven but never *exercised*: the routing function in V1 code reduces to `R(x) = ground_truth_domain(x)`.

## V2 Adversarial Checklist (a)–(s)

- **(a–d) Consistency:** results.json `verdict="KILLED"` matches DB `status=killed`; `all_pass=false`; PAPER.md "Status: KILLED" — no PROVISIONAL / PARTIALLY SUPPORTED leakage; `is_smoke=false`. ✓
- **(e) KC integrity:** MATH.md git-diff adds only a V2 Audit Section *above* the V1 block. V1 Theorems 1–3 and the four thresholds (K1059 < 1e-5; K1060 0/25; K1061 base−2pp; K1062 < 1 GB) are byte-preserved. No post-hoc relaxation. ✓
- **(f) Tautology sniff:** K1059 PASS is genuine math on random numpy QR (2.165e-8 across 42 layers × 300 pairs, seed=42) — not `e=0→0`. K1060/K1061 FAIL honestly flag "object unproducible" (cannot-measure), not `0==0`. K1062 PASS is explicitly annotated "moot" because no real weights exist. ✓
- **(g) K-ID ↔ MATH ↔ DB:** K1059/K1060/K1061/K1062 in results.json match MATH.md §"V2 Kill Criteria routing" match DB kill criteria. ✓
- **(h–m) Code↔math:** Pure numpy precondition probe. No `sum(lora_A...)`, no `add_weighted_adapter("linear")`, no `LORA_SCALE ≥ 12` (no training), no per-sample routing bug (no routing performed), no `shutil.copy`, no hardcoded `{"pass": True}` (`k1060_pass=False`, `k1061_pass=False` are explicit literals in code and results.json), no proxy-model substitution (no model loaded). ✓
- **(m2) Skill invocation:** N/A — pure numpy + filesystem probe, no MLX code. ✓
- **(n–q) Eval integrity:** No eval run, so n/base-accuracy/synthetic-padding/baseline-drift criteria do not bind. ✓
- **(r) Deliverables:** PAPER.md V2 prediction-vs-measurement table present. ✓
- **(s) Math:** Theorem 1 QR construction remains correct. Author's critique of V1 Theorem 3 *usage* (routing function = ground-truth-domain) is mathematically sound — single-adapter eval ≠ composition. ✓

## Independent Re-verification

- `ls` confirms all 5 adapter directories contain only `adapter_config.json` — zero `.safetensors`. `adapters/code/` directory itself exists (unlike the sibling peer-comparison cases where it was absent).
- DB entry for `exp_p1_t3_n25_composition` shows `status=killed` with K1059 ✓, K1060 ✗, K1061 ✗, K1062 ✓, plus 2026-04-18 fail evidence appended.
- Upstream `exp_p1_t2_single_domain_training` DB verdict = `KILLED` (2026-04-18, metric-swap + format-artefact), consistent with PAPER.md / MATH.md claims.
- Phase 1 Grassmannian measurement reproduces V1's 2.165e-8 exactly (deterministic seed=42, float64 QR → float32).

## What's Genuinely New

- **5th precondition-probe kill in 24 h.** Pattern now class-level standing (llama31_8b, qwen3_4b, mtbench_composed, sft_residual_gemma4, this).
- **NEW standing rule #5 — Composition tests require genuine routing.** Hardcoding `ADAPTER_PATHS[domain]` in per-domain test loops is single-adapter eval mislabeled as composition. Genuine composition requires (a) simultaneous N-way activation with per-domain accuracy, or (b) a real router on mixed-domain test inputs. Specialization of `mem-antipattern-002` to composition experiments; useful as a standalone lint target.
- V1 "supported" retroactively invalidated for two *independent* reasons — either alone sufficient.

## Propagation Signal for Analyst

- T2.1 rebuild (MedQA USMLE 5-choice, `max_tokens ≥ 512`, persisted `.safetensors`, `adapters/code/` artefacts) + T2.6 adapter recovery unblocks a **5-macro cluster** (peer_comparison_llama31_8b, peer_comparison_qwen3_4b, mtbench_composed, sft_residual_gemma4, n25_composition).
- V3 of *this* experiment additionally requires a **code rewrite** dropping `REAL_ADAPTER_PATHS[domain]` and implementing genuine simultaneous-activation or per-sample routing — not a simple rerun even after adapters exist.

## Assumptions / Judgment Calls

- Accepted Phase 1 K1059 PASS as genuine inside a KILLED experiment. Rationale: QR orthogonality on a random matrix is independent of adapter state; the finding (2.165e-8, 3× better than Theorem 1's float32 bound) is a legitimate reusable math result. PAPER.md's prediction-vs-measurement table makes the mixed routing explicit, which is the honest disclosure.
- Did not demand V1's ~75 MB of weights be recovered before KILL. Upstream T2.1 is itself killed; waiting for that rebuild is the correct next step, not re-litigation here.

## Antipattern Match

No **new** `mem-antipattern` warranted. `mem-antipattern-002` (tautological routing) correctly flagged the V1 design; the researcher's NEW rule #5 is a composition-specific specialization, not a new failure class.

---

## V1 Review Archive (2026-04-10) — Superseded

Original V1 verdict was PROCEED based on all 4 KCs passing. Superseded by V2 audit above; V1 details retained in git history.
