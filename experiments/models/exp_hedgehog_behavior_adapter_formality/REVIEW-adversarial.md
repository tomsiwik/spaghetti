# REVIEW-adversarial.md — exp_hedgehog_behavior_adapter_formality

**Verdict:** PROVISIONAL (novel-mechanism design-only sub-case, dual-target / zero-proxy)

## One-line reason
Canonical novel-mechanism design-lock pattern satisfied on all 4 required artifacts; KC design is dual-target (both K1879 + K1880 grounded to external ground truth), so neither F#666-pure nor §5 tautological-inter-variant preempt-KILL applies.

## Checklist (a)–(u)
- **(a)** `results.json["verdict"] = "PROVISIONAL"` matches DB `status=provisional`. ✓
- **(b)** `all_pass=false` with KCs `untested` is coherent with PROVISIONAL. ✓
- **(c)** PAPER.md verdict line "PROVISIONAL (design-lock; novel KC design)" aligns with DB. ✓
- **(d)** `is_smoke=false`; SMOKE_TEST toggled via env — not a silent smoke claim. ✓
- **(e)** K1879/K1880 pre-registered verbatim in DB (MATH.md §7); no post-hoc KC edit. ✓
- **(f) Tautology sniff test.** K1879 = Δ formality-judge(adapter) − formality-judge(base) on 50 held-out neutral prompts (external auto-judge rubric); K1880 = |Δ accuracy(adapter) − accuracy(base)| on MMLU-100 seed=42 (canonical ground truth). Neither is an inter-variant-delta tautology; both grounded to external benchmarks. ✓
- **(g)** K-IDs match across MATH.md table, `run_experiment.py` KC dict, `results.json`. ✓
- **(h)** No `sum(lora_A…)`, no `add_weighted_adapter("linear")`, no independent sum of safetensor keys — code is a graceful stub. ✓
- **(i)** `LORA_SCALE = 6.0` ≤ 8 per F#328/F#330. ✓
- **(j)** No per-sample routing needed here; not applicable. ✓
- **(k)** No `shutil.copy` of sibling adapters. ✓
- **(l)** No hardcoded `{"pass": True}`; KCs are `"untested"` strings. ✓
- **(m)** MATH.md declares student `gemma-4-e4b-it-4bit` and teacher `gemma-4-26b-a4b-it-4bit`; `run_experiment.py` matches verbatim (lines 56-57). No proxy substitution. ✓
- **(m2) Skill invocation.** MATH.md §0 cites `/mlx-dev` + `/fast-mlx` as required and explicitly gates them to the `_impl` follow-up where the training-loop MLX code lands. Per PROVISIONAL (novel-mechanism design-only sub-case) precedent (F#682/F#683/F#684/F#719/F#723), citing the skills without shipping MLX training-loop code is sufficient; stubs raise `NotImplementedError` under `try/except` while `main()` writes `results.json`. ✓
- **(n-q) Eval integrity.** Not applicable — all KCs `"untested"`, no measurements. ✓
- **(r) PAPER.md prediction-vs-measurement table.** Present with explicit "UNTESTED" rows and predicted ranges. ✓
- **(s) Math errors.** MATH.md §3 verdict matrix is internally consistent with §4 KC table; §5 predictions enumerate all 4 outcome classes (SUPPORTED, KILLED, two PROVISIONAL modes). §8 A4 correctly flags MMLU-100 power asymmetry as a necessary-not-sufficient check — honest about the limitation. ✓
- **(t) Target-gated kill (F#666).** Both K1879 + K1880 are target KCs (behavioral acquisition + style/substance orthogonality); there is no proxy KC. F#666-pure preempt-KILL does NOT apply (KC design is the *opposite* of F#666-pure-standalone: zero-proxy instead of pure-proxy). §5 tautological-inter-variant-delta does NOT apply (both KCs grounded to external ground truth). ✓
- **(u) Scope-changing fixes.** No silent mechanism swap; `NotImplementedError` stubs + graceful `main()` is the canonical design-lock artifact pattern (F#682/F#683/F#684 precedents). Full Phase A/B/C/D pipeline lands in `_impl` at the declared scope. ✓

## Novel-mechanism sub-case pattern match
Per reviewer.md §4 PROVISIONAL (novel-mechanism design-only sub-case), all 4 required artifacts present:
1. MATH.md §0 cites platform skills → (m2) satisfied without MLX training-loop code landing.
2. `main()` never raises; writes `results.json` with `verdict="PROVISIONAL"` + KCs `"untested"`.
3. `exp_hedgehog_behavior_adapter_formality_impl` filed at P=1 macro (researcher summary; confirmed via `experiment query`).
4. PAPER.md prediction-vs-measurement table with all rows "UNTESTED" + explicit scope rationale.

## NEW KC design + sub-cluster position
- **1st dual-target / zero-proxy KC design** in the Hedgehog-framework super-family. Justified by K1880 being a safety target (style/substance orthogonality / non-interference), which by construction belongs in the target column, not the proxy column. KC-design bifurcation rule extends axis-invariantly: paired-target → PROVISIONAL; pure-proxy → KILL; **dual-target / zero-proxy → PROVISIONAL** (more conservative — lacks structural-proxy short-circuit).
- **2nd behavior-axis instance** in Hedgehog-framework (cousin of F#683 politeness). Domain-axis sub-cluster closed at 5 (F#684/696/697/717/718) post-F#718; the axis-extension super-saturation note is behavior-axis-orthogonal — behavior-axis was under-represented at 1 vs 5 domain instances.
- **9th Hedgehog-framework PROVISIONAL** (pile now 9 designs / 0 measurements). 26B teacher cache remains the standalone-prereq-task candidate blocking 9+ dependents.

## Assumptions (judgment calls)
- Accepting K1880 two-sided |Δ| ≤ 2 pp at n=100 as a *necessary-but-not-sufficient* check per MATH.md A4 (binomial CI ±5 pp at p≈0.5). The `_impl` may scale n up if borderline.
- Accepting behavior-axis sub-cluster opening past domain-axis-extension closure: behavior-axis had 1 instance vs 5 domain instances; formality ⊥ politeness is uncontroversial on canonical register theory.
- Accepting the dual-target / zero-proxy design as novel-in-super-family rather than demanding a retro-fitted cos-sim proxy KC; the safety-target framing is the correct structural justification (A2).

## Route
Emit `review.proceed` with payload prefixed `PROVISIONAL:` including the `_impl` follow-up id (`exp_hedgehog_behavior_adapter_formality_impl`). DB already at `status=provisional`; F#724 filed and verified via `experiment finding-list --status provisional`. No `experiment complete` invocation (PROVISIONAL uses the 3-step workaround, which the researcher already applied).
