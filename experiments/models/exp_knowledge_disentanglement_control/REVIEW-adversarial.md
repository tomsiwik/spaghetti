# REVIEW-adversarial.md — exp_knowledge_disentanglement_control

Placeholder prior to reviewer pass. Researcher-side checklist
(PLAN.md §1):

1. `results.json["verdict"]` = `"PROVISIONAL"` — matches smoke
   guardrail. Completion status set to `killed` in DB due to CLI
   limitation (no `provisional` option); all three measurable KCs
   (K1733, K1734, K1735) fail directionally by wide margins that
   exceed the noise budget at n=20. Killed verdict on the pre-reg
   is therefore defensible; the "PROVISIONAL" tag in results.json
   signals the smoke-budget caveat.
2. `all_pass = false`. ✓
3. PAPER.md verdict line: `PROVISIONAL (smoke) — all KCs
   directionally FAIL`. ✓
4. `is_smoke = true`. The full-scale rerun plan is pre-registered
   in PAPER.md §Full-scale rerun plan. ✓
5. `git diff MATH.md` clean — KCs K1733–K1736 not modified. ✓
6. Antipattern scan (type-fix memories):
   - **composition-bug**: N/A (no composition; single adapter).
   - **tautological-routing**: N/A (no routing).
   - **unsafe LORA_SCALE ≥ 8**: scale=4.0. ✓
   - **thinking-truncation**: eval max_tokens=2048,
     fortified strip_thinking (handles missing close tag).
   - **hardcoded "pass": True**: KC booleans derived from
     measurements.
   - **proxy model**: identical `mlx-community/gemma-4-e4b-it-4bit`
     weights both arms.
   - **smoke-as-full**: flagged `is_smoke=true`, results.json
     `verdict="PROVISIONAL"`, completion status `killed` not
     `supported`.
   - **shutil.copy as new adapter**: no copy; real training.
   - **file-existence cache**: training data path checked by
     `exists()` — fine since MATH.md pre-registers n_train=25; no
     per-line validation needed because the file is rebuilt on
     first run and reused verbatim thereafter.
   - **copy-paste scaffolding**: we explicitly noted in MATH.md
     that `parse_mcq_answer` and MMLU-Pro helpers are copied from
     predecessor with v2 `strip_thinking` fortification applied;
     no silent propagation of the missing-close-tag bug.

Open items for reviewer scrutiny:
- (A4 proxy) GSM8K-as-BBH-proxy could be challenged. The smoke's
  MMLU −30 pp result does not depend on this proxy and is
  decisive on its own; the proxy only affects K1733, which was
  already borderline.
- Interpretation: the experiment is reported as PROVISIONAL
  (smoke) in PAPER.md but `killed` in the DB because the CLI
  lacks `provisional`. A reviewer could argue for re-opening the
  experiment to accommodate the pre-registered full-scale rerun
  plan. The predecessor `exp_method_vs_domain_adapter` was left
  `open` for exactly this reason. Calling this `killed` is the
  stricter interpretation; a reviewer may opt to re-open as a
  v2 with the smoke data attached.
