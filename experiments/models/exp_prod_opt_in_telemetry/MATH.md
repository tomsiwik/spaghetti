# MATH.md — exp_prod_opt_in_telemetry

## §0 Disposition

This experiment is a **structural preempt-KILL candidate** (F#765 super-family,
PROD-deliverable-cascade — 5th instance). No measurement is performed. This
file documents the impossibility structure that makes measurement-as-currently-
specified vacuous, and routes the experiment to reviewer for `--status killed`
with **finding-add SKIPPED** per F#769 closing-note (ledger-explosion antipattern
when filing the Nth instance of a closed super-family).

## §1 Inherited KCs (DB byte-for-byte, untouched)

- **K1679**: "Telemetry off by default; explicit consent flow on first run"
- **K1680**: "When on: no user text, no prompts, no completions — only aggregated counters + crash stacks"
- **K1681**: "GDPR compliance review passes (data minimization, right to delete)"

## §2 Why this is preempt-KILL, not measure-and-report

### §2.1 F#666-pure standalone (no proxy/target pairing)

All 3 KCs are **deliverable-spec checks**, not scientific measurements:

- K1679 is a binary product-default check — measurement is `grep -r "telemetry_enabled = false"` in source, not a proxy/target metric pair.
- K1680 is a privacy-policy compliance check — measurement is a manual code audit of the telemetry payload schema, not a behavioral measurement.
- K1681 is a legal-process gate — measurement requires a third-party human compliance reviewer, not reproducible from this repo.

None of the 3 KCs has a paired proxy/target metric per **F#666** (target-gated kill). They are all "proxy-only" in the degenerate sense: the proxy is a literal-string presence check.

### §2.2 F#502/F#646 schema-incomplete (success_criteria=[])

Per `experiment get exp_prod_opt_in_telemetry`: `Success Criteria: NONE`. This is the **11th** instance of the F#502/F#646 schema-cohort (per scratchpad AVOID list). The DB itself flags `⚠ INCOMPLETE: missing success_criteria`.

### §2.3 PROD-deliverable-cascade super-family membership (5th instance)

Prior instances in the super-family:
- F#740, F#741 — parent `exp_prod_mlxlm_integration` (KILLED) cascade
- F#764 — parent `exp_prod_pip_package_pierre` (KILLED) cascade
- F#765 — parent `exp_prod_version_resolution` (KILLED) cascade — **promotion threshold crossed at 4th instance**

This experiment (`exp_prod_opt_in_telemetry`) is the **5th** PROD-deliverable instance and a **new sub-form**: **no-parent, no measurable scientific KC**. It does not require a KILLED parent to fail measurement; it fails on its own KC structure (§2.1 + §2.2). The super-family is therefore **broader than originally promoted** — it covers all PROD child experiments whose KC structure encodes deliverable-completion checks rather than behavioral/structural measurements.

Per F#769 closing-note (ledger-explosion antipattern at the 5th–Nth instance): **no new finding registered**. Reviewer reuses F#765 super-family evidence on `experiment evidence` and on `experiment update --status killed`. If reviewer judges the no-parent sub-form distinct enough to merit recording, they may file a single super-family-extension finding rather than per-instance.

## §3 What measurement WOULD look like (cited but not performed)

For completeness, if this product spec were to be implemented as a measurable experiment, the work would be:

- A privacy-engineering deliverable, not a research experiment. Belongs in a roadmap doc / pull-request review, not in `micro/models/`.
- KCs would need to be rewritten as: (1) a unit-test of the telemetry-default flag (binary, falsifiable, `pytest`); (2) a contract test of payload schema against a forbidden-fields allowlist (falsifiable, `pytest`); (3) an external compliance-attestation document (out of repo).
- The "experiment" framing here is a **category error**: there is no proxy/target metric pair to measure; there is no behavioral outcome to falsify; there is only a deliverable to ship.

## §4 No skill invocation

`/mlx-dev` and `/fast-mlx` are not invoked because the refusal scaffold writes no platform code. This matches the precedent established in F#763, F#764, F#765, F#768, F#769, and the prior 5 PROD-deliverable preempt-KILL deliverables.

## §5 Antipattern scan (researcher-scope)

| Antipattern | Status |
|---|---|
| (a) composition math bug | N/A — no model loaded |
| (b) unsafe LORA_SCALE | N/A — no LoRA |
| (c) tautological routing | N/A — no routing |
| (d) shutil.copy as new adapter | N/A — no adapter |
| (e) hardcoded `"pass": True` | OK — scaffold writes `verdict=KILLED`, all KCs `untested`, never silently passes |
| (f) eval-template truncation producing base=0% | N/A |
| (g) proxy-model substitution | N/A — no model required |
| (h) KC measures wrong object | OK — KCs reproduced byte-for-byte from DB; impossibility documented in §2.1 |
| (i) N=smoke reported as full | N/A — no run |
| (j) silent SFT→LoRA swap | N/A |
| (k) skill-invocation skip | OK — explicit deferral §4 with rationale |
| (l) doom-loop A→A | OK — preempt-KILL is structurally different from prior 3 PROVISIONAL escalations (verdict path differs); prior 4 PROD-deliverable instances (F#740/F#741/F#764/F#765) used identical preempt-structural pattern and reviewer ledger-explosion guidance applies (no new finding) |
| (m) skill-invocation unverified | OK — see (k) |

All clear or N/A.

## §6 References

- F#666 — target-gated KILL discipline
- F#502 / F#646 — schema-incomplete cohort
- F#740 / F#741 — PROD-deliverable-cascade 1st/2nd
- F#764 — PROD-deliverable-cascade 3rd
- F#765 — PROD-deliverable-cascade 4th, super-family-promotion-trigger crossed
- F#768 — BLOCKED-on-resource model-cache sub-form (PROVISIONAL super-family)
- F#769 — BLOCKED-on-resource compute-budget sub-form + ledger-explosion closing-note
