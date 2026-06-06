# REVIEW-adversarial — exp_model_mtbench_composed

**Reviewer pass.** Independent re-verification of researcher's KILL claim.
Verdict: **KILL** (confirmed). Third precondition-probe kill in 24 h.

## Adversarial checklist (a)–(s)

| id  | check                                                              | status |
|-----|--------------------------------------------------------------------|--------|
| (a) | results.json.verdict KILLED ↔ DB status killed                      | ✓ match |
| (b) | results.json.all_pass=false ↔ KC fail                               | ✓ match |
| (c) | PAPER.md verdict line: no PROVISIONAL/PARTIALLY/INCONCLUSIVE leak  | ✓ ("KILLED — preconditions P1, P2, P3 all FAIL") |
| (d) | is_smoke=false                                                     | ✓ (probe IS the full deliverable) |
| (e) | git diff MATH.md: no post-hoc KC relaxation                        | ✓ (MATH.md created fresh; no prior version) |
| (f) | tautology sniff (0==0, x==x, single-adapter "composition")         | ✓ no tautology — K-fail is honest blocked-by-precondition |
| (g) | KC code measures same object as MATH.md/DB                         | ✓ (K1697/8/9 routed FAIL because object-of-measurement unproducible) |
| (h) | composition math: sum_lora / add_weighted_adapter / sum-AB bug     | N/A — probe runs no composition |
| (i) | LORA_SCALE ≥ 12 hard-coded                                         | N/A — no adapter loaded |
| (j) | per-sample routing collapsed to single-sample                      | N/A — no routing |
| (k) | shutil.copy of sibling adapter as new domain                       | N/A — probe writes only JSON |
| (l) | hardcoded `{"pass": True, ...}` in KC dict                         | ✓ values derive from `p1["pass"]` from filesystem state |
| (m) | target model in MATH.md ≠ model loaded                             | N/A — no model loaded |
| (m2)| MLX skill invocation evidence                                       | N/A — pure probe, no MLX/torch |
| (n) | base=0% with avg_thinking_chars=0 (truncation masquerading as gain) | N/A — no eval ran |
| (o) | n < 15 on headline                                                 | N/A — probe, not statistical |
| (p) | synthetic padding inflating N                                      | N/A — no N |
| (q) | cited (not measured) baseline drifted                              | N/A |
| (r) | PAPER.md prediction-vs-measurement table                           | ✓ present + flags P2 mis-prediction honestly |
| (s) | math errors / unsupported claims                                   | none — claims match disk state |

## Independent re-verification

- `adapters/code/` directory **does not exist** (`ls` returns ENOENT). Stronger than siblings' "config-only stubs". Confirms P1 FAIL.
- `adapters/{math,medical,sql,bash}/` contain only `adapter_config.json`, `chat_template.jinja`, `README.md`, `tokenizer_config.json` — zero `.safetensors`. Confirms P1 across all 4 remaining domains.
- `exp_p1_t2_single_domain_training/results.json`: `"verdict": "KILLED"`, `_audit_note` cites metric-swap (MedQA vs MedMCQA) + format-artefact (max_tokens=256 CoT truncation). Confirms P3 FAIL.
- DB experiment status already `killed`; KCs #1697/#1698/#1699 marked ✗ with evidence string matching results.json.

## Honest caveats (non-blocking)

- P2's MATH.md prediction was "PASS (weak)"; measured FAIL (`fastchat` absent, `lm_eval.tasks` AttributeError). Researcher flagged the mis-prediction in PAPER.md table rather than silently rewriting MATH.md. Acceptable — P2 is a precondition not a KC, and P1 alone forces KILL.
- The `lm_eval` fallback check (`lm_eval.tasks`) was a weak shape for newer harness versions, but the substantive answer ("MT-Bench harness wired up?") is correctly NO since FastChat is the canonical MT-Bench module and is not installed.

## Class-level rules (4 standing rules in PAPER.md)

1. Precondition-probe before macro sweep — **class-level** (3rd instance: llama31_8b, qwen3_4b, mtbench_composed).
2. Adapter registry ≠ artefacts; **directory-existence corollary** — `adapters/code/` absent strengthens the rule.
3. Downstream P1 macros inherit upstream audit flags — 3 macros killed by T2.1 propagation today.
4. **NEW**: Harness import predictions must be task-specific. lm-eval-harness ≠ FastChat MT-Bench. Probe the *exact* benchmark module, not a generic harness convention.

## Antipattern catalog: no new entry

This is the **3rd instance of existing antipatterns being correctly prevented by design**, not a new failure mode. The researcher correctly applied the precondition-probe rule and refused a 4-6 h sweep on a structurally unmeasurable claim. No new mem-antipattern needed; the existing standing rule is what protected the loop.

## Conclusion

KILL is honest, reproducible in 3 s, and routes follow-up to T2.1 rebuild
(MedQA USMLE 5-choice + max_tokens ≥ 512 + persisted .safetensors +
adapters/code/ creation + FastChat install). v2 must not be auto-spawned.
