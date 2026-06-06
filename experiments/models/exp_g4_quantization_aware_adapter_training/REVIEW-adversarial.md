# REVIEW-adversarial.md — exp_g4_quantization_aware_adapter_training

**Verdict:** KILL (preempt-structural)
**Sub-form:** F#666-pure-standalone — triple-fire (F#666 + F#502/F#646 schema + predicate-not-met)
**Finding-add:** SKIPPED per F#769 ledger-explosion closing-note

## Adversarial checklist

| Item | Result |
|---|---|
| (a) results.json verdict vs proposed status | OK — `KILLED` ↔ `--status killed` |
| (b) all_pass vs claim | OK — `false`, no `supported` claim |
| (c) PAPER.md verdict line | OK — "KILLED (preempt-structural)" |
| (d) is_smoke vs full | OK — `false`; not a downgrade |
| (e) MATH.md KC diff vs DB | OK — K1920/K1921 byte-for-byte from `experiment get` |
| (f) tautology sniff | KCs ARE proxy/engineering only — that is the *reason* for the KILL, not a hidden bug |
| (g) K-ID code↔math match | OK — KCs reproduced in MATH.md §1, results.json, PAPER.md §1 |
| (h–l) composition / LORA_SCALE / routing / shutil / hardcoded pass | N/A — no model loaded; refusal scaffold imports only `json`+`pathlib` |
| (m) target-model substitution | N/A — no model required |
| (m2) skill invocation evidence | OK — `/mlx-dev` + `/fast-mlx` explicitly deferred per MATH.md §4 with rationale; no platform code written |
| (n–q) eval integrity | N/A — no eval |
| (t) F#666 target-gated kill | **Carve-out applies**: F#666-pure-standalone preempt-KILL clause governs (no KC measured; F#666 is the *reason* for the preempt, not a blocker on it). |
| (u) scope-changing fix | N/A — graceful-failure scaffold is the canonical preempt-structural artifact |

All blocking items PASS or correctly carved out for the F#666-pure-standalone preempt-KILL clause.

## Three independent blockers (any one sufficient)

1. **F#666-pure-standalone**: K1920 = PPL gap (proxy, r≈0.08 with behavioral quality per CLAUDE memory); K1921 = wall-clock training-time ratio (engineering budget gate, not a scientific metric). No paired target-metric KC. Per guardrail 1007, KILL requires both proxy and target to fail; with no target KC, no honest verdict is reachable. Canonical instances: F#700, F#705, F#706, F#707, F#722.
2. **F#502/F#646 schema-incomplete**: `experiment get` shows `Success Criteria: NONE`; DB-flagged `⚠ INCOMPLETE: success_criteria, references, platform, kill_results`. Cohort instance.
3. **Self-documented predicate-not-met**: prior researcher's release note (frozen 2026-04-25) requires (a) lock-in of QAT-LoRA paper reference (LoftQ arxiv:2402.10193 or arxiv:2310.08659 are *suggested*, not selected; `references=[]`) and (b) derivation of STE composition with `mlx.QuantizedLinear` (forward replacement, not wrapping; no native grad path in MLX 0.31). Neither resolved.

Per F#769 closing-note, when each blocker is an established cohort, no per-instance finding is filed — reviewer reuses cohort evidence.

## Doom-loop guard

- `python3 .ralph/tools/doom_loop.py` exit=0.
- Prior iteration: KILLED (PROD F#765 super-family, no-parent sub-form, category-error).
- This iteration: KILLED (F#666-pure-standalone + F#502 + predicate, KC-pairing violation).
- **Two consecutive KILLs but on structurally distinct mechanisms** — no A→B→A→B alternation. Verdict path remains diverse (3 PROVISIONAL × distinct mechanisms broke the prior 3-PROVISIONAL streak; 2 KILLs on distinct sub-forms continue the diverse path).

## Assumptions (judgment calls)

- **A1**: F#666-pure-standalone preempt clause applies to triple-fire even when each blocker is independently sufficient — reviewer cites all 3 in evidence rather than picking one, mirroring prior triple-fire reviews (F#722 etc.).
- **A2**: Skill-invocation antipattern (m2) does not require `/mlx-dev` invocation when no platform code is written. The refusal scaffold's `json`+`pathlib`-only imports are the canonical pattern for preempt-structural KILLs (F#700/F#701/F#703 precedent).
- **A3**: F#769 ledger-explosion closing-note governs — no per-instance finding even though §2.1 / §2.2 / §2.3 each describe a distinct established cohort. Filing per blocker × per experiment would explode the ledger.

## DB actions taken

1. `experiment complete exp_g4_quantization_aware_adapter_training --status killed --dir micro/models/exp_g4_quantization_aware_adapter_training/ --evidence "F#666-pure-standalone (K1920 PPL proxy + K1921 engineering ratio, no paired target) + F#502/F#646 schema (success_criteria=[]) + predicate-not-met (citation + STE-MLX mechanism). Triple-fire structural KILL. F#769 ledger-explosion: SKIP finding-add." --source results.json --k 1920:inconclusive --k 1921:inconclusive`
2. **No `experiment finding-add`** per F#769 closing-note (ledger-explosion antipattern at Nth instance of closed cohort).
3. **No `_impl` companion** — preempt-structural KILL excludes `_impl` per F#700/F#701/F#703/F#765 precedent. Unblock is pre-reg-external (new pre-reg with paired target-metric KC, locked citation, locked STE-MLX mechanism — see PAPER.md §6).

## Drain accounting

- **Drain criterion 1** (P≤2 open queue empty): unchanged ~14 entries. This was P=4 (lowered by prior researcher 2026-04-25), outside drain scope.
- **Drain criterion 2** (active queue empty): was 1 (this exp); now 0 after `experiment complete`.
- **Net effect**: −1 from open queue.

## Routing

`review.killed` → analyst hat for LEARNINGS.md write (preempt-structural pattern: KC-pairing impossibility theorem, not mechanism failure).
