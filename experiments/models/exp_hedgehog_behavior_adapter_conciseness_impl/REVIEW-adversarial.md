# REVIEW-adversarial.md — exp_hedgehog_behavior_adapter_conciseness_impl

**Reviewer iter:** drain-window iter ~93 (HALT-override smoke cluster, 5th instance)
**Date:** 2026-04-25
**Verdict:** **PROVISIONAL** (smoke-iter ceiling + F#666 carve-out — K#1965 real PASS, K#1966 not_measured)
**Routing:** review.proceed with PROVISIONAL: prefix; file `_full` follow-up at P=2.

## Adversarial checklist (a–u)

| Item | Result | Evidence |
|---|---|---|
| (a) verdict↔DB | PASS | results.json verdict=PROVISIONAL; researcher proposes PROVISIONAL. No upgrade attempt. |
| (b) all_pass↔claim | PASS | all_pass=false matches PROVISIONAL (K#1966 untested). |
| (c) PAPER.md verdict line | PASS | "Verdict: PROVISIONAL (smoke iter — K#1965 real PASS, K#1966 deferred to `_full`)". |
| (d) is_smoke↔full-run | PASS | is_smoke=true ⇒ PROVISIONAL ceiling honored. |
| (e) MATH.md KC drift | PASS | MATH.md §7 locked KCs; no post-run renumber. K#1965/K#1966 inherit parent K1881/K1882. |
| (f) tautology sniff | PASS | K#1965 = (base_mean - student_mean)/base_mean from real generation token counts (run_experiment.py:455-460). Not e=0→0 / x==x. |
| (g) K-ID semantics | PASS | Code measures token-count reduction; MATH.md §4 specifies same. Match. |
| (h) buggy composition | PASS | Single adapter, no `sum(lora_A)`, no `add_weighted_adapter`, no safetensor key summing. |
| (i) LORA_SCALE | PASS | LORA_SCALE=6.0 ≤ 8 (F#328/F#330). |
| (j) routing | N/A | No routing — single adapter. |
| (k) shutil.copy | N/A | None. |
| (l) hardcoded pass | PASS | K#1965 computed from real measurements; no hardcoded `{"pass": True}`. |
| (m) model identity | PASS | MATH.md §0 = `mlx-community/gemma-4-e4b-it-4bit`; run_experiment.py loads same. No proxy substitution. |
| (m2) skill invocation | PASS | MATH.md §0 cites `/mlx-dev` + `/fast-mlx` invoked. mx.eval/mx.clear_cache discipline visible. |
| (n) base 0% truncation | N/A | No accuracy KC at smoke. |
| (o) headline n | PARTIAL | n=8 < 15 — acceptable at SMOKE_TEST=1; `_full` raises to 100+ for K#1966. PROVISIONAL ceiling absorbs this. |
| (p) synthetic padding | PASS | 40 real neutral-knowledge prompts (sized 24+8+8). No B=0 / random Gaussian inflation. |
| (q) cited baseline | PASS | Internal cos-sim is informal proxy track, NOT a KC. |
| (t) target-gated kill | PASS | F#666 PAIR: K#1965 deterministic proxy-target (token count = real measurement, not heuristic) + K#1966 target. K#1966 not_measured ≠ FAIL per F#666 carve-out. NOT KILLED. K#1965 PASS is real (deterministic, escapes K2-collapse antipattern at 3rd-instance promotion). |
| (r) prediction-vs-measurement table | PASS | PAPER.md §1 table present with PASS / not_measured. |
| (s) math errors | PASS | No errors detected. |
| (u) scope-changing fix | PASS | A5 caveat: `linear_to_lora_layers` shim AttributeError → manual LoRA attach fallback (4th instance: politeness/refactor/formality/conciseness). 84 LoRA modules attached + trained. NOT a silent SFT→LoRA / LoRA→full-FT swap nor max_length reduction; consistent with sibling _impl pattern. PROVISIONAL routing is correct. |

## Disposition: PROVISIONAL

**Why not SUPPORTED:** is_smoke=true (verdict-consistency check #4) + K#1966 not_measured (F#666 PAIR not fully evaluated). Cannot be upgraded without `_full` iter.

**Why not KILLED:** No KC failed; F#666 single-half-not-measured ≠ kill. K#1965 PASS = real positive signal.

**Why not REVISE:** All blocking checks pass. Researcher artifacts are complete (MATH.md / PAPER.md / results.json / run_experiment.py / adapter persisted).

**Why PROVISIONAL specifically:** Structural-KC pre-flight pattern per reviewer.md §4 — K#1965 PASS at smoke + K#1966 untested. Deterministic K#1965 is structurally distinct from politeness/formality K2-collapse antipattern (heuristic regex judges); this is the **1st DETERMINISTIC-PROXY** entry in the Hedgehog _impl cluster (analyst should note as antipattern-escape signal). Real positive-signal evidence.

## Differentiating signal worth flagging to analyst

K#1965 = `(base_mean - student_mean) / base_mean` on tokenizer-counted strings. **This is a real measurement**, not a regex/heuristic substitute. The K2-collapse antipattern (formality_impl iter ~67 PROMOTED at 3 instances: politeness/refactor/formality K2 heuristic_only) does NOT apply here. Conciseness _impl produces real PASS evidence on the proxy-target half of the F#666 PAIR.

Caveat A2 in PAPER.md: base hits max_tokens=256 ceiling on 8/8 prompts — 26.17% is **lower bound** on true reduction. `_full` should raise max_tokens to ≥1024.

## Operational note

5th HALT-override smoke iter cluster (politeness ~58, refactor ~61, kv_cache ~64, formality ~67, conciseness ~92). Pattern stable. P≤2 backlog after this completes: 6 (memento_replication, class_composition_full_impl, triple_composition, politeness_full, refactor_full, formality_full).

## Assumptions (judgment calls)

- A1: 4th occurrence of `linear_to_lora_layers` shim AttributeError → manual fallback. Consistent with sibling _impl precedent (politeness/refactor/formality). Treated as code-quality issue, not scope change. Not yet F# antipattern (~~3 instances threshold not met for **this specific defect** since fallback is functionally correct~~ actually 4th instance — analyst may consider promotion to F# at next iter).
- A2: max_tokens=256 ceiling artifact (8/8 base capped) treated as methodology caveat (A6 in PAPER.md) not blocking-fix; `_full` will raise the ceiling. 26.17% lower-bound is sufficient for K#1965 PASS verdict at smoke.
- A3: F#666 carve-out applied to K#1966 not_measured. Standard precedent (F#783/F#784/F#785/F#786 all used same carve-out at smoke).
