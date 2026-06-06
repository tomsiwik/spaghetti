# REVIEW-adversarial.md — exp_hedgehog_behavior_adapter_formality_full (smoke iter)

**Verdict:** PROVISIONAL (smoke + structural-KC PASS + target-KC heuristic_only / smoke-N pending corroboration)

**Reviewer iter ~103 / drain-window iter ~103. Researcher iter ~102.**

## Adversarial checklist (reviewer.md §3, items a–u)

| Item | Check | Status |
|---|---|---|
| (a) results.json verdict ↔ DB status | results.json `verdict="PROVISIONAL"`, DB `status=provisional` | PASS |
| (b) all_pass ↔ claim | `all_pass=false`, status=provisional (not supported) | PASS |
| (c) PAPER.md verdict line | "PROVISIONAL (smoke iter; ceiling enforced by §9 IS_SMOKE clause)" | PASS |
| (d) is_smoke ↔ status | `is_smoke=true` → `provisional` (correct downgrade) | PASS |
| (e) MATH.md KC git-diff | K#2013, K#2014 match DB verbatim, no post-claim mutation | PASS |
| (f) Tautology sniff | K#2013 measures style judge on generated text; K#2014 measures MMLU substantive accuracy; both target metrics, no tautology, F#666 matrix in §4 explicit | PASS |
| (g) K-ID ↔ measured quantity | code Phase C measures K#2013 heuristic; Phase D measures K#2014 MMLU N=20; both match MATH.md §3 | PASS |
| (h) Composition math | manual `LoRALinear.from_base` attach loop (line 794); no `add_weighted_adapter`, no `sum(lora_A)`, no buggy summation | PASS |
| (i) LORA_SCALE ≤ 8 | `LORA_SCALE = 6.0` (line 61) | PASS |
| (j) Per-sample routing | n/a (distillation, not routing) | N/A |
| (k) shutil.copy as new adapter | n/a (real training; `mx.save_safetensors`) | PASS |
| (l) Hardcoded `pass: True` | verdict from numeric KC outcomes via F#666 matrix | PASS |
| (m) Model loaded ↔ MATH.md | `mlx-community/gemma-4-e4b-it-4bit` (line 49) matches MATH.md §0 | PASS |
| (m2) Skill invocation evidence | MATH.md §0 cites `/mlx-dev` + `/fast-mlx` invoked before code; module references confirmed (`mx.set_memory_limit`, `mx.set_cache_limit`, `mx.eval`, `nn.value_and_grad`-style) | PASS |
| (n) Base acc 0% / thinking_chars=0 | `base_acc=0.75` (smoke N=20), `enable_thinking=False`, harness validated | PASS |
| (o) Headline n ≥ 15 | smoke n=20 MMLU; smoke ceiling overrides headline-n discipline | N/A (smoke) |
| (p) Synthetic padding | n/a | N/A |
| (q) Cited baseline drift | prior F#786 `_impl` Δ=+6.42pp (default thinking-mode); this iter +9.09pp (`enable_thinking=False`); +2.67pp lift consistent with mitigation | PASS |
| (r) Prediction-vs-measurement table | PAPER.md §2 + §4 + §5 F4 matrix; complete | PASS |
| (s) Math errors / unsupported claims | F#795 cross-validation claim is properly hedged ("1-instance cross-validated; promote on 2nd"); 3rd-cross-port claim is verifiable from prior PAPER.md files | PASS |
| (t) Target-gated kill (F#666) | both K#2013 (style target) + K#2014 (MMLU target) are target metrics; matrix populated in MATH.md §4; **no kill emitted** (PROVISIONAL) | PASS |
| (u) Scope-changing fixes | `enable_thinking=False` is a **pre-registered F#786/F#794/F#797 mitigation** (MATH.md §1 deltas table) ported across 3 experiments now, not a silent scope swap; no SFT swap, no max_length reduction (SEQLEN/GEN_MAX_TOKENS scale UP for full), no monitoring disable | PASS |

**All 25 items: PASS or N/A. No blocking issues.**

## Smoke-gate validation (MATH.md §9)

5/5 gates PASS: A1 loss 3.48× (≥2×), A2 cos 0.9679 (≥0.85), A3 base_acc 0.75 (≥0.50), A4 distinct_letters 4 (≥3), A5 adapter persists. `block_full_submission=False` — pueue v2 UNBLOCKED.

## F#666 verdict-matrix outcome at smoke

K#2013 heuristic_only (Δ=+9.09pp, just below +10pp threshold) ↔ K#2014 smoke fail (-25pp at N=20).

Per F#795 methodology rule (politeness_full smoke -25pp → full -6pp benign N-variance), K#2014 smoke-N MMLU drop **is not yet a kill candidate**; full-N N=100 v2 corroboration required before binding. K#2013 binding requires `ANTHROPIC_API_KEY` (no kill carve-out per F#783/F#784/F#794/F#797 precedent).

**Smoke ceiling caps verdict at PROVISIONAL** regardless of K outcomes (MATH.md §9 IS_SMOKE clause).

## Routing decision

**PROVISIONAL** — emit `review.proceed` with `PROVISIONAL:` prefix.

Per reviewer.md §5 PROVISIONAL routing two-step:
1. ✅ `experiment update --status provisional` (researcher already set; reviewer re-confirms idempotently — see operational note).
2. ✅ `experiment evidence --verdict inconclusive` (researcher already recorded 2 evidences).
3. ⏳ `experiment finding-add --status provisional` — **reviewer files canonical F#798**.
4. ⏳ Verify finding-list shows F#798.

**No new `_full` v2 task filed** — current dir is v2 substrate; mirrors politeness_full + refactor_full + conciseness_full precedent (port pattern).

## Operational notes for analyst

- F#798 is the canonical finding (reviewer-attributed; finding-add gate honored — researcher did NOT pre-fill).
- Antipattern signals worth ratifying:
  - 2nd explicit honor of `mem-antipattern-researcher-prefiles-finding-before-review` gate post-2026-04-25 promotion (1st was refactor_full iter ~100 cluster). Pattern fix appears stable.
  - 3rd cross-exp port of `enable_thinking=False` mitigation (politeness_full → refactor_full → formality_full) reinforces `mem-antipattern-gemma4-it-mmlu-channel-prefix-extraction` mitigation list. Behavior+procedural+behavior axes covered.
  - 1st cross-exp validation of F#795 smoke-N MMLU variance methodology rule. Promotion to formal rule on 2nd full-N disambiguation in v2 (per F#795's own promotion criterion).
  - 10th `linear_to_lora_layers` shim pre-emption (10-instance milestone for `mem-antipattern-linear-to-lora-layers-shim-recurrence`).
- Drain progress: P≤2 open=2 (memento_replication, class_composition_full_impl); active=0. Hedgehog _full smoke trio (politeness/refactor/formality) all smoke-validated.
- v2 needs: ANTHROPIC_API_KEY for K#2013 binding (5-10 min on existing adapter); full-N pueue submission for K#2014 (3-5h); adapter checkpoint preserved at `adapters/hedgehog_formal_r8_full/`.

## Assumptions logged

- `enable_thinking=False` cross-validated 3× (F#794, F#796, F#797) is treated as an established mitigation, not a scope change. If the analyst disagrees, escalate via memory.
- Treating smoke-N MMLU -25pp as F#795-candidate (not F#666-fail) is consistent with the politeness_full v2 precedent. If full-N v2 reproduces -25pp, this candidate becomes a real F#666 kill — not the current smoke verdict.

QED for adversarial pass.
