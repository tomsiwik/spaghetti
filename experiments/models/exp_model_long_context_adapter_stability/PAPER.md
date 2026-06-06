# PAPER — `exp_model_long_context_adapter_stability`

**Verdict.** PROVISIONAL (BLOCKED on compute budget; F#263 frontier-extension partial coverage at 8k/32k, 128k empirically open).

---

## Hypothesis (restated)

Does N=5 PoLAR adapter composition on Gemma 4 E4B preserve needle-in-haystack retrieval (K1706, within 5pp of base) and RULER subtask scores (K1707, within 3pp on every subtask) across 8k / 32k / 128k context windows?

## Prediction vs. measurement

| Prediction (MATH.md §4) | Mechanism | Measured | Status |
|---|---|---|---|
| P1 (8k): NIAH within 1pp of base | §3.2 V/O structural protection within base training length | NOT MEASURED — compute budget exceeds drain iteration | untested |
| P2 (32k): NIAH within 5pp of base | §3.2 protection + mild §3.3 extrapolation drift | NOT MEASURED | untested |
| P3 (128k): NIAH within 5pp of base | §3.2 protection vs §3.3 full-extrapolation compounding (genuinely open) | NOT MEASURED — load-bearing empirical hinge | untested |
| P4 (RULER): score within 3pp of base on every subtask | aggregate target-behavioral pair per F#666 | NOT MEASURED — RULER is 13-subtask multi-hour benchmark | untested |
| P5 (degradation grows monotonically with context length) | linear-superposition bound + extrapolation drift | NOT MEASURED | untested |

All five predictions are untested. No empirical claim filed.

## Kill criteria resolution

| KC | Text | Result |
|---|---|---|
| K1706 (proxy/structural) | NIAH recall under N=5 composition within 5pp of base at 8k, 32k, AND 128k | **untested** |
| K1707 (target/behavioral, paired per F#666) | RULER score under composition within 3pp of base across all 13 subtasks | **untested** |

## Measurement blockers

1. **Compute budget.** Per the experiment's `notes` field (set by prior researcher iteration on 2026-04-24): `128k prefill @ Gemma 4 E4B 4bit on M5 Pro ≈ 10 min/sample`; `RULER is 13-subtask multi-hour benchmark`; `N=5 composition × {8k,32k,128k} × ≥5-needle positions = 4-8 h total compute`. Lower-bound estimate (240 min) is **8× the single-iteration researcher budget** (30 min, guardrail 1009).
2. **RULER baseline absent.** Gemma 4 E4B baseline RULER scores are not present in `experiment query`; would need to be measured in the same iteration to compute deltas. This adds 13 subtasks × 3 lengths to the budget — already accounted in the 4-8h figure.
3. **Adapter availability.** The 5 PoLAR adapters required (code/math/medical from `exp_p1_t2_single_domain_training` + legal/finance from `exp_p1_t2_multi_domain_5`) exist on disk per F#627 + multi-domain-5 deliverables; this is NOT a blocker, the script verifies presence at IMPL time.

## Proof-first prior coverage

MATH.md §3 derives a partial proof-first coverage:

- **§3.1 (linear-perturbation regime, F#263 extension):** N=5 adapter sum is operator-norm-bounded; under Grassmannian-orthogonal A (F#562) the bound is sub-linear in practice. Predicts mild degradation, not catastrophic failure.
- **§3.2 (V/O-only structural protection):** Position-encoded retrieval at long context is structurally protected for `v_proj + o_proj` adapters because the attention pattern `softmax(QK^T/√d)` is unaffected by V/O perturbations. Composition error enters only via the value-aggregation channel, bounded by `‖Σ_i ΔW_v‖`. Predicts **K1706 PASS at 8k**, mild drift at 32k.
- **§3.3 (range-extrapolation regime, LongLoRA):** Beyond Gemma 4 E4B's base training context, position encoding extrapolates per its frequency schedule. LongLoRA shows V/O perturbations can compound with positional drift at full-attention 128k. Makes the **128k row genuinely open** — the empirical hinge.

**The 128k point is load-bearing and cannot be resolved by extrapolation from 8k/32k measurement** (that is the antipattern (m) trap on the context-length axis: silent proxy-substitution by measuring only the easy regime and claiming the hard one).

## Why this experiment is filed PROVISIONAL rather than KILLED or RELEASED

Three options were considered:
1. **RELEASED-to-OPEN** — what the prior 2026-04-24 researcher iteration did (downgrade P1→P3). Repeating this would constitute a doom-loop signal per researcher.md §0; the prior pattern triggered the F#768 PROVISIONAL escalation.
2. **KILLED on §3.1 monotonic prior** — defensible at 8k/32k (proof-predicted PASS, would just be confirmation), but **not** at 128k where §3.3 creates genuine uncertainty. Filing as KILLED would discard the load-bearing 128k question.
3. **PROVISIONAL with reclaim path** — preserves the well-specified reclaim path from `notes` (schedule >=4h session → reprioritize to P=2 → invoke skills → run NIAH then RULER), and treats the proof-first prior as partial coverage rather than full resolution.

This filing chooses (3), matching the canonical F#768 BLOCKED-on-resource pattern (compute-budget variant rather than model-cache variant).

## Verdict-consistency pre-flight (all 6 checks per PLAN.md §1)

1. `results.json["verdict"]` = `"PROVISIONAL"` — not KILLED, not SUPPORTED ✓
2. `results.json["all_pass"]` = `false` — consistent with PROVISIONAL ✓
3. PAPER.md verdict line reads `PROVISIONAL` — not `supported` ✓
4. `is_smoke` = `false` — no smoke-as-full issue ✓
5. No KC was modified between MATH.md and now. K1706 (NIAH within 5pp at 8k/32k/128k) and K1707 (RULER within 3pp on all subtasks) match DB-registered text byte-for-byte. ✓
6. Antipattern scan:
   - composition math bug — N/A (no run; IMPL path requires `Σ_i B_i @ A_i`, documented in MATH.md §6) ✓
   - tautological routing — N/A (no routing; per-sample composition) ✓
   - LORA_SCALE — would be ≤8 per MATH.md §6 ✓
   - KC-swap-after-failure — no data collected; no KC swap ✓
   - shutil.copy as new adapter — N/A ✓
   - hardcoded `pass: True` — no KCs marked PASS ✓
   - proxy-model substitution — scaffold explicitly refuses to proxy on context-length axis (running only 8k and claiming 128k by extrapolation). BLOCKED path emits PROVISIONAL ✓
   - eval-template truncation — N/A in design-only iteration ✓
   - silent base-model swap — script verifies `gemma-4-e4b-it-4bit` cached path; never swaps to `e2b` or `26b-a4b` variants ✓
   - thinking-mode truncation — would use `enable_thinking=True` per MATH.md §6 / mem-antipattern-008 ✓

All 6 checks clear **for a PROVISIONAL verdict**. A SUPPORTED verdict is not claimed.

## Assumptions (per researcher autonomy guardrail 1008)

- **A1.** Gemma 4 E4B native context is 128k (per Gemma 4 model card). If only 32k is reliable, the 128k row of the prediction table becomes "untestable on this base" rather than "open"; the experiment should then be redesigned around 8k/32k only with explicit base-model-context disclosure.
- **A2.** The §3.2 V/O structural-protection argument requires the proven F#627 target-module configuration. Substituting q/k targets would invalidate the derivation.
- **A3.** N=5 PoLAR adapter set has Grassmannian-orthogonal A matrices (F#562), giving sub-linear operator-norm growth under sum.
- **A4.** Researcher-hat guardrail 1009 caps single-iteration work at 30 min / 40 tool calls; a 4-8 h NIAH+RULER sweep is explicitly out of scope without authorization.

## Suggested follow-ups (content-level, not workflow-required)

1. **NIAH-only quick screen** (~1 h budget): if a future iteration has only ~1 h of authorized compute, run NIAH (5 depths × 3 lengths × 5 samples × 1 needle = manageable) but explicitly DO NOT mark K1706 PASS without the RULER pair (F#666 forbids); file as PROVISIONAL with K1706-only data.
2. **Range-decomposition study**: split into three sibling micro-experiments at 8k / 32k / 128k each. The 8k experiment is proof-predicted PASS and probably not worth running. The 32k and 128k are independently informative and each fits in a single dedicated session.
3. **Single-adapter long-context stability**: F#263 was N=1 NTP/SFT at short context. A single-adapter long-context measurement establishes the per-adapter baseline before testing N=5; would be a cheaper precursor.

## Doom-loop break note

This experiment was previously released-to-OPEN by the 2026-04-24 researcher iteration (P1→P3 downgrade). Releasing again in this iteration would constitute a doom-loop pattern per researcher.md §0 (same hat / same experiment / same release action). Filing PROVISIONAL with the F#768 pattern is the structurally-different action.
