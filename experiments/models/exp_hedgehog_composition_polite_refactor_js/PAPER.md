# PAPER.md — exp_hedgehog_composition_polite_refactor_js

## Verdict: KILLED (preempt-structural, F#669 4th+ reuse)

This experiment was preempt-killed before any MLX code was written. The kill is structural, not empirical: all 5 kill criteria transitively require three trained Hedgehog-distilled adapters (politeness, refactor, JS) to exist as target-SUPPORTED parents. None of the three do:

- `exp_hedgehog_behavior_adapter_politeness` → PROVISIONAL (F#683, design-only scaffold, no MLX training loop run)
- `exp_hedgehog_procedural_adapter_refactor` → PROVISIONAL (F#684, design-only scaffold, no MLX training loop run)
- `exp_hedgehog_domain_adapter_js` → OPEN (P3, never claimed, never run)

Testing K1–K5 against composition-over-nonexistent-adapters produces either vacuous PASS (zero-init co-occurrence) or vacuous FAIL (uninterpretable uninformative), i.e. an unidentifiable sample per F#669.

## Prediction vs measurement

| KC  | Prediction                                                                  | Measurement        | Verdict       |
| --- | --------------------------------------------------------------------------- | ------------------ | ------------- |
| K1  | composed per-layer cos > 0.70; predicted ∈ [0.70, 0.82]                     | not measured       | untested      |
| K2  | politeness +22pp, refactor ≈ LoRA baseline, JS +3pp (all simultaneously)    | not measured       | untested      |
| K3  | ablate-polite → politeness -18pp, refactor/JS drop < 3pp                    | not measured       | untested      |
| K4  | ablate-refactor → refactor -12pp, polite/JS drop < 3pp                      | not measured       | untested      |
| K5  | non-code polite prompts → composition within 2pp of polite-alone            | not measured       | untested      |

**All KC rows are "not measured" because 0/3 Hedgehog adapters exist to compose.** The triple-parent preempt is strictly sharper than the single-parent cases (F#671/F#672/F#687): the child is unidentifiable even if only 1/3 parents were unverified — here all three are.

## Assumptions

- All three parent experiments will eventually be re-run to full scale via their respective follow-ups:
  - P1, P2 have `_impl` companions filed at P3 (already).
  - P3 (`exp_hedgehog_domain_adapter_js`) is itself OPEN at P3 and can be claimed directly without needing an `_impl` companion — its own status is `open`, not `provisional`.
- When all three parents reach `supported` with target KCs verified, this child becomes re-claimable with the original design (MATH.md §6) intact.
- No redesign attempted this iteration to avoid the triple-parent dependency (e.g. pair composition at N=2, or synthetic-teacher-deltas as null adapters). The alternative-unblock path is mentioned in MATH.md §4 but is out of scope for drain.

## Related findings

- **Finding #669** — defining precedent for preempt-KILL on target-unverified parent (single-parent case).
- **Finding #671, #672, #687** — prior reuse of F#669 (3 prior applications; this is the 4th+).
- **Finding #683** — parent P1 PROVISIONAL.
- **Finding #684** — parent P2 PROVISIONAL.
- **Finding #666** — target-gated KC discipline; per reviewer.md §5 canonical clause, does NOT gate preempt-KILL.
- **Finding #627** — N=24 SFT-LoRA runtime composition supported on Gemma 4 E4B; design precedent for §6 theorem.
- **Finding #571** — pre-merge composition killed 4× (motivates runtime-only composition).

## Unblock path

Re-claim this experiment when **all three conditions** simultaneously hold:

1. `exp_hedgehog_behavior_adapter_politeness` → `status=supported` with K2 (auto-judge politeness Δ ≥ +15pp, n≥100) SUPPORTED.
2. `exp_hedgehog_procedural_adapter_refactor` → `status=supported` with K2/K3 (Fowler-judge refactor quality + F#666 target pair) SUPPORTED.
3. `exp_hedgehog_domain_adapter_js` → `status=supported` with K2 (HumanEval-JS / JS-nuance benchmark) SUPPORTED.

Then the three ΔW_i deltas exist as target-validated trained tensors. The original design (MATH.md §6) becomes executable at full scale; the 5 KCs become measurable against predictable thresholds.

**Alternative unblock (out of scope now):** redesign child with fewer parent dependencies — e.g. pair composition (N=2) as a weaker test, or null-adapter ablations using random-init deltas matched for norm. Would require new experiment id.

## Follow-up filed

None. Preempt-structural kill does not spawn an `_impl` companion (per reviewer.md §5 canonical clause promoted from F#669 reuse). Unblock is parent-external via 3 independent paths:

- P1 unblock → `exp_hedgehog_behavior_adapter_politeness_impl` (P3, filed)
- P2 unblock → `exp_hedgehog_procedural_adapter_refactor_impl` (P3, filed)
- P3 unblock → `exp_hedgehog_domain_adapter_js` itself (P3, already open)
