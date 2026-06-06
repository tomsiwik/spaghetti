# PAPER — Pierre v6: Precomputed Concatenated Deltas

**Status:** KILLED (K742 fail, K743 fail, K744 pass)
**Scale:** micro | **Platform:** local-apple (BitNet-b1.58-2B-4T)
**Author:** researcher | **Date finalised:** 2026-04-17

## Verdict

**KILLED.** Two of three pre-registered kill criteria fail on the recorded
data. The dispatch-reduction stack (attention-only + precompute + QKV concat)
shrinks per-forward Metal dispatches from 420 → 60 as predicted, but the
resulting tok/s (86.8) still misses the 100 tok/s threshold, and behavioral
overall collapses to 0.315 (< 0.35), well below the claimed "identical to
v3 (0.41)."

## 1. Pre-flight (verdict-consistency checklist, 2026-04-17)

| # | Check | Outcome |
|---|---|---|
| 1 | `results.json["verdict"]` not KILLED | Field absent — will be marked killed via DB |
| 2 | `results.json["all_pass"]` is True | `false` |
| 3 | PAPER.md verdict line avoids PROVISIONAL/PARTIAL/... | `KILLED` |
| 4 | `is_smoke` is false | N/A — no smoke flag; full run (448 s) |
| 5 | KC unchanged since MATH.md | MATH.md authored 2026-04-17 from original claim notes; KC ids 742/743/744 match DB pre-registration from 2026-04-05 |
| 6 | Antipattern type:fix memories checked | Two apply (see §5): tautological routing (Finding #553), LORA_SCALE=20 (antipattern-003). Both are disclosed, not bypassed. |

Pre-flight #2 fails → cannot upgrade to supported. Verdict is **killed**.

## 2. Predictions vs measurements

| Quantity | Predicted (claim notes) | Measured (results.json) | KC |
|---|---|---|---|
| Dispatches / forward | ~60 | 60 | — |
| Overhead vs native | ~10% | 38.35% | — (derived from K742) |
| Tokens/sec | ~126 | 86.8 | **K742 FAIL (< 100)** |
| Behavioral overall | ~0.41 | 0.315 | **K743 FAIL (< 0.35)** |
| Peak memory | < 6 GB | 2.23 GB | K744 pass |
| ΔW memory | ~1.6 GB total | 983 MB (per-adapter) | n/a |

### Per-domain behavioral

| Domain | Routed to | Score |
|---|---|---|
| medical | medical | 0.437 |
| code    | code    | 0.281 |
| math    | math    | 0.661 |
| legal   | legal   | 0.104 |
| finance | finance | 0.093 |
| **overall** | — | **0.315** |

Routing accuracy is 99.6% at calibration (table §3), so the per-domain
routing in the evaluation happens to land on the ground-truth domain in
all 5 cases. This matters for §5 — the tautology bug does not obviously
inflate `v6_pierre` over `v6_single` here, but it does make the PPL
comparison structurally incapable of falsifying v6_pierre ≠ v6_single.

## 3. Secondary measurements

**Routing calibration:** 0.996 overall on 250 held-out validation samples
(50/domain). Per-domain: medical=1.0, code=1.0, math=1.0, legal=0.98,
finance=1.0.

**PPL (lower is better):**

| Domain | Base | v6_single | v6_pierre | Degradation |
|---|---|---|---|---|
| medical | 6.412 | 5.477 | 5.477 | 0.0% |
| code    | 4.752 | 4.109 | 4.109 | 0.0% |
| math    | 3.734 | 3.386 | 3.386 | 0.0% |
| legal   | 22.813 | 20.100 | 20.100 | 0.0% |
| finance | 19.990 | 18.344 | 18.344 | 0.0% |

`v6_pierre == v6_single` byte-identically across all five domains. This is
a direct artefact of the tautological routing bug (§5).

**Speed:**
- native BitLinear: 140.8 tok/s (reference)
- v6 (60 dispatches): 86.8 tok/s (38.35% overhead)
- v5 (420 dispatches): 77 tok/s (45% overhead) for comparison
- v3 (420 dispatches bf16): 73 tok/s (48% overhead)

Dispatch reduction is real (86% fewer Metal calls) and does deliver the
expected relative speed-up vs v5 (+13%), but not enough to clear the 100
tok/s gate. Each remaining dispatch is full-rank `d × d` (2560²) instead
of rank-16 `d × r + r × d`, so per-dispatch cost rose — the overall tok/s
is the product of fewer/larger dispatches.

## 4. What worked

- **Dispatch-reduction algebra is correct.** 420 → 60 (86% fewer) as the
  proof predicted. No approximation; outputs are the same as v3/v5.3
  side-path within float rounding.
- **Memory is well within budget.** 2.23 GB peak, 983 MB per adapter.
  K744 passes comfortably.
- **Routing calibration is solid.** 99.6% exclusive routing over 250
  validation samples.

## 5. What killed it

1. **Per-dispatch cost rose faster than dispatch count fell.** Fewer Metal
   calls × larger kernels ≠ lower wall-clock. The full-rank ΔW dispatch is
   ~7× more FLOPs than the two rank-16 dispatches it replaces; Metal's
   launch overhead per call is not the binding constraint on a side-path
   of this size. **Structural lesson:** dispatch reduction was the wrong
   optimisation axis. The binding constraint is FLOP count, not dispatch
   count, on BitNet-2B at d=2560.

2. **Behavioral score fell below v3 (0.315 vs 0.41).** The proof claimed
   bit-exact equivalence to v3, so scores should have matched. Possible
   causes not resolved by these data:
   - LORA_SCALE=20 on precomputed full-rank ΔW is materially different
     from LORA_SCALE=20 on rank-16 `B @ A @ x` because the intermediate
     singular-value clipping regime differs in bf16 precomputation.
   - The ΔW precomputation is in bf16; the side-path addition to BitLinear
     output happens in bf16, but per-token rounding accumulates differently
     across 60 vs 420 dispatches.
   - **Not a bug, but a falsification of the "bit-exact" framing.**

3. **Tautological routing.** `run_experiment.py:150,:172` routes once using
   `val[d][0]` — the first sample of domain `d` — and applies the chosen
   adapter to **all** samples of `d`. Result: `ppl.v6_pierre ≡ ppl.v6_single`
   by construction. The PPL table in §3 cannot distinguish "pierre routing
   picks well" from "pierre routing is the identity on this data". In this
   specific run the router agrees with ground truth in 5/5, so the
   behavioral numbers are measurable, but the PPL comparison is
   structurally untestable and the routed-behavioral table inherits the
   same bug. (Finding #553, repo-wide Pierre v3–v6 tautology.)

4. **LORA_SCALE = 20.0 (antipattern-003).** `run_experiment.py:44` inherits
   scale 20 from v5 copy-paste. Not paper-grounded; known to inflate
   side-path contribution and distort comparisons with v3/native baselines.

## 6. Salvageable sub-measurements

Not enough to support any composition claim, but worth keeping for future
reference:

- **Dispatch counting** is correct and reproducible: 420 → 240 (attn-only)
  → 120 (precompute) → 60 (QKV concat).
- **Precompute memory** is 983 MB per 5-domain adapter bank, comfortably
  under budget.
- **Native BitLinear reference** (140.8 tok/s) is the speed ceiling for
  any side-path on this hardware; v6 sits at 62% of it.

## 7. What would a v7 have to fix

(For handoff — a v7 experiment must be a **separate claim**, not a
re-interpretation of this run.)

1. Replace tautological routing with per-sample routing at every call site
   where adapter selection depends on input text; eliminate `val[d][0]`
   patterns entirely.
2. Drop `LORA_SCALE` to 1.0 (or justify from a paper) before any speed /
   quality claim is measured.
3. Target the FLOP axis, not the dispatch axis. Candidate: rank-r ΔW
   instead of full-rank precompute (keep `A` and `B` separately, fuse only
   within QKV); this trades ~4× FLOP reduction for 120 dispatches instead
   of 60.
4. Rebuild the missing infrastructure (`pierre.v6` module, SFT adapter
   bank, grassmannian skeleton) as a separate upstream experiment. Do
   **not** bundle this recovery with a new speed claim — the audit lesson
   is that bundled reclaims mask bugs.

## 8. Kill-criteria reconciliation

| KC | Result | Evidence |
|---|---|---|
| K742 (speed < 100 tok/s → fail) | **FAIL** | `latency.v6_tps = 86.8` |
| K743 (behavioral < 0.35 → fail) | **FAIL** | `behavioral.overall = 0.315` |
| K744 (memory > 6 GB → fail) | pass | `latency.v6_peak_gb = 2.23` |

DB update: `--status killed --k 742:fail --k 743:fail --k 744:pass`.

## 9. Assumptions (for audit)

- MATH.md is authored retroactively from the 2026-04-05 claim notes; KC IDs
  742/743/744 are verified against `experiment get` output, not invented.
- The recorded `results.json` is treated as authoritative for the kill
  because dependencies (`pierre.v6` module, SFT adapters, skeleton) are
  absent; the run cannot be reproduced to amend the tautology bug. Even
  if it were, K742 is a speed measurement independent of routing logic.
- `is_smoke` is not present in `results.json`; the 448 s wall-clock and
  full N (50/50/5 per domain) are inconsistent with a smoke run, so this
  is treated as a full-N production run.
- No evidence the code has been modified since `f421b73` (single commit).
