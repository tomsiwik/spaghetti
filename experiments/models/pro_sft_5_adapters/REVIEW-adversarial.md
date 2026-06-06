# Adversarial Review -- exp_pro_sft_5_adapters (Re-Review #2)

## Verdict: PROCEED

Both blocking fixes from the previous re-review have been correctly applied.
No remaining blocking issues.

---

## Fix Verification (This Round)

### Fix 1: Math and Finance Rows Corrected

The Training Loss Instability table (PAPER.md lines 193-199) now matches
results.json for ALL five domains. Cell-by-cell verification:

**Math row (previously wrong):**
| Step | results.json | PAPER.md | Match? |
|------|-------------|----------|--------|
| 50   | 0.9605      | 0.96     | YES    |
| 100  | 0.6593      | 0.66     | YES    |
| 150  | 0.8047      | 0.80     | YES (was 0.61) |
| 200  | 0.5250      | 0.53     | YES (was 0.42) |
| 250  | 0.6894      | 0.69     | YES (was 0.42) |
| 300  | 0.4187      | 0.42     | YES (was 0.45) |

**Finance row (previously wrong):**
| Step | results.json | PAPER.md | Match? |
|------|-------------|----------|--------|
| 50   | 4.8222      | 4.82     | YES    |
| 100  | 3.9360      | 3.94     | YES    |
| 150  | 3.5118      | 3.51     | YES (was 2.98) |
| 200  | 3.3833      | 3.38     | YES (was 3.11) |
| 250  | 3.8455      | 3.85     | YES (was 2.46) |
| 300  | 3.4727      | 3.47     | YES (was 2.94) |

Medical, code, and legal rows remain correct (verified in prior review,
re-confirmed this round -- all 18 cells match within 2-decimal rounding).

**Total: 30/30 cells in the Training Loss Instability table match results.json.**

### Fix 2: Monotonic Convergence Claim Corrected

Previous claim (wrong): "Only math shows monotonic convergence"

Current text (PAPER.md line 202): "No domain shows monotonic convergence. Even
math, which has the strongest overall improvement (45.6%), oscillates:
0.66->0.80 (up at step 150), 0.53->0.69 (up at step 250)."

**Verification against results.json -- every domain has at least one step-over-step increase:**
- Medical: step 50->100 UP (2.14->2.89), step 250->300 UP (0.88->1.28)
- Code: step 50->100 UP (2.26->3.49), step 150->200 UP (0.75->1.95), step 200->250 UP (1.95->2.16)
- Math: step 100->150 UP (0.66->0.80), step 200->250 UP (0.53->0.69)
- Legal: step 50->100 UP (2.42->3.17), step 100->150 UP (3.17->4.23), step 250->300 UP (2.93->3.74)
- Finance: step 200->250 UP (3.38->3.85)

The claim is correct. No domain shows monotonic convergence. The cited
oscillation points for math (0.66->0.80 and 0.53->0.69) are exact matches
to results.json rounded values.

---

## Previously Verified (Not Re-Reviewed)

The following were verified as correctly applied in the prior re-review and
are not re-examined here:

- Fix 1 (v1): Type 2 reclassification -- APPLIED
- Fix 2 (v1): "Proposition" renamed to "Hypothesis" -- APPLIED
- Fix 3 (v1): Finding #319 downgraded to PROVISIONAL -- APPLIED
- Fix 4 (v1): K813 vacuity analysis -- APPLIED
- Fix 5 (v1): Behavioral evaluation limitations section -- APPLIED
- Type 2 compliance -- VERIFIED
- Loss Convergence table -- VERIFIED (all 5 domains match)
- Behavioral Quality table -- VERIFIED (all 5 domains match)
- Summary statistics -- VERIFIED

---

## Non-Blocking Notes (Carried Forward)

1. **P2 assessment labeling:** P2 (mean val loss reduction 10-25%) is marked
   PARTIAL when 11.4% is technically within the predicted range. The nuance
   (2/5 domains degraded) is fair but could be stated more clearly. Not blocking.

2. **Math step 200 rounding:** 0.525 is rendered as 0.53 in PAPER.md. This is
   correct under round-half-up convention but would be 0.52 under banker's rounding.
   Trivial; not blocking.

---

## Verdict: PROCEED

Both blocking issues from the prior re-review are resolved:
1. All 30 cells in the Training Loss Instability table now match results.json
2. The monotonic convergence claim has been corrected to "no domain shows monotonic convergence"

The experiment is ready to close as PROVISIONAL (Finding #319).
