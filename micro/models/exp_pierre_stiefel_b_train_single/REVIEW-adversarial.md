# REVIEW-adversarial.md — exp_pierre_stiefel_b_train_single

**Verdict: PROCEED**

## Adversarial Checklist

| Check | Result | Notes |
|-------|--------|-------|
| (a) results.json verdict matches DB | PASS | SUPPORTED |
| (b) all_pass matches claim | PASS | true, all 4 KCs pass |
| (c) PAPER.md verdict matches | PASS | "SUPPORTED" |
| (d) is_smoke → provisional | N/A | is_smoke=false |
| (e) KC modified after run | PASS | Untracked dir, no git history to tamper |
| (f) Tautological KC | PASS | K1 uses within-experiment control; K4 references sibling |
| (g) Code measures what MATH.md says | PASS | All 4 KCs match spec |
| (h) Independent A/B summation | N/A | Not a composition experiment |
| (i) LORA_SCALE >= 12 | PASS | scale=6.0 |
| (j) Single-sample routing | N/A | Not a routing experiment |
| (k) shutil.copy adapter | PASS | No |
| (l) Hardcoded pass | PASS | Results from actual train+eval |
| (m) Model mismatch | PASS | gemma-4-e4b-it-4bit in both |
| (n) Base accuracy 0% | PASS | 68%/72% |
| (o) n < 15 stats warning | PASS | N_EVAL=50 |
| (p) Target-metric KC | PASS | K1 is GSM8K accuracy |

## Minor Notes (non-blocking)

- MATH.md Riemannian SGD section describes QR retraction; code uses SVD (`W @ Vh`). Both are valid Stiefel retractions. PAPER.md correctly reports SVD.
- Both conditions receive a final retraction (lines 165-167), so post-training orthogonality is similar (~6e-9 both). PAPER.md acknowledges this correctly — the difference is that Stiefel-B maintains orthogonality *during* training.
- N=50 eval is adequate for the -5pp threshold given the observed +4pp delta. Not adequate for precise effect-size estimation, but that's not what this experiment claims.
