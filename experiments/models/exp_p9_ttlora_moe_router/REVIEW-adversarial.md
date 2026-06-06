# REVIEW-adversarial.md -- exp_p9_ttlora_moe_router

**Verdict: PROCEED**

## Round 1 (pre-experiment): REVISE -- 2 blocking fixes (PAPER.md missing, code data caveat)
## Round 2 (post-experiment): PROCEED -- both fixes applied, experiment complete

---

## Checklist

| Check | Status |
|---|---|
| Prediction-vs-measurement table | Present, all 3 kill criteria |
| Kill criteria match evidence | Verified against results.json |
| Finding status appropriate | Yes -- "supported" for guided exploration |
| Math errors | One non-blocking concern (see below) |

## Verified Claims

1. **K1360 PASS:** Router 97.7% matches results.json `router_eval_acc: 97.7`. Per-domain
   breakdown consistent across PAPER.md and results.json. Prediction (95%) met.

2. **K1361 FAIL:** MoE 25.8% vs SingleBest 27.2% = -1.4pp. Cross-domain accuracy matrix
   in results.json confirms all adapters at random (~25%). Oracle routing = 25.1% confirms
   this is not a routing problem. Honest reporting.

3. **K1362 PASS:** 814,434 bytes in results.json = 795 KB. Slight overshoot vs 652 KB
   prediction (format overhead in saved files). PAPER.md reports correctly.

## Non-Blocking Concerns

**1. Theorem 3 input assumptions were ungrounded.**
MATH.md assumed q_bar ~ 60% and q_off ~ 35% without measurement basis. Finding #516
measured PPL quality retention (84.4%), not MCQ accuracy. The experiment itself proved
PPL doesn't predict MCQ accuracy (r=0.08 from prior work). A tighter MATH.md would have
flagged this as the key unknown and made the prediction conditional: "IF q_bar >= 60%
THEN Delta >= 17.5pp." This would have made K1361's failure mode a predicted possibility
rather than a surprise. PAPER.md correctly diagnoses this post-hoc, so no REVISE needed.

**2. Base model MCQ accuracy not measured.**
The experiment never measures the base model (no adapter) MCQ accuracy. If base = 25%,
then adapters add nothing. If base = 35%, adapters actually hurt. This baseline would
strengthen the impossibility argument. Not blocking because the conclusion (adapters at
random) is clear regardless.

## What the Review Validates

- Router mechanism is proven (97.7%, 12K params, 25 KB). Reusable component.
- Impossibility structure is correct: v_proj-only 64K TT-LoRA is below the behavioral
  change threshold. The 4 proposed next steps (multi-projection, higher rank, FFN, full LoRA)
  are the right design space to explore.
- PAPER.md is well-structured: prediction table, root cause analysis, data scarcity caveat,
  impossibility structure. Meets proof-first standards.
- Prior round 1 fixes (PAPER.md + code data caveat) both applied.
