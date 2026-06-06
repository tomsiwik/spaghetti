# LEARNINGS — exp_followup_jointly_trained_routing_molora

**Verdict:** KILLED (preemptive, structural)
**Date:** 2026-04-19

## Key learnings

1. **Absolute-pp thresholds against near-oracle baselines are
   structurally tight.** K1551 (+3pp) against a 96.6% baseline leaves
   only 3.4pp headroom. Any KC that requires an architectural change
   to close >=50% of the ceiling-to-100% gap needs to be checked
   against prior measurements before pre-reg — especially on small-N
   splits where baselines saturate.

2. **Jointly-training the router + adapters doesn't escape F#305's
   per-token full-sequence null.** The null comes from cross-attention
   contamination in the attention matrix, which is a property of
   full-sequence forward passes, not of the routing signal. Swapping
   TF-IDF for a Gumbel-sigmoid router doesn't change which tokens the
   query attends to.

3. **F#312's MLP-only +3.3% is a structural ceiling for any
   contamination-free per-token routing.** Jointly-trained router over
   MLP-only gives the same contamination-free regime as F#312 but
   replaces oracle with learned routing — the upper bound cannot
   exceed the oracle-driven +3.3%.

4. **F#340 is the closest analog already killed.** Ridge router +
   single-pass E2E on mixed-domain dropped 8.6pp because the router
   was context-dependent without boundary awareness. Jointly-trained
   MoLoRA shares that architectural class; the failure mode is
   inherited.

5. **Four-finding structural proof is a stronger preempt than
   single-finding reuse.** Any single lemma (Lemma 1 alone, Lemma 2
   alone) could potentially be circumvented by design changes. The
   logical AND of all four (ceiling AND null AND MLP-cap AND analog-
   killed) covers the full design space of "jointly-trained per-token
   router at N=5" — no mechanism bridges all four.

## Tripwire (analyst-owed when cap lifts)

**near-oracle-vs-absolute-pp-KC:** Before registering any KC of the
form "A_new − A_baseline ≥ Xpp" where baseline is measured >=(100-X-1)%,
check:
- Is headroom (100 − baseline) < 2·X? → KC is tight; likely preempt-able.
- Does the mechanism purportedly closing the gap already have a null
  (e.g. F#305 per-token full-sequence null)?
- Does a closer architectural analog already have a killed finding?

If any yes, the KC should be reframed as relative-gain-vs-oracle or
relative-gain-under-segment-isolation rather than absolute-pp over
baseline.

## Findings reused
F#431 (96.6% TF-IDF N=5), F#305 (per-token null), F#312 (MLP-only
+3.3%), F#193 (representation-limited), F#340 (ridge-router E2E KILL).

## No new finding registered
Preempt is pure reuse of F#305/F#312/F#193/F#340 under F#431 ceiling.
Analyst hat would register the `near-oracle-vs-absolute-pp-KC`
tripwire sub-axis when cap lifts; 50/50 currently.
