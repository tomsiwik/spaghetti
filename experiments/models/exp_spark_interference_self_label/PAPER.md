# PAPER — interference as a self-label (zero-training domain detector)

## Question
Can the *damage* an off-domain LoRA does to the base model's own next-token prediction serve as a free,
training-free, router-free domain label? Concretely: does the scalar margin shift

    s_i = (1/K) Σ_t [ logit^{B+A_i}_t(ŷ_t) − logit^B_t(ŷ_t) ],   ŷ_t = base greedy pick, K=8

rank on-domain prompts above off-domain prompts (AUROC > 0.70)?

## Setup (real, not mock)
- Frozen base: `mlx-community/gemma-4-e4b-it-4bit` (real 4-bit Gemma, real forward passes).
- Adapters: real trained LoRA rank-6 on `self_attn.q_proj`, scale 6.0 (≤ 8 guard OK), domains
  {code, math, medical}, from `exp_p1_t2_single_domain_training/adapters`.
- Prompts: 30 held-out `valid.jsonl` user turns per domain (90 total). No training, no merging, no router.
- AUROC = Mann–Whitney U; positives = adapter's own domain, negatives = other two domains.
- `is_smoke: false`, total wall clock 52.9 s.

## Prediction vs measurement

| Quantity | Predicted | Measured | Kill threshold |
|---|---|---|---|
| mean AUROC (3 adapters) | ≥ 0.80 | **0.278** | < 0.70 ⇒ killed |
| sign-at-zero accuracy | > 0.60 | 0.556 | (reported) |

Per-adapter AUROC (on-domain vs off-domain):

| Adapter | AUROC | mean s_on | mean s_off |
|---|---|---|---|
| code | 0.047 | −2.153 | +0.144 |
| math | 0.738 | −1.407 | −1.919 |
| medical | 0.049 | −7.270 | −4.527 |

## Verdict: KILLED

Mean AUROC = 0.278 < 0.70 (KILL 2300, `result: fail`). `verdict: killed`, `all_pass: false`.

The hypothesis is not merely unmet — it is **inverted**. The theorem predicted E[s_d | on-domain] ≥ 0 and
E[s_i | off-domain] < 0, i.e. an adapter should *raise* the base's own margin on its own domain and *lower*
it elsewhere. The data shows the opposite structure:

- For `code` and `medical`, mean s_on is *more negative* than mean s_off (−2.15 vs +0.14; −7.27 vs −4.53),
  giving AUROC ≈ 0.05 — strongly anti-correlated. The adapter damages the base's own greedy pick **most** on
  its own training domain.
- s_i is negative almost everywhere (every adapter lowers the base margin on average across all domains),
  so there is no sign-based on/off operating point: sign-at-zero accuracy 0.556 ≈ chance.
- Only `math` clears 0.5 (0.738), and even there both means are negative; the separation is a weak magnitude
  effect, not the predicted sign flip.

## Why the theorem failed
The premise "base greedy ŷ_t ≈ the correct on-domain continuation, so A_d reinforces it" is false for an
instruction-tuned 4-bit base teacher-forced on its *own* greedy trajectory. The on-domain adapter was trained
to move probability mass toward the *training-data* continuation, which diverges from the base's greedy
continuation precisely where the adapter has learned something — so on its own domain the adapter pushes
*away* from ŷ_t, driving s_on negative. The margin shift therefore measures "how much the adapter disagrees
with the frozen base's greedy habit," which is largest where the adapter was actually trained. Interference
is real, but it is not a clean self-label: its sign does not encode domain ownership.

## Status
Real run, falsifiable, refuted by its own pre-registered threshold. No router-free / training-free domain
detector emerges from the margin-shift sign. Reviewer gates `experiment complete`.
