# MATH.md — exp_g4_grassmannian_ap_pretrain

## Pre-registered hypothesis

**Theorem (claim):** Grassmannian antipodal-packed (AP) pre-training of LoRA
skeletons on Gemma 4 E4B, applied to `q_proj + v_proj` across all 42 layers,
lowers inter-expert interference by ≥1.5× vs random Gaussian init, measured on
N=25 disjoint-domain experts.

**Motivation (citation):** Finding #132 (`exp_grassmannian_expert_init`),
which measured AP-init interference reduction on Qwen-0.6B. Experiment ports
that finding to Gemma 4 E4B at the cohort audit-2026-04-17 scale.

## Kill criterion

**K1589:** interference ratio (AP init vs random Gaussian init) ≤ 0.67
on ≥3 of 5 held-out domains at end of pre-training. Failure → KILLED.

## Pre-registered precondition probe (tripwire)

Per the audit-2026-04-17 cohort-wide standing rule (Findings
#605/#606/#608/#610/#611/#612/#613/#615/#616/#617), this experiment MUST NOT
launch ~4h of MLX training before verifying three structural preconditions
exist on disk:

- **P1.** Gemma 4 E4B N=25 disjoint-domain adapters (q_proj + v_proj, 42
  layers) exist as `*.safetensors` at a canonical location. These are the
  reference experts against which interference is measured.
- **P2.** A Gemma 4 port of the Grassmannian AP skeleton from Finding #132
  (Qwen-0.6B) exists as runnable code. Dimensional/rank assumptions in the
  Qwen path do not auto-transfer — the skeleton must be re-derived for
  Gemma 4's hidden/head dimensions.
- **P3.** The upstream T2.1 single-domain training recipe has been rerun at
  LORA_SCALE=5, max_tokens≥512, rank-sweep {2,4,6,12,24}, grad-SNR logging,
  and `all_pass=true` in its `results.json` — this is the same rebuild every
  prior cohort downstream has blocked on.

If any of P1/P2/P3 fail: K1589 is UNMEASURABLE → status=killed, NO heavy
training, follow-up experiment becomes the upstream rebuild.

## Predicted verdict path

- 3/3 preconditions PASS → proceed to heavy MLX run, decide K1589 on
  observed interference ratio.
- ANY precondition FAIL → status=killed, K1589 result=fail (unmeasurable),
  evidence logs exact missing artifacts.

## Assumptions logged (guardrail 1007 autonomy)

- "Interference" is measured as expected cosine between expert update
  directions on disjoint domain batches; same operationalization as
  Finding #132. No MATH.md edit needed if probe fires.
- "≥1.5× lowers" ≡ AP_interference / random_interference ≤ 0.67.
