# REVIEW (adversarial) — exp_spark_interference_self_label  →  KILL upheld

Verdict under review: killed, mean AUROC 0.278 (< 0.70 KILL 2300), signal inverted. Upheld.

## Mock / not-real — PASS
- verify-experiment.sh exits 0; is_smoke:false; real model load via mlx_lm.load("mlx-community/gemma-4-e4b-it-4bit").
- Model in MATH.md == model loaded in code == config.base_model. Match.
- 3 adapters exist, distinct SHA-256 (no shutil.copy stand-in), real q_proj LoRA rank-6.
- Real held-out prompts (valid.jsonl, 199 lines/domain, 30 used). No hardcoded pass.

## Consistency — PASS
- results.json verdict=killed, all_pass=false, kill_criteria.2300.result=fail; PAPER.md "Verdict: KILLED". All agree.

## Integrity — PASS
- Threshold 0.70 identical in MATH.md, code (KILL_AUROC), results.json. Dir is untracked (no commit history),
  but for a KILL the only tamper concern is raising the bar; measured 0.278 fails any plausible bar — moot.
- Code measures what MATH.md claims: s_i = mean_t[ logit^{B+A_i}(ŷ_t) − logit^B(ŷ_t) ] over base-greedy picks.
- Independently recomputed AUROC from raw_scores: code 0.047, math 0.738, medical 0.049, mean 0.278 — exact match.
- No tautology: composition is single low-rank delta per proj; per-sample scoring; AUROC threshold-free.

## Evidence quality — PASS (strong refutation)
- Not a proxy artifact: the signal is behaviorally meaningful (margin on the base's own greedy token) and it
  is *inverted*, not merely flat. code/medical: mean_s_on MORE negative than mean_s_off (AUROC≈0.05).
- Honest reporting: PAPER admits the premise "base greedy ŷ_t ≈ on-domain correct continuation" is false for
  an instruction-tuned base teacher-forced on its own trajectory. Refuted by its own pre-registered threshold.

## Route: PROCEED-as-KILL. Falsifiable prediction crossed its pre-registered refutation line on a real run.
