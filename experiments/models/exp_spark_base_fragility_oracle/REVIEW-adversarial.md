# REVIEW-adversarial — exp_spark_base_fragility_oracle (verdict: KILLED)

Reviewer with fresh context. Goal: break the null, not confirm it. Could not.

## Mock / not-real: PASS
verify-experiment.sh exits 0. is_smoke=false. Real gemma-4-e4b-it-4bit loaded twice
(base + composed). Three adapters have DISTINCT md5 (math/python/medical) — no shutil.copy
sibling. No numpy/random stand-in for the model output.

## (a) Predictor uses ZERO adapter info: CONFIRMED
fragility (run_experiment.py:323-337) is built only from h, h_norm, and a unit-norm random
Gaussian u from mx.random.normal seeded mx.random.seed(EPS_SEED+1000*pi+k), EPS_SEED=20250609.
eps = NOISE_FRAC*‖h‖*u. No lora_a/lora_b/adapter tensor enters the fragility branch — the
adapters are only loaded/applied in Phase 2 (composed logprobs). ε is fixed-norm random noise,
not a disguised adapter delta. Seed is time-independent: time.time() appears only for wall-clock.

## (b) Damage label real with genuine spread: CONFIRMED
damage = base_lp - composed_lp from two independent full forwards (base vs (1/N)ΣsBᵢAᵢ q_proj).
Spread is real: frac_tokens_damaged=0.352 (both signs present), mean_damage=-0.103 (composition
net-helps — a signed real difference, not degenerate), top-decile split 387 pos / 3441 neg
(=3828=n_tokens_scored). Not all-zero; a meaningful top decile exists.

## (c) Near-zero rho is a real null, not a shape/align bug: CONFIRMED
rho=0.008492860612514402 is NOT exactly 0.0 → spearman()'s da==0/db==0 guard (lines 254-255)
did NOT fire → BOTH fragility and damage arrays have nonzero variance. So the ~0 correlation is
genuine independence, not a constant-array artifact. Alignment is enforced: both phases iterate
identical prompts with identical encode_capped, asserted per-token via rec["prompt_idx"]==pi and
ptr==len(per_tok). fragility[t] and damage[t] share position t (predicting token t+1) by
construction. No off-by-one zeroing.

## Integrity
RHO_FLOOR=0.30, AUC_FLOOR=0.62 match MATH.md kill-2305 verbatim and the stored config. MATH.md
is a new untracked file (no post-run threshold edit possible in git history). Both clauses fired
(rho<0.30 AND AUC<0.62); either alone kills. results.json (killed/all_pass:false/is_smoke:false)
and PAPER.md verdict agree.

## Verdict: KILLED — sealed.
Isotropic base-curvature fragility carries no rank signal for per-token composition damage.
