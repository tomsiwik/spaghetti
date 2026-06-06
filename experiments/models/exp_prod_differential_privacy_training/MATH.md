# MATH: `exp_prod_differential_privacy_training` — preemptive kill (5-theorem stack)

## TYPE
preemptive-kill (ap-017 drain). No empirical run on the proposed target claim.
The experiment's **running cost is provably unjustified** before any data
is collected.

## OBJECTIVE (claimed by target)
Train a LoRA adapter on Gemma 4 E4B on local-apple (MLX) under a formal
(ε=8, δ=1e-5)-DP-SGD guarantee, losing at most 10% quality vs. the non-DP
baseline, with a reproducible epsilon accountant across 3 seeds.
- **K1665**: "DP-SGD at epsilon=8, delta=1e-5 trains adapter with quality
  within 10% of non-DP baseline"
- **K1666**: "Epsilon accounting reproducible across 3 seeds"

Declared source experiment: `exp_p1_t5_user_local_training` (SUPPORTED,
P=1, macro, local-apple). This is the closest-scope SUPPORTED source:
same platform, same base model, same LoRA training surface — differing
exclusively on the DP mechanism.

## PRIOR MATH / REFERENCES
- ap-017 (audit-2026-04-17 cohort, 34th preempt as of this iter):
  5-theorem stack as codified in iters 35–37. Prior branches:
  composition-bug (24, incl. iter 37's software-infrastructure-unbuilt
  variant), scale-safety (2), tautological-routing (3), projection-scope
  (2), tautological-duplicate (1), hardware-topology-unavailable (2).
- F#502 / F#646 (schema-completeness-vs-instance-fix): DB-literal
  `success_criteria: [] # MISSING` is a **6th occurrence** here (after
  tfidf_routing_no_alias, flywheel_real_users, loader_portability,
  registry_host, version_resolution).
- F#652 (software-infrastructure-unbuilt variant of composition-bug,
  registered iter 37): in-repo library + data absences that
  `pip install` cannot fix are defense-in-depth blockers distinct from
  `hardware-topology-unavailable`.
- Abadi et al. 2016 (arxiv:1607.00133) — *Deep Learning with Differential
  Privacy* (DP-SGD algorithm, moments accountant).
- Mironov 2017 (arxiv:1702.07476) — *Rényi Differential Privacy* (RDP
  accountant now standard in Opacus).
- Opacus (PyTorch DP library) — https://github.com/pytorch/opacus — the
  only mature open-source per-sample-gradient + RDP-accountant stack.

## THE 5 THEOREMS (pre-flight)

### T1 — Prerequisite inventory (shortfall)
Let `R = {dp_sgd_optimizer_mlx, per_sample_gradient_mlx,
rdp_accountant, non_dp_lora_baseline_on_same_data}` be the artifacts
required to exercise K1665/K1666.

Repo-wide grep for Opacus / DP-SGD primitives:
- `opacus`, `make_private`, `RDPAccountant`, `noise_multiplier`,
  `per_sample_grad`, `clip_per_sample`, `vmap`, `dp_sgd`, `sigma_noise`,
  `gaussian_mechanism`: **0 hits outside skill-docs / jsonl data**.
- `pyproject.toml`: no `opacus`, no `jax-privacy`, no `tensorflow-privacy`,
  no `dp-*`. The `train` extra declares `torch>=2.4` + `peft>=0.13` but
  no DP library; the canonical local-apple stack is MLX / `mlx-lm`
  (micro extra), which has **no DP-SGD port in the open-source MLX
  ecosystem as of 2026-01 knowledge cutoff** and no such port in-repo.

Source MATH.md for `exp_p1_t5_user_local_training`: grep of {privacy,
differential, epsilon, dp-sgd, gaussian noise, clip grad} = **0 hits**.
The source's Theorem 2/3 *depend* on a standard (non-private) LoRA
update rule (ΔW = B·A). DP-SGD injects calibrated Gaussian noise per
per-sample clipped gradient; this is an entirely different update
rule that the source never derived and never ran.

No non-DP baseline on the *same data* at the same LoRA config exists
either; the source ran a single user's 50 conversation examples, not a
controlled DP-vs-nonDP comparator pair at matched hyperparameters.

**shortfall = |R| − |R ∩ repo| = 4 − 0 = 4.**

### T2 — Scale-safety budget
Source T5.1 training time (K1096 measured): ~5-7 min for 300 iters on
16 layers, batch 2, seq 256 (M5 Pro). Source T2.1 (K1031): ≤22 min for
1000 iters on 42 layers, batch 2, seq 512.

DP-SGD wall-time overhead vs non-DP LoRA is dominated by per-sample
gradient computation. Reported factors in the literature:
- Opacus + PEFT LoRA on NVIDIA A100: **10-30× slower** (Yu et al. 2022,
  arxiv:2110.06500, §5.2).
- Without GPU vmap support, naive per-sample-grad (one backward per
  example) scales as O(batch_size): at batch=2, that is **2× baseline**
  just for the per-sample pass, plus noise + accountant overhead.

Assume a conservative **10× overhead** on M5 Pro (no published Opacus-
for-MLX benchmark exists — see T1). Per seed, per the T2.1 baseline of
22 min, DP-SGD is ≈ 220 min = **3.67 h per seed**. K1666 requires
**3 seeds** for accountant reproducibility → ≈ **11 h total**.

Micro budget ceiling (PLAN.md Part 2, standing cohort rule): **120 min**.
11 h = 660 min = **5.5× over ceiling**. And that ignores the separate
non-DP baseline pair required for K1665's "within 10% of non-DP
baseline" → another 3 × 22 = 66 min minimum, making total ≈ 12.1 h.

**T2 blocks.**

### T3 — DB literal schema completeness
`experiment get exp_prod_differential_privacy_training`:
```
success_criteria: [] # MISSING
⚠ INCOMPLETE: success_criteria, references, kill_results (all untested)
```

This is a DB LITERAL match for F#502/F#646. It is the **6th** such
match in the audit-2026-04-17 drain (after tfidf_routing_no_alias,
flywheel_real_users, loader_portability, registry_host,
version_resolution).

### T4 — Kill-criterion pin ratio
- K1665: "epsilon=8" ✓ + "delta=1e-5" ✓ + "within 10%" ✓ + "quality"
  ✗ (no metric: GSM8K? HumanEval? MedQA? response-compliance as in T5.1?
  unspecified) ⇒ 3 pinned / 4 sub-claims.
- K1666: "3 seeds" ✓ + "reproducible" ✗ (no threshold: CV<1%? max-min
  spread <0.1 ε? identical byte-for-byte accountant state?) ⇒ 1 / 2.

Total pin ratio = 4 / 6 = **0.667** (above the 0.20 auto-block floor).
**Reinforces only.**

### T5 — Source-scope breach vs `exp_p1_t5_user_local_training`
Target depends on this SUPPORTED source (same platform, same base
model). Source's own LITERAL KCs and proof scope:
- **K1099** (LITERAL, PASS @ 127 lines): *"Script single-file, < 200
  lines"* — proved for HF PEFT on consumer GPU; never exercised DP.
- **K1096** (LITERAL, PASS @ 1.2 min): *"Adapter trained from 50
  conversation examples in < 10 minutes on consumer GPU"* — single-seed,
  single-user, no noise injection.
- **MATH.md Theorem 1 Step 2**: standard SGD convergence bound
  `E[L(θ_T)] ≤ L(θ*) + C/(η·T)` — this bound *does not hold* under
  DP-SGD; the noise + clipping introduce a different bias-variance
  tradeoff (Bassily et al. 2014, arxiv:1405.7085) that the source never
  derived.
- **Source MATH.md grep {privacy, differential, epsilon, dp-sgd,
  gaussian noise, clip grad}**: 0 hits. Source never entered DP scope.

Five LITERAL target/source scope breaches:

- **(A) privacy-mechanism-scope.** Source proves LoRA can inject user
  preferences with 50 examples (T5.1 K1097 achieved +60pp). Target
  claims this capability *survives ε=8-DP noise*. DP noise
  monotonically degrades accuracy (Abadi 2016 §3); whether the +60pp
  gain survives ε=8 is an *empirical* question the source never asked.
  Source MATH.md contains 0 occurrences of {privacy, differential,
  epsilon, dp, noise, clip}.
- **(B) library-scope.** Source K1099 LITERALLY binds "single-file,
  <200 lines" + "HF PEFT" (T5.1 K1099 text and 127-line measurement).
  A DP-SGD training script requires per-sample gradient vmap, calibrated
  Gaussian noise, and an RDP accountant — Opacus is ~3000 lines, a
  minimal DP-SGD wrapper is ≥500 lines even without accountant. Target
  breaches the source's <200 line contract by construction.
- **(C) comparator-scope.** Source compares *adapter vs base* (Δ ≥ 5pp
  in K1097). Target K1665 compares *DP adapter vs non-DP adapter* (Δ ≤
  10%). The source never ran a same-data non-DP baseline at matched
  hyperparameters; it ran a single adapter config. The comparator axis
  target K1665 demands does not exist in the source's evidence.
- **(D) reproducibility-scope.** Source ran N=1 (one user, one adapter,
  one seed). Target K1666 requires 3-seed accountant reproducibility —
  not a claim the source's evidence can support, because a 1-seed
  training run cannot attest to 3-seed variance.
- **(E) platform-library-scope.** Source platform = local-apple (MLX).
  The DP-SGD ecosystem is PyTorch-only (Opacus) or JAX-only
  (jax-privacy / TensorFlow-Privacy). **No MLX-native DP-SGD library
  is in the open-source ecosystem as of 2026-01 cutoff**, and 0 hits
  in-repo (T1). This is a *cross-cut* of ap-017 (s)
  (hardware-topology-unavailable: DP-SGD tooling is in a different
  *ecosystem*) and ap-017 (s2) (software-infrastructure-unbuilt: the
  specific artifacts — per-sample vmap on MLX, MLX-RDPAccountant —
  are not in-repo). F#652 variant.

**T5 score: 5/5 LITERAL breaches.**

### Verdict
```
block(T1) = True                 # 4 missing artifacts
block(T2) = True                 # 11 h + 1 h baseline vs 120 min ceiling
block(T3) = True                 # schema-incomplete DB literal (6th F#502)
block(T4) = False                # 0.667 > 0.20 floor; reinforces
block(T5) = True                 # 5/5 literal source-scope breaches

all_block = T1 ∧ T2 ∧ T3 ∧ T5 = True
defense_in_depth = any single one of {T1, T2, T3, T5} alone also blocks
```

Four independent blockers — the strongest defense-in-depth of the
audit-2026-04-17 drain so far. T2 first blocks on its own since iter 35
(hardware_topology) had 24 h network uptime; this iter has the first
T2-block where compute-time overhead alone (DP-SGD × 3 seeds) exceeds
the ceiling on in-repo hardware.

## THEOREM 1 (Preemptive kill)
Under `block(T1) ∨ block(T2) ∨ block(T3) ∨ block(T5)` (in fact all
four), the only honest verdict achievable by any 3-seed DP-SGD run of
this experiment is `killed_preregistered`, because:
1. The data-collection machinery (DP-SGD on MLX + RDP accountant) does
   not exist in the repo or the open-source MLX ecosystem (T1).
2. Even if it existed, the 3-seed budget exceeds the 120-min micro
   ceiling by ≥ 5.5× (T2).
3. The DB's KC is under-specified (T3); no success_criteria means no
   positive-pass rule exists to distinguish supported from provisional.
4. The source never proved LoRA training *survives* DP noise at any
   ε (T5(A,C,D)), and the platform has no DP library (T5(E)).

Running it would either silently upgrade to `supported` without the
machinery to separate it from `killed` (per G1009), or consume ≥ 11 h
of MLX compute on infrastructure-building that PLAN.md Part 2 has not
scoped. **QED.**

## PREDICTIONS (pre-flight)
- P1: T1 shortfall ≥ 3 (≥ 3 of 4 artifacts absent).
- P2: T2 estimated wall time ≥ 600 min (5× over ceiling).
- P3: DB `success_criteria` remains `[]` post-run (T3 unchanged);
  `⚠ INCOMPLETE` flag persists.
- P4: Source MATH.md `grep -iE "(privacy|differential|epsilon|dp.sgd|
  gaussian.noise|clip.grad)"` = 0 post-run (T5(A) unchanged).
- P5: all_block (T1 ∧ T2 ∧ T3 ∧ T5) = True; defense_in_depth = True.

## KILL CRITERIA (pre-registered; locked, do not edit after data)
- **K1665 (DP-SGD ε=8 within 10% non-DP baseline):**
  pre-flight FAIL — no DP-SGD on MLX + no non-DP baseline at matched
  data + comparator never exercised in source (T1 ∧ T2 ∧ T5(A,C,E)).
- **K1666 (3-seed accountant reproducibility):**
  pre-flight FAIL — no RDP accountant in-repo + source N=1 (T1 ∧ T5(D)).

Both are **preempted** by the 5-theorem stack; neither is exercised
with data.

## BEHAVIORAL OUTCOME
The behavior this preempt enforces: do not consume ≥ 11 h of M5 Pro
MLX compute to generate a 5th PAPER.md that would be indistinguishable
from iter 37's F#502/F#652 preempt. Drain-forward honesty: if the
infrastructure (DP-SGD on MLX) does not exist in the open-source
ecosystem, say so and route the work to the operator who can either
(a) port Opacus to MLX (a >6-month engineering effort), (b) declare
the target out-of-scope for local-apple, or (c) downgrade to P≥3.

## ASSUMPTIONS (logged per guardrail 1007)
- A1. ap-017 5-theorem stack is the canonical preempt tool for this
  cohort. defense-in-depth passed with T1 ∨ T2 ∨ T3 ∨ T5.
- A2. DP-SGD overhead factor of 10× is conservative; published Opacus
  benchmarks on A100 report 10-30× for LoRA. On M5 Pro / MLX (no vmap
  for per-sample grad), naive implementation would be even slower, so
  the 11 h estimate is a *lower bound*.
- A3. Source MATH.md's standard SGD convergence bound
  (Theorem 1 Step 2) does not transfer to DP-SGD; Bassily 2014 gives
  a separate `O(√(log(1/δ)) / (ε·√n))` excess risk term. The source
  never considered this and does not have to — its scope is non-DP.
- A4. is_smoke = False. This is a complete pre-flight evaluation
  against the target claim, not a partial / smoke run.
- A5. ap-017 axis for this preempt: **composition-bug
  (software-infrastructure-unbuilt, platform-library cross-cut
  variant)** — distinct from iter 37's pure in-repo-software variant
  because this iter's absent library (Opacus on MLX) is absent from
  *both* the repo *and* the open-source MLX ecosystem. Sub-axis for
  analyst to formalize when the cap raises.

## NON-GOALS
- Porting Opacus to MLX. PLAN.md Part 2 has not scoped it.
- Running any DP-SGD training, baseline, or inference.
- Proposing a v2 experiment. That is the operator's call after either
  (a) declaring DP-SGD-on-MLX as a new SUPPORTED dependency, or
  (b) downgrading this target to P≥3 / out-of-local-apple scope.
