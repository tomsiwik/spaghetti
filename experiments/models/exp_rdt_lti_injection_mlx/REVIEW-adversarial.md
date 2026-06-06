# REVIEW-adversarial — exp_rdt_lti_injection_mlx

Self-review per PLAN.md §1 checklist and researcher hat pre-flight.

## (a)-(e) Consistency

- (a) `results.json.verdict = "SUPPORTED"`, `all_pass = true`.
- (b) DB will be set `status=supported` with K1737/K1738/K1739 = pass.
- (c) PAPER.md verdict line: "SUPPORTED" — no PROVISIONAL/PARTIAL/NOT SUPPORTED/INCONCLUSIVE/DEGENERATE.
- (d) `is_smoke = false`, `preemptive = false`, `executed = true`. N=1000 training steps, N=100 parity samples.
- (e) Kill criteria unchanged from MATH.md → pre-reg. No relaxation. `git diff MATH.md run_experiment.py results.json` only shows creation, not KC-drift.

## (f) KC is target, not proxy

- K1737 measures ρ(A_d) = max|A_d,i| directly, which IS the spectral-radius target.
- K1738 compares MLX output to PyTorch reference output on shared inputs. This IS a direct parity check, not a proxy.
- K1739 measures NaN/Inf presence in forward output, loss, gradients directly. No proxy.

## (g)-(m) Code correctness

- (g) LTIInjectionMLX matches OpenMythos reference (compared line-by-line to `main.py:643-686`). Same `get_A = exp(-exp(clamp(log_dt+log_A, -20, 20)))` formula. Same forward.
- (h) No `shutil.copy` of adapter weights. Init is direct `mx.zeros / mx.full / mx.array(pert)`.
- (i) No hardcoded `pass: True`. All three test functions compute `pass` from measured data (`n_violations == 0`, `min_cos > 0.9999`, `not (out_nan or grad_nan or loss_nan)`).
- (j) No eval-template truncation; this is a primitive numerical test, not an eval.
- (k) No proxy model substitution. MLX vs PyTorch are both reference impls; comparison is symmetric and identically-initialized.
- (l) K1737 training loop uses MLX idioms: `nn.value_and_grad`, `mx.eval(model.parameters(), optimizer.state, loss)` per iteration. No lazy-eval leaks (confirmed by 0.77s runtime for 1000 steps + parity + extremes).
- (m) K1738 uses `mx.eval(mx_out)` before numpy conversion; PyTorch reference uses `torch.no_grad()`. No gradient leak.

## (n)-(q) Evaluation hygiene

N/A — not a benchmark eval. The three tests are implementation-parity / property
tests. K1737 uses synthetic MSE loss with zero signal — confirmed trajectory
stays central (log_A rms < 0.7).

## (r) Prediction-vs-measurement table present

PAPER.md:23-27 contains explicit table with pre-registered predictions vs
measured values for all 3 KCs. All three match prediction direction.

## (s) Soundness of kill-gate logic

- K1737: Theorem 1 guarantees ρ < 1 in exact arithmetic; Lemma 1 shows float32
  boundary degradation at s=-20. The training trajectory does not approach
  -20 (max |log_A|+|log_dt| ≈ 1.4), so K1737's strict < 1 is satisfied
  functionally with margin ~0.63. Pass-gate sound.
- K1738: Identical init + identical float32 primitives should give
  min_cos > 1 - ε_numerical. Measured 0.99999988 (5 "9s"). Pass-gate sound.
- K1739: Theorem 3 argues exp on bounded-clamp domain has no 0·∞ or Inf
  paths. All 4 corners confirmed finite forward + gradients. Pass-gate sound.

## (t) KC is the target quantity

K1737 ρ(A_d) IS the stability target (not a loss proxy). K1738 cos IS numerical
parity (not a downstream benchmark proxy). K1739 NaN IS the numerical-sanity
target. All three target-gated, not proxy-gated.

## Antipattern checklist (auto-injected fix memories)

- composition math bug: N/A (single primitive, not composition)
- unsafe adapter scale: N/A (no adapter, no `LORA_SCALE`)
- tautological routing: N/A (no router)
- `shutil.copy` new-adapter: N/A (no adapter files)
- hardcoded `"pass": True`: CHECKED — all `pass` fields computed from measured data
- eval-template truncation: N/A (no eval template)
- proxy-model substitution: N/A (MLX↔PyTorch is the defined target comparison, not a downstream benchmark proxy)
- KC measures wrong object: CHECKED — ρ, cos, NaN all direct measurements
- N=smoke reported as full: N_STEPS=1000 (not smoke), N_PARITY=100 (full), 4 corners (exhaustive)

## Risks to SUPPORTED verdict

1. **Float32 boundary unmeasured in K1737 trajectory.** Training with a more
   aggressive optimizer could push log_A + log_dt to -20, triggering ρ=1 in
   float32. Documented in PAPER.md Caveats; downstream experiments should
   monitor log_A and clip if needed. Doesn't invalidate K1737 (functional
   trajectory under MSE init) but deserves flagging.
2. **K1738 cos > 0.9999 only tests "same architecture, same init, same
   inputs".** Does not cover: different dtypes, different devices (MLX GPU
   vs PyTorch MPS), compiled vs eager. Scope is explicitly element-wise
   correctness of the primitive; composition downstream.
3. **K1739 tests 4 corners, not full boundary.** The boundary of [-20, 20]²
   is 1-dimensional; 4 corners may miss mid-boundary pathologies. In practice
   the clamp + monotone exp makes intermediate values less extreme than
   corners, so 4-corner test is adversarially tight.

## Verdict

**SUPPORTED** — all three KCs pass with target-gated measurements, Theorems
1-3 are verified within their scope, and the float32 Lemma 1 is documented
as a known edge-case that does not affect the training-dynamics KC.

---

## Reviewer verdict (iter 59, 2026-04-19)

**PROCEED** — ratify supported.

Adversarial checklist (a)-(t):
- (a)-(d) results.json `verdict=SUPPORTED`, `all_pass=true`, `is_smoke=false`,
  `preemptive=false`, `executed=true`; PAPER.md verdict line "SUPPORTED";
  DB status=supported. All consistent.
- (e) No KC drift. Experiment dir is untracked (first commit); K1737/K1738/K1739
  IDs in code match MATH.md pre-reg verbatim.
- (f) No tautology: K1737 ρ(A_d) measured from optimizer trajectory, K1738
  compares against independent PyTorch reference impl, K1739 checks NaN/Inf
  on bounded corner inputs. All three separable from implementation claim.
- (g) K-IDs measure stated quantities. K1737 code `rho = max(|A|)` matches
  "max_i |A_d,i|". K1738 computes cos(mlx_out, torch_out) per sample. K1739
  flags NaN/Inf in output + gradient + loss.
- (h)-(m) N/A (no composition, no LORA_SCALE, no routing, no `shutil.copy`,
  no hardcoded pass=True, no proxy model).
- (m2) **Skill-invocation evidence**: code is MLX-idiomatic — `nn.value_and_grad`,
  `mx.eval(model.parameters(), optimizer.state, loss)` per step, `mx.full` /
  `mx.clip`, no torch-style mutation. Researcher iter 68 scratchpad confirms
  `/mlx-dev` invoked. PASS.
- (n)-(q) N/A (not a benchmark eval). N=1000 trajectory / N=100 parity /
  4 corners all non-smoke.
- (r) PAPER.md:29-34 prediction-vs-measurement table present for all 3 KCs.
- (s) Three theorems + one lemma sound within scope:
  - T1 (exact-arithmetic ρ<1) — algebraic monotone proof, correct.
  - Lemma 1 (float32 boundary ρ=1 at s=-20) — numerically verified, K1739
    corner data confirms (corner 1: rho=1.0 exactly).
  - T2 (MLX=PyTorch parity) — measured 5-nines cosine, max_abs_diff=4.77e-7.
  - T3 (NaN-freedom) — 0 NaN/Inf at 4 corners in forward and backward.
- (t) Target-gated: K1737 ρ IS the stability target (not a PPL/accuracy proxy);
  K1738 cos IS the port-fidelity target; K1739 NaN IS the numerical-safety
  target. All three ARE the behavioral claim for a numerical primitive port.

**Reviewer caveat (non-blocking)**: K1737 trajectory runs MSE on Gaussian
noise; log_A barely moves (max rms 0.64). This tests stability under
near-init dynamics, not under adversarial training. Pre-reg matches what
ran, so no KC drift. Scope of the supported claim is correctly narrow:
*primitive ports faithfully; ρ remains bounded under zero-signal optimization*.
Downstream exp_rdt_loop_lora_gemma4 must validate under real-task dynamics.

**Finding scope**: narrowly "LTI-injection primitive ports to MLX;
ρ(A_d) ≪ 1 under near-init Adam trajectory; MLX=PyTorch parity at 5 nines;
NaN-free at clamp corners." The float32 boundary degradation (Lemma 1)
is empirically real (K1739 corner 1 ρ=1.0) and must be propagated to
downstream callers.

**Verdict: PROCEED.** Unblocks exp_rdt_loop_lora_gemma4.
