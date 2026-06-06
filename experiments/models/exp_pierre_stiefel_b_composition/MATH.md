# MATH.md — Composition behavior of joint-Stiefel-trained adapters

## Position in the arc

Runs AFTER `exp_pierre_joint_stiefel_b_train` produces a set of K=7
jointly-Stiefel-trained PoLAR adapter weights. Tests composition methods
on those weights without further training.

> **Q1**: Does simple uniform 1/K mean composition reach TIES-B baseline
> on joint-Stiefel-trained adapters, beating both Fisher-Rao and TIES?
>
> **Q2**: Does the Karcher mean on Stiefel (Riemannian Frechet mean) do
> better than Euclidean mean?
>
> **Q3**: How does composition robustness compare to unconstrained
> adapter composition under noise / weight perturbation?

## Composition methods under test

For K=7 joint-Stiefel-trained adapters:

1. **`simple_mean`** — uniform 1/K, no rescaling. Should be the natural
   composition for Stiefel-trained adapters (orthogonal rows → Pythagoras
   gives clean magnitude prediction).

2. **`fisher_rao`** — norm-rescaled mean (Pierre's old default). May still
   help if Stiefel orthogonality isn't perfectly preserved.

3. **`ties_b`** — trim + sign-elect + disjoint merge. Heuristic that wins
   on unconstrained adapters; tests whether structure renders it
   redundant.

4. **`karcher_stiefel`** — Riemannian Frechet mean on Stiefel(K·r, d_out).
   The "right" mean for the joint manifold:
   ```
   Initialize: M_0 = simple_mean(B_all)
   Iterate:
     T_k = mean_k Log_M(B_k)  # tangent-space mean of log-mapped B's
     M_{k+1} = Exp_M(T_k)
   Converges to the manifold-aware mean.
   ```
   Expensive (iterative) but mathematically the canonical operator.

## Pre-registered Kill Criteria

- **K1 (SIMPLE MEAN WINS)** `simple_mean` on Stiefel-trained adapters
  ≥ TIES-B on standard adapters (71.3%).
  PASS → the mathematical-guarantee path delivers.

- **K2 (NO COMPOSITION HEURISTICS NEEDED)** `simple_mean` ≥ best of
  {fisher_rao, ties_b} on Stiefel-trained adapters by ≥ 1pp.
  PASS → heuristic corrections are redundant when constraint is built-in.

- **K3 (KARCHER UPSIDE)** `karcher_stiefel` ≥ `simple_mean` by ≥ 1pp.
  PASS → Riemannian Frechet mean adds value beyond Euclidean.
  FAIL = OK: simple_mean is good enough; karcher is theoretical garnish.

- **K4 (ROBUSTNESS)** Under 5% multiplicative noise on B_k weights,
  composition accuracy drops ≤ 2pp.
  PASS → Stiefel-trained composition is robust to small perturbations.

## Verdict outcomes

K1+K2 PASS → ship simple_mean composition for Stiefel-trained Pierre.
Whole composition arc closes with a 4-line composer.

K1 PASS, K2 FAIL → Stiefel training helps but heuristics still add value;
ship TIES-B-on-Stiefel.

K1 FAIL → joint-Stiefel training didn't deliver the promised composition.
Either training didn't converge to true joint Stiefel (check K2 of
sibling training experiment) or the math doesn't translate to
behavioral wins. Important negative finding.

## Implementation status

**SPEC — depends on `exp_pierre_joint_stiefel_b_train` weights.**

Required when ready:
1. Load joint-Stiefel-trained adapter weights from sibling experiment.
2. Implement Karcher mean on Stiefel(K·r, d_out) (~80 LoC, iterative).
3. Eval rig same as existing `_pierre_shared/eval_runner`.
4. ~1h eval runtime for 4 methods × 3 benchmarks at N=50.

## References

- Karcher mean on Stiefel: classical reference is Edelman/Arias/Smith
  (1998), "The geometry of algorithms with orthogonality constraints."
  arxiv: math/9806030.
- OrthoMerge (arxiv 2602.05943) uses inverse-Cayley + Riemannian mean;
  reference implementation.
- Sibling: `exp_pierre_joint_stiefel_b_train` (P2) — must complete first
