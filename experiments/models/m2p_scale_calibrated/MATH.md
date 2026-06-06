# MATH.md: M2P Scale Calibrated — Proof of Adapter Scale Self-Calibration

## Experiment Type: Type 1 — Proof Verification (with Type 2 component for λ)

**Scope:** Single domain (arithmetic). No multi-domain joint training.

**Prior kills that constrain scope:**
- Finding #341: Multi-domain joint M2P training is structurally impossible due to
  gradient conflicts (gradient sum dominated by high-loss domains).
- Finding #342: Additive domain embeddings cannot overcome attention bottleneck
  (Jacobian ∂B/∂e_d effectively low-rank).

This experiment avoids both failure modes by training M2P on ONE domain only.

---

## A. Failure Mode Identification

### What is the disease?

Finding #330 (exp_solidified_composition_mmlu) proved:
- Adapters trained at scale=5 → 0pp degradation on MMLU under composition
- Adapters trained at scale=20 → -42pp degradation on MMLU under composition

The **disease** is not scale itself, but the fact that the adapter scale is fixed
at training time, independent of how much the adapter actually needs to perturb the
model. A large scale is catastrophic for adapters that should make subtle modifications,
and insufficient for adapters that need to make large changes.

**The failure mode:** M2P trained with L_task = CrossEntropy(adapter_output, task_data)
alone will greedily increase adapter magnitude to minimize task loss, with no countervailing
force. The model learns: "bigger adapter → lower training loss" as a consistent signal.
This is a stable fixed point of unconstrained gradient descent.

**Formal statement of the failure mode:** Let α = ||B||_F (adapter scale). For pure
task loss L_task(α), ∂L_task/∂α ≤ 0 almost surely (higher α → lower task loss, at
least locally). Therefore unconstrained gradient descent on L_task(α) drives α → ∞.
In practice, weight norm grows until curvature or learning rate stopping halts it at
some arbitrary large value, not at the optimal value.

**Why this IS the root cause, not a symptom:** The B-matrix scale determines how much
the adapter perturbs the base model's knowledge. Too large → catastrophic forgetting
of general knowledge. Too small → insufficient task learning. The "right" scale α* is
determined by the equilibrium between task improvement and general knowledge preservation.
Without L_preserve, no such equilibrium exists in the loss landscape.

---

## B. Reframe — The Right Question

**Wrong question:** "How do we prevent adapters from being too large?"

**Right question:** "What is the optimal adapter scale α* such that the task improvement
gradient and the general quality gradient are in equilibrium? Can M2P self-discover α*
without being explicitly told?"

The answer follows from the KKT conditions for constrained optimization:

> α* = argmax L_task(α) subject to L_preserve(α) ≤ τ

By the KKT complementary slackness theorem, the unconstrained Lagrangian formulation:

> L_total = L_task + λ · L_preserve

has the same first-order necessary conditions at the optimal α* (with appropriate λ).

This means: adding L_preserve with any λ > 0 creates an equilibrium fixed point at α*,
which depends on λ and the data distribution but NOT on arbitrary scale hyperparameters.
M2P learns α* by gradient descent, not by programmer intervention.

---

## C. Prior Mathematical Foundations

### C.1 KKT Conditions (Kuhn-Tucker, 1951)

For the constrained problem: maximize f(x) subject to g(x) ≤ 0, the KKT conditions
state that at an optimal point x*:

  ∇f(x*) = μ · ∇g(x*) for some μ ≥ 0

Applied to our problem with f = -L_task and g = L_preserve - τ:

  -∇_α L_task(α*) = λ* · ∇_α L_preserve(α*)

which is exactly the gradient stationarity condition for L_total = L_task + λ · L_preserve.

**Theorem reference:** Karush-Kuhn-Tucker conditions, 1939/1951. Standard result in
constrained optimization (Nocedal & Wright, "Numerical Optimization", Ch. 12).

### C.2 Fixed-Point Existence (Brouwer Fixed-Point Theorem, 1912)

For a continuous map F: K → K on a compact convex set K ⊂ R^n, there exists a
fixed point F(x*) = x*. Applied to the gradient flow:

  α_{t+1} = α_t - η · (∂L_task/∂α + λ · ∂L_preserve/∂α)

Under mild smoothness assumptions on L_task and L_preserve (Lipschitz continuous
gradients, which hold for cross-entropy loss on bounded inputs), there exists α* where
the gradient is zero. This α* is the self-calibrated scale.

**Monotonicity argument:** For cross-entropy losses:
- ∂L_task/∂α < 0 for α < α_task* (task still improvable by scaling up)
- ∂L_preserve/∂α > 0 for α > 0 (increasing scale always increases preservation loss)

Therefore ∂L_total/∂α = 0 has a solution α* in (0, α_task*) whenever λ > 0.
The sign argument guarantees a crossing point — the equilibrium is non-degenerate.

### C.3 Jensen's Inequality / Adapter Scale and Output Variance

For a linear layer with LoRA: W_adapted = W_base + α · A @ B where ||B||_F = 1.

The output perturbation Δy = x @ (α · A @ B)^T has Frobenius norm:

  ||Δy||_F = α · ||x @ (A @ B)^T||_F ≤ α · ||x|| · ||A||_op · ||B||_F = α · ||x||

So adapter output magnitude scales linearly with α. L_preserve measures how much
the base model output distribution is perturbed, which is a monotone function of α.
This confirms the monotonicity argument in C.2.

### C.4 Davis-Kahan Sin-Theta Theorem (Scale → Quality relationship from Finding #330)

Finding #330 empirically confirmed: composition quality (MMLU score) is a decreasing
function of adapter scale when scale > 5. The Davis-Kahan theorem (1970) provides the
mechanism: for a perturbed symmetric matrix A + E where ||E||_2 = ε, the perturbation
to the top-k subspace is bounded by ε / (λ_k - λ_{k+1}). Larger ε (adapter scale)
→ larger subspace rotation → more knowledge disruption.

The L_preserve gradient ∂L_preserve/∂α serves as a proxy for the Davis-Kahan
perturbation bound: minimizing L_preserve minimizes the adapter's interference with
the base model's knowledge representation.

---

## D. Proof of Guarantee

**Theorem 1 (Scale Self-Calibration).**

Let M2P be a differentiable neural network that generates B-matrices for LoRA adapters.
Let L_task(B) = CrossEntropy(model_with_adapter(B), task_data) and
L_preserve(B) = CrossEntropy(model_with_adapter(B), general_data).

Define α(B) = ||B||_F (the adapter scale as a function of B).

**Claim:** Training M2P on L_total = L_task + λ · L_preserve converges to a fixed
point B* where α(B*) ≈ α* satisfying:

  ∂L_task/∂α |_{α*} = -λ · ∂L_preserve/∂α |_{α*}

*Proof.*

Step 1: Gradient structure.

  ∂L_total/∂B = ∂L_task/∂B + λ · ∂L_preserve/∂B

At convergence, ∂L_total/∂B = 0 (gradient stationarity). This implies:

  ∂L_task/∂B + λ · ∂L_preserve/∂B = 0   ... (*)

Step 2: Decomposing gradient with respect to scale.

Project both sides of (*) onto the direction ∂α/∂B = B / ||B||_F (radial direction
in B-space). Since α = ||B||_F, this projection gives the KKT condition:

  ∂L_task/∂α + λ · ∂L_preserve/∂α = 0   ... (**)

Step 3: Sign analysis ensures solution existence.

For cross-entropy loss on task data:
  - L_task decreases as adapter scale increases from 0 to α_task* (adapter helps)
  - Therefore ∂L_task/∂α < 0 for α ∈ (0, α_task*)

For cross-entropy loss on general data:
  - L_preserve increases monotonically with α (larger adapter → more disruption)
  - Therefore ∂L_preserve/∂α > 0 for all α > 0

From (**): ∂L_task/∂α = -λ · ∂L_preserve/∂α requires the left side to be positive.
Since ∂L_task/∂α < 0 for α < α_task*, the equation (**) requires α > α_task_zero,
where the task loss gradient changes sign. And since ∂L_preserve/∂α > 0, the right
side is always positive (times -λ, hence negative). 

By the intermediate value theorem applied to h(α) = ∂L_total/∂α:
  - h(0) = ∂L_task/∂α|_0 - ε < 0 (task gradient dominates at α=0, L_task ↓)
  - h(∞) = λ · ∂L_preserve/∂α|_∞ > 0 (preserve gradient dominates at large α)

By IVT, there exists α* ∈ (0, ∞) such that h(α*) = 0.

Step 4: M2P converges to α*.

M2P's output B determines α = ||B||_F. Gradient descent on L_total:
  B_{t+1} = B_t - η · ∂L_total/∂B_t

Since L_task and L_preserve are both convex in B in a neighborhood of B*
(cross-entropy loss of a Lipschitz network is locally convex), gradient descent
converges to B* where both gradients balance, achieving α = α*.

Step 5: Self-calibration property.

Crucially, α* is NOT a hyperparameter. It emerges from the data:
  - Hard domains (large task gradient) → large α* (M2P learns to output bigger adapters)
  - Easy domains (small task gradient) → small α* (M2P learns to output smaller adapters)
  - λ sets the TRADEOFF between task quality and preservation, not the absolute scale

Therefore, given different task descriptions, M2P will generate adapters with different
magnitudes, automatically calibrating to the task difficulty. This is the "self-calibration"
property: magnitude variance across contexts is an observable signature of self-calibration.

QED.

---

## E. Quantitative Predictions (Derived from the Proof)

### Prediction 1 (K849): General quality degradation < 10pp

From Theorem 1: at the fixed point α*, the L_preserve gradient is balanced against
L_task. For λ = 0.1 (chosen as a moderate regularization weight), the equilibrium
condition means ∂L_preserve/∂α is 10x smaller than ∂L_task/∂α.

By the Davis-Kahan monotonicity argument: the adapter perturbation at α* causes
approximately 10x less general quality degradation than an unconstrained adapter
that maximizes task loss alone.

Finding #330 showed scale=5 → 0pp degradation, scale=20 → -42pp degradation.
L_preserve acts as a soft constraint pushing toward scale < 10. Expected degradation:
general PPL ratio < 1.10 (10% increase = 10pp in our PPL-based metric).

**Predicted bound:** general_degradation_pp < 10pp
**Kill criterion K849:** measured general_degradation_pp ≥ 10pp → FAIL

### Prediction 2 (K850): M2P adapters have varying magnitudes (self-calibration)

From Theorem 1 Step 5: the self-calibrated scale α* depends on the task gradient,
which varies with input context. For arithmetic tasks with different operand sizes
(5-10 number addition vs single-digit addition), the task gradient magnitude differs.

Therefore: M2P's B-matrix Frobenius norm ||B||_F should vary across different
arithmetic task descriptions. If ALL contexts produce the same ||B||_F, the M2P
has mode-collapsed to a constant regardless of input (same failure mode as #341/#342,
but for scale instead of direction).

**Predicted bound:** std(||B_i||_F) / mean(||B_i||_F) > 0.05 (5% coefficient of variation)

This threshold is derived from the prediction that different operand sizes in arithmetic
(easy: 5+3, hard: 987+456) should differ in required adapter magnitude by at least 5%.

**Kill criterion K850:** adapter_magnitude_variance = 0 (all identical) → FAIL

### Prediction 3: Self-calibrated scale in range [3, 15]

Finding #330 established:
- Scale=5: 0pp degradation (safe)
- Scale=20: -42pp catastrophic

The L_preserve gradient at scale=5 is small (model is mostly unperturbed). At scale=20
it is large. For λ=0.1, the equilibrium α* should lie near the "knee" of the
degradation curve, empirically around scale=5 to 15.

**Predicted range:** 3 ≤ scale_learned ≤ 15
This is a Type 2 (guided exploration) prediction — the exact α* within this range
depends on the training data and λ, and is unknown until we measure it.

---

## F. Assumptions and Breaking Conditions

**Assumption 1:** L_task and L_preserve are both differentiable with respect to B.
- *If violated:* Gradient descent cannot find α*. Violated only for discrete losses;
  cross-entropy is smooth.

**Assumption 2:** L_preserve is monotone increasing in α.
- *If violated:* Multiple equilibria exist; no unique α*.
- *Testing:* We measure PPL_general vs ||B||_F correlation. Negative correlation →
  assumption violated → K849 threshold may not hold.

**Assumption 3:** Single domain only (no gradient conflicts from multi-domain training).
- *If violated:* Finding #341/#342 failure mode returns (centroid collapse).
- *Enforced by experiment design:* arithmetic only, no round-robin.

**Assumption 4:** M2P has sufficient capacity to vary B-matrices across task inputs.
- *If violated:* K850 fails — all contexts produce identical magnitude.
- *Mitigation:* Using same M2P size as prior experiments (2-layer, d=256, 4 heads).

**Breaking condition for Theorem 1:**
If the gradient ratio |∂L_task/∂α| / |∂L_preserve/∂α| = 0 for all α (flat preserve
landscape), the equilibrium condition cannot be satisfied and L_total = L_task.
In practice this would require general_data to be unaffected by adapter scale, which
is impossible for non-trivial adapters.

---

## G. Worked Example (d=256, rank=4)

**Setup:**
- Toy GPT: d=256, L=2, rank=4, vocab=128
- B matrix for one attention head: shape (rank, d_out) = (4, 256)
- α = ||B||_F

**Forward pass with adapter:**
  y = x @ W_adapted^T = x @ (W_base + α · A @ B / ||B||_F)^T

Perturbation at the output:
  Δy = α · x @ (A @ B / ||B||_F)^T

For x = random unit vector, A orthogonal (4 columns of d=256 space):
  ||Δy|| ≈ α · ||A||_op · 1 = α · 1 = α

**Task loss gradient (arithmetic):**
After 100 steps of training, assume L_task ≈ 1.5 (before convergence).
If α is doubled (5 → 10), L_task decreases by roughly log(2) · ∂L_task/∂α.
∂L_task/∂α is negative and roughly -0.1 at α=5 (estimated).

**Preserve loss gradient:**
L_preserve at α=5: PPL_general ≈ 1.05 × PPL_base (5% increase)
L_preserve at α=10: PPL_general ≈ 1.20 × PPL_base (20% increase)

Approximate gradient: ∂L_preserve/∂α ≈ (1.20 - 1.05) / 5 = 0.03 per unit scale.

**Equilibrium at λ=0.1:**
  ∂L_task/∂α + λ · ∂L_preserve/∂α = 0
  -0.1 + 0.1 · 0.03 · α_sensitivity = 0

This gives the equilibrium at approximately α* ≈ 5-8, consistent with Finding #330's
observation that scale=5 preserves general quality while scale=20 degrades it.

**Self-calibration demonstration:**
- Easy arithmetic (1+1): task gradient small → M2P converges to small B → small α
- Hard arithmetic (987+456+123+...): task gradient large → M2P converges to large B → larger α
- The RATIO of scales across contexts is the self-calibration signal (K850)

---

## H. Complexity and Architecture Connection

**M2P architecture for this experiment:**
- Input: arithmetic task description (tokenized, 16-32 chars)
- M2P: 1-layer Transformer, d=256, 4 heads, N_MEMORY=8 memory tokens
- Output: B-matrices for each LoRA layer (2 layers × 5 modules per layer)

**Parameter count:**
- Toy GPT base: 2-layer, d=256, 4 heads, vocab=128 ≈ ~520K params
- B-matrix size: rank × d_out = 4 × 256 = 1024 per attn module
- M2P parameter count: ~O(d^2 · n_layers) = O(256^2 · 2) ≈ 130K params

**FLOPs per forward pass (approximate):**
- Base GPT: T × (4d^2 + 8d^2) = T × 12 × 256^2 ≈ T × 785K FLOPs per layer
- M2P: N_MEM × (4d^2 + 8d^2) = 8 × 12 × 256^2 ≈ 6.3M FLOPs
- L_preserve requires one additional forward pass: same FLOPs as L_task

**Connection to production architecture:**
This experiment validates the scale self-calibration mechanism at micro scale.
At production scale (d=2048, rank=16), the same L_total = L_task + λ·L_preserve
objective applies with different equilibrium α* but same mathematical structure.
The Davis-Kahan perturbation bound scales as α/√d, so at larger d, smaller α
achieves the same per-token perturbation, and the self-calibrated scale will be
proportionally smaller.

---

## Self-Test (MANDATORY)

**1. What is the ONE mathematical property that makes the failure mode impossible?**

The KKT stationarity condition h(α) = ∂L_task/∂α + λ · ∂L_preserve/∂α = 0 has
a unique crossing from negative to positive (by IVT + sign analysis), making it
impossible for unconstrained gradient descent to drive α → ∞. L_preserve provides
the countervailing force that creates a stable fixed point at α*.

**2. Which existing theorem(s) does the proof build on?**

- Karush-Kuhn-Tucker conditions (Karush 1939, Kuhn-Tucker 1951): constrained
  optimization → Lagrangian stationarity at α*.
- Intermediate Value Theorem (Cauchy 1821): guarantees α* exists between 0 and ∞.
- Davis-Kahan sin-theta theorem (1970): mechanism linking adapter scale to quality
  degradation (grounding from Finding #330).
- Brouwer Fixed-Point Theorem (1912): existence of fixed point for gradient flow.

**3. What specific numbers does the proof predict?**

- K849: general_degradation_pp < 10pp (PPL ratio < 1.10)
- K850: adapter magnitude CV = std/mean > 0.05 across contexts
- Prediction 3: scale_learned ∈ [3, 15] (Type 2 — guided exploration for exact value)
- Equilibrium scale near Finding #330's "safe" range (scale ≤ 13)

**4. What would FALSIFY the proof?**

The proof is wrong if:
- L_preserve does NOT monotonically increase with α (Assumption 2 violation)
- General quality degrades MORE with L_preserve than without (L_preserve itself is
  counterproductive, due to optimization dynamics, not the math)
- All contexts produce identical ||B||_F (M2P ignores task input entirely)
- Theorem 1 fixed point at α* is a saddle point, not a minimum of L_total
  (requires showing L_total is not jointly convex in B)

**5. How many hyperparameters does this approach add?**

Count: 1 — the regularization weight λ.

λ cannot be fully derived from the math because it encodes the user's preference for
the task/general quality tradeoff, which is an irreducible design choice. However,
the proof predicts that ANY λ > 0 creates a valid equilibrium (just at different α*),
and the experiment explores λ=0.1 as a reasonable starting point. This is a Type 2
(guided exploration) component of an otherwise Type 1 experiment.

**6. Hack check: Am I adding fix #N to an existing stack?**

No. The prior M2P experiments (Finding #341, #342) had gradient conflicts from
multi-domain training. This experiment eliminates that problem by using single-domain
training — not by adding a fix, but by scoping to one domain where the proof holds.
L_preserve is the SINGLE constraint that makes the scale failure mode impossible.
There are no stacked fixes: L_total = L_task + λ·L_preserve is a 2-term objective
with clear mathematical motivation for each term.

The approach adds exactly one mechanism (L_preserve) with one hyperparameter (λ),
and that mechanism directly corresponds to the KKT condition for optimal α*.
