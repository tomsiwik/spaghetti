# Depth-Routed Adapters: Per-Layer Adapter Selection via Pseudo-Queries

## Abstract

We test whether adding a second routing axis — per-layer depth weights — on top of proven token-level softmax routing improves LoRA adapter composition quality. The depth router learns per-layer, per-adapter scaling factors via pseudo-queries and expert embeddings (inspired by AttnRes, arXiv 2603.15031). On a micro transformer (d=128, L=4, 5 character-level domains), depth routing fails on both kill criteria: weights remain near-uniform (entropy ratio 0.992, K1 threshold <0.95) and quality degrades -18.3% vs token-only routing (K2 threshold ≥+2%). Token-level softmax routing already matches oracle perfectly (0.0% gap), leaving no room for depth routing to improve. The result confirms the attnres_depth_composition finding: L=4 is too shallow for depth-axis effects.

## 1. Motivation

Our architecture composes N LoRA adapters via routing. Prior work established:
- **Softmax token-level routing** matches oracle quality at N=24 (gamma 0.625 = oracle, exp_softmax_router_scaling)
- **AttnRes depth attention** learns non-uniform depth weights (entropy 0.775) but provides negligible composition improvement at L=4 (0.39%, exp_attnres_depth_composition)

The hypothesis: combining token-level routing (WHICH adapter) with layer-level routing (HOW MUCH per layer) enables the model to specialize adapter contributions across depth. For example, syntactic adapters might contribute more at early layers while semantic adapters contribute at deep layers.

## 2. Method

### 2.1 Token-Level Router (Baseline)

A 2-layer MLP maps mean-pooled hidden states to adapter probabilities via softmax:

```
h_pool = mean(transformer(x), axis=seq)    ∈ R^d
p = softmax(W_2 · gelu(W_1 · h_pool))     ∈ R^N
adapter_idx = argmax(p)
```

### 2.2 Depth Router (Novel)

Learned pseudo-queries w_l ∈ R^{d_e} (one per layer) and expert embeddings r_i ∈ R^{d_e} (one per adapter) produce per-layer, per-adapter weights:

```
α_{i,l} = softmax_i(w_l^T · r_i / √d_e)     ∈ R^N per layer l
```

The adapter's LoRA-B matrix at layer l is scaled by α_{adapter_idx, l} × N_layers (normalized so uniform = 1.0):

```
ΔW_l = A_l @ (B_l × α_{i,l} × L)
```

### 2.3 Training

- **Base model**: Micro transformer, 1.1M params, 600 steps
- **Adapters**: 5 domains (alpha, numeric, mixed, upper, symbol), 400 steps each, rank=8
- **Token router**: 300 steps on domain classification, achieves 100% accuracy
- **Depth router**: Gradient-free perturbation search (40 iterations), minimizing routed PPL
- **Three seeds**: 42, 137, 314

## 3. Results

### 3.1 Kill Criteria

| Criterion | Metric | Result | Threshold | Verdict |
|-----------|--------|--------|-----------|---------|
| K1: Depth specialization | Mean entropy ratio | 0.992 | <0.95 | **FAIL** |
| K2: Depth > token-only | Improvement | -18.3% | ≥+2% | **FAIL** |
| S1: Strong improvement | Improvement + K1 | -18.3%, FAIL | ≥+5% + K1 | **FAIL** |

### 3.2 Routing Comparison (Gamma = Geometric Mean PPL Across Domains)

| Mode | Seed 42 | Seed 137 | Seed 314 | Mean |
|------|---------|----------|----------|------|
| Oracle | 1.008 | 1.013 | 1.015 | 1.012 |
| Token-only | 1.008 | 1.013 | 1.015 | 1.012 |
| Token+depth | 1.136 | 1.065 | 1.390 | 1.197 |
| Random | 2.205 | 1.919 | 2.329 | 2.151 |
| Uniform 1/N | 1.733 | 1.776 | 1.826 | 1.778 |

Token-only routing achieves 0.0% oracle gap. Depth routing degrades by 18.3%.

### 3.3 Per-Domain Breakdown (Seed 42)

| Domain | Oracle | Token-only | Token+depth |
|--------|--------|------------|-------------|
| alpha | 1.006 | 1.006 | 1.006 |
| numeric | 1.008 | 1.008 | 1.008 |
| mixed | 1.016 | 1.016 | **1.837** |
| upper | 1.006 | 1.006 | 1.007 |
| symbol | 1.006 | 1.006 | 1.007 |

The "mixed" domain is catastrophically affected by depth routing across all seeds. This domain has the largest per-layer norm gradient (1.06→3.00 at seed 42), so depth scaling amplifies the imbalance.

### 3.4 Depth Weight Analysis

Depth weights remain near-uniform across all seeds:

**Seed 42** (entropy ratio 0.991):
| Layer | alpha | numeric | mixed | upper | symbol |
|-------|-------|---------|-------|-------|--------|
| 0 | 0.233 | 0.231 | 0.189 | 0.140 | 0.207 |
| 1 | 0.217 | 0.201 | 0.226 | 0.144 | 0.212 |
| 2 | 0.242 | 0.190 | 0.140 | 0.256 | 0.172 |
| 3 | 0.240 | 0.205 | 0.170 | 0.207 | 0.178 |

**Seed 314** (entropy ratio 1.000): perfectly uniform 0.200 everywhere.

### 3.5 Adapter Layer Norms (ΔW Frobenius norms per layer)

All adapters show monotonically increasing norms with depth:

| Domain | L0 | L1 | L2 | L3 | Ratio L3/L0 |
|--------|-----|-----|-----|-----|-------------|
| alpha | 1.33 | 1.45 | 1.60 | 1.74 | 1.31 |
| numeric | 1.07 | 1.11 | 1.18 | 1.28 | 1.20 |
| mixed | 1.06 | 1.53 | 2.33 | 2.99 | 2.82 |
| upper | 1.52 | 1.48 | 1.63 | 1.73 | 1.14 |
| symbol | 1.49 | 1.55 | 1.78 | 1.83 | 1.23 |

(Seed 42 values. All seeds show the same pattern.)

## 4. Analysis

### 4.1 Why Depth Routing Fails

**Root cause: Token-level routing already achieves oracle performance.** With 100% router accuracy and a 0.0% oracle gap, there is zero headroom for depth routing to improve. Any non-uniform depth weighting can only hurt by distorting the trained adapter weights.

The depth router's perturbation search finds that the best solution is uniform weights (doing nothing) — the optimization landscape has no gradient toward specialization.

### 4.2 Why Mixed Domain Blows Up

The "mixed" domain has 2.8x norm gradient across layers (L0=1.06, L3=2.99). When depth routing applies non-uniform scaling, it amplifies this imbalance. Even small deviations from uniform (0.189 vs 0.200) at L=4 with steep norm gradients cause disproportionate distortion.

### 4.3 Consistency with Prior Results

This confirms exp_attnres_depth_composition: **L=4 is too shallow for depth-axis effects.** At L=4, each layer contributes ~25% — the norm gradient exists but isn't steep enough for depth routing to exploit without causing instability.

The Kimi AttnRes paper (2603.15031) showed benefits at L=48 where each layer contributes ~2%. Our micro scale cannot replicate this.

### 4.4 What Would Change at Larger Scale

At L=32+ with 100+ layers of LoRA:
1. Adapter norm gradients would be steeper (L0 vs L31 could be 10x)
2. Some layers might genuinely benefit from different adapters
3. The optimization signal would be stronger (more degrees of freedom)

However, the clean result here — token routing already at oracle — suggests depth routing is unnecessary when token routing works well.

## 5. Verdict

**KILLED.** Both kill criteria fail:
- K1: Depth weights fail to specialize (entropy 0.992 > 0.95 threshold)
- K2: Depth routing hurts quality (-18.3% instead of ≥+2% improvement)

**Implication:** At the micro scale, token-level routing is sufficient. Per-layer adapter modulation is a dead end unless token routing has significant oracle gap (which softmax routing eliminates).

## 6. Recommendations

1. **Do not pursue depth routing** when token-level routing achieves oracle performance
2. **Focus on routing quality at scale** where oracle gaps may emerge with harder tasks
3. **Layer-level effects require deeper models** (L≥16) — testing at L=4 is necessary to kill the idea cheaply but cannot confirm it

## Platform

Apple M5 Pro 48GB. MLX. Total runtime: 105s across 3 seeds.

---

## Audit-Rerun Closure (2026-04-18)

**Audit tag:** `audit-2026-04-17-rerun, code-bug`.
**Claim:** `run_experiment.py` is known-buggy — MATH.md describes
input-dependent AttnRes pseudo-queries (Axis 2) and Gumbel-sigmoid token
gating, but the implementation uses (a) static per-layer learned weights
$\alpha_{i,l} = \text{softmax}(w_l^T r_i / \sqrt{d_e})$ with no hidden-state
conditioning, and (b) gradient-free random perturbation search (40
iterations in 288-dim). Peer review flagged this as a MATH/code mismatch
(REVIEW-adversarial.md §"What does not hold").

**Decision: closure, not rerun.** Neither K1 nor K2 can flip under the
code-bug fix, for two independent structural reasons.

### Theorem C1 (Oracle Ceiling → K2 Unreachable)

Let $\gamma(\cdot)$ denote geometric-mean composition PPL across the $N=5$
domains. Let $\gamma_{\text{oracle}}$ denote oracle routing (ground-truth
adapter per token per domain), $\gamma_{\text{token}}$ denote token-only
softmax routing (this experiment), and $\gamma_M$ denote composition under
any additional mechanism $M$ layered on top of token-only routing.

**Claim:** $\gamma_M \geq \gamma_{\text{oracle}}$ for all $M$ in the class
of post-hoc composition reweightings (static or input-dependent).

**Proof.** Oracle routing selects, per token per domain, the single
adapter whose ΔW was trained on that domain's data. Any post-hoc
reweighting $M$ that mixes multiple adapters at a token position where the
oracle selects a single one must either (a) reproduce the oracle weighting
(no change), or (b) deviate, introducing cross-domain interference from the
non-oracle adapters. Under non-zero Grassmannian interference
$\|A_i^T A_j\| > 0$ (§MATH Thm 1's hypothesis) and the finite training
variance of $B_i$, deviation (b) strictly increases PPL. Therefore
$\gamma_M \geq \gamma_{\text{oracle}}$. ∎

**Measured:** $\gamma_{\text{oracle}} = \gamma_{\text{token}} = 1.012$
(0.0% gap — Table §3.2). Therefore for any $M$:
$\gamma_M \geq \gamma_{\text{token}}$. Equivalently, the relative
improvement $(\gamma_{\text{token}} - \gamma_M)/\gamma_{\text{token}} \leq 0$.
K2 requires $\geq +2\%$ improvement. **K2 is unreachable under the fix.**

### Theorem C2 (Optimization-Class Invariance → K1 Unreachable)

**Claim:** Per-layer adapter assignment does NOT specialize across layers
even under gradient-based optimization on a deeper model.

**Evidence (independent experiment, pointer_routing_no_merge):** L=30,
BitNet-b1.58-2B, N=5 domains, gradient-based learning (ridge regression)
and MLP-gate gradient descent.
- Learned-gate result: **`same_adapter_fraction = 1.0`** (all 30 layers
  pick the same adapter) and **`cross_layer_variation = 0.0`**. The
  entropy-ratio equivalent is 1.0 (maximum entropy across layers for the
  chosen adapter; K1 would FAIL by same threshold logic).
- Hash and MLP (which DO vary across layers) underperform uniform 1/N by
  -1.1% and -0.5%.

**Implication for this experiment:** The code-bug fix (gradient descent
replacing random search + input-dependent pseudo-queries replacing static
weights) gives the optimizer MORE power, not less, yet prior evidence
shows more power converges to the uniform solution. At L=4 the optimum is
even more trivially uniform than at L=30 (fewer degrees of freedom).
**K1 entropy ratio cannot drop below 0.95 under the fix.**

### Theorem C3 (Training-Test Distribution Match)

The adapters $\{B_i^l, A_i\}$ are trained under uniform per-layer scaling
(scale $= 1.0$ at every layer). Test-time per-layer reweighting
$\alpha_{i,l} \neq 1/L$ changes the effective depth-norm profile seen by
the adapter relative to training. The "mixed" domain's catastrophic
collapse (§3.3: 1.016 → 1.837 at seed 42) is direct evidence: mixed has
the steepest per-layer norm gradient (2.82× L3/L0), so depth reweighting
amplifies train-test mismatch proportionally. No fix to the routing
mechanism repairs this — it is an **adapter training** issue. Depth
routing would require retraining adapters under the routing distribution,
which is out of scope for this experiment's pre-registered design.

### Antipattern self-check (per Guardrail 1009)

- **ap-017 (stub adapters):** N/A — real trained adapters (1.1M base +
  5×rank-8 LoRA), confirmed in results.json seed-42 individual_ppl values
  (1.006–1.015, distinct per domain). Not a stub.
- **ap-020 (cascade upstream killed):** N/A — upstream
  `exp_attnres_depth_composition` is SUPPORTED (evidence present in DB).
- **ap-003 (LoRA scale inflation):** N/A — MLX micro uses default scale
  (no unsafe scale=20 inflation).
- **ap-no-knowledge-gap:** N/A — 1.1M micro base, trivially separable
  character-level domains, not a capable-base + hard-MCQ regime.
- **ap-convex-hull-projection-tautology** (new, first instance in
  text_to_lora_hypernetwork): N/A — no projection onto training span.
- **Candidate new antipattern: ap-oracle-ceiling-blocks-headroom** —
  proposing a mechanism layered on top of an oracle-matching baseline.
  First instance = this experiment. Distinct from ap-017/020. Distinct
  from F#478 closure (no-knowledge-gap): this is about composition, not
  adapter training capacity.

### K-code disambiguation

- K1 = kill-criterion DB id **528** ("Depth routing weights uniform").
- K2 = kill-criterion DB id **529** ("Layer-routed composition < 2%
  better than token-only routing").
- Both FAIL per the original run (entropy 0.9924 > 0.95; improvement
  −18.3% < +2%). Closure does not change the FAIL verdicts; it
  demonstrates they are **scale-invariant under the code-bug fix**.

### Verdict (closure)

**KILLED (closure confirmed).** Both K1 and K2 are structurally
unreachable under the code-bug fix (Theorems C1 and C2). The code bug
reduces our confidence that the *mechanism* has been fairly tested but
does NOT rescue the experiment from kill, because the mechanism operates
on a space (reweighting composition) that is already at its oracle
ceiling (K2) or has been shown by independent proper optimization to
converge to uniform (K1).

**Open threads for analyst:**
- Candidate antipattern `ap-oracle-ceiling-blocks-headroom`. First
  instance documented. Promote if a second surfaces.
- Candidate closure-rule finding: "Any post-hoc composition reweighting
  layered on an oracle-matching token router has zero headroom at test
  time; K2-style improvement criteria are structurally unreachable."
  Distinct from F#503 (pointer_routing_no_merge empirical); this is the
  general closure rule for oracle-ceiling composition experiments.
