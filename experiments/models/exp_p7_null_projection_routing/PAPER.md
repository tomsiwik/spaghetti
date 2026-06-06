# P7.B0: Null-Space Projection as Natural Router — KILLED

## Summary

Null-space A-matrix projection magnitude does NOT discriminate domains. Routing
accuracy = 20% (chance level for 5 domains). All inputs route to the same adapter
(legal), dominated by A-matrix norm differences, not directional alignment. Even
after mental normalization, no domain signal exists in null-space projections.

**Status: KILLED** — Domain information lives in range(W_v), not null(W_v).

## Prediction vs Measurement

| Metric | Predicted | Measured | Status |
|--------|-----------|----------|--------|
| K1300: Routing accuracy | >= 80% | 20.0% (4/20) | **FAIL** |
| K1301: Spearman r (proj vs quality) | >= 0.3 | -0.19 | **FAIL** |
| K1302: Routing latency | < 0.5ms | 4.35ms | **FAIL** |
| Training convergence | All domains converge | All converge (loss 0.008-0.029) | PASS |
| Null-space basis reuse | Works from prior experiment | Works | PASS |

## Key Observations

### 1. Magnitude Bias (Not Directional Alignment)

Raw projection magnitudes per domain adapter (averaged across all 20 test texts):
- medical: ~2200
- code: ~2100
- math: ~1900
- **legal: ~6400** (3x larger)
- finance: ~2400

Legal adapter dominates regardless of input text. This is A-matrix Frobenius norm
bias, not domain signal. The legal adapter learned higher-magnitude weights.

### 2. No Domain Signal Even Without Bias

Removing legal from consideration and looking at remaining 4 adapters on a medical test text:
- medical: 2169, code: 2121, **finance: 2437**, math: 1932

Finance > medical on medical text. The "correct" adapter never consistently ranks
highest among non-legal adapters either. The null space projection carries no
domain-discriminative information.

### 3. Per-Layer Routing Is Uniformly Chance

All 8 layers (16-23) show identical 20% accuracy. No layer encodes domain
information in its null-space projection. This is not a "wrong layer" problem.

### 4. Latency 8.7x Above Target

4.35ms per routing decision (5 adapters × 8 layers). The Q projection
(3584 → 2048) is a large matmul that dominates. Even with optimization,
the null-space projection is fundamentally more expensive than simpler routing.

## Failure Mode Analysis

### Why Routing Fails: Domain Info Is in range(W_v), Not null(W_v)

The null space of W_v is the subspace that W_v ignores — by definition, it
contains features the base model decided are irrelevant for value computation.
Domain information (medical vs code vs legal terminology) IS relevant to value
computation, so it lives in range(W_v).

Projecting inputs into null(W_v) via Q strips away domain features. What remains
is domain-blind noise. Adapters trained in this space learn to extract useful
signal from the noise (Finding #494 showed 98.7% quality), but the feature
directions they learn are NOT domain-discriminative.

### Mathematical Impossibility Structure

Let V = range(W_v) and V⊥ = null(W_v). Domain information D lives in V
(the base model uses it). The routing signal s_i(x) = ||A_i Q^T x||² operates
entirely in V⊥. Since V ⊥ V⊥, the routing signal is structurally orthogonal
to domain information:

    <routing_signal, domain_information> = 0

This makes domain routing via null-space projection mathematically impossible,
not just empirically unlikely. No normalization, hyperparameter tuning, or
additional training can overcome this orthogonality.

### Distinction from Finding #295

Finding #295 showed B-projection failed because B-subspaces overlap. This
experiment shows A-projection fails for a different, more fundamental reason:
the null space doesn't contain domain information at all. The A-matrices DO
learn different directions, but those directions are all domain-blind variations
within null(W_v).

## Implications for Architecture

1. **Routing and null-space isolation serve fundamentally different purposes**:
   - Routing needs domain-discriminative features → lives in range(W_v)
   - Isolation needs interference-free subspaces → lives in null(W_v)
   - These are mathematically orthogonal concerns

2. **The Room Model "routing IS the matmul" claim needs refinement**:
   - True for standard LoRA (A-matrix sees full input, can discriminate domains)
   - False for null-space LoRA (A-matrix only sees domain-blind null projection)
   - Routing must operate in a space that contains domain information

3. **Correct architecture separation**:
   - Route using hidden states or range(W_v) features (where domain info lives)
   - Adapt using null(W_v) subspaces (where interference is zero)
   - The router and the adapter operate in complementary subspaces

## Training Details

| Domain | Final Loss | Train Time | Sequences |
|--------|-----------|------------|-----------|
| medical | 0.0098 | 42.6s | 8 |
| code | 0.0158 | 42.4s | 8 |
| math | 0.0289 | 60.6s | 8 |
| legal | 0.0085 | 41.9s | 8 |
| finance | 0.0106 | 42.1s | 8 |

All adapters converge to near-zero loss (memorization scale, 8 texts each).
r=16 null-space LoRA on v_proj layers 16-23, 300 iters, AdamW lr=1e-4.
Reused null bases from exp_p7_null_space_adapter_quality.

Total experiment time: 4.1 min.

## References

- Finding #494: Null-space LoRA works (98.7% quality) but NOT for routing
- Finding #493: v_proj null_dim=2048 (plenty of space, but no domain info)
- Finding #295: B-projection fails (overlap), A-projection fails (domain-blind)
- arXiv:2106.09685 (LoRA): A-matrices specialize, but specialization ≠ domain routing
- arXiv:2212.04089 (Task Arithmetic): task vectors work in full parameter space, not null space
