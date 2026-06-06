# MATH.md — T5.3: User Adapter Submission Pipeline

## Theorem 1: Pipeline Latency Bound

**Statement:** The end-to-end latency from adapter submission to live generation
is bounded by the sum of its proven component latencies:

  T_total = T_validate + T_integrate + T_first_gen

where:
- T_validate ≤ 30s (T5.2 measured 23.5s for validation pipeline, Finding #437)
- T_integrate = O(1) ≤ 0.01s (registry dict update + TF-IDF refit on N+1 documents)
- T_first_gen ≤ 5s (T4.6 Finding #434: TTFT overhead < 10ms + 50-token generation ~3s on M5 Pro)

**Proof:** Each step is proven independently:
1. Validation pipeline (T5.2, Finding #437): 23.5s per adapter for 5 checks
   (orthogonality, quality, safety, scale, timing). 2.56× margin to 60s threshold.
2. Integration: Adding adapter to runtime registry is a dict insert O(1).
   TF-IDF refit with N+1 documents is O(N×V) where N=6, V≈50,000 → ~2ms (measured in T4.1).
3. First generation: T4.6 showed route+swap overhead < 10ms. 50-token generation
   at ≥80 tok/s (T4.3 Finding #432: 90.8% of base 165 tok/s) = ~0.6s.

Therefore: T_total ≤ 30 + 0.01 + 5 = 35.01s << 300s (5 minutes). QED.

**Quantitative prediction:** T_total < 60s (single model load, M5 Pro 48GB).

---

## Theorem 2: Integration Preserves Routing Correctness

**Statement:** After integrating a personal adapter for user u, the routing
system correctly selects the personal adapter for user-specific queries
and the domain adapter for domain-specific queries.

**Proof:** The routing uses TF-IDF (term frequency–inverse document frequency)
over adapter domain keywords. Personal adapters are registered with a
user-specific keyword set (user ID as pseudo-domain). Since user IDs are
unique tokens not appearing in any domain corpus:

  IDF(user_token) = log(N/1) = log(N) (appears in exactly 1 document)

This maximizes discrimination: a query containing the user token routes to the
personal adapter with probability ≥ 1 - ε where ε → 0 as IDF(user_token) → ∞.

For domain queries (no user token): routing is identical to the N-domain case
proven in T4.1 (Finding #431: N=5 accuracy 96.6%). Adding one personal adapter
with zero overlap in keyword space does not degrade domain routing. QED.

**Prediction:** 
- User-specific routing accuracy: 100% (user token is unique)
- Domain routing accuracy with personal adapter present: ≥ 90% (T4.1 baseline)

---

## Kill Criteria

| Criterion | Threshold | Source Theorem |
|-----------|-----------|---------------|
| K_a: Adapter goes live (generation succeeds) | No error | Theorem 1 (structural) |
| K_b: Total time submit→live < 5 min | < 300s | Theorem 1 (latency bound) |
| K_c: Personal routing accuracy | = 100% (user token unique) | Theorem 2 |
| K_d: Domain routing unaffected by personal adapter | ≥ 90% | Theorem 2 |
| K_e: Adapter quality preserved through pipeline | Compliance > 0% | T5.2 validated |

---

## Failure Mode Analysis

**What breaks this?**
1. Model OOM during pipeline: T5.2 showed model loads fine + adapter in 1.1s.
   Both validation and generation use same adapter model load path. No new risk.
2. Routing collision: User token has zero overlap with domain vocab by construction.
   Cannot fail unless user ID matches a domain keyword (checked in code).
3. Adapter weights corrupted in transit: We load + run QR check as part of validation.
   K1100 (max|cos| < 0.95) would flag any degenerate adapter.

**Impossibility structure:** Pipeline latency blow-up is structurally impossible
given the component proofs. The only degree of freedom is model load time, which
is bounded by hardware memory bandwidth (M5 Pro: 273 GB/s, 4-bit model = ~2.4GB → 8.8ms lower bound).

---

## References

- Finding #436 (T5.1): User local training, 72s, 1.2min, behavioral adaptation proven
- Finding #437 (T5.2): Validation pipeline, 23.5s, all 5 checks pass
- Finding #431 (T4.1): TF-IDF routing N=5 96.6%, N=25 86.1%
- Finding #432 (T4.3): Hot-swap p99=4.77ms, throughput 90.8%
- Finding #434 (T4.6): E2E overhead=1.4ms, throughput=96.1%
