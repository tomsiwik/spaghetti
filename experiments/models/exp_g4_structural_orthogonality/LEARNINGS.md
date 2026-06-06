# LEARNINGS — exp_g4_structural_orthogonality

## Core Finding
Partition-QR construction of N=25 rank-6 LoRA-A matrices at Gemma 4 native hidden
dims (d=2816, d=5376) yields cross-adapter `max|cos|` of **2.74·10⁻⁹** and
**1.92·10⁻⁹** in float32 — **9 orders below** kill threshold `100·√(r/d)` and
**7 orders below** the random-subspace baseline `√(r/d)`. Float64 reference
lands at ≈2·u_f64 (3·10⁻¹⁶) as Theorem 2 predicts. Finding #562 posted.

## Why
Partition QR assembles the N·r adapter-A columns as contiguous blocks of a
single tall QR factor. `Q^T Q = I` is algebraic zero in exact arithmetic, so
cross-block inner products are zero by construction (Theorem 1). In float32,
LAPACK Householder QR is backward-stable (Higham §19.3), giving an
`O(√(Nr)·u)` error bound that at r=6, N=25, u≈1.2·10⁻⁷ predicts ≤1·10⁻⁵.
Measured value sits 4 orders below even that bound. The construction is a
**pre-training** guarantee — distinct from Finding #3 (trained adapters,
d=896 Qwen proxy) and Finding #42 (plateau at convergence). Triangle
inequality extends it to `‖ΣBᵢAᵢ‖ ≤ N·‖B‖‖A‖` for composition.

## Implications for Next Experiment
1. **Drop the Qwen-proxy caveat.** Any downstream experiment that cites
   "Grassmannian init → structural orthogonality" can now cite this result at
   Gemma 4 native dims directly. Update PLAN.md Part 2 next time the
   Grassmannian-init claim is touched.
2. **Composition at Gemma 4 dims is safe by construction at init.** When the
   adapter-rebuild unblock (P11.ADAPTER-REBUILD) lands, the 5 retrained
   domain adapters should be initialized with partition QR (not random) so
   this guarantee carries into training. Finding #42 then asserts it
   plateaus rather than degrading.
3. **Complements sibling** `exp_p1_t0_grassmannian_gemma4` (rank=16, Frobenius
   metric, K=1e-6). This result covers rank=6 (PoLAR production rank) with
   the column-cosine metric. Together the two bracket the rank regime.
4. **No antipattern follow-up.** Pure algebra, no model, no cascade — the
   kind of experiment the backlog was waiting for. Antipattern-017 count
   remains at 7; this does not touch adapter artifacts.
5. **Next high-value P=1 claim:** same pattern (pure-algebra verification)
   could confirm bf16 behavior if bf16 ever becomes the serving precision.
   Not currently blocking anything — kill threshold is so loose that bf16
   passes trivially per PAPER Assumptions. Defer.
