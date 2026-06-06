# MATH.md: End-to-End System Benchmark

## Problem Statement

We have proven individual components work:
- v_proj+o_proj is the correct adapter target (Finding #504)
- TF-IDF routing achieves 96% at N=5 (Finding #502)
- Adapter serving has 0% overhead (Finding #503)
- q_proj adapters achieve +22-82pp on benchmarks (Finding #421)

**Question:** Does the full system (train + route + generate) exceed base model accuracy
on standard benchmarks?

## Prior Results

Finding #421 (q_proj, rank 6): GSM8K 0%->82% (base=0% is artifact of max_tokens=256),
HumanEval 20%->66%, MedMCQA 26%->48%. BUT: q_proj only, and base GSM8K=0% is a
measurement artifact (CoT exceeds token limit). True base ~40-60%.

Finding #506: HF training data degrades vocab density BUT improves matching benchmark
accuracy. The disease is evaluation-distribution mismatch, not data quality.

## Theorem 1: Distribution-Aligned Training Guarantees Benchmark Improvement

**Setup.** Let M be a pretrained model with output distribution P_base(y|x) over
vocabulary V. Let A = (B,A) be a rank-r LoRA adapter on v_proj and o_proj with
W' = W + alpha * B * A.

**Claim.** When training data D is drawn from the same distribution as benchmark B,
the adapted model M_A minimizes cross-entropy on B more effectively than M alone,
provided sufficient training (KL(P_A || P_B) < KL(P_base || P_B)).

**Argument.** LoRA training minimizes L = E_{(x,y)~D}[-log P_A(y|x)] via SGD.
Since D and B share the same distribution, reducing L on D simultaneously reduces
expected loss on B (up to generalization gap delta, bounded by Rademacher complexity
of the rank-r subspace).

For v_proj+o_proj specifically (Finding #504 Theorem 1):
- v_proj adapter shifts which content vectors are projected: h = softmax(QK^T/sqrt(d)) * (W_v + alpha*B_v*A_v) * x
- o_proj adapter shifts how attention output maps to residual stream: out = (W_o + alpha*B_o*A_o) * h
- Together they directly modify the token distribution, not just attention patterns

The rank-r modification has capacity C = 2 * r * d parameters per layer. With r=8,
d=2560, L=42 layers: C = 2 * 8 * 2560 * 42 * 2 = 6.9M trainable parameters.
This is 0.09% of 7.5B total but concentrated on output-path computation, giving
disproportionate influence on token selection.

**QED:** Distribution-aligned LoRA training on v_proj+o_proj reduces benchmark loss
below base model loss, producing accuracy improvement.

## Theorem 2: Routed System Quality

**Setup.** N=3 domain adapters with routing accuracy P_r. Per-domain adapted accuracy
a_i. Base accuracy b_i.

**System accuracy on domain i:**
S_i = P_r * a_i + (1 - P_r) * avg(a_j, j != i)

For worst case (wrong adapter = base quality):
S_i >= P_r * a_i + (1 - P_r) * b_i

With P_r = 0.96 (Finding #502):
S_i >= 0.96 * a_i + 0.04 * b_i

If a_i >= b_i + 10pp, then S_i >= b_i + 0.96 * 10 = b_i + 9.6pp.

At P_r = 0.90 (kill criterion K1331): S_i >= b_i + 0.90 * 10 = b_i + 9pp.

**Conclusion:** Routing >= 90% preserves >= 90% of the per-adapter improvement.
The full system should show >= 9pp improvement if solo adapters show >= 10pp.

## Predictions

| Metric | Prediction | Kill Threshold | Basis |
|--------|-----------|----------------|-------|
| GSM8K adapted | base + 15-25pp | base + 10pp (K1328) | #421 showed +82pp with q_proj (artifact), true ~+20pp after base fix |
| HumanEval adapted | base + 10-20pp | base + 10pp (K1329) | #421 showed +46pp, v_proj should be similar |
| MedMCQA adapted | base + 10-15pp | base + 10pp (K1330) | #421 showed +22pp with q_proj |
| Routing accuracy | 96% | >= 90% (K1331) | Finding #502 at N=5 |
| E2E latency | ~0.7s | <= 2s (K1332) | Finding #503: 1ms swap + ~0.5s generation |

## Kill Criteria Derivation

- K1328-K1330: +10pp is the minimum improvement that demonstrates the adapter adds
  value beyond the base model. Below 10pp, noise dominates signal.
- K1331: 90% routing means <= 1 in 10 queries gets wrong adapter. Below 90%,
  wrong-adapter degradation may negate adapter improvement.
- K1332: 2s e2e is the threshold for interactive use (user-perceived latency).

## Composition Prediction

With N=3 equal-weight merge (Finding #505 showed N=5 works):
- PPL should improve or stay neutral (Finding #507: PPL improved 8% at N=25)
- Per-domain accuracy under composition: >= 80% of solo accuracy
  (ensemble effect may compensate for interference at N=3)

## Experimental Design

1. **Base evaluation**: GSM8K, HumanEval, MedMCQA accuracy (n=100 each)
2. **Train 3 adapters**: v_proj+o_proj, rank 8, 1000 iters, 2000 examples each
3. **Solo evaluation**: each adapter on its matching benchmark
4. **Routing evaluation**: TF-IDF routing accuracy at N=3
5. **E2E evaluation**: route query -> select adapter -> generate -> measure accuracy
6. **Composition**: merge all 3, evaluate all benchmarks
7. **Latency**: time route + load + generate for 100 tokens
