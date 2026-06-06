# P0: TT-LoRA N=10 Scaling — Mathematical Framework

## Type: Frontier Extension
- **Proven:** N=3 TT-LoRA E2E works (Finding #508 + e2e benchmark: 93% GSM8K retention, 87% HumanEval, 98.3% routing)
- **Proven:** N=25 synthetic routing works (Finding #406: 99.96% routing accuracy)
- **Proven:** N=25 standard LoRA real routing works (Finding #502: 84.2% TF-IDF accuracy)
- **Gap:** No test with N>=10 **real TT-LoRA adapters** on **real benchmarks** simultaneously

## Theorem 1: TF-IDF Routing Scaling (Hoeffding + Union Bound)

**Setup:** N domains with vocabulary distributions P_1,...,P_N in TF-IDF feature space R^d.
Let mu_i = E[phi(x) | x in domain_i] be the TF-IDF centroid for domain i,
and delta_min = min_{i != j} ||mu_i - mu_j||_2 be the minimum pairwise centroid distance.

**Claim:** Ridge classifier routing accuracy satisfies:
  acc >= 1 - (N-1) * exp(-n * delta_min^2 / 8)
where n is the number of TF-IDF features (d=5000).

**Proof sketch:** By Hoeffding's concentration inequality, the empirical TF-IDF vector
of a test sample concentrates around its domain centroid. The probability of
misclassification to domain j given true domain i is bounded by
exp(-n * ||mu_i - mu_j||^2 / 8). Union bound over N-1 alternative domains gives
the per-sample error bound. For 10 semantically distinct domains (medicine vs law
vs code vs...), delta_min remains bounded away from zero because domain-specific
terminology creates well-separated clusters. QED.

**Prediction 1:** Routing accuracy >= 85% at N=10. The 98.3% at N=3 degrades primarily
from vocabulary overlap between semantically adjacent domains (science/engineering,
philosophy/psychology). Finding #502 achieved 84.2% with N=25 standard LoRA — N=10
with 10 semantically distinct domains should exceed this.

**Prediction 2:** No single domain routing accuracy < 70%. Even the most overlapping
domain pair (e.g., science/engineering) retains distinguishing vocabulary.

## Theorem 2: Independent Adapter Quality Invariance

**Claim:** Per-domain TT-LoRA quality is invariant to total number of adapters N,
since each adapter is independently trained, saved, and loaded at inference.

**Proof:** The inference computation for domain i is:
  y = f(x; theta_base + alpha * DW_i)
where DW_i is reconstructed from TT cores of adapter i.
This is independent of the existence of adapters j != i.
N only affects the routing decision (which adapter to load), not the adapter's quality.

**Prediction 3:** Math/code/medical adapters trained identically to e2e benchmark
should achieve within +/-5pp of:
- GSM8K: 68% (e2e benchmark result)
- HumanEval: 55% (e2e benchmark result)
- MedMCQA: 21% (e2e benchmark result)

## Theorem 3: Linear Footprint Scaling

**Claim:** Total adapter footprint = N * (params_per_adapter * bytes_per_param + overhead).

Each TT-LoRA rank-6 adapter on Gemma 4 E4B (v_proj + o_proj):
- Parameters: 135,492
- Float16 size: 135,492 * 2 = 270,984 bytes
- Safetensors overhead: ~62 KB (metadata + alignment)
- Total per adapter: ~333 KB

**Prediction 4:** 10 * 333 KB = 3.33 MB total.
K1435 (< 2 MB) will FAIL at float16.
To pass: int8 quantization -> 10 * 198 KB = 1.98 MB (borderline).

## Kill Criteria Predictions

| Kill   | Criterion                          | Prediction     | Confidence |
|--------|------------------------------------|----------------|------------|
| K1433  | Routing >= 85% at N=10             | 88-92%         | Medium     |
| K1434  | Mean retention >= 75% top-5        | ~80% (math 93% + code 87% help; medical 42% hurts) | Low-Medium |
| K1435  | Total size < 2 MB                  | FAIL (3.33 MB) | High       |
| K1436  | No domain routing < 70%            | PASS (~80% min) | Medium     |

## Domains

| # | Domain      | Training Source                    | Eval Benchmark |
|---|-------------|------------------------------------|----------------|
| 1 | math        | GSM8K train (NTP)                  | GSM8K test     |
| 2 | code        | CodeAlpaca (NTP)                   | HumanEval      |
| 3 | medical     | MedMCQA train (NTP)                | MedMCQA val    |
| 4 | science     | MMLU: astronomy, biology, chemistry, physics | MMLU MCQ |
| 5 | legal       | MMLU: professional_law, jurisprudence, international_law | MMLU MCQ |
| 6 | finance     | MMLU: accounting, econometrics, marketing | MMLU MCQ |
| 7 | history     | MMLU: us_history, world_history, prehistory | MMLU MCQ |
| 8 | psychology  | MMLU: professional_psychology, hs_psychology | MMLU MCQ |
| 9 | philosophy  | MMLU: philosophy, formal_logic, logical_fallacies | MMLU MCQ |
| 10| engineering | MMLU: electrical_eng, comp_security, college_cs | MMLU MCQ |

MMLU domains use MCQ-format training (Finding #522: MCQ loss works with TT-LoRA).
The 3 original domains use NTP (matching e2e benchmark).
