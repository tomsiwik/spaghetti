# LEARNINGS.md — P4.A0: 5-Domain TF-IDF Ridge Routing

**Finding #474 — SUPPORTED**
**Experiment:** 5-domain TF-IDF Ridge routing on real corpora (medical/code/math/legal/finance)

---

## What We Learned

### 1. TF-IDF Ridge routing scales to 5 real domains without modification
- 97.3% weighted accuracy, 0.247ms p99 latency, 76ms training time
- Finding #458 (98.8% on 25 synthetic MMLU categories) generalizes to real data
- No hierarchical router or learned query embedding required at N=5

### 2. Math × Legal is the hard pair (cos=0.237)
- Both domains use formal argumentation language as their primary register (not noise)
- "therefore", "given that", "it follows", "proof", "stipulated" appear in both
- All 13 routing errors concentrate around this pair and its spillover into medical
- **Corrected model:** For any N>5 system including math + legal, expect cosine ≈ 0.2–0.25 and set thresholds accordingly

### 3. Training time is not a bottleneck
- 76ms training (790× under the 60s budget)
- TF-IDF feature extraction is the bottleneck, not Ridge regression
- P4.A1 (new domain in <10 min) should be achievable: the routing component costs <0.1s

### 4. Theorem 2 calibration lesson
- Vocabulary divergence formula E[cos] ≈ |V_shared|/sqrt(|V_i|×|V_j|) is structurally correct
- Single-pair calibration (medical_vs_code) transfers accurately to similarly specialized pairs
- Fails for pairs where shared vocabulary is primary register (math, legal) not incidental
- Fix for P4 scale-up: use 2 calibration pairs (specialized+formal) to set thresholds

### 5. Medical is the weakest domain (lowest precision: 93.2%)
- Incoming confusions: finance→medical (4), legal→medical (3), math→medical (3)
- Medical vocabulary overlaps with finance (clinical economics), legal (clinical documents), math (quantitative methods)
- Likely improves at N=1000+ training examples per domain

---

## What Still Needs Testing

- P4.A1: Domain adapter training speed — can we add a new domain in <10 minutes?
- P4.A2: Personal style integration with domain routing active
- N>5 domains: does 97.3% hold at N=10, N=25?

---

## Connection to Architecture

| Component | Status | Metric |
|-----------|--------|--------|
| Domain routing (N=5) | ✅ Production-ready | 97.3% weighted, 0.247ms |
| Domain adapter quality | → P4.A1 | pending |
| Personal style E2E | ✅ Verified (P3.D0) | 93.3% style, 0pp degradation |
| Full pipeline | → P4.A3+ | pending |

The router is the fastest component in the pipeline (0.247ms). Adapter inference will dominate latency in production.
