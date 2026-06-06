# LEARNINGS.md — exp_p1_p0_finance_routing_fix

## Core Finding
FiQA finance data *worsened* TF-IDF routing accuracy (91% → 87%) because quantitative finance queries ("calculate the P/E ratio") activate math vocabulary identical to GSM8K math queries. TF-IDF nearest-centroid routing cannot separate domains that share an operational vocabulary — this is a structural impossibility, not a data quality problem.

## Why
The Theorem 1 prediction assumed FiQA finance vocabulary has ~3% overlap with math corpus. Empirically, FiQA is a calculation benchmark: ~60% of queries are quantitative (rates, ratios, amortization). These queries use identical tokens to math (calculate, rate, ratio, value, formula), so the math–finance centroid cosine increased 0.277 → 0.353. When w_calc ≈ 0.6, the lower bound cos(C_finance, C_math) ≥ w_calc makes centroid separation impossible by construction. (Maia et al., FiQA 2018)

## Implications for Next Experiment
Discriminative boundary learning (ridge regression) directly optimizes the margin W·(φ_finance - φ_math) > δ rather than centroid proximity — this is the structural fix. exp_p1_p1_ridge_routing_n25 should raise N=25 accuracy from 86.1% to ≥92% and also resolve the finance/math confusion regardless of data source. (Citation: MixLoRA, arXiv:2401.15016; Finding #431)
