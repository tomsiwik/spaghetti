# LEARNINGS.md — exp_p1_p1_ridge_routing_n25

**Status: SUPPORTED — Finding #458**

## Core Finding
Ridge regression routing raises N=25 accuracy from 86.1% → 98.8% (+12.7pp), eliminating the production bottleneck. Finance domain specifically recovers from 74% → 93%, directly fixing the math-finance confusion that killed exp_p1_p0_finance_routing_fix.

## Why It Works
Centroid routing fails at N=25 because many MMLU domains share generic academic vocabulary (argument, theory, evidence). Ridge regression learns W* that suppresses shared terms and amplifies domain-specific ones via joint cross-domain optimization. With M=7500 training examples >> d_eff, low regularization (α=0.1) is optimal. Citation: MixLoRA (2312.09979) predicted this; result confirms and quantifies the gap.

## Implications for Next Experiment
Routing is no longer the bottleneck. Ridge W* ∈ ℝ^{d×N} trains in 0.57s, infers at 0.40ms p99, costs 2MB memory, and hot-adds a new domain in <1s (closed-form). The remaining open question is behavioral quality: domain adapters (Finding #457) don't improve generation quality for capable base models (δ_D ≈ 0). The next priority is understanding when δ_D > 0 — when does the base have a genuine behavioral gap that adapters can fill?
