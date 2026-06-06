# LEARNINGS.md — exp_knowledge_disentanglement_control

Placeholder for analyst synthesis. Key smoke-level observations
pending full-scale validation:

1. **Method/knowledge disentanglement is NOT automatic with
   rank-16 LoRA on `v_proj+o_proj`.** At n_train=25 × N_STEPS=60
   the adapter collapses MMLU by 30 pp (90 → 60) while *also*
   failing to lift GSM8K reasoning (−5 pp). The prediction from
   ROME-localisation (factual recall in MLPs, not attention) does
   not transfer mechanically to Gemma-4-E4B-4bit — attention's
   output projection participates in factual retrieval too.
2. **Overfit-at-smoke is an experiment-pattern risk, not an
   experiment-specific bug.** The predecessor
   `exp_method_vs_domain_adapter` observed the same regime
   (multi-adapter *below* base at n_train=15 × N_STEPS=40).
   Naïvely running smoke experiments with rank-16 on 25
   examples produces adapters that generalise poorly by
   construction. Smoke should be interpreted as a viability
   check of the pipeline, not of the hypothesis.
3. **Disentanglement claim remains open at full scale.** The
   smoke is not evidence against disentanglement at
   `n_train=100 × N_STEPS=300`; it is evidence against
   disentanglement at the *current* hyperparameters. A pilot
   at `n_train=50 × N_STEPS=150` (pre-registered in PAPER.md
   §Full-scale) is the right next step to decide whether to
   spend the full 3-seed budget.

### Candidate new antipattern memory

Proposed by researcher for analyst confirmation:

> **`rank-16-on-v_proj+o_proj-damages-knowledge-on-gemma4`** (type:
> fix). When rank-16 LoRA on `{self_attn.v_proj, self_attn.o_proj}`
> is trained on small (n ≤ 50) SFT datasets for ≥ 60 steps on
> Gemma-4-E4B-it-4bit, MMLU accuracy drops by ≥ 10 pp — the
> attention output projection is not a safe "procedural-only"
> target, it participates in factual retrieval. Use a pilot
> (n_train=50, N_STEPS=150) before the full-scale rerun of any
> method-adapter hypothesis; if MMLU drop > 5 pp at pilot, do
> not spend the full 3-seed budget.
