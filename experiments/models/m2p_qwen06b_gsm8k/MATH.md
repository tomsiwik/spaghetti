# MATH.md: M2P on Qwen3-0.6B + GSM8K — First Real Language Test

## TYPE: frontier-extension (Type 3)
## PROVEN FRAMEWORK: M2P quality scaling at d=1024 (#362), layer depth L=28-36 (#365)
## FRONTIER QUESTION: Does M2P generate useful adapters from REAL language on a REAL model?

---

## Why This Is the Credibility Inflection Point

All 362+ prior M2P findings are on synthetic domains (arithmetic, sort, reverse, cipher)
with toy transformers (d=64-1024, L=2-36, vocab=128). The adversarial review (critique #3)
correctly states: "one real result > all toy experiments combined."

This experiment answers: can M2P generate a LoRA adapter on Qwen3-0.6B that improves
GSM8K math reasoning?

## Model: Qwen3-0.6B-4bit

- d_model = 1024 (proven: #362, 99.6% at d=1024)
- n_layers = 28 (proven: #365, 89.1% at L=36 > L=28)
- n_heads = 16, n_kv_heads = 8 (GQA — proven: #318, Grassmannian holds on GQA)
- intermediate_size = 3072
- vocab = 151936 (BPE, NOT character-level)
- head_dim = 128 (2x toy)

## M2P Configuration

Proven recipe (unchanged): d_M2P=64, L_m2p=2, N_memory=32.
The ONE question: does d_M2P=64 bottleneck capture enough structure from real NLP
hidden states? At toy scale, d_int < 64 holds (Aghajanyan #362). For real language
tasks, Aghajanyan et al. (2012.13255 Table 1) measured d_int = 100-1000 for various
NLP tasks. GSM8K may require d_int > 64.

**If d_M2P=64 fails:** retry with d_M2P=128 (doubles M2P params, still tiny).

## Approach

1. Load Qwen3-0.6B-4bit frozen
2. Generate Grassmannian A-matrices for GQA architecture (28 layers)
3. Train SFT LoRA adapter on GSM8K train split (rank=4, scale=5, 300 steps) → quality ceiling
4. Train M2P to generate B-matrices from GSM8K context → adapter
5. Evaluate both via answer accuracy on GSM8K test subset (parse final numeric answer)
6. Measure MMLU preservation (scale=5, proven safe #330)

## Data

- GSM8K train: ~7.5K examples (format: question → chain-of-thought → #### answer)
- GSM8K test: ~1.3K examples
- For M2P: use n=2000 randomly sampled train examples as context
- For SFT: same 2000 examples, same training budget T=300 steps
- Evaluation: 200 test examples (parse "#### <number>" for accuracy)

## Kill Criteria

**K_real:** M2P quality_ratio ≥ 70% of SFT accuracy on GSM8K test subset.
  Relaxed from 85% because this is the first real-language test.
  SFT accuracy on GSM8K with QLoRA rank=4 on 0.6B model: expect ~5-15% accuracy
  improvement over base. 70% of that improvement is the bar.

**K_mmlu:** MMLU accuracy degradation ≤ -3pp with M2P adapter applied.

**K_KILL:** M2P quality_ratio < 30%. If M2P captures < 30% of SFT improvement on
  real language, the hypernetwork approach is toy-only.

## Predictions

| Metric | Predicted | Reasoning |
|--------|-----------|-----------|
| SFT accuracy gain | +5-15pp over base | QLoRA rank=4 on small model |
| M2P quality ratio | 70-90% | d_M2P=64 may be tight for real NLP |
| MMLU degradation | < 1pp | scale=5, proven safe (#330) |
| M2P generation time | < 50ms | Single forward pass, model is small |

## What It Proves If It Passes

If K_real PASS: M2P generates useful adapters from real language context on a real
model. The hypernetwork approach works beyond synthetic tasks. This resolves
adversarial critiques #3 (no natural language) and partially #10 (hypernetworks
don't scale — counter-evidence at 0.6B).

## What It Proves If It Fails

If K_KILL: M2P cannot capture real NLP task structure at d_M2P=64. Either:
- d_M2P must increase (try 128, 256)
- M2P architecture needs modification for real language (cross-attention over tokens instead of mean-pool)
- The Aghajanyan d_int < 64 assumption breaks for real NLP (possible — their Table 1 shows d_int=100+ for some tasks)

In this case, Pierre falls back to SFT-only adapters on real models (proven: #319, #332).
The composition and routing math still holds; only the generation speed advantage is lost.
