# Learnings: exp_top2_output_space_falcon

## Core Finding

Output-space top-2 composition is **mechanism-sound but experimentally untested** — the kill applies to "NTP adapters on instruction-tuned Falcon," not to output-space composition itself. The composition mechanism never received a fair trial because ALL adapters individually degraded the base (0.390 avg vs 0.540 base). This is the third independent confirmation that NTP-trained adapters degrade instruction-tuned bases on QA benchmarks, establishing it as a **systematic architectural constraint**, not an incidental failure.

## Why This Happened

### 1. Training-evaluation objective mismatch is the root cause (not composition)

The adapters were trained with next-token prediction on domain text (medical dialogs, code completions, etc.). The evaluation is MMLU (multiple-choice QA). This mismatch is well-documented: arXiv:2502.14502 ("How Much Knowledge Can You Pack into a LoRA Adapter without Harming LLM?") shows that knowledge-injecting LoRA adapters can degrade the base model's pre-existing capabilities when the training distribution diverges from the evaluation distribution. The instruction-tuned base already occupies the "good region" of output space for QA formatting — adapter deltas push it out.

**Three-experiment convergence:**
- exp_falcon_e3b_composition: uniform merge degrades 5/6 benchmarks on Falcon-E-3B
- exp_task_accuracy_real_benchmarks: oracle routing hurts MMLU on 4/5 domains even with perfect adapter selection
- This experiment: ALL composition methods (single, merge, OS-top2) worse than base

The pattern is now unambiguous: NTP-trained adapters are **format-specialized** (good at domain text continuation) but **format-destructive** (bad at QA format on instruction-tuned bases). This is not a composition failure — it's a training objective failure that no composition mechanism can fix.

### 2. Output-space cross-term elimination is real but irrelevant here

LoRI (arXiv:2504.07448) proves that output-space composition eliminates cross-terms through nonlinearities — each adapter's contribution is computed in isolation. This is mathematically correct and validated by production MoE systems (Mixtral top-2/8, DeepSeek-V3 top-2/256). However, the guarantee is about **composition safety** (no destructive interference between adapters), not **composition quality** (composed output is better than base).

When individual adapters are harmful (single adapter avg = 0.390, 28% below base), eliminating cross-terms between them is eliminating interactions between negative contributions. OS-top2 (0.410) slightly beat single adapter (0.390) precisely because averaging two harmful outputs dilutes each adapter's damage — this is cross-term elimination working as designed, but the adapters have nothing positive to contribute.

### 3. Naive adapter-swap speed is a known systems problem, not fundamental

The 17x overhead (45.1 → 2.7 tok/s) comes from adapter swapping invalidating KV caches, forcing full recomputation per adapter pass. This is a known problem in multi-LoRA serving:

- arXiv:2505.03756 ("Improving Multi-LoRA Serving via Efficient LoRA and KV Cache Management") shows that adapter switches without KV cache management create catastrophic TTFT degradation. Their solution: jointly optimize LoRA and KV cache placement in HBM.
- arXiv:2512.17910 ("Cross-Model KV-Cache Reuse with Activated LoRA") demonstrates that adapters sharing the same base can reuse KV caches across adapter switches, dramatically reducing overhead. The key insight: KV cache differences between adapters are small when adapters are low-rank perturbations.

Our naive implementation (apply adapter → forward → remove → apply next → forward) is the worst case of exactly what these papers solve. The 17x overhead is implementation-dependent, not mechanism-dependent. The reviewer correctly flagged this: K2 kills the implementation, not the concept.

### 4. The math+code signal is noise (p~0.37)

The reviewer's statistical analysis is definitive: keyword-routed OS-top2 scoring 0.650 vs 0.550 base on math is 13 vs 11 correct out of 20 questions. Fisher exact test gives p ≈ 0.37 — not remotely significant. The PAPER.md treated this as "the ONE domain where composition helped," but it is indistinguishable from random variation at n=20. No follow-up experiments should be designed around this signal without replication at n≥100.

### 5. lora_scale=20.0 is an untested confound

The reviewer identified that lora_scale=20.0 was never ablated. This means each adapter delta is amplified 20x before being added to the base weights. For NTP-trained adapters on an instruction-tuned base, this may dramatically amplify the format mismatch: a large scale pushes the model further from the instruction-tuned equilibrium toward the domain-text distribution. A sweep over {1, 5, 10, 20} could reveal whether the adapters are harmful at all scales or only at extreme amplification. This is cheap (~4 minutes) and could change the interpretation of all three NTP-adapter experiments.

## Confirming Evidence

1. **exp_falcon_e3b_composition**: Uniform composition degrades 5/6 benchmarks. Instruction-tuned Falcon base beats all adapter methods. Direct replication of the same failure mode.

2. **exp_task_accuracy_real_benchmarks**: Oracle routing hurt MMLU on 4/5 domains. "Adapters are format-specialized, not knowledge-additive" (from that experiment's review). Even perfect routing cannot overcome training objective mismatch.

3. **exp_lora_soups_cat**: No superlinear composition with orthogonal adapters — orthogonality prevents both destructive AND constructive interference. Task Arithmetic λ=0.5 is optimal for static merge.

4. **arXiv:2502.14502** (Knowledge Packing in LoRA): Shows LoRA adapters can degrade base model knowledge when training/eval distributions diverge. The degradation scales with adapter rank and training duration — consistent with our lora_scale=20.0 amplifying the effect.

5. **arXiv:2603.03535** (Routing > Merging at Scale): Systematic comparison showing ensembling > routing > merging for multi-LoRA composition. Our output-space approach is ensembling (the strongest method), yet still fails — confirming the problem is upstream of composition.

6. **Our own exp_ppl_vs_task_performance**: PPL does NOT predict task accuracy (Pearson r=0.08). The adapters improve domain PPL but degrade benchmark accuracy — these are fundamentally different metrics measuring different capabilities.

## Contradicting Evidence

1. **arXiv:2506.13479** (Top-k Composition): k=2 gives superlinear gains on models where adapters individually help. The mechanism is valid — our adapters are not. The paper used task-matched adapters (adapters trained for the evaluation objective), not generic NTP adapters.

2. **MoLoRA (arXiv:2603.15965)**: Qwen3-1.7B + 4 per-token-routed adapters beats 8B monolithic model. Key differences: (a) standard (not instruction-tuned) base, (b) task-specific adapter training, (c) per-token routing (not per-sequence). The composition mechanism works when adapters are beneficial.

3. **arXiv:2505.03756** (Multi-LoRA KV Cache Management): Shows 17x overhead is solvable with proper KV cache management. Adapter switching overhead drops dramatically when KV caches are shared across adapter variants. Our speed kill is implementation-specific.

4. **LoTA-QAF (arXiv:2407.11024)**: Ternary adapters CAN improve MMLU by +5.14% when specifically trained for quantization recovery on factual benchmarks. Proves ternary adapters are not inherently harmful — the training objective determines whether they help or hurt.

## Alternative Approaches

### A. Fix the adapters (training objective)

1. **QA-format adapter training**: Train adapters on instruction-formatted domain QA data (e.g., medical MMLU-style questions, not medical dialogs). LoTA-QAF (arXiv:2407.11024) proved this works for ternary adapters. This directly addresses the root cause identified across three experiments.

2. **Knowledge-preserving fine-tuning**: arXiv:2502.14502 proposes techniques to inject domain knowledge via LoRA without degrading base capabilities. Apply their balanced adaptation approach to our adapter training pipeline.

### B. Fix the evaluation (measure what adapters actually improve)

3. **Domain PPL evaluation**: Our adapters improve domain PPL by -26.3% (proven in multiple experiments). Evaluate composition on domain PPL, where adapters are known to help, to test whether output-space composition provides additional gains over parameter merge.

### C. Fix the serving (speed)

4. **KV-cache-aware adapter application**: arXiv:2512.17910 shows cross-model KV cache reuse for LoRA adapters sharing a base. Implementing shared KV caches across adapter passes would reduce the 17x overhead toward the theoretical 2x minimum. This is engineering, not research.

5. **Per-token MoLoRA routing**: arXiv:2603.15965 avoids adapter-swap entirely by routing at the FFN level. Each token is dispatched to 1-2 expert LoRA blocks within a layer, with the base model's attention layers shared. This is the production-viable version of what we attempted.

## Implications for Next Experiments

1. **The NTP-adapter-on-instruction-tuned-base problem is SETTLED.** Three experiments confirm it. Stop testing composition mechanisms with these adapters on instruction-tuned models. Either: (a) use BitNet-2B-4T (not instruction-tuned) where adapters improve PPL, or (b) retrain adapters for the evaluation objective.

2. **Output-space composition is NOT falsified.** The mechanism was never properly tested. A fair test requires: adapters that individually beat base → then test whether composing 2 exceeds the best single. This experiment's MATH.md correctly predicted the mechanism but missed the prerequisite (Assumption 1: "adapters are trained and specialized" — they were specialized for NTP, not QA).

3. **Prerequisite gate needed for all future composition experiments.** The reviewer's recommendation is correct: add K0 = "single best adapter must beat base by >X on ≥3/5 domains" before testing any composition method. This prevents wasting compute on composition of broken components.

4. **lora_scale ablation should precede new adapter training.** Before committing to retraining adapters, test whether scale={1, 5, 10} produces less degradation. If scale=1 adapters are neutral or slightly beneficial, the amplification (not the training objective) may be the primary issue.

5. **Speed is solvable.** The 17x overhead is a known, studied problem with published solutions (arXiv:2505.03756, arXiv:2512.17910). Do not kill output-space composition for speed reasons — kill it only if the quality case fails with proper adapters.

## Recommended Follow-Up

**No new experiment recommended from this kill.** The critical path (CLAUDE.md P0) is the deployment track on BitNet-2B-4T, where:
- The base is NOT instruction-tuned (adapters improve PPL by -26.3%)
- exp_generation_quality_test is the existential test (does composition produce better TEXT?)
- The adapter training objective mismatch does not apply to BitNet-2B-4T + domain PPL evaluation

If output-space composition is revisited, it should be on BitNet-2B-4T with a prerequisite gate (adapters individually improve the metric being tested) and domain PPL evaluation (not MMLU).
