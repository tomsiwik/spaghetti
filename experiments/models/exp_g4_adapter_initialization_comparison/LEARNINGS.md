# LEARNINGS.md — exp_g4_adapter_initialization_comparison

**Status:** PROVISIONAL (K1925 target PASS, K1924 proxy FAIL-by-confound per F#666).

## Core Finding

At Gemma 4 E4B d=2560 q_proj r=6, 100 iters on medical-MCQ: final eval-PPL is init-invariant within **3.54%** across {Grassmannian 1.201, Kaiming 1.210, Gaussian 1.168}; baseline 2070.79. Grassmannian intra-column |cos| grows from 4.77e-9 to 0.032 — still **2.8× cleaner** than Kaiming (0.090). Cross-init cos-sim (K1924 proxy) is confounded: all three inits shared `mx.random.key(42)` sub-keys, so start cos is 0.977–0.9995 (should be ~√(r/d)≈0.054).

## Why

1. **Behavioral invariance** is a B-compensation result: with B=0 at init, the only signal entering the loss is Y=BAx=0; gradient flow through B decouples from A's column-basis for small steps. PPL-init-invariance at d=2560 generalizes F#169 (BitNet, same d) to attention-only LoRA on 4-bit Gemma 4.
2. **Grassmannian persistence** (F#562 extension): at r=6 d=2560, random-Gaussian columns are already near-orthogonal (concentration of measure, cos ~0.05); QR only marginally improves init, and 100 iters slightly erodes it. Kaiming's uniform-bounded draws drift further (0.016→0.090) — training introduces column alignment more aggressively under uniform init.
3. **Grassmannian train-loss lag** (P3 FAIL, 17% gap vs 10% threshold): uniform singular values =1 force B to grow uniformly across rank before any direction is exploited; heterogeneous Gaussian A gives B an early concentrated direction. This is dynamics, not capability — eval PPL converges.
4. **PRNG confound**: `mx.random.normal(key=k)` and `mx.random.uniform(key=k)` draw from the SAME underlying sequence; sharing a seed across init variants produces co-linear starting matrices. Cross-init cos-sim is structurally uninformative under this design.

## Implications for Next Experiment

1. **Primary follow-up `exp_g4_adapter_initialization_comparison_v2`** (already filed): distinct per-init top-level seeds (42/43/44), 3 seeds each for variance, 1000 iters to address both the PRNG confound AND the P3 train-loss convergence question. This will resolve whether the 3.5% PPL spread survives the clean design.
2. **Intra-init column-orthogonality is the confound-free structural probe**: future init-comparison KCs should use intra-method metrics (self-orthogonality, spectrum) as primary, reserve cross-method cos-sim for seed-separated designs only.
3. **Antipattern to propagate** (new memory below): shared PRNG key across method variants is a silent confound — any cross-method structural KC must pre-register distinct seeds.
4. **Non-blocking fix**: MATH.md §0/§3 typos `d=2048` should read `d=2560` for Gemma 4 E4B; code reads shape dynamically so measurements stand.
5. **Open axis**: does Grassmannian's train-loss lag close by iter-1000? Answerable by the v2 1000-iter rerun without new scaffolding.
