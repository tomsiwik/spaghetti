# Pierre — Architecture

> Technical architecture for the composable strategy-adapter platform. Every component references a proven finding. No speculative designs.

---

## System Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                      Pierre Cloud                            │
│                                                              │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Router   │  │  Adapter     │  │  Inference Engine    │  │
│  │          │  │  Registry     │  │                      │  │
│  │ Ridge    │  │              │  │  Gemma 4 E4B 4-bit   │  │
│  │ 98.8%    │  │ Strategy(10) │  │  (frozen base,       │  │
│  │ 0.4ms    │  │ Domain(20)   │  │   has all knowledge) │  │
│  │          │  │ Personal(/u) │  │                      │  │
│  └────┬─────┘  └──────┬───────┘  │  + composed adapters  │  │
│       │               │          │  + per-domain mods   │  │
│       ▼               ▼          │  + skip redun layers  │  │
│  ┌────────────────────────┐      │                      │  │
│  │   Adapter Compositor   │─────▶│  Multi-tenant KV     │  │
│  │                        │      │  Batched serving     │  │
│  │  NRE norm-rescaled     │      │                      │  │
│  │  Precomputed QKV delta │      │  Target: 100+ tok/s  │  │
│  │  Per-domain modules    │      └──────────────────────┘  │
│  │  Skip redundant layers │                                 │
│  └────────────────────────┘                                 │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Personal Extraction Pipeline (M2P + MEMENTO)  │   │
│  │                                                      │   │
│  │  Sessions → MEMENTO compress → M2P extract           │   │
│  │  → personal adapters (approach, domain, preferences,  │   │
│  │    style, codebase) → register in Adapter Registry    │   │
│  │                                                      │   │
│  │  Same methodology, different extraction lenses.       │   │
│  │  One forward pass per dimension. No fine-tuning.      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            Adapter Training Pipeline                  │   │
│  │                                                      │   │
│  │  Strategy: Hedgehog cos-sim distillation (behavioral) │   │
│  │  Domain: Grassmannian A-init + NTP training           │   │
│  │  Personal: M2P one-shot extraction                    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│    Pierre Local Agent    │
│                         │
│  Codebase indexer       │
│  Privacy-gated queries  │
│  Offline adapter cache  │
│  Anonymized sync        │
└─────────────────────────┘
```

---

## Adapter Taxonomy

Three types, all Grassmannian LoRA deltas, all orthogonally composable:

### Strategy Adapters (~10, universal)

Encode problem-solving approaches. Transfer across all domains. This is Pierre's core differentiation — strategies compose, knowledge doesn't.

**Training:** Hedgehog per-layer cos-sim distillation (F#683/F#684).
- Loss: `L = sum_l(1 - cos(A_l_teacher, A_l_student))`
- Teacher: base model + strategy prompt (e.g. "Break this into sub-problems...")
- Student: base model + adapter
- No next-token CE loss — captures attention-routing behavior only

| Strategy | Distillation prompt | F#203 transfer evidence |
|---|---|---|
| `systematic` | "Break into sub-problems, solve each" | Code adapter helped math 10%→70% (F#204) |
| `hypothesis-driven` | "Generate hypotheses, test, eliminate" | Every adapter improves every domain (F#203) |
| `iterative` | "Draft, critique, improve" | Code SFT iterative approach universal (F#204) |
| `constraint-based` | "Enumerate constraints, satisfy all" | — |
| `creative` | "Generate diverse candidates" | — |
| `conservative` | "Prefer known-correct, flag uncertainty" | Scale≤4 preserves knowledge (F#248) |
| `skeptical` | "Seek disconfirming evidence" | — |
| `step-by-step` | "Explicit chain-of-thought, verify each" | NTP preserves reasoning +10pp (F#262) |
| `pattern-matching` | "Recognize known patterns, apply template" | — |
| `analogical` | "Find structural similarity to known solution" | Cross-domain transfer DDR=1.13 (F#202) |

**Properties:**
- Each: 2-5MB, attn-only (v_proj+o_proj)
- Orthogonal by construction: cos=2e-8 (F#428)
- Wrong strategy still helps: 87% of benefit retained (F#203)

### Domain Context Adapters (~20, specialized)

Not domain knowledge — domain-specific *context* for how to apply strategies. The base model has the knowledge; these provide the conventions.

**Training:** Grassmannian A-init + NTP training (F#262: NTP preserves reasoning, SFT destroys it).

| Adapter | What it contextualizes | Modules |
|---|---|---|
| `code-context` | Syntax, idioms, debugging conventions | v_proj+o_proj+gate+up (F#304) |
| `medical-context` | Terminology, clinical workflow | v_proj+o_proj only (F#304) |
| `legal-context` | Citation patterns, argumentation | v_proj+o_proj only |
| `math-context` | Proof conventions, notation | v_proj+o_proj only |
| `finance-context` | Numerical precision, regulatory language | v_proj+o_proj only |

**Properties:**
- Each: 2-5MB, rank 6-8
- 13-30 min training per adapter (F#508)
- +19-56pp on domain benchmarks (F#508)

### Personal Adapters (per-user, M2P-extracted)

Same extraction methodology (M2P from MEMENTO session history), different lenses on the same behavioral data. Not "your knowledge" — "your patterns."

**Training:** M2P one-shot generation from MEMENTO buffer (F#362: 99.6% of SFT quality, 512:1 compression).

| Adapter | What it extracts | Extraction signal |
|---|---|---|
| `user-{id}-approach` | Your problem-solving tendencies | Which strategies you gravitate toward |
| `user-{id}-domain` | Your domain expertise | Domain-specific patterns in your work |
| `user-{id}-preferences` | Output preferences (language, format, verbosity) | Length, structure, formatting patterns |
| `user-{id}-style` | Your writing/communication style | Vocabulary, tone, register |
| `user-{id}-codebase` | Your code patterns and conventions | Naming, architecture, idiom choices |

**Properties:**
- Same M2P forward pass can produce all dimensions
- Generated from MEMENTO-compressed session buffer (50 sessions rolling)
- One forward pass. No fine-tuning. Continuous regeneration.
- Domain-conditional retraining prevents covariate shift (F#466: 92% compliance)

---

## Core Components

### 1. Base Model

**Gemma 4 E4B 4-bit** (`mlx-community/gemma-4-e4b-it-4bit`)

- 4B parameter model, 4-bit quantized
- 42 decoder layers, d_model=2816 (majority), with 7 wide layers at d=1024/4096 (F#766)
- **This IS the knowledge layer.** Frozen, never modified. Contains the facts.
- Served via MLX on Apple Silicon or Metal-compatible GPU

**Why 4B not 400B:** Knowledge is in the base. Expertise comes from adapters. A 4B model with the right strategy adapters outperforms a 400B monolith with one compromised approach (F#204: 10%→70% with a single adapted strategy).

### 2. Adapter Format

```
Adapter = {
  id: string,
  type: strategy | domain | personal,
  lens: string,                          // For personal: approach|domain|preferences|style|codebase
  targets: ["v_proj", "o_proj"],         // Default attn-only (F#304)
  rank: 6 | 8,
  A_matrices: Grassmannian QR,           // Orthogonal by construction (F#428)
  B_matrices: Learned weights,
  scale: float,                          // Strategy-dependent calibration (F#248)
  layers: [0..41],                       // Skip L23-L40 where redundant (F#747)
  metadata: {
    adapter_type: string,
    training_method: string,             // hedgehog | ntp-loRA | m2p
    findings: [F#...],
    quality_scores: {...}
  }
}
```

**Size:** 2-5MB per adapter. **Hot-swap:** <1ms (F#766).

### 3. Router

**Ridge regression on TF-IDF features** (F#458).

```
Input:  user query (text)
Output: strategy adapter stack + domain context adapter

Pipeline:
  1. TF-IDF vectorize query
  2. Ridge regression: query_vector → adapter scores
  3. Strategy selection: which problem-solving approach(es) fit
  4. Domain selection: which context adapter
  5. Personal overlay: always include user's personal adapters

Robustness: routing errors cost only 13% of adapter benefit (F#203).
Wrong adapter still captures 87% of benefit. The system is forgiving.
```

### 4. Adapter Compositor

**Algorithm (NRE — Norm-Rescaled Euclidean, F#275):**
```
W_composed = W_base + sum_i(alpha_i * Delta_W_i / ||Delta_W_i||_F)
```

**Optimizations:**
- **Precomputed QKV deltas** (F#292): 86.8 tok/s
- **Per-type module selection** (F#304): strategy adapters = attn-only, code domain = +MLP
- **Skip redundant layers** (F#747): 18/42 layers skippable for non-code, 43% fewer ops
- **Per-type scale calibration** (F#248): strategies need s≥6 (capability regime), domain context needs s≤4 (format regime), personal needs per-user calibration

**Composition residual:** tau=0.48 (F#752). Real but bounded. NRE handles gracefully.

### 5. Inference Engine

```
Serving stack:
  - MLX for Apple Silicon deployment
  - Standard GPU (CUDA/Metal) for cloud deployment
  - Multi-tenant KV sharing (F#455)
  - Batched adapter composition across concurrent requests

Performance targets:
  - Single-query: 100+ tok/s
  - Batched (N=10 concurrent): 50+ tok/s per user
  - First-token latency: <500ms
  - Adapter hot-swap: <1ms (F#766)
```

### 6. Personal Extraction Pipeline (M2P + MEMENTO)

```
Session N
    │
    ▼
MEMENTO block-masking (F#685)
    │  Compress reasoning into mementos (~3-10x compression)
    ▼
Memento buffer (per-user, rolling 50 sessions)
    │
    ├──→ M2P extract lens=approach    → user-{id}-approach adapter
    ├──→ M2P extract lens=domain      → user-{id}-domain adapter
    ├──→ M2P extract lens=preferences → user-{id}-preferences adapter
    ├──→ M2P extract lens=style       → user-{id}-style adapter
    └──→ M2P extract lens=codebase    → user-{id}-codebase adapter
                                    │
                                    ▼
                          Register in Adapter Registry
                                    │
                                    ▼
                          Available for next query (<1ms hot-swap)
```

**Key properties:**
- Same M2P Transformer, different extraction lenses (different output heads or conditioning)
- No gradient steps. No fine-tuning. One forward pass per dimension.
- Scales to L=36 depth (F#364: 89.1% quality at Qwen3-4B)
- Domain-conditional retraining prevents covariate shift (F#466)

---

## Weak Link Matrix

What breaks when components are removed or replaced?

| Component | If Removed | If Replaced Simple | Fragility | F# |
|---|---|---|---|---|
| Grassmannian A-init | Composition fails (0% all benchmarks) | Random init → interference cos~0.1 | **CRITICAL** | F#510, F#428 |
| NTP training objective | SFT kills reasoning -20pp | SFT OK for formatting only | **HIGH** (reasoning) | F#262 |
| Scale calibration (s≥6 vs s≤4) | Wrong regime → no capability gain or knowledge loss | Uniform s=6 as compromise | **HIGH** | F#248, F#250 |
| Strategy adapters | Lose expertise, still have knowledge | Base model alone (degraded) | Medium | F#203 |
| Domain context adapters | Lose domain conventions | Strategies alone still transfer | **Low** | F#203 |
| Ridge router | Falls back to base model | TF-IDF argmin works at 88% | **Low** | F#203, F#458 |
| M2P personalization | No personal adaptation | Few-shot prompting fallback | Medium | F#362 |
| Per-domain modules | Wasted params, not broken | Full-module everywhere | Low | F#304 |
| Layer skipping | Wasted compute | All layers adapted | Low | F#747 |
| Base model (Gemma 4) | Entire system breaks | Any 4B+ model, retrain adapters | **Low** | F#97 |
| Precomputed QKV | Slower (40 vs 87 tok/s) | Runtime composition | Low | F#292 |

**Three genuinely fragile components:** Grassmannian init, NTP objective, scale calibration. Everything else degrades gracefully.

---

## Data Flow

### Standard query

```
1. User sends query via API/app/CLI
2. Router: TF-IDF → strategy stack + domain context + personal adapters (0.4ms)
3. Compositor: load adapter stack, NRE compose (precomputed QKV)
4. Inference: Gemma 4 (knowledge) + composed adapters (strategies) → response
5. MEMENTO: compress session into memento, append to buffer
6. Background: M2P regenerates personal adapters periodically
7. Response returned to user
```

### Personal adapter update

```
1. User accumulates 5+ new sessions since last update
2. MEMENTO compresses new sessions into mementos
3. M2P extracts updated personal adapters (one forward pass per dimension)
4. Register in Adapter Registry (replaces previous versions)
5. Next query uses updated adapters automatically
```

---

## Deployment

**Cloud serving:**
- Apple Silicon clusters (M5 Pro / M5 Ultra) or Metal-compatible GPUs
- MLX inference backend
- One base model copy shared across all users (knowledge is shared)
- Per-user adapter stacks loaded on demand (<1ms swap) (expertise is personal)
- Multi-tenant KV cache sharing for common prefixes

**Local agent:**
- Runs on user's Mac
- Indexes local codebase → M2P extracts codebase adapter
- Handles privacy-sensitive queries locally
- Syncs anonymized adapter weights to cloud
- Offline mode: cached adapters + quantized base

---

## Security & Privacy

- Base model + strategy adapters + domain adapters: shared, no user data
- Personal adapters: extracted behavioral patterns, not raw content (F#262: NTP learns structure, not tokens)
- MEMENTO compression: lossy 3-10x, original interactions not reconstructable
- Local agent: raw data stays on-device, only adapter weights sync
- Enterprise: on-premise, personal adapters never leave VPC
