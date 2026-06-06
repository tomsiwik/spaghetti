# Pierre — Gameplan

> Execution roadmap from research to product. Each phase has a clear exit criterion. Research only runs experiments that directly unblock a phase.

---

## Phase 0: Foundation (COMPLETE)

**Goal:** Prove the core math works.

**Status:** Done. 856 experiments, 783 findings, 36 conclusive.

| Result | Finding | Significance |
|---|---|---|
| Orthogonal composition works at N=100 | F#428, F#440 | Composition scales without interference |
| Routing at 98.8%, 0.4ms | F#458 | Routing is solved |
| E2E benchmarks: +19-56pp | F#508 | Adapters produce real quality gains |
| Pre-merge fails without orthogonality | F#510 | Orthogonality is mandatory (defensible) |
| M2P one-shot at 99.6% SFT quality | F#362 | Personalization in one forward pass |
| Hot-swap <1ms | F#766 | Production serving viable |
| NTP preserves reasoning, SFT destroys it | F#262 | Training objective matters |
| Wrong adapter still captures 87% benefit | F#203 | **Strategies transfer, not just knowledge** |
| Code adapter helps math 10%→70% | F#204 | **Strategies, not domain facts** |
| Two behavioral regimes (format vs capability) | F#248 | Scale determines strategy activation |

---

## Phase 1: Integration Sprint (CURRENT)

**Goal:** Wire proven components into a single end-to-end pipeline. One binary, one API endpoint, one composed query.

**Exit criterion:** `POST /v1/chat` returns a response composed from 3+ adapters (at least 1 strategy + 1 domain context), with routing, on Gemma 4 E4B, at >50 tok/s.

### 1.1 Serving Infrastructure

Build the inference server.

- [ ] Single-process MLX server: load Gemma 4 E4B 4-bit, accept adapter stack, generate
- [ ] NRE compositor module: take adapter ID list, produce composed deltas (precomputed QKV)
- [ ] Hot-swap integration: swap adapter stack between requests without reloading base model
- [ ] Benchmark: tok/s with N=0 (base), N=1, N=3, N=5
- [ ] Latency budget: P95 first-token <500ms, generation >50 tok/s

**Dependencies:** F#766, F#292, F#275

### 1.2 Router Integration

Wire the router.

- [ ] Load TF-IDF vectorizer + ridge model
- [ ] Route query → strategy adapter stack + domain context adapter
- [ ] Confidence gate: low confidence → base model only
- [ ] Benchmark: routing accuracy at N=5, N=10

**Dependencies:** F#458

### 1.3 First Strategy Adapters (3)

Train the first 3 strategy adapters to validate the strategy-not-knowledge thesis.

- [ ] `systematic` — Hedgehog distillation from "break into sub-problems" teacher prompts
- [ ] `step-by-step` — Hedgehog distillation from explicit chain-of-thought teacher
- [ ] `conservative` — Hedgehog distillation from "prefer known-correct, flag uncertainty" teacher

**Validation:** Each strategy adapter must improve at least 2 out of 3 domains (math, code, medical) to prove cross-domain transfer. If `systematic` helps math AND code but not medical, that's a pass. If it only helps one, the thesis needs revision.

**Dependencies:** F#683, F#684 (Hedgehog design proven), F#203 (transfer evidence)

### 1.4 First Domain Context Adapters (5)

Train 5 thin domain context adapters.

- [ ] `code-context` — NTP, full-module, rank-8
- [ ] `math-context` — NTP, attn-only, rank-6
- [ ] `medical-context` — NTP, attn-only, rank-6
- [ ] `legal-context` — NTP, attn-only, rank-6
- [ ] `finance-context` — NTP, attn-only, rank-6

**Dependencies:** F#627, F#262, F#304

### 1.5 E2E Quality Benchmark

Measure the full pipeline.

- [ ] Benchmark: GSM8K, HumanEval, MedMCQA, MMLU-Pro at n=500
- [ ] Compare: base model, strategies only, domain only, strategies+domain composed
- [ ] Compare: Pierre vs GPT-4o-mini vs Claude Haiku
- [ ] **Critical test:** do strategy adapters transfer across domains? (Expected yes per F#203)
- [ ] Document quality gaps

**Dependencies:** 1.1 + 1.2 + 1.3 + 1.4

---

## Phase 2: Personalization

**Goal:** Extract personal adapters (approach, domain, preferences, style, codebase) from user sessions via same M2P methodology.

**Exit criterion:** A user completes 10 sessions. M2P extracts personal adapters across all 5 dimensions. Adapted model outperforms un-adapted by >10pp on the user's own tasks.

### 2.1 M2P on Gemma 4 E4B

Scale M2P from d=1024 to d=2816.

- [ ] Port M2P Transformer to Gemma 4 dimensions
- [ ] Train M2P with multi-lens extraction (different conditioning for approach vs domain vs style vs preferences vs codebase)
- [ ] Verify quality_ratio >= 95% per lens
- [ ] Measure generation latency (target: <100ms per lens)

**Dependencies:** F#362, F#364

### 2.2 MEMENTO Compression Pipeline

Implement session-to-memento compression.

- [ ] Implement block-masking compression in MLX
- [ ] Validate compression ratio (3-10x) and accuracy retention (>90%)
- [ ] Build per-user rolling memento buffer (50 sessions)

**Dependencies:** F#685, F#686

### 2.3 Multi-Lens Personal Extraction

Wire MEMENTO → M2P → 5 personal adapter dimensions.

- [ ] M2P extract lens=approach → your problem-solving tendencies
- [ ] M2P extract lens=domain → your domain expertise
- [ ] M2P extract lens=preferences → your output preferences (language, format, verbosity)
- [ ] M2P extract lens=style → your writing/communication style
- [ ] M2P extract lens=codebase → your code patterns (from local agent)
- [ ] Validate: personal adapters compose with strategy + domain adapters without interference
- [ ] A/B test: with vs without personal adapters on user tasks

**Dependencies:** 2.1 + 2.2

---

## Phase 3: Full Strategy Library

**Goal:** Complete the strategy adapter library and validate full-stack composition.

**Exit criterion:** 10 strategy adapters trained. Full stack (3 strategies + 1 domain + 3 personal) composes at >50 tok/s with measurable quality gain on every axis.

### 3.1 Complete Strategy Library

- [ ] `systematic`, `step-by-step`, `conservative` (from Phase 1)
- [ ] `hypothesis-driven` — generate hypotheses, test, eliminate
- [ ] `iterative` — draft, critique, improve
- [ ] `constraint-based` — enumerate constraints, satisfy all
- [ ] `creative` — generate diverse candidates
- [ ] `skeptical` — seek disconfirming evidence
- [ ] `pattern-matching` — recognize known patterns, apply template
- [ ] `analogical` — find structural similarity to known solution

### 3.2 Cross-Strategy Composition Validation

- [ ] Verify strategies compose with each other (systematic + conservative works)
- [ ] Verify strategies don't interfere with domain context adapters
- [ ] Verify strategies don't interfere with personal adapters
- [ ] Measure composition quality at N=7 (3 strategies + 1 domain + 3 personal)
- [ ] **Critical test:** does strategy composition produce emergent behavior? (e.g. systematic + skeptical = better code review than either alone)

### 3.3 Complete Domain Context Library

- [ ] `sql-context`, `rust-context`, `typescript-context`, `scientific-writing-context`
- [ ] `japanese-context`, `german-context` (language context as domain variant)

---

## Phase 4: Product

**Goal:** Ship a usable product.

**Exit criterion:** 100 beta users, >80% retention after 2 weeks, measurable quality advantage over GPT-4o-mini on user-specific tasks.

### 4.1 API & SDK

- [ ] REST API: `/v1/chat` with strategy + domain + personal adapter selection
- [ ] Streaming responses (SSE)
- [ ] Python SDK, TypeScript SDK

### 4.2 Web Application

- [ ] Chat interface showing active strategy/domain stack
- [ ] Session history → MEMENTO → personal adapter feedback loop
- [ ] Strategy playground: let users try different strategy combinations

### 4.3 IDE Extensions

- [ ] VS Code extension: task-aware strategy routing (debugging → systematic+skeptical)
- [ ] JetBrains extension
- [ ] Local agent: codebase indexing → M2P codebase adapter

### 4.4 Local Agent

- [ ] Thin Mac client
- [ ] Privacy-sensitive queries stay local
- [ ] Codebase indexing → codebase personal adapter
- [ ] Offline mode with cached adapters

---

## Phase 5: Scale

**Goal:** Production infrastructure, adapter marketplace, enterprise features.

### 5.1 Serving Infrastructure

- [ ] Multi-GPU serving (Apple Silicon cluster or cloud Metal)
- [ ] Multi-tenant KV sharing (F#455)
- [ ] P99 <1s at 1000 concurrent users

### 5.2 Adapter Marketplace

- [ ] Community strategy adapter sharing
- [ ] Quality scoring / review system
- [ ] Adapter versioning
- [ ] Monetization for adapter creators

### 5.3 Enterprise

- [ ] On-premise / VPC deployment
- [ ] Organization-wide strategy and domain adapters
- [ ] Compliance adapters (HIPAA-safe strategies, SOX documentation)
- [ ] SSO / SCIM integration

---

## Research Alignment Rules

Every experiment must answer: **"Which phase does this unblock?"**

| Phase | Allowed research | Disallowed research |
|---|---|---|
| **Phase 1** | Serving, composition, strategy training, domain training, benchmarking | MEMENTO, M2P multi-lens, new loss functions, theoretical abstractions |
| **Phase 2** | M2P scaling, MEMENTO compression, multi-lens extraction, personal quality | New routing, new strategies, new composition methods |
| **Phase 3** | Full strategy library, cross-strategy composition, full domain library | New base models, spectral analysis, open exploration |
| **Phase 4+** | Product-driven optimizations only | Anything not blocking a product feature |

**Kill criterion:** If an experiment doesn't directly unblock a Phase 1-4 task, don't claim it. Every experiment must ship code toward the product.

**Strategy-first principle:** When in doubt, test a strategy adapter before a domain adapter. The thesis is that strategies compose better than knowledge. Prove it early.

---

## Timeline (aggressive)

| Phase | Duration | Cumulative |
|---|---|---|
| Phase 1: Integration Sprint | 2-3 weeks | Week 3 |
| Phase 2: Personalization | 2-3 weeks | Week 6 |
| Phase 3: Full Strategy Library | 2-3 weeks | Week 9 |
| Phase 4: Product | 4-6 weeks | Week 15 |
| Phase 5: Scale | Ongoing | — |

---

## Key Risks

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Strategy adapters don't transfer cross-domain | **Low** | Kills core thesis | F#203/F#204 already show transfer; validate early in Phase 1.3 |
| Hedgehog distillation can't capture strategies | Medium | Blocks strategy adapters | Fallback: NTP training with strategy-prompted data + post-hoc audit |
| M2P doesn't scale to Gemma 4 d=2816 | Medium | Blocks Phase 2 | Fallback: direct LoRA fine-tuning (slower but proven) |
| Multi-lens extraction doesn't separate dimensions | Medium | Personal adapters merge | Fallback: single combined personal adapter |
| Composition quality degrades at N>5 real adapters | Low | Limits adapter count | F#440 tested N=100; validate early |
| Serving speed <50 tok/s | Low | Blocks product | F#292 hit 86.8 tok/s |
| Thinking-mode incompatibility (F#536) | Medium | Retraining needed | Retrain with enable_thinking=True |

---

## Weak Link Summary

Three fragile components, everything else degrades gracefully:

| Component | If broken | Why fragile |
|---|---|---|
| **Grassmannian A-init** | 0% on all benchmarks | Without orthogonality, composition is catastrophic (F#510) |
| **NTP training objective** | Reasoning destroyed (-20pp) | SFT response-masking kills chain-of-thought (F#262) |
| **Scale calibration (s≥6 vs s≤4)** | No capability gain or knowledge loss | Phase transition at s∈[4,6] (F#248, F#250) |

Everything else (router, domain adapters, layer skipping, precomputed QKV) is robust — the system works even when these are removed or simplified, just with degraded performance.
