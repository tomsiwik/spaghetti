# Pierre — Vision

> Composable Intelligence as a Service. A hosted AI platform that beats frontier models by composing orthogonal problem-solving strategies, not by cramming in more knowledge.

---

## The Problem with Monoliths

Frontier models (GPT-4, Claude Opus, Gemini) are knowledge encyclopedias — one giant parameter blob that memorized everything. When you ask a monolith to refactor JavaScript politely, it activates a fuzzy compromise between code knowledge, refactoring patterns, and politeness. These compete for the same attention patterns. Scaling up delays the problem but doesn't solve it. A 10T parameter model is still one set of compromises.

Worse: knowledge doesn't compose. Medical knowledge + coding knowledge doesn't make you better at either. But **strategies do compose.** A hypothesis-driven approach helps in medicine AND coding AND legal analysis. A systematic-decomposition strategy transfers everywhere. Humans prove this daily — different people with different approaches solve the same problems well, and the best problem-solvers aren't the ones who know the most facts, but the ones with the best strategies.

## The Pierre Thesis

**The base model already has all the knowledge. What it needs is better thinking.**

Pierre composes orthogonal problem-solving strategies — not domain knowledge — to turn a generalist model into a per-task specialist. The evidence from our research (783 findings, 856 experiments) is overwhelming:

1. **Adapters encode strategies, not knowledge.** A code adapter applied to math (0% routing accuracy) improved math from 10% to 70% (F#204). Every adapter improves every domain — wrong adapter still captures 87% of benefit (F#203). This is inexplicable if adapters encode domain knowledge. It makes perfect sense if they encode problem-solving approaches.

2. **Strategies compose orthogonally.** Grassmannian-initialized adapters occupy mathematically disjoint subspaces (cos=2e-8, F#428). `hypothesis-driven` + `conservative` + `pattern-matching` composes without interference. Medical knowledge + coding knowledge doesn't compose at all.

3. **Personalization extracts your approach, not your knowledge.** The M2P hypernetwork generates personal adapter weights from session history in one forward pass at 99.6% of fine-tuned quality (F#362). It extracts your problem-solving style — whether you're systematic or intuitive, conservative or exploratory — not your domain knowledge.

4. **Composition beats compromise.** At N=3 strategy adapters, Pierre has ~3x the effective specialization vs a monolith. At N=7 (strategies + domain context + personal), it's structurally impossible for a monolith to match. The monolith has one approach. Pierre has the right combination of approaches for each task.

## The Insight: Strategies Transfer, Knowledge Doesn't

| Compose these... | Result | Example |
|---|---|---|
| Medical knowledge + Coding knowledge | Nothing useful | Medical facts don't help you code |
| Hypothesis-driven + Conservative + Pattern-matching | Expert diagnostician | Works for medical, debugging, legal analysis |
| Systematic decomposition + Skeptical verification | Expert debugger | Works for code, audit, fact-checking |
| Iterative refinement + Creative + Constraint-based | Expert designer | Works for architecture, product, research |

The base model has the knowledge. The strategies are what make it expert.

## Why This Wins

| Dimension | Monolith | Pierre |
|---|---|---|
| Approach | One fuzzy compromise for all tasks | Composes the right strategies per task |
| Expertise mechanism | Memorize more facts | Apply better thinking strategies |
| Personalization | Few-shot prompting, context-limited | Extracts your problem-solving approach from sessions |
| Continuous learning | Fixed until next training cut | M2P regenerates personal adapter after every session |
| Cross-domain transfer | Poor — knowledge is domain-locked | Strong — strategies transfer everywhere |
| Cost per query | $3-15/M tokens (massive model) | Commodity GPU + 4B model + adapter swap |
| Privacy | Data leaves your org | Personal adapters trainable on-premise, served in your VPC |

## The Product

**Pierre is a hosted API and application** that delivers specialized, personalized AI through composable strategy adapters:

- **Web app** — chat interface with automatic strategy routing
- **API** — `POST /v1/chat` with strategy hints or auto-routing
- **IDE extensions** — VS Code / JetBrains with task-aware strategy selection
- **CLI** — `pierre "review this PR" --strategies systematic,skeptical,conservative`
- **Local agent** — indexes local codebase, handles privacy-sensitive queries locally, syncs anonymized adapter updates to cloud

## Adapter Catalog

### Strategy Adapters (~10, universal)

These encode problem-solving approaches, not domain knowledge. Each one transfers across all domains:

| Adapter | What it does | Applies to |
|---|---|---|
| `systematic` | Break problem into sub-problems, solve each | Math, code, diagnosis, legal analysis |
| `hypothesis-driven` | Generate hypotheses, test against evidence, eliminate | Diagnosis, debugging, scientific reasoning |
| `iterative` | Draft → critique → improve cycle | Writing, code review, analysis |
| `constraint-based` | Enumerate constraints, find solution satisfying all | Legal, engineering, scheduling |
| `creative` | Generate diverse candidates, evaluate later | Ideation, brainstorming, design |
| `conservative` | Prefer known-correct over novel, flag uncertainty | Medical, legal, finance |
| `skeptical` | Actively seek disconfirming evidence | Code review, fact-checking, audit |
| `step-by-step` | Explicit chain-of-thought with verification at each step | Math, proof, formal reasoning |
| `pattern-matching` | Recognize known patterns, apply template | Code patterns, symptoms, precedents |
| `analogical` | Find structural similarity to known solution | Cross-domain innovation, creative problem-solving |

### Domain Context Adapters (~20, specialized)

Not "domain knowledge" but "how to apply strategies in this domain." Thin contextual layer on top of strategies:

- `code-context` — syntax patterns, common idioms, debugging conventions
- `medical-context` — terminology conventions, clinical workflow norms
- `legal-context` — citation patterns, argumentation structures
- `math-context` — proof conventions, notation standards
- `finance-context` — numerical precision norms, regulatory language

### Personal Adapters (per-user, extracted via same methodology)

Same extraction methodology (M2P from MEMENTO session history) produces multiple personal dimensions:

- `user-{id}-approach` — your problem-solving tendencies (systematic vs intuitive, conservative vs exploratory)
- `user-{id}-domain` — your domain expertise and specializations
- `user-{id}-preferences` — your output preferences (language, tone, format, verbosity)
- `user-{id}-style` — your writing and communication style
- `user-{id}-codebase` — your code patterns and conventions

All generated by M2P from the same session buffer. Same one forward pass. Same 99.6% of fine-tuned quality (F#362). Different extraction lenses on the same behavioral data.

## Business Model

**Individual ($20-50/mo):** All strategy + domain adapters, personal adapter extraction, M2P continuous learning, API access.

**Professional ($100-200/mo):** Unlimited personal adapters, priority inference, custom adapter training on proprietary data, local agent.

**Enterprise (custom):** Private adapter registry, on-premise/VPC deployment, organization-wide adapters, compliance adapters, SLA guarantees.

## What Makes This Defensible

1. **The orthogonality math.** Without Grassmannian initialization, composition fails catastrophically (F#510: 0% on all benchmarks). This is a mathematical requirement, not an optimization. Patentable.

2. **The strategy-not-knowledge insight.** Competitors building "domain adapter" systems will hit the knowledge composition wall. Strategies compose; knowledge doesn't. This is a structural advantage.

3. **M2P one-shot personalization.** Years ahead of fine-tuning. Extracts your approach from every session without retraining anything.

4. **Adapter ecosystem network effects.** More users → more strategy refinements → better extraction → better for all users.

5. **Structural cost advantage.** Serving 4B + adapter swap is orders of magnitude cheaper per query than serving a frontier monolith.

## Success Criteria

Pierre succeeds when it demonstrably outperforms frontier models through better thinking, not more knowledge:

- Strategy-composed Pierre beats base model by >20pp on behavioral benchmarks
- Personal adapter improves user-specific tasks by >10pp over un-adapted Pierre
- Cross-domain transfer: same strategies help on unseen domains (zero-shot)
- <2s end-to-end latency at P95
- <$0.001 marginal cost per query
