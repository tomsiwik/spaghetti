# Reasoning from Scratch (Sebastian Raschka)

**Source:** https://github.com/rasbt/reasoning-from-scratch
**Type:** Git submodule at `references/reasoning-from-scratch/`

**Key Insight:** Educational resource showing how to build reasoning capabilities
into LLMs from the ground up. Covers chain-of-thought, tool use, and structured
reasoning — all with clean, understandable implementations.

**Relevance to our work:**
- Understanding reasoning mechanics informs how expert composition affects
  downstream capabilities (not just perplexity)
- Reasoning chains may activate different expert subsets — relevant to
  routing and expert selection research
- Functional eval (macro/eval_functional.py) tests basic code generation;
  reasoning evaluation would be the next level

**How to use:**
Together with LLMs-from-scratch, this forms the foundational knowledge base
for all agents/researchers. Check this repo when designing evaluation
protocols or understanding how model capabilities emerge from architecture.
