# exp_hedgehog_domain_adapter_js

## Paper
Stage 1 of [arXiv:2604.14191](https://arxiv.org/abs/2604.14191). Third axis: **language-domain knowledge** (JavaScript nuance) — parallel to politeness (behavior) and refactor (procedural).

## Teacher corpus
- MDN JavaScript reference (open web data)
- *Eloquent JavaScript* 4th ed. by Marijn Haverbeke ([eloquentjavascript.net](https://eloquentjavascript.net), Creative Commons licensed — redistributable)
- Node.js official documentation

Focus domains: hoisting, closures, `this`-binding, event loop, prototype chain, async/await nuances.
~200 question-answer pairs covering these.

## Reference implementations
Same as refactor/politeness experiments — `HazyResearch/lolcats` Hedgehog reference.

## MLX translation notes
Same attention-capture pattern. JS documentation blocks can be ~2-4k tokens; set `SEQLEN=2048`.

## Dataset acquisition
1. Scrape MDN pages for the 6 focus topics (robots.txt-respecting)
2. Eloquent JS is CC-BY-NC 3.0 — can be redistributed with attribution; download from eloquentjavascript.net
3. Generate Q-A pairs via larger model with the source text in context; split 80/20

## Quick start
```bash
experiment claim <worker-id> --id exp_hedgehog_domain_adapter_js
experiment run exp_hedgehog_domain_adapter_js
```
