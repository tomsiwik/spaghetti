# spark/ — prompt bank for emergent exploration (under eval)

Goal: find the atomic `prompts/*.md` building blocks whose **side-effect** is exploration/hunch-discovery, and **prove with promptfoo** that they out-produce a plain baseline at generating genuine *reframes* — before composing anything into the agent loop.

Nothing here is wired into the research loop. It's a measurement bench.

## Layout
- `prompts/*.md` — atomic building blocks, one idea each (`{{problem}}` is the test var):
  - `baseline.md` — control ("solve it")
  - `invert-assumption.md` · `representation-shift.md` · `analogy-collision.md` · `self-question.md` · `constraint-removal.md` — perturbations
- `promptfooconfig.yaml` — the eval (generator + judge = local ollama; two rubrics: `reframe`, `novelty`; neutral CS problems).

## Run
```bash
promptfoo eval -c tooling/spark/promptfooconfig.yaml -o tooling/spark/eval_<v>.json
promptfoo view            # interactive, or read the JSON / printed table
```

## Reading the proof
Per-prompt average of `reframe` (pass-rate) and `novelty` (0-1). A building block earns its place only if it beats `baseline` on these, repeatably. Prompts that don't are revised or dropped — the bank is curated by eval, not assertion.
