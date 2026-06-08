#!/usr/bin/env python
"""Complementary-model spark pipeline (roles assigned from the cross-model data):
  Stage 1 EXPLORE     -> Gemini   (widest diversity)
  Stage 2 SYNTHESIZE  -> Opus 4.8 (novelty + applied scientific foundation)
  Stage 3 SYSTEMATIZE -> GPT-5.5  (strict, systematic, proof-first rigor)

Each stage shells out to the real model's CLI. Generates testable, breakthrough-shaped
hunches that push PAST an established finding.

Usage: python pipeline.py <target_context_file> [out_dir]
"""
import subprocess, sys, os, tempfile

target = open(sys.argv[1]).read().strip()
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "pipeline_out")
os.makedirs(out, exist_ok=True)

RECIPE = ("Breakthrough recipe (this lab's own, mined from its best results): "
          "(1) invert a hidden field axiom — name the unspoken assumption every paper shares and ask 'what if the opposite carries the information?'; "
          "(2) import a distant primitive as an INVARIANT or IMPOSSIBILITY, not a faster-router gadget (reject data-structure-as-router imports); "
          "(3) relocate a known primitive onto an unused axis — depth/iteration, time/version, decode-step, frequency; "
          "(4) a clean impossibility-kill that yields a reusable theorem.")

def gemini(prompt):
    r = subprocess.run(["gemini", "-p", prompt, "-o", "text", "--skip-trust"],
                       capture_output=True, text=True, timeout=400)
    return "\n".join(l for l in r.stdout.splitlines()
                     if not l.startswith(("Ripgrep", "Falling back", "Loaded cached"))).strip()

def opus(prompt):
    r = subprocess.run(["claude", "-p", "--model", "opus"], input=prompt,
                       capture_output=True, text=True, timeout=400)
    return r.stdout.strip()

def gpt(prompt):
    f = tempfile.mktemp(suffix=".txt")
    subprocess.run(["codex", "exec", "-m", "gpt-5.5", "--skip-git-repo-check",
                    "-s", "read-only", "--output-last-message", f, prompt],
                   capture_output=True, text=True, timeout=500)
    return open(f).read().strip() if os.path.exists(f) else "(codex produced no output)"

# Stage 1 — EXPLORE (Gemini)
print(">>> Stage 1: Gemini exploring...", file=sys.stderr)
explore_prompt = f"""You are an exploration engine. Here is an established, surprising, PROVEN result:

{target}

This is a launch point, NOT a destination. Generate 8 DISTINCT, non-obvious hunches that push PAST it — new directions it opens that nobody has tried. {RECIPE}
Be bold and diverse; do NOT restate the result and do NOT propose a faster/better router. One line each, numbered 1-8."""
s1 = gemini(explore_prompt); open(f"{out}/1_explore.txt", "w").write(s1)

# Stage 2 — SYNTHESIZE (Opus)
print(">>> Stage 2: Opus synthesizing...", file=sys.stderr)
synth_prompt = f"""Proven result:

{target}

Raw hunches generated to push past it (from a divergent explorer):

{s1}

{RECIPE}

Select and MERGE these into the 3 most INGENIOUS AND scientifically-foundable candidate hypotheses, for a frozen Gemma-4 + LoRA-adapter research setup. Each must be a genuine reframe (NOT a faster router), and grounded enough to actually implement. For each, give: HUNCH (one line) / MECHANISM (why it should hold) / WHY NON-OBVIOUS. Discard the boring ones. Number them 1-3."""
s2 = opus(synth_prompt); open(f"{out}/2_synthesize.txt", "w").write(s2)

# Stage 3 — SYSTEMATIZE (GPT-5.5)
print(">>> Stage 3: GPT-5.5 systematizing...", file=sys.stderr)
sys_prompt = f"""Three candidate research hypotheses:

{s2}

For EACH, impose rigor in a strict proof-first style: (1) a falsifiable one-line CLAIM; (2) MECHANISM / why it should hold; (3) a pre-registered KILL criterion using a TARGET behavioral metric (NOT a proxy like PPL or cosine); (4) the MINIMAL experiment that tests it; (5) what existing math/result it builds on. Be strict and systematic; explicitly FLAG any hypothesis that cannot be made falsifiable. Number them 1-3."""
s3 = gpt(sys_prompt); open(f"{out}/3_systematize.txt", "w").write(s3)

print("\n\n========== STAGE 3 — SYSTEMATIZED, TESTABLE HUNCHES (GPT-5.5) ==========\n")
print(s3)
print(f"\n(intermediate explore/synthesize stages saved in {out}/)")
