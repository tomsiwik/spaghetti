#!/usr/bin/env python3
"""Proper throughput benchmark for BitNet + LoRA on llama.cpp.

Uses consistent prompts with fixed token counts.
"""
import re
import subprocess
import statistics
from pathlib import Path

LLAMA_CLI = "/tmp/llama-cpp-build/build-cpu/bin/llama-simple"
GGUF = str(Path(__file__).parent / "bitnet-2b-4t.gguf")
ADAPTERS = str(Path(__file__).parent / "adapters")
DOMAINS = ["math", "python", "legal", "medical", "creative"]

# Long prompt to get stable timing
PROMPT = "In the following analysis we will examine multiple aspects of modern computing, including hardware design, software architecture, programming paradigms, and the intersection of theory and practice in computer science."
N_PREDICT = 64
N_RUNS = 3


def run_bench(label, extra_args=None):
    """Run llama-cli with given args and extract speeds."""
    cmd = [
        LLAMA_CLI, "-m", GGUF,
        "-p", PROMPT,
        "-n", str(N_PREDICT),
        "-ngl", "0",
        "-s", "42",
        "--temp", "0",
    ]
    if extra_args:
        cmd.extend(extra_args)

    prompt_speeds = []
    gen_speeds = []

    for i in range(N_RUNS):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr

        pm = re.search(r"prompt eval time.*?(\d+\.\d+) tokens per second", output)
        gm = re.search(r"eval time.*?=\s+(\d+\.\d+) ms.*?(\d+\.\d+) tokens per second", output)

        if pm:
            prompt_speeds.append(float(pm.group(1)))
        if gm:
            gen_speeds.append(float(gm.group(2)))

    if prompt_speeds and gen_speeds:
        pp_avg = statistics.mean(prompt_speeds)
        pp_std = statistics.stdev(prompt_speeds) if len(prompt_speeds) > 1 else 0
        tg_avg = statistics.mean(gen_speeds)
        tg_std = statistics.stdev(gen_speeds) if len(gen_speeds) > 1 else 0
        print(f"  {label:25s}  pp: {pp_avg:6.1f} +/- {pp_std:4.1f} t/s  |  tg: {tg_avg:6.1f} +/- {tg_std:4.1f} t/s")
        return {"pp_avg": pp_avg, "pp_std": pp_std, "tg_avg": tg_avg, "tg_std": tg_std}
    else:
        print(f"  {label:25s}  FAILED")
        return None


print("BitNet + llama.cpp LoRA Throughput Benchmark")
print("=" * 70)
print(f"Prompt tokens: ~40, Generation tokens: {N_PREDICT}, Runs: {N_RUNS}")
print()

# Base
base = run_bench("Base (no adapter)")

# Single adapter
single = run_bench("1x LoRA (math)", ["--lora", f"{ADAPTERS}/math.gguf"])

# 3 adapters with 1/3 scaling
three = run_bench("3x LoRA (1/3 scale)", [
    "--lora-scaled", f"{ADAPTERS}/math.gguf:0.333",
    "--lora-scaled", f"{ADAPTERS}/python.gguf:0.333",
    "--lora-scaled", f"{ADAPTERS}/legal.gguf:0.333",
])

# 5 adapters with 1/5 scaling
five = run_bench("5x LoRA (1/5 scale)", [
    "--lora-scaled", f"{ADAPTERS}/math.gguf:0.2",
    "--lora-scaled", f"{ADAPTERS}/python.gguf:0.2",
    "--lora-scaled", f"{ADAPTERS}/legal.gguf:0.2",
    "--lora-scaled", f"{ADAPTERS}/medical.gguf:0.2",
    "--lora-scaled", f"{ADAPTERS}/creative.gguf:0.2",
])

print()
if base and single and five:
    single_oh = (1 - single["tg_avg"] / base["tg_avg"]) * 100
    five_oh = (1 - five["tg_avg"] / base["tg_avg"]) * 100
    print(f"Overhead vs base:")
    print(f"  1x LoRA: {single_oh:+.1f}%")
    if three:
        three_oh = (1 - three["tg_avg"] / base["tg_avg"]) * 100
        print(f"  3x LoRA: {three_oh:+.1f}%")
    print(f"  5x LoRA: {five_oh:+.1f}%")
    print(f"\nK2 verdict: {'PASS' if five_oh < 50 else 'FAIL'} (5x overhead = {five_oh:.1f}%, threshold = 50%)")
