#!/usr/bin/env python3
"""Throughput benchmark using llama-server API for proper measurement.

Tests base, 1x, 3x, 5x adapter configurations.
"""
import json
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

LLAMA_SERVER = "/tmp/llama-cpp-build/build-cpu/bin/llama-server"
GGUF = str(Path(__file__).parent / "bitnet-2b-4t.gguf")
ADAPTERS = Path(__file__).parent / "adapters"
DOMAINS = ["math", "python", "legal", "medical", "creative"]

PROMPT = "In the following analysis we will examine multiple aspects of modern computing, including hardware design, software architecture, programming paradigms, and the intersection of theory and practice in computer science."
N_PREDICT = 64
N_RUNS = 3


def find_port():
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def start_server(lora_args=None, port=None):
    if port is None:
        port = find_port()
    cmd = [LLAMA_SERVER, "-m", GGUF, "--port", str(port), "-ngl", "0"]
    if lora_args:
        cmd.extend(lora_args)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Wait for server
    for _ in range(30):
        time.sleep(1)
        if proc.poll() is not None:
            _, err = proc.communicate(timeout=5)
            print(f"  Server died: {err.decode()[-300:]}")
            return None, port
        try:
            req = urllib.request.Request(f"http://localhost:{port}/health")
            with urllib.request.urlopen(req, timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return proc, port
        except:
            continue
    return proc, port


def api_completion(port, prompt, n_predict, seed=42):
    data = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "seed": seed,
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/completion",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    elapsed = time.perf_counter() - t0

    # Extract timing from result
    timings = result.get("timings", {})
    return {
        "prompt_tps": timings.get("prompt_per_second", 0),
        "gen_tps": timings.get("predicted_per_second", 0),
        "prompt_n": timings.get("prompt_n", 0),
        "predicted_n": timings.get("predicted_n", 0),
        "content": result.get("content", ""),
        "wall_time": elapsed,
    }


def set_adapters(port, adapter_scales):
    """Set adapter scales via API. adapter_scales is list of (id, scale) tuples."""
    data = json.dumps([{"id": aid, "scale": scale} for aid, scale in adapter_scales]).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/lora-adapters",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def run_bench(label, port, n_runs=N_RUNS):
    """Run N completions and report stats."""
    prompt_speeds = []
    gen_speeds = []
    for i in range(n_runs):
        r = api_completion(port, PROMPT, N_PREDICT, seed=42 + i)
        if r["gen_tps"] > 0:
            prompt_speeds.append(r["prompt_tps"])
            gen_speeds.append(r["gen_tps"])

    if prompt_speeds and gen_speeds:
        pp = statistics.mean(prompt_speeds)
        pp_std = statistics.stdev(prompt_speeds) if len(prompt_speeds) > 1 else 0
        tg = statistics.mean(gen_speeds)
        tg_std = statistics.stdev(gen_speeds) if len(gen_speeds) > 1 else 0
        print(f"  {label:25s}  pp: {pp:7.1f} +/- {pp_std:5.1f} t/s  |  tg: {tg:6.1f} +/- {tg_std:4.1f} t/s")
        return {"pp": pp, "pp_std": pp_std, "tg": tg, "tg_std": tg_std}
    else:
        print(f"  {label:25s}  FAILED")
        return None


def main():
    print("BitNet + llama.cpp LoRA Server Throughput Benchmark")
    print("=" * 70)
    print(f"Generation tokens: {N_PREDICT}, Runs per config: {N_RUNS}")
    print()

    results = {}

    # === Config 1: All 5 adapters loaded, test different active configs ===
    lora_args = []
    for d in DOMAINS:
        lora_args.extend(["--lora", str(ADAPTERS / f"{d}.gguf")])

    print("Starting server with 5 adapters pre-loaded...")
    proc, port = start_server(lora_args)
    if proc is None:
        print("Failed to start server with adapters")
        sys.exit(1)

    try:
        # Disable all adapters (base-only)
        set_adapters(port, [(i, 0.0) for i in range(5)])
        results["base"] = run_bench("Base (all disabled)", port)

        # Enable 1 adapter
        set_adapters(port, [(0, 1.0)] + [(i, 0.0) for i in range(1, 5)])
        results["1x"] = run_bench("1x LoRA (math)", port)

        # Enable 3 adapters with 1/3 scaling
        scales = [(0, 0.333), (1, 0.333), (2, 0.333)] + [(i, 0.0) for i in range(3, 5)]
        set_adapters(port, scales)
        results["3x"] = run_bench("3x LoRA (1/3 scale)", port)

        # Enable 5 adapters with 1/5 scaling
        set_adapters(port, [(i, 0.2) for i in range(5)])
        results["5x"] = run_bench("5x LoRA (1/5 scale)", port)

    finally:
        proc.terminate()
        proc.wait(timeout=10)

    # === Config 2: Base-only server (no adapters loaded) ===
    print("\nStarting server with NO adapters (true baseline)...")
    proc2, port2 = start_server()
    if proc2:
        try:
            results["true_base"] = run_bench("True base (no adapters)", port2)
        finally:
            proc2.terminate()
            proc2.wait(timeout=10)

    # === Summary ===
    print()
    print("=" * 70)
    print("OVERHEAD ANALYSIS")
    print("=" * 70)

    base_tg = results.get("base", {}).get("tg", 0) if results.get("base") else 0
    true_base_tg = results.get("true_base", {}).get("tg", 0) if results.get("true_base") else 0
    ref_tg = true_base_tg if true_base_tg > 0 else base_tg

    for label, key in [("True base", "true_base"), ("Base (adapters loaded)", "base"),
                       ("1x LoRA", "1x"), ("3x LoRA", "3x"), ("5x LoRA", "5x")]:
        r = results.get(key)
        if r and ref_tg > 0:
            oh = (1 - r["tg"] / ref_tg) * 100
            print(f"  {label:30s}  {r['tg']:6.1f} t/s  ({oh:+6.1f}% vs true base)")

    five_r = results.get("5x")
    if five_r and ref_tg > 0:
        oh = (1 - five_r["tg"] / ref_tg) * 100
        print(f"\n  K2 verdict: {'PASS' if oh < 50 else 'FAIL'} (5x overhead = {oh:.1f}%, threshold = 50%)")

    # Save results
    out_path = Path(__file__).parent / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
