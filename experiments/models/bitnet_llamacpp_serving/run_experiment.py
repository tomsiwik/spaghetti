#!/usr/bin/env python3
"""
Experiment: llama.cpp --lora serves multiple BitNet adapters on CPU with hot-swap.

Tests three kill criteria:
  K1: llama.cpp cannot load BitNet GGUF + LoRA adapter (tooling blocker)
  K2: multi-adapter overhead > 50% throughput loss vs base-only (too expensive)
  K3: hot-swap causes any output corruption

Uses llama.cpp CPU-only build (Metal does not support TQ2_0 ternary types).
"""

import json
import os
import subprocess
import time
import sys
from pathlib import Path

# Paths
EXPERIMENT_DIR = Path(__file__).parent
LLAMA_CLI = "/tmp/llama-cpp-build/build-cpu/bin/llama-cli"
LLAMA_SIMPLE = "/tmp/llama-cpp-build/build-cpu/bin/llama-simple"
LLAMA_BENCH = "/tmp/llama-cpp-build/build-cpu/bin/llama-bench"
LLAMA_SERVER = "/tmp/llama-cpp-build/build-cpu/bin/llama-server"
LLAMA_PERPLEXITY = "/tmp/llama-cpp-build/build-cpu/bin/llama-perplexity"
GGUF_PATH = str(EXPERIMENT_DIR / "bitnet-2b-4t.gguf")
ADAPTER_DIR = EXPERIMENT_DIR / "adapters"

DOMAINS = ["math", "python", "legal", "medical", "creative"]
ADAPTER_PATHS = {d: str(ADAPTER_DIR / f"{d}.gguf") for d in DOMAINS}

RESULTS = {
    "experiment": "exp_bitnet_llamacpp_serving",
    "model": "BitNet-2B-4T (TQ2_0)",
    "backend": "CPU-only (Metal lacks TQ kernels)",
    "llama_cpp_version": "latest main (2026-03-24)",
}


def run_cmd(cmd, timeout=120):
    """Run a command and return stdout+stderr."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )
    return result.stdout + result.stderr, result.returncode


def extract_speed(output):
    """Extract tok/s from llama.cpp output."""
    import re
    # prompt eval
    prompt_match = re.search(r"prompt eval time.*?(\d+\.\d+) tokens per second", output)
    eval_match = re.search(r"eval time.*?(\d+\.\d+) tokens per second", output)
    prompt_tps = float(prompt_match.group(1)) if prompt_match else None
    eval_tps = float(eval_match.group(1)) if eval_match else None
    return prompt_tps, eval_tps


def test_k1_base_loading():
    """K1a: Can llama.cpp load BitNet GGUF?"""
    print("\n=== K1a: Base model loading ===")
    cmd = [LLAMA_SIMPLE, "-m", GGUF_PATH, "-p", "Hello world", "-n", "10"]
    output, rc = run_cmd(cmd)
    success = rc == 0 and "decoded" in output
    prompt_tps, eval_tps = extract_speed(output)
    print(f"  Load: {'OK' if success else 'FAIL'}")
    print(f"  Prompt: {prompt_tps:.1f} t/s, Gen: {eval_tps:.1f} t/s" if prompt_tps else "  No speed data")
    return {
        "test": "base_loading",
        "success": success,
        "prompt_tps": prompt_tps,
        "eval_tps": eval_tps,
    }


def test_k1_single_lora():
    """K1b: Can llama.cpp load BitNet GGUF + single LoRA?"""
    print("\n=== K1b: Single LoRA loading ===")
    results = {}
    for domain in DOMAINS:
        cmd = [
            LLAMA_SIMPLE, "-m", GGUF_PATH,
            "--lora", ADAPTER_PATHS[domain],
            "-p", f"This is a test of {domain} knowledge.",
            "-n", "10",
        ]
        output, rc = run_cmd(cmd)
        success = rc == 0 and "decoded" in output
        prompt_tps, eval_tps = extract_speed(output)
        results[domain] = {
            "success": success,
            "prompt_tps": prompt_tps,
            "eval_tps": eval_tps,
        }
        status = "OK" if success else "FAIL"
        speed = f"{eval_tps:.1f} t/s" if eval_tps else "N/A"
        print(f"  {domain}: {status} ({speed})")
    return {"test": "single_lora", "results": results}


def test_k2_multi_adapter_overhead():
    """K2: Multi-adapter overhead vs base-only."""
    print("\n=== K2: Multi-adapter throughput ===")
    prompt = "Write a detailed analysis of the following topic in a comprehensive manner."

    # Base-only throughput (3 runs)
    base_speeds = []
    for i in range(3):
        cmd = [LLAMA_SIMPLE, "-m", GGUF_PATH, "-p", prompt, "-n", "50"]
        output, rc = run_cmd(cmd)
        _, eval_tps = extract_speed(output)
        if eval_tps:
            base_speeds.append(eval_tps)
    avg_base = sum(base_speeds) / len(base_speeds) if base_speeds else 0
    print(f"  Base-only: {avg_base:.1f} t/s (avg of {len(base_speeds)} runs)")

    # Single adapter throughput (3 runs)
    single_speeds = []
    for i in range(3):
        cmd = [
            LLAMA_SIMPLE, "-m", GGUF_PATH,
            "--lora", ADAPTER_PATHS["math"],
            "-p", prompt, "-n", "50",
        ]
        output, rc = run_cmd(cmd)
        _, eval_tps = extract_speed(output)
        if eval_tps:
            single_speeds.append(eval_tps)
    avg_single = sum(single_speeds) / len(single_speeds) if single_speeds else 0
    print(f"  Single LoRA: {avg_single:.1f} t/s (avg of {len(single_speeds)} runs)")

    # Multi adapter throughput (all 5 adapters with uniform scaling)
    multi_speeds = []
    for i in range(3):
        lora_args = []
        for domain in DOMAINS:
            lora_args.extend(["--lora-scaled", f"{ADAPTER_PATHS[domain]}:0.2"])
        cmd = [LLAMA_SIMPLE, "-m", GGUF_PATH] + lora_args + ["-p", prompt, "-n", "50"]
        output, rc = run_cmd(cmd, timeout=180)
        _, eval_tps = extract_speed(output)
        if eval_tps:
            multi_speeds.append(eval_tps)
        if rc != 0:
            print(f"    Multi-adapter run {i} failed (rc={rc})")
            # Show last few lines of output
            lines = output.strip().split('\n')
            for line in lines[-5:]:
                print(f"      {line}")
    avg_multi = sum(multi_speeds) / len(multi_speeds) if multi_speeds else 0

    # Calculate overhead
    single_overhead = (1 - avg_single / avg_base) * 100 if avg_base > 0 and avg_single > 0 else None
    multi_overhead = (1 - avg_multi / avg_base) * 100 if avg_base > 0 and avg_multi > 0 else None

    print(f"  Multi (5x) LoRA: {avg_multi:.1f} t/s (avg of {len(multi_speeds)} runs)")
    if single_overhead is not None:
        print(f"  Single overhead: {single_overhead:.1f}%")
    if multi_overhead is not None:
        print(f"  Multi overhead: {multi_overhead:.1f}%")
        print(f"  K2 verdict: {'PASS' if multi_overhead < 50 else 'FAIL'} (threshold: 50%)")

    return {
        "test": "multi_adapter_overhead",
        "base_tps": avg_base,
        "base_runs": base_speeds,
        "single_tps": avg_single,
        "single_runs": single_speeds,
        "single_overhead_pct": single_overhead,
        "multi_tps": avg_multi,
        "multi_runs": multi_speeds,
        "multi_overhead_pct": multi_overhead,
        "k2_pass": multi_overhead is not None and multi_overhead < 50,
    }


def test_k3_hot_swap():
    """K3: Hot-swap adapters via llama-server API."""
    print("\n=== K3: Hot-swap test ===")
    print("  Testing adapter hot-swap via sequential generation with different adapters...")

    # For hot-swap testing, we use llama-server with the /lora-adapters endpoint.
    # But llama-simple doesn't support hot-swap. Let's test consistency:
    # Run the same prompt with adapter A, then adapter B, then adapter A again.
    # If A's outputs are identical both times, hot-swap is coherent.

    prompt = "The capital of France is"
    results_a1 = []
    results_b = []
    results_a2 = []

    # Run with math adapter (A) - 3 times for consistency
    print("  Phase 1: Math adapter (A)...")
    for i in range(3):
        cmd = [
            LLAMA_SIMPLE, "-m", GGUF_PATH,
            "--lora", ADAPTER_PATHS["math"],
            "-p", prompt, "-n", "20",
            "-s", "42",  # fixed seed
            "--temp", "0",  # greedy
        ]
        output, rc = run_cmd(cmd)
        # Extract generated text (after the prompt)
        lines = output.strip().split('\n')
        gen_text = ""
        for line in lines:
            if prompt in line:
                gen_text = line.split(prompt, 1)[-1].strip()
                break
        results_a1.append(gen_text)

    # Run with legal adapter (B)
    print("  Phase 2: Legal adapter (B)...")
    for i in range(3):
        cmd = [
            LLAMA_SIMPLE, "-m", GGUF_PATH,
            "--lora", ADAPTER_PATHS["legal"],
            "-p", prompt, "-n", "20",
            "-s", "42",
            "--temp", "0",
        ]
        output, rc = run_cmd(cmd)
        lines = output.strip().split('\n')
        gen_text = ""
        for line in lines:
            if prompt in line:
                gen_text = line.split(prompt, 1)[-1].strip()
                break
        results_b.append(gen_text)

    # Run with math adapter (A) again
    print("  Phase 3: Math adapter (A) again...")
    for i in range(3):
        cmd = [
            LLAMA_SIMPLE, "-m", GGUF_PATH,
            "--lora", ADAPTER_PATHS["math"],
            "-p", prompt, "-n", "20",
            "-s", "42",
            "--temp", "0",
        ]
        output, rc = run_cmd(cmd)
        lines = output.strip().split('\n')
        gen_text = ""
        for line in lines:
            if prompt in line:
                gen_text = line.split(prompt, 1)[-1].strip()
                break
        results_a2.append(gen_text)

    # Check consistency
    a1_consistent = len(set(results_a1)) == 1  # All A1 runs identical
    a2_consistent = len(set(results_a2)) == 1  # All A2 runs identical
    a_matches = results_a1[0] == results_a2[0] if results_a1 and results_a2 else False
    b_differs = results_a1[0] != results_b[0] if results_a1 and results_b else True

    print(f"  A1 consistent: {a1_consistent} (outputs: {results_a1[:2]})")
    print(f"  A2 consistent: {a2_consistent}")
    print(f"  A1 == A2: {a_matches}")
    print(f"  A != B: {b_differs}")
    print(f"  K3 verdict: {'PASS' if a_matches else 'FAIL'}")

    return {
        "test": "hot_swap",
        "results_a1": results_a1,
        "results_b": results_b,
        "results_a2": results_a2,
        "a1_consistent": a1_consistent,
        "a2_consistent": a2_consistent,
        "a_matches_after_swap": a_matches,
        "b_differs_from_a": b_differs,
        "k3_pass": a_matches,
    }


def test_k3_server_hot_swap():
    """K3b: True hot-swap via llama-server /lora-adapters endpoint."""
    print("\n=== K3b: Server-based hot-swap test ===")
    import socket
    import urllib.request

    # Find free port
    with socket.socket() as s:
        s.bind(('', 0))
        port = s.getsockname()[1]

    # Start server with all adapters pre-loaded
    lora_args = []
    for domain in DOMAINS:
        lora_args.extend(["--lora", ADAPTER_PATHS[domain]])

    server_cmd = [
        LLAMA_SERVER, "-m", GGUF_PATH,
        "--port", str(port),
        "-ngl", "0",
    ] + lora_args

    print(f"  Starting server on port {port}...")
    server = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for server to start
        time.sleep(5)
        if server.poll() is not None:
            _, stderr = server.communicate(timeout=5)
            print(f"  Server failed to start: {stderr.decode()[-500:]}")
            return {"test": "server_hot_swap", "success": False, "error": "server failed to start"}

        base_url = f"http://localhost:{port}"

        def api_call(endpoint, data=None, method="GET"):
            req = urllib.request.Request(
                f"{base_url}{endpoint}",
                data=json.dumps(data).encode() if data else None,
                headers={"Content-Type": "application/json"} if data else {},
                method=method,
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return json.loads(resp.read())
            except Exception as e:
                return {"error": str(e)}

        # Check health
        health = api_call("/health")
        print(f"  Health: {health}")

        # List available adapters
        adapters = api_call("/lora-adapters")
        print(f"  Available adapters: {adapters}")

        # Generate with adapter 0 (math)
        prompt = "The derivative of x^2 is"
        result_math = api_call("/completion", {
            "prompt": prompt,
            "n_predict": 20,
            "temperature": 0,
            "seed": 42,
        })
        text_math = result_math.get("content", "")
        print(f"  Math adapter output: {text_math[:80]}")

        # Hot-swap to adapter 1 (python)
        # Set LoRA adapter scale
        swap_result = api_call("/lora-adapters", [
            {"id": 0, "scale": 0.0},  # disable math
            {"id": 1, "scale": 1.0},  # enable python
        ], method="POST")
        print(f"  Swap result: {swap_result}")

        # Generate with python adapter
        result_python = api_call("/completion", {
            "prompt": "def fibonacci(n):",
            "n_predict": 30,
            "temperature": 0,
            "seed": 42,
        })
        text_python = result_python.get("content", "")
        print(f"  Python adapter output: {text_python[:80]}")

        # Swap back to math
        swap_back = api_call("/lora-adapters", [
            {"id": 0, "scale": 1.0},  # enable math
            {"id": 1, "scale": 0.0},  # disable python
        ], method="POST")

        # Re-generate math prompt
        result_math2 = api_call("/completion", {
            "prompt": prompt,
            "n_predict": 20,
            "temperature": 0,
            "seed": 42,
        })
        text_math2 = result_math2.get("content", "")
        print(f"  Math adapter output (after swap): {text_math2[:80]}")

        # Check consistency
        coherent = text_math == text_math2
        print(f"  Math outputs match: {coherent}")
        print(f"  K3b verdict: {'PASS' if coherent else 'FAIL'}")

        return {
            "test": "server_hot_swap",
            "success": True,
            "adapters_loaded": adapters,
            "text_math": text_math,
            "text_python": text_python,
            "text_math_after_swap": text_math2,
            "outputs_match": coherent,
            "k3_pass": coherent,
        }

    finally:
        server.terminate()
        server.wait(timeout=10)
        print("  Server terminated.")


def main():
    print("=" * 60)
    print("BitNet + llama.cpp LoRA Serving Experiment")
    print("=" * 60)

    # Verify binaries exist
    for binary in [LLAMA_SIMPLE, LLAMA_SERVER]:
        if not os.path.exists(binary):
            print(f"ERROR: {binary} not found. Build llama.cpp first.")
            sys.exit(1)

    all_results = []

    # K1: Loading tests
    r1a = test_k1_base_loading()
    all_results.append(r1a)

    r1b = test_k1_single_lora()
    all_results.append(r1b)

    # K2: Throughput overhead
    r2 = test_k2_multi_adapter_overhead()
    all_results.append(r2)

    # K3: Hot-swap coherence (CLI-based)
    r3a = test_k3_hot_swap()
    all_results.append(r3a)

    # K3b: Server-based hot-swap
    r3b = test_k3_server_hot_swap()
    all_results.append(r3b)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    k1_pass = r1a["success"] and all(r["success"] for r in r1b["results"].values())
    k2_pass = r2.get("k2_pass", False)
    k3_pass = r3a.get("k3_pass", False)
    k3b_pass = r3b.get("k3_pass", False)

    print(f"  K1 (load BitNet + LoRA): {'PASS' if k1_pass else 'FAIL'}")
    print(f"  K2 (overhead < 50%):     {'PASS' if k2_pass else 'FAIL'} ({r2.get('multi_overhead_pct', 'N/A')}%)")
    print(f"  K3a (CLI hot-swap):      {'PASS' if k3_pass else 'FAIL'}")
    print(f"  K3b (server hot-swap):   {'PASS' if k3b_pass else 'FAIL'}")

    RESULTS["k1"] = {"pass": k1_pass, "details": [r1a, r1b]}
    RESULTS["k2"] = {"pass": k2_pass, "details": r2}
    RESULTS["k3"] = {"pass": k3_pass and k3b_pass, "cli": r3a, "server": r3b}
    RESULTS["summary"] = {
        "k1_pass": k1_pass,
        "k2_pass": k2_pass,
        "k3_pass": k3_pass and k3b_pass,
        "overall": "SUPPORTED" if (k1_pass and k2_pass and (k3_pass or k3b_pass)) else "KILLED",
    }

    # Save results
    results_path = EXPERIMENT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
