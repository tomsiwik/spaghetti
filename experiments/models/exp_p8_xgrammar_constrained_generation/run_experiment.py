#!/usr/bin/env python3
"""P8.A0: Grammar-Constrained Code Generation via Self-Repair.

Tests whether the think-then-constrain + self-repair protocol achieves:
  K1333: ≤2% syntax errors (self-repair N=3)
  K1334: Think-then-code accuracy ≥ direct - 5pp
  K1335: Grammar check overhead < 5% of latency

Implementation: MLX-native self-repair loop using ast.parse as grammar oracle.

Type: Verification (Theorem 1 proof: self-repair converges to near-0% errors).
Cite: arXiv:2411.15100 (XGrammar), arXiv:2601.07525 (Think-then-constrain)
"""

import ast
import gc
import json
import os
import time
import traceback
from pathlib import Path

import mlx.core as mx

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

# Code adapter path (Finding #421: HumanEval 63%)
CODE_ADAPTER_PATH = str(REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters/code")

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
MAX_TOKENS = 300
TEMPERATURE = 0.0  # Greedy — deterministic for reproducibility
N_RETRY = 3        # Self-repair attempts

# ── 20 Python generation problems ───────────────────────────────────────────

# Each problem: (description, function_sig, test_cases)
# test_cases: list of (args_str, expected_result)
PROBLEMS = [
    (
        "Return the sum of a list of integers.",
        "def sum_list(nums: list[int]) -> int:",
        [("[1, 2, 3]", 6), ("[]", 0), ("[-1, 1]", 0), ("[5]", 5)],
    ),
    (
        "Return True if a string is a palindrome, False otherwise.",
        "def is_palindrome(s: str) -> bool:",
        [('\"racecar\"', True), ('\"hello\"', False), ('\"\"', True), ('\"a\"', True)],
    ),
    (
        "Return the factorial of n (n >= 0).",
        "def factorial(n: int) -> int:",
        [("0", 1), ("1", 1), ("5", 120), ("6", 720)],
    ),
    (
        "Return the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1).",
        "def fib(n: int) -> int:",
        [("0", 0), ("1", 1), ("6", 8), ("10", 55)],
    ),
    (
        "Return the maximum element in a non-empty list.",
        "def list_max(nums: list[int]) -> int:",
        [("[3, 1, 4, 1, 5]", 5), ("[-1, -2, -3]", -1), ("[7]", 7)],
    ),
    (
        "Return a list with duplicate elements removed, preserving order.",
        "def deduplicate(lst: list) -> list:",
        [("[1, 2, 1, 3, 2]", [1, 2, 3]), ("[]", []), ("[1, 1, 1]", [1])],
    ),
    (
        "Return the number of vowels (a, e, i, o, u) in a string (case-insensitive).",
        "def count_vowels(s: str) -> int:",
        [('\"hello\"', 2), ('\"aeiou\"', 5), ('\"xyz\"', 0), ('\"\"', 0)],
    ),
    (
        "Return True if n is prime, False otherwise.",
        "def is_prime(n: int) -> bool:",
        [("2", True), ("3", True), ("4", False), ("17", True), ("1", False)],
    ),
    (
        "Reverse a string.",
        "def reverse_string(s: str) -> str:",
        [('\"hello\"', "olleh"), ('\"\"', ""), ('\"ab\"', "ba")],
    ),
    (
        "Return the GCD of two positive integers.",
        "def gcd(a: int, b: int) -> int:",
        [("12, 8", 4), ("100, 75", 25), ("7, 13", 1)],
    ),
    (
        "Flatten a nested list by one level.",
        "def flatten_one(lst: list[list]) -> list:",
        [("[[1, 2], [3, 4]]", [1, 2, 3, 4]), ("[]", []), ("[[1]]", [1])],
    ),
    (
        "Return the second largest unique element in a list of at least 2 unique values.",
        "def second_largest(nums: list[int]) -> int:",
        [("[1, 3, 2]", 2), ("[5, 5, 3]", 3), ("[9, 1]", 1)],
    ),
    (
        "Return the list sorted in ascending order without using built-in sort.",
        "def bubble_sort(lst: list[int]) -> list[int]:",
        [("[3, 1, 2]", [1, 2, 3]), ("[]", []), ("[1]", [1])],
    ),
    (
        "Return the number of occurrences of target in a list.",
        "def count_occurrences(lst: list, target) -> int:",
        [("[1, 2, 1, 3], 1", 2), ("[], 5", 0), ("[5, 5, 5], 5", 3)],
    ),
    (
        "Return True if a number is a perfect square.",
        "def is_perfect_square(n: int) -> bool:",
        [("0", True), ("1", True), ("4", True), ("9", True), ("3", False), ("8", False)],
    ),
    (
        "Compute the product of all elements in a list. Return 1 for empty list.",
        "def list_product(lst: list[int]) -> int:",
        [("[1, 2, 3, 4]", 24), ("[]", 1), ("[0, 5]", 0)],
    ),
    (
        "Return a string with all whitespace removed.",
        "def remove_whitespace(s: str) -> str:",
        [('\"hello world\"', "helloworld"), ('\" \"', ""), ('\"no spaces\"', "nospaces")],
    ),
    (
        "Return True if a list is sorted in non-decreasing order.",
        "def is_sorted(lst: list[int]) -> bool:",
        [("[1, 2, 3]", True), ("[3, 1, 2]", False), ("[]", True), ("[1, 1, 2]", True)],
    ),
    (
        "Convert a temperature from Celsius to Fahrenheit: F = C * 9/5 + 32.",
        "def celsius_to_fahrenheit(c: float) -> float:",
        [("0", 32.0), ("100", 212.0), ("37", 98.6)],
    ),
    (
        "Return the index of the first occurrence of target in lst, or -1 if not found.",
        "def find_index(lst: list, target) -> int:",
        [("[1, 2, 3], 2", 1), ("[1, 2, 3], 5", -1), ("[], 1", -1)],
    ),
]

if IS_SMOKE:
    PROBLEMS = PROBLEMS[:3]
    N_RETRY = 2


def log(m):
    print(m, flush=True)


def cleanup():
    gc.collect()
    mx.clear_cache()


def build_direct_prompt(desc, sig):
    """Build a direct generation prompt (P0)."""
    return f"""<start_of_turn>user
Write a Python function with the following signature and behavior.
Return ONLY the function definition (no explanation, no test code, no ``` markers).

{sig}
    \"\"\"{desc}\"\"\"
<end_of_turn>
<start_of_turn>model
{sig}
"""


def build_think_prompt(desc, sig):
    """Build a think-then-code prompt (P1): forces reasoning before code."""
    return f"""<start_of_turn>user
Write a Python function with the following signature and behavior.
First, briefly reason about the algorithm, then write the function.
Return ONLY the function definition (no explanation, no test code, no ``` markers).

{sig}
    \"\"\"{desc}\"\"\"
<end_of_turn>
<start_of_turn>model
# Reasoning: """


def build_repair_prompt(desc, sig, broken_code, error_msg):
    """Build a self-repair prompt with the syntax error feedback."""
    return f"""<start_of_turn>user
The following Python function has a syntax error. Fix it.
Return ONLY the corrected function definition (no explanation, no ``` markers).

Original request: {sig}
    \"\"\"{desc}\"\"\"

Broken code:
{broken_code}

Syntax error: {error_msg}
<end_of_turn>
<start_of_turn>model
{sig}
"""


def extract_function(text: str, sig: str) -> str:
    """Extract the function body from generated text."""
    # Find the function definition line
    fn_name = sig.split("(")[0].replace("def ", "").strip()

    lines = text.split("\n")

    # Find where the function starts
    start_idx = None
    for i, line in enumerate(lines):
        if f"def {fn_name}" in line:
            start_idx = i
            break

    if start_idx is None:
        # Try to return what we have, cleaning up markdown
        cleaned = text.replace("```python", "").replace("```", "").strip()
        return cleaned

    # Collect function lines (stop at next top-level def or class)
    fn_lines = []
    for i in range(start_idx, len(lines)):
        line = lines[i]
        if i > start_idx and line and not line[0].isspace() and line.strip():
            if line.startswith("def ") or line.startswith("class "):
                break
        fn_lines.append(line)

    result = "\n".join(fn_lines).strip()
    if not result:
        return text.strip()
    return result


def check_syntax(code: str) -> tuple[bool, str]:
    """Check Python syntax. Returns (is_valid, error_message)."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def run_tests(code: str, test_cases: list) -> tuple[int, int]:
    """Execute function against test cases. Returns (passed, total)."""
    passed = 0
    total = len(test_cases)

    for args_str, expected in test_cases:
        try:
            # Build test execution
            exec_globals = {}
            exec(code, exec_globals)

            # Get function name from code
            tree = ast.parse(code)
            fn_name = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    fn_name = node.name
                    break

            if fn_name is None:
                continue

            result = eval(f"{fn_name}({args_str})", exec_globals)

            # Handle float comparison (Celsius test)
            if isinstance(expected, float):
                if abs(result - expected) < 0.1:
                    passed += 1
            else:
                if result == expected:
                    passed += 1
        except Exception:
            pass

    return passed, total


def generate_code(model, tokenizer, prompt: str, max_tokens: int = MAX_TOKENS) -> tuple[str, float]:
    """Generate code using mlx_lm.generate. Returns (text, latency_ms)."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=TEMPERATURE)  # TEMPERATURE=0.0 → greedy argmax

    t0 = time.perf_counter()
    output = generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
        sampler=sampler,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    return output, latency_ms


def evaluate_mode(model, tokenizer, mode: str, adapter_name: str) -> dict:
    """Evaluate one generation mode on all problems."""
    results = {
        "mode": mode,
        "adapter": adapter_name,
        "problems": [],
    }

    total_syntax_errors = 0
    total_tests_passed = 0
    total_tests = 0
    total_latency_ms = 0.0
    total_check_time_ms = 0.0
    total_gen_time_ms = 0.0
    total_retries = 0

    for i, (desc, sig, test_cases) in enumerate(PROBLEMS):
        log(f"  [{i+1}/{len(PROBLEMS)}] {sig[:50]}...")

        if mode == "P0_direct":
            prompt = build_direct_prompt(desc, sig)
        elif mode == "P1_think":
            prompt = build_think_prompt(desc, sig)
        elif mode == "P2_repair":
            prompt = build_direct_prompt(desc, sig)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Generation
        t_total_start = time.perf_counter()
        code_raw, gen_ms = generate_code(model, tokenizer, prompt)
        code = extract_function(code_raw, sig)

        # Syntax check
        t_check = time.perf_counter()
        valid, err = check_syntax(code)
        check_ms = (time.perf_counter() - t_check) * 1000

        n_retries = 0
        if not valid and mode == "P2_repair":
            # Self-repair loop
            for retry in range(1, N_RETRY):
                repair_prompt = build_repair_prompt(desc, sig, code, err)
                code_raw2, gen_ms2 = generate_code(model, tokenizer, repair_prompt)
                code_candidate = extract_function(code_raw2, sig)
                gen_ms += gen_ms2

                t_check2 = time.perf_counter()
                valid2, err2 = check_syntax(code_candidate)
                check_ms += (time.perf_counter() - t_check2) * 1000

                n_retries += 1
                if valid2:
                    code = code_candidate
                    valid = True
                    break
                code = code_candidate
                err = err2

        total_latency_ms += (time.perf_counter() - t_total_start) * 1000
        total_gen_time_ms += gen_ms
        total_check_time_ms += check_ms
        total_retries += n_retries

        if not valid:
            total_syntax_errors += 1

        # Run tests
        tests_passed, tests_total = run_tests(code, test_cases) if valid else (0, len(test_cases))
        total_tests_passed += tests_passed
        total_tests += tests_total

        results["problems"].append({
            "idx": i,
            "sig": sig[:50],
            "syntax_valid": valid,
            "syntax_error": err if not valid else "",
            "tests_passed": tests_passed,
            "tests_total": tests_total,
            "gen_ms": round(gen_ms, 1),
            "check_ms": round(check_ms, 3),
            "n_retries": n_retries,
        })

        log(f"    syntax={valid} tests={tests_passed}/{tests_total} gen={gen_ms:.0f}ms retries={n_retries}")

    n = len(PROBLEMS)
    syntax_error_rate = total_syntax_errors / n
    test_pass_rate = total_tests_passed / total_tests if total_tests else 0
    check_overhead_pct = (total_check_time_ms / total_gen_time_ms * 100) if total_gen_time_ms > 0 else 0

    results["summary"] = {
        "n_problems": n,
        "n_syntax_errors": total_syntax_errors,
        "syntax_error_rate": round(syntax_error_rate, 4),
        "syntax_valid_rate": round(1 - syntax_error_rate, 4),
        "tests_passed": total_tests_passed,
        "tests_total": total_tests,
        "test_pass_rate": round(test_pass_rate, 4),
        "total_latency_ms": round(total_latency_ms, 1),
        "avg_latency_ms": round(total_latency_ms / n, 1),
        "check_overhead_pct": round(check_overhead_pct, 3),
        "total_retries": total_retries,
        "avg_retries_per_problem": round(total_retries / n, 2),
    }

    log(f"  SUMMARY: syntax_errors={total_syntax_errors}/{n} ({syntax_error_rate:.1%}) "
        f"tests={test_pass_rate:.1%} check_overhead={check_overhead_pct:.3f}%")

    return results


def main():
    from mlx_lm import load

    # Memory setup (MLX best practice)
    _dev = mx.device_info()
    mx.set_memory_limit(_dev["memory_size"] - 8 * 1024**3)
    mx.set_cache_limit(2 * 1024**3)

    results = {
        "is_smoke": IS_SMOKE,
        "model": MODEL_ID,
        "code_adapter": CODE_ADAPTER_PATH,
        "n_problems": len(PROBLEMS),
        "n_retry": N_RETRY,
        "phases": {},
    }

    # ── Phase 1: Base model ───────────────────────────────────────────────────
    log("=" * 60)
    log("Phase 1: Base model (no adapter)")
    log("=" * 60)

    log("Loading base model...")
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    log("[MEM post-load] "
        f"active={mx.get_active_memory()/1e9:.2f}GB "
        f"peak={mx.get_peak_memory()/1e9:.2f}GB")

    phase1_results = {}
    for mode in ["P0_direct", "P1_think", "P2_repair"]:
        log(f"\n  Mode: {mode}")
        phase1_results[mode] = evaluate_mode(model, tokenizer, mode, "base")
        cleanup()

    results["phases"]["phase1_base"] = phase1_results
    del model, tokenizer
    cleanup()
    log(f"[MEM post-phase1] active={mx.get_active_memory()/1e9:.2f}GB")

    # ── Phase 2: Code adapter ─────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("Phase 2: Code adapter (code-codealpaca-knowledge-v0)")
    log("=" * 60)

    # Check adapter exists
    adapter_path = Path(CODE_ADAPTER_PATH)
    if not adapter_path.exists():
        log(f"  WARNING: Adapter not found at {adapter_path}")
        log("  Skipping phase 2 (adapter unavailable)")
        results["phases"]["phase2_code_adapter"] = {"error": f"adapter not found: {adapter_path}"}
    else:
        log(f"Loading model with adapter: {adapter_path}")
        model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
        mx.eval(model.parameters())
        log(f"[MEM post-load] active={mx.get_active_memory()/1e9:.2f}GB")

        phase2_results = {}
        for mode in ["P0_direct", "P1_think", "P2_repair"]:
            log(f"\n  Mode: {mode}")
            phase2_results[mode] = evaluate_mode(model, tokenizer, mode, "code-adapter")
            cleanup()

        results["phases"]["phase2_code_adapter"] = phase2_results
        del model, tokenizer
        cleanup()
        log(f"[MEM post-phase2] active={mx.get_active_memory()/1e9:.2f}GB")

    # ── Kill Criteria ─────────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("KILL CRITERIA EVALUATION")
    log("=" * 60)

    # K1333: Self-repair (N=3) achieves ≤2% syntax errors
    # Use best result (code adapter P2 if available, else base P2)
    best_repair = None
    for phase_key in ["phase2_code_adapter", "phase1_base"]:
        p = results["phases"].get(phase_key, {})
        if isinstance(p, dict) and "P2_repair" in p:
            best_repair = p["P2_repair"]["summary"]
            break

    k1333_pass = None
    if best_repair:
        error_rate = best_repair["syntax_error_rate"]
        k1333_pass = error_rate <= 0.02
        log(f"K1333 (syntax_error_rate ≤ 2%): {'PASS' if k1333_pass else 'FAIL'} — {error_rate:.1%}")

    # K1334: Think-then-code accuracy ≥ direct - 5pp
    k1334_pass = None
    for phase_key in ["phase2_code_adapter", "phase1_base"]:
        p = results["phases"].get(phase_key, {})
        if isinstance(p, dict) and "P0_direct" in p and "P1_think" in p:
            direct_acc = p["P0_direct"]["summary"]["test_pass_rate"]
            think_acc = p["P1_think"]["summary"]["test_pass_rate"]
            k1334_pass = think_acc >= direct_acc - 0.05
            log(f"K1334 (think_acc ≥ direct_acc - 5pp): {'PASS' if k1334_pass else 'FAIL'} "
                f"— think={think_acc:.1%} direct={direct_acc:.1%} delta={think_acc - direct_acc:+.1%}")
            break

    # K1335: Grammar check overhead < 5%
    k1335_pass = None
    if best_repair:
        overhead = best_repair["check_overhead_pct"]
        k1335_pass = overhead < 5.0
        log(f"K1335 (check_overhead < 5%): {'PASS' if k1335_pass else 'FAIL'} — {overhead:.3f}%")

    results["kill_criteria"] = {
        "k1333_syntax_error_rate": best_repair["syntax_error_rate"] if best_repair else None,
        "k1333_pass": k1333_pass,
        "k1334_pass": k1334_pass,
        "k1335_check_overhead_pct": best_repair["check_overhead_pct"] if best_repair else None,
        "k1335_pass": k1335_pass,
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults written to {RESULTS_FILE}")

    # Final summary
    log("\n" + "=" * 60)
    log("FINAL VERDICT")
    log("=" * 60)
    log(f"K1333 (≤2% syntax errors after repair): {'PASS' if k1333_pass else 'FAIL' if k1333_pass is not None else 'N/A'}")
    log(f"K1334 (think ≥ direct - 5pp): {'PASS' if k1334_pass else 'FAIL' if k1334_pass is not None else 'N/A'}")
    log(f"K1335 (overhead < 5%): {'PASS' if k1335_pass else 'FAIL' if k1335_pass is not None else 'N/A'}")


if __name__ == "__main__":
    main()
