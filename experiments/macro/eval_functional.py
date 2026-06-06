"""Functional code generation smoke test.

Hand-curated trivial Python function prompts with test assertions.
NOT a benchmark — a smoke test that catches "composition broke generation"
without requiring HumanEval/MBPP infrastructure.

Protocol:
1. Feed function signature + docstring as prompt
2. Generate completion via mlx_lm.generate
3. Execute prompt + completion + assertions in subprocess (5s timeout)
4. Score: fraction of tests that pass
"""

import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass

PROBLEMS = [
    {
        "name": "add",
        "prompt": 'def add(a, b):\n    """Return a + b."""\n',
        "tests": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(0, 0) == 0",
    },
    {
        "name": "is_even",
        "prompt": 'def is_even(n):\n    """Return True if n is even."""\n',
        "tests": "assert is_even(2) is True\nassert is_even(3) is False\nassert is_even(0) is True",
    },
    {
        "name": "reverse_string",
        "prompt": 'def reverse_string(s):\n    """Return s reversed."""\n',
        "tests": 'assert reverse_string("hello") == "olleh"\nassert reverse_string("") == ""\nassert reverse_string("a") == "a"',
    },
    {
        "name": "max_of_two",
        "prompt": 'def max_of_two(a, b):\n    """Return the larger of a and b."""\n',
        "tests": "assert max_of_two(3, 5) == 5\nassert max_of_two(7, 2) == 7\nassert max_of_two(4, 4) == 4",
    },
    {
        "name": "absolute_value",
        "prompt": 'def absolute_value(n):\n    """Return the absolute value of n."""\n',
        "tests": "assert absolute_value(-5) == 5\nassert absolute_value(3) == 3\nassert absolute_value(0) == 0",
    },
    {
        "name": "factorial",
        "prompt": 'def factorial(n):\n    """Return n! for non-negative integer n."""\n',
        "tests": "assert factorial(0) == 1\nassert factorial(1) == 1\nassert factorial(5) == 120",
    },
    {
        "name": "count_vowels",
        "prompt": 'def count_vowels(s):\n    """Return the number of vowels (aeiou) in s, case-insensitive."""\n',
        "tests": 'assert count_vowels("hello") == 2\nassert count_vowels("AEIOU") == 5\nassert count_vowels("xyz") == 0',
    },
    {
        "name": "is_palindrome",
        "prompt": 'def is_palindrome(s):\n    """Return True if s reads the same forwards and backwards."""\n',
        "tests": 'assert is_palindrome("racecar") is True\nassert is_palindrome("hello") is False\nassert is_palindrome("") is True',
    },
    {
        "name": "fibonacci",
        "prompt": 'def fibonacci(n):\n    """Return the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1)."""\n',
        "tests": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55",
    },
    {
        "name": "list_sum",
        "prompt": 'def list_sum(lst):\n    """Return the sum of all elements in lst."""\n',
        "tests": "assert list_sum([1, 2, 3]) == 6\nassert list_sum([]) == 0\nassert list_sum([-1, 1]) == 0",
    },
    {
        "name": "remove_duplicates",
        "prompt": 'def remove_duplicates(lst):\n    """Return a new list with duplicates removed, preserving order."""\n',
        "tests": "assert remove_duplicates([1, 2, 2, 3]) == [1, 2, 3]\nassert remove_duplicates([]) == []\nassert remove_duplicates([1, 1, 1]) == [1]",
    },
    {
        "name": "clamp",
        "prompt": 'def clamp(value, lo, hi):\n    """Clamp value to the range [lo, hi]."""\n',
        "tests": "assert clamp(5, 0, 10) == 5\nassert clamp(-1, 0, 10) == 0\nassert clamp(15, 0, 10) == 10",
    },
    {
        "name": "capitalize_words",
        "prompt": 'def capitalize_words(s):\n    """Capitalize the first letter of each word."""\n',
        "tests": 'assert capitalize_words("hello world") == "Hello World"\nassert capitalize_words("a") == "A"',
    },
    {
        "name": "flatten",
        "prompt": 'def flatten(lst):\n    """Flatten a list of lists into a single list."""\n',
        "tests": "assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]\nassert flatten([[], [1]]) == [1]\nassert flatten([]) == []",
    },
    {
        "name": "gcd",
        "prompt": 'def gcd(a, b):\n    """Return the greatest common divisor of a and b."""\n',
        "tests": "assert gcd(12, 8) == 4\nassert gcd(7, 13) == 1\nassert gcd(0, 5) == 5",
    },
]


@dataclass
class SmokeTestResult:
    name: str
    passed: bool
    error: str | None = None
    generated_code: str = ""


def run_single_test(prompt: str, completion: str, tests: str, timeout: float = 5.0) -> tuple[bool, str | None]:
    """Execute prompt + completion + tests in a subprocess. Returns (passed, error)."""
    code = prompt + completion + "\n" + tests
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode == 0:
                return True, None
            return False, result.stderr[-500:] if result.stderr else "non-zero exit"
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, str(e)


def run_smoke_test(generate_fn, problems: list[dict] | None = None) -> dict:
    """Run the functional smoke test suite.

    Args:
        generate_fn: callable(prompt: str) -> str that generates a completion
        problems: optional override of the problem set

    Returns:
        dict with score, total, results list
    """
    problems = problems or PROBLEMS
    results = []
    for prob in problems:
        completion = generate_fn(prob["prompt"])
        passed, error = run_single_test(prob["prompt"], completion, prob["tests"])
        results.append(SmokeTestResult(
            name=prob["name"],
            passed=passed,
            error=error,
            generated_code=completion,
        ))

    passed = sum(1 for r in results if r.passed)
    return {
        "score": passed / len(results) if results else 0.0,
        "passed": passed,
        "total": len(results),
        "results": results,
    }


def print_smoke_report(report: dict):
    """Print a human-readable smoke test report."""
    print(f"\n{'=' * 50}")
    print("FUNCTIONAL SMOKE TEST")
    print(f"{'=' * 50}")
    print(f"\n{'Problem':<20} {'Result':>8}")
    print("-" * 30)
    for r in report["results"]:
        status = "PASS" if r.passed else "FAIL"
        print(f"  {r.name:<18} {status:>6}")
        if r.error:
            # Truncate error for display
            err_line = r.error.strip().split("\n")[-1][:60]
            print(f"    {err_line}")

    print(f"\nScore: {report['passed']}/{report['total']} ({report['score']:.0%})")
