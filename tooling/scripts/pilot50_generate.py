#!/usr/bin/env python3
"""Generate training data for all 50 domains via Groq API.

Runs locally. Calls Groq API with Llama 3.3 70B to generate
1000 instruction-response pairs per domain.

Uses concurrent requests within each domain for speed.
Resumes from existing data (idempotent).

Usage:
    python scripts/pilot50_generate.py [--n-examples 1000] [--workers 8]
"""

import argparse
import json
import os
import sys
import time
import re
import threading
import concurrent.futures
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

import yaml

# ── Config ────────────────────────────────────────────────────────────
DOMAINS_FILE = REPO_ROOT / "data" / "distillation" / "domains.yml"
OUTPUT_DIR = REPO_ROOT / "data" / "distillation"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_API_KEY_ENV = "GROQ_API_KEY"

SYSTEM_PROMPT = """You are an expert training data generator. Given a domain description,
generate a high-quality instruction-response pair for fine-tuning a language model.

The instruction should be a realistic user question or task in this domain.
The response should be thorough, correct, and demonstrate domain expertise.
Vary the difficulty: mix simple factual questions with complex multi-step problems.

Output ONLY valid JSON with exactly two fields:
{"instruction": "...", "response": "..."}"""


def generate_one(client, model: str, domain: str, description: str, temperature: float = 0.9):
    """Generate a single instruction-response pair."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Domain: {domain}\nDescription: {description}\n\nGenerate one training example."},
            ],
            temperature=temperature,
            max_tokens=1024,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        text = text.replace("\t", "\\t")
        text = text.replace("\r", "\\r")
        text = re.sub(r'(?<!\\)\n', '\\\\n', text)
        pair = json.loads(text)
        if "instruction" in pair and "response" in pair:
            return pair
    except (json.JSONDecodeError, KeyError, IndexError):
        pass
    except Exception as e:
        if "rate_limit" in str(e).lower() or "429" in str(e):
            raise
    return None


def generate_batch(client, model, domain, description, count):
    """Generate a batch of examples concurrently for one domain."""
    results = []
    for _ in range(count):
        try:
            pair = generate_one(client, model, domain, description)
            if pair:
                results.append(pair)
        except Exception:
            time.sleep(2)
    return results


def generate_domain(client, model: str, domain: str, description: str,
                    n_examples: int, output_dir: Path, workers: int = 8):
    """Generate training data for one domain with concurrent workers."""
    out_file = output_dir / domain / "train.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    existing = 0
    if out_file.exists():
        with open(out_file) as f:
            existing = sum(1 for _ in f)
        if existing >= n_examples:
            return existing, 0.0

    needed = n_examples - existing
    print(f"  {domain}: generating {needed} examples (have {existing})...", flush=True)
    start = time.time()
    generated = 0
    retries = 0
    max_retries = 5
    lock = threading.Lock()

    with open(out_file, "a") as f:
        while generated < needed and retries < max_retries:
            batch_size = min(needed - generated, workers * 10)
            # Split into worker chunks
            chunk_size = max(1, batch_size // workers)

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = []
                for _ in range(min(workers, batch_size)):
                    futures.append(executor.submit(
                        generate_batch, client, model, domain, description, chunk_size))

                for future in concurrent.futures.as_completed(futures):
                    try:
                        pairs = future.result(timeout=300)
                        for pair in pairs:
                            record = {
                                "messages": [
                                    {"role": "user", "content": pair["instruction"]},
                                    {"role": "assistant", "content": pair["response"]},
                                ]
                            }
                            with lock:
                                if existing + generated < n_examples:
                                    f.write(json.dumps(record) + "\n")
                                    f.flush()
                                    generated += 1
                    except Exception as e:
                        if "429" in str(e) or "rate_limit" in str(e).lower():
                            retries += 1
                            wait = min(60, 2 ** retries)
                            print(f"    {domain}: rate limited, waiting {wait}s...", flush=True)
                            time.sleep(wait)

            if generated > 0 and generated % 200 == 0:
                elapsed = time.time() - start
                rate = generated / elapsed
                print(f"    {domain}: {existing + generated}/{n_examples} ({rate:.1f}/s)", flush=True)

    total = existing + generated
    elapsed = time.time() - start
    est_cost = generated * (200 * 0.59 + 300 * 0.79) / 1_000_000
    rate = generated / max(elapsed, 1)
    print(f"  {domain}: {total}/{n_examples} complete ({elapsed:.0f}s, {rate:.1f}/s, ~${est_cost:.3f})", flush=True)
    return total, est_cost


def main():
    parser = argparse.ArgumentParser(description="Generate pilot 50 training data via Groq")
    parser.add_argument("--n-examples", type=int, default=1000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=8, help="Concurrent API requests per domain")
    parser.add_argument("--domains", nargs="*", help="Specific domains to generate")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(DOMAINS_FILE) as f:
        all_domains = yaml.safe_load(f.read())

    if args.domains:
        domains = {k: v for k, v in all_domains.items() if k in args.domains}
    else:
        domains = all_domains

    needs_gen = {}
    for domain, desc in domains.items():
        fpath = OUTPUT_DIR / domain / "train.jsonl"
        existing = 0
        if fpath.exists():
            with open(fpath) as f:
                existing = sum(1 for _ in f)
        if existing < args.n_examples:
            needs_gen[domain] = (desc, existing)

    print(f"50-Domain Distillation Data Generation")
    print(f"  Total domains: {len(domains)}")
    print(f"  Need generation: {len(needs_gen)}")
    print(f"  Already complete: {len(domains) - len(needs_gen)}")
    print(f"  Examples/domain: {args.n_examples}")
    print(f"  Workers: {args.workers}")
    print(f"  Model: {args.model}")
    remaining = sum(args.n_examples - ex for _, ex in needs_gen.values())
    est_cost = remaining * (200 * 0.59 + 300 * 0.79) / 1_000_000
    print(f"  Remaining examples: {remaining}")
    print(f"  Estimated cost: ${est_cost:.2f}")
    print(flush=True)

    if args.dry_run:
        for domain, (desc, existing) in sorted(needs_gen.items()):
            print(f"  {domain}: {existing}/{args.n_examples}")
        return

    api_key = os.environ.get(DEFAULT_API_KEY_ENV, "")
    if not api_key:
        print(f"ERROR: Set {DEFAULT_API_KEY_ENV} environment variable")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=DEFAULT_BASE_URL)

    total_cost = 0.0
    for domain in sorted(domains.keys()):
        desc = domains[domain]
        _, cost = generate_domain(client, args.model, domain, desc.strip(),
                                  args.n_examples, OUTPUT_DIR, workers=args.workers)
        total_cost += cost

    print(f"\n{'='*60}")
    print(f"Generation complete. Estimated total cost: ${total_cost:.2f}")
    print(f"\nData summary:")
    total_examples = 0
    for domain in sorted(domains.keys()):
        fpath = OUTPUT_DIR / domain / "train.jsonl"
        n = 0
        if fpath.exists():
            with open(fpath) as f:
                n = sum(1 for _ in f)
        total_examples += n
        status = "OK" if n >= args.n_examples else f"INCOMPLETE ({n})"
        print(f"  {domain}: {n} [{status}]")
    print(f"\nTotal: {total_examples} examples across {len(domains)} domains")


if __name__ == "__main__":
    main()
