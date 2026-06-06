#!/usr/bin/env python3
"""Generate training data for 10 selected domains using Llama 3.1 8B teacher via Groq.

Reuses pilot50_generate.py logic but with 8B teacher model.
Stores data separately in data/distillation_8b/ to compare with 70B data.

Usage (runs locally):
    python experiments/models/distillation_quality_vs_teacher/generate_8b_data.py
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

REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
# Try worktree .env first, fall back to main repo root
load_dotenv(REPO_ROOT / ".env")
# Also try the main repo root (in case this is a git worktree)
main_repo = REPO_ROOT
while main_repo != main_repo.parent:
    if (main_repo.parent / ".env").exists():
        load_dotenv(main_repo.parent / ".env")
        break
    main_repo = main_repo.parent
# Direct fallback: common location for this project
load_dotenv(Path("/Users/tom/Code/tomsiwik/llm/.env"))
load_dotenv(Path.home() / ".env")

# ── Config ────────────────────────────────────────────────────────────
TEACHER_8B = "llama-3.1-8b-instant"
TEACHER_70B = "llama-3.3-70b-versatile"
BASE_URL = "https://api.groq.com/openai/v1"

# 10 domains selected for this experiment
# 5 code/factual (expected: 8B sufficient), 5 reasoning/nuanced (expected: 70B needed)
SELECTED_DOMAINS = {
    # Code/Factual
    "python": "Python programming: write functions, classes, algorithms, data structures, "
              "decorators, generators, async/await patterns, and debugging techniques.",
    "sql": "SQL database queries: SELECT, JOIN, GROUP BY, subqueries, window functions, "
           "CTEs, indexing strategies, query optimization, and schema design.",
    "bash": "Bash shell scripting: file manipulation, text processing with sed/awk/grep, "
            "pipelines, process management, system administration, and automation scripts.",
    "physics": "Physics problems and explanations: classical mechanics, electromagnetism, "
               "thermodynamics, quantum mechanics basics, and dimensional analysis.",
    "accounting": "Accounting: financial statements, GAAP principles, journal entries, tax "
                  "implications, auditing procedures, and managerial accounting.",
    # Reasoning/Nuanced
    "ethics": "Ethics and moral reasoning: ethical frameworks (utilitarian, deontological, "
              "virtue), applied ethics, moral dilemmas, and stakeholder analysis.",
    "creative-fiction": "Creative fiction writing: character development, plot structure, dialogue, "
                        "world-building, narrative voice, and literary techniques across genres.",
    "causal-reasoning": "Causal reasoning: cause-and-effect analysis, counterfactual thinking, "
                        "confounding variables, causal inference methods, and root cause analysis.",
    "legal": "Legal analysis: contract law, constitutional principles, case analysis, "
             "legal reasoning, statutory interpretation, and legal writing conventions.",
    "game-theory": "Game theory: Nash equilibrium, prisoner's dilemma, mechanism design, "
                   "strategic thinking, auction theory, and cooperative vs non-cooperative games.",
}

SYSTEM_PROMPT = """You are an expert training data generator. Given a domain description,
generate a high-quality instruction-response pair for fine-tuning a language model.

The instruction should be a realistic user question or task in this domain.
The response should be thorough, correct, and demonstrate domain expertise.
Vary the difficulty: mix simple factual questions with complex multi-step problems.

Output ONLY valid JSON with exactly two fields:
{"instruction": "...", "response": "..."}"""


def generate_one(client, model, domain, description, temperature=0.9):
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
    """Generate a batch of examples for one domain."""
    results = []
    for _ in range(count):
        try:
            pair = generate_one(client, model, domain, description)
            if pair:
                results.append(pair)
        except Exception:
            time.sleep(2)
    return results


def generate_domain(client, model, domain, description, n_examples, output_dir, workers=8):
    """Generate training data for one domain with concurrent workers."""
    out_file = output_dir / domain / "train.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    existing = 0
    if out_file.exists():
        with open(out_file) as f:
            existing = sum(1 for _ in f)
        if existing >= n_examples:
            print(f"  {domain}: already have {existing}/{n_examples}, skipping")
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
    rate = generated / max(elapsed, 1)
    print(f"  {domain}: {total}/{n_examples} complete ({elapsed:.0f}s, {rate:.1f}/s)", flush=True)
    return total, elapsed


def main():
    parser = argparse.ArgumentParser(description="Generate 8B teacher data for teacher size comparison")
    parser.add_argument("--n-examples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--domains", nargs="*", help="Specific domains (default: all 10)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = REPO_ROOT / "data" / "distillation_8b"
    domains = SELECTED_DOMAINS
    if args.domains:
        domains = {k: v for k, v in domains.items() if k in args.domains}

    print(f"Teacher Size Experiment: 8B Data Generation")
    print(f"  Teacher: {TEACHER_8B}")
    print(f"  Domains: {len(domains)}")
    print(f"  Examples/domain: {args.n_examples}")
    print(f"  Output: {output_dir}")
    print(f"  Workers: {args.workers}")

    # Check what needs generation
    needs_gen = 0
    for domain in domains:
        fpath = output_dir / domain / "train.jsonl"
        existing = 0
        if fpath.exists():
            with open(fpath) as f:
                existing = sum(1 for _ in f)
        if existing < args.n_examples:
            needs_gen += 1

    remaining = sum(
        args.n_examples - (sum(1 for _ in open(output_dir / d / "train.jsonl"))
                          if (output_dir / d / "train.jsonl").exists() else 0)
        for d in domains
    )
    # 8B pricing: $0.05/M input, $0.08/M output (Groq)
    est_cost = remaining * (200 * 0.05 + 300 * 0.08) / 1_000_000
    print(f"  Need generation: {needs_gen} domains")
    print(f"  Remaining examples: {remaining}")
    print(f"  Estimated cost: ${est_cost:.2f}")
    print(flush=True)

    if args.dry_run:
        for domain in sorted(domains.keys()):
            fpath = output_dir / domain / "train.jsonl"
            existing = 0
            if fpath.exists():
                with open(fpath) as f:
                    existing = sum(1 for _ in f)
            print(f"  {domain}: {existing}/{args.n_examples}")
        return

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("ERROR: Set GROQ_API_KEY environment variable")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    total_time = 0.0
    for domain in sorted(domains.keys()):
        desc = domains[domain]
        _, elapsed = generate_domain(client, TEACHER_8B, domain, desc.strip(),
                                     args.n_examples, output_dir, workers=args.workers)
        total_time += elapsed

    print(f"\n{'='*60}")
    print(f"8B data generation complete in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"\nData summary:")
    total_examples = 0
    for domain in sorted(domains.keys()):
        fpath = output_dir / domain / "train.jsonl"
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
