#!/usr/bin/env python3
"""Test composition interference: does reasoning LoRA degrade domain experts?

Kill criterion K2: reasoning LoRA composed with domain expert degrades domain
quality >5%. Tests multiple pilot50 domain experts with and without the
reasoning adapter composed via pre-merge.

For each domain expert:
  1. Baseline: base + domain expert PPL on domain eval data
  2. Composed: base + domain expert + reasoning LoRA PPL on same data
  3. Measure degradation: (composed_ppl - baseline_ppl) / baseline_ppl

Also measures orthogonality: cosine similarity between reasoning adapter
and each domain expert in weight space.

Usage (on RunPod):
    cd /workspace/llm
    python experiments/models/reasoning_expert_distillation/eval_composition_interference.py

Expected runtime: ~30-60 min on RTX 4090
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/Qwen2.5-7B")
HF_CACHE = os.environ.get("HF_CACHE", "/workspace/hf_cache")
REPO_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "micro" / "models" / "reasoning_expert_distillation"
REASONING_ADAPTER = OUTPUT_DIR / "reasoning_adapter"
ADAPTER_DIR = REPO_ROOT / "data" / "adapters"
DATA_DIR = REPO_ROOT / "data" / "distillation"

# Test these domain experts from pilot50
# Diverse domains: programming, science, professional, writing, reasoning
TEST_DOMAINS = [
    "python", "math", "physics", "medical", "creative-fiction",
    "legal", "bash", "statistics", "logic-puzzles", "rust",
]

# Eval config
MAX_EVAL_EXAMPLES = 50  # Per domain (from tail of training data, same as pilot50_bench)
MAX_SEQ_LENGTH = 512
LORA_RANK = 16
LORA_ALPHA = 16

SEED = 42


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Orthogonality Measurement ─────────────────────────────────────────────────

def extract_lora_delta_flat(adapter_path: Path) -> np.ndarray:
    """Extract flattened LoRA delta vector. Returns numpy array."""
    from safetensors.torch import load_file
    import torch

    weights_file = adapter_path / "adapter_model.safetensors"
    if not weights_file.exists():
        weights_file = adapter_path / "adapter_model.bin"
        weights = torch.load(weights_file, map_location="cpu", weights_only=True)
    else:
        weights = load_file(str(weights_file), device="cpu")

    modules = {}
    for key, tensor in weights.items():
        clean = key.replace("base_model.model.", "")
        if "lora_A" in clean:
            mod = clean.split(".lora_A")[0]
            modules.setdefault(mod, {})["A"] = tensor.float()
        elif "lora_B" in clean:
            mod = clean.split(".lora_B")[0]
            modules.setdefault(mod, {})["B"] = tensor.float()

    scaling = LORA_ALPHA / LORA_RANK
    parts = []
    for mod_name in sorted(modules.keys()):
        ab = modules[mod_name]
        if "A" in ab and "B" in ab:
            delta = scaling * (ab["B"] @ ab["A"])
            parts.append(delta.numpy().flatten())

    return np.concatenate(parts)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── PPL Measurement ───────────────────────────────────────────────────────────

def prepare_eval_data(domain: str, tokenizer, max_eval: int = MAX_EVAL_EXAMPLES) -> list[str]:
    """Extract eval data from the tail of domain training data."""
    train_file = DATA_DIR / domain / "train.jsonl"
    if not train_file.exists():
        return []

    with open(train_file) as f:
        lines = f.readlines()

    eval_lines = lines[-max_eval:]
    texts = []
    for line in eval_lines:
        record = json.loads(line)
        if "messages" in record:
            text = tokenizer.apply_chat_template(
                record["messages"], tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
    return texts


def compute_perplexity(model, tokenizer, texts: list[str],
                       max_length: int = MAX_SEQ_LENGTH) -> float:
    """Compute average perplexity over texts."""
    import torch

    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=max_length).to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            n_tokens = inputs["input_ids"].shape[1]
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test composition interference of reasoning LoRA"
    )
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--reasoning-adapter", default=str(REASONING_ADAPTER))
    parser.add_argument("--domains", nargs="*", default=TEST_DOMAINS)
    parser.add_argument("--max-eval", type=int, default=MAX_EVAL_EXAMPLES)
    parser.add_argument("--output",
                        default=str(OUTPUT_DIR / "interference_results.json"))
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Verify reasoning adapter exists
    if not Path(args.reasoning_adapter).exists():
        log(f"ERROR: Reasoning adapter not found at {args.reasoning_adapter}")
        log("Run train_reasoning_expert.py first.")
        sys.exit(1)

    # Find available domain adapters
    available_domains = []
    for domain in args.domains:
        adapter_dir = ADAPTER_DIR / domain
        if (adapter_dir / "adapter_config.json").exists():
            available_domains.append(domain)
        else:
            log(f"  WARNING: Adapter not found for {domain}, skipping")

    if not available_domains:
        log("ERROR: No domain adapters found!")
        sys.exit(1)

    log("=" * 72)
    log("COMPOSITION INTERFERENCE TEST")
    log(f"  Base model: {args.base_model}")
    log(f"  Reasoning adapter: {args.reasoning_adapter}")
    log(f"  Domains to test: {available_domains}")
    log("=" * 72)

    # ── Phase 1: Orthogonality measurement ────────────────────────────────
    log("\n=== Phase 1: Orthogonality (weight space) ===")
    reasoning_delta = extract_lora_delta_flat(Path(args.reasoning_adapter))
    log(f"  Reasoning adapter delta dim: {len(reasoning_delta):,}")
    log(f"  Reasoning adapter delta norm: {np.linalg.norm(reasoning_delta):.6f}")

    ortho_results = {}
    for domain in available_domains:
        domain_delta = extract_lora_delta_flat(ADAPTER_DIR / domain)
        cos = cosine_sim(reasoning_delta, domain_delta)
        ortho_results[domain] = {
            "cosine": float(cos),
            "abs_cosine": float(abs(cos)),
            "domain_norm": float(np.linalg.norm(domain_delta)),
        }
        log(f"  reasoning x {domain}: cos = {cos:+.6f} (|cos| = {abs(cos):.6f})")

    cos_values = [v["abs_cosine"] for v in ortho_results.values()]
    mean_cos = np.mean(cos_values)
    max_cos = np.max(cos_values)
    log(f"\n  Mean |cos|: {mean_cos:.6f}")
    log(f"  Max  |cos|: {max_cos:.6f}")

    # Random baseline for this dimensionality
    D = len(reasoning_delta)
    random_expected = math.sqrt(2 / (math.pi * D))
    log(f"  Random baseline E[|cos|]: {random_expected:.2e} (D={D:,})")
    log(f"  Ratio to random: {mean_cos / random_expected:.1f}x")

    # ── Phase 2: PPL interference measurement ─────────────────────────────
    log("\n=== Phase 2: PPL Interference ===")

    # Load base model (fp16 for accurate PPL)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, cache_dir=HF_CACHE, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16, device_map="auto",
        cache_dir=HF_CACHE, trust_remote_code=True
    )

    ppl_results = {}
    for domain in available_domains:
        log(f"\n  --- {domain} ---")
        eval_texts = prepare_eval_data(domain, tokenizer, args.max_eval)
        if not eval_texts:
            log(f"  No eval data for {domain}, skipping")
            continue

        log(f"  Eval examples: {len(eval_texts)}")

        # Baseline: base + domain expert
        domain_model = PeftModel.from_pretrained(
            base_model, str(ADAPTER_DIR / domain)
        )
        baseline_ppl = compute_perplexity(domain_model, tokenizer, eval_texts)
        log(f"  Domain only PPL: {baseline_ppl:.4f}")
        del domain_model
        gc.collect()
        torch.cuda.empty_cache()

        # Composed: base + domain expert + reasoning adapter (pre-merge)
        # Load domain adapter, merge it, then load reasoning adapter on top
        domain_model = PeftModel.from_pretrained(
            base_model, str(ADAPTER_DIR / domain)
        )
        merged_model = domain_model.merge_and_unload()

        # Now load reasoning adapter on top of the merged model
        composed_model = PeftModel.from_pretrained(
            merged_model, args.reasoning_adapter
        )
        composed_ppl = compute_perplexity(composed_model, tokenizer, eval_texts)
        log(f"  Composed PPL:    {composed_ppl:.4f}")

        degradation = (composed_ppl - baseline_ppl) / baseline_ppl * 100
        log(f"  Degradation:     {degradation:+.2f}%")

        ppl_results[domain] = {
            "baseline_ppl": round(float(baseline_ppl), 4),
            "composed_ppl": round(float(composed_ppl), 4),
            "degradation_pct": round(float(degradation), 2),
            "n_eval": len(eval_texts),
        }

        del composed_model, merged_model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Kill Criteria Assessment ──────────────────────────────────────────
    log("\n" + "=" * 72)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 72)

    degradations = [v["degradation_pct"] for v in ppl_results.values()]
    mean_degradation = np.mean(degradations) if degradations else 0
    max_degradation = np.max(degradations) if degradations else 0
    domains_degraded_gt_5 = sum(1 for d in degradations if d > 5.0)

    log(f"\n  PPL Degradation Summary:")
    log(f"    Mean degradation: {mean_degradation:+.2f}%")
    log(f"    Max degradation:  {max_degradation:+.2f}%")
    log(f"    Domains > 5% degradation: {domains_degraded_gt_5}/{len(degradations)}")

    # K2: reasoning LoRA degrades domain quality >5%
    k2_pass = mean_degradation <= 5.0
    log(f"\n  K2: Mean domain degradation <= 5%")
    log(f"      Mean: {mean_degradation:+.2f}%")
    log(f"      Verdict: {'PASS' if k2_pass else 'KILL'}")

    # Orthogonality summary
    log(f"\n  Orthogonality Summary:")
    log(f"    Mean |cos| with domain experts: {mean_cos:.6f}")
    log(f"    Expected at d=3584: ~0.0002 (from macro/ortho_scaling)")
    log(f"    Ratio to random: {mean_cos / random_expected:.1f}x")

    # Save results
    results = {
        "experiment": "reasoning_composition_interference",
        "base_model": args.base_model,
        "reasoning_adapter": args.reasoning_adapter,
        "domains_tested": available_domains,
        "orthogonality": {
            "per_domain": ortho_results,
            "mean_abs_cos": round(float(mean_cos), 8),
            "max_abs_cos": round(float(max_cos), 8),
            "random_baseline": round(float(random_expected), 8),
            "delta_dim": D,
        },
        "ppl_interference": {
            "per_domain": ppl_results,
            "mean_degradation_pct": round(float(mean_degradation), 2),
            "max_degradation_pct": round(float(max_degradation), 2),
            "domains_gt_5pct": domains_degraded_gt_5,
        },
        "kill_criteria": {
            "K2_mean_degradation_le_5pct": {
                "threshold": 5.0,
                "actual": round(float(mean_degradation), 2),
                "pass": k2_pass,
            },
        },
        "verdict": "PASS" if k2_pass else "KILLED (composition interference)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
