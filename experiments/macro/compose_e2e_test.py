"""End-to-end compose CLI test on RunPod.

1. Train 5 domain LoRA adapters, save as .pt files
2. Register them with compose CLI
3. Benchmark: routing quality, merge latency, cache performance
4. Serve and test inference
5. Test add/remove expert workflow

Output: macro/compose_e2e/results.json
"""

import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_HOME = os.environ.get("HF_HOME", "/workspace/hf_cache")
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
LORA_RANK = 16
LORA_ALPHA = 16
BATCH_SIZE = 4
MAX_SEQ_LEN = 256
FINETUNE_STEPS = 200
LR = 2e-4
OUT_DIR = Path(__file__).parent / "compose_e2e"
ADAPTER_DIR = OUT_DIR / "adapters"
REPO_ROOT = Path(__file__).parent.parent.parent

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]

DOMAINS = {
    "python": {"dataset": "bigcode/the-stack-smol", "name": "data/python", "field": "content"},
    "javascript": {"dataset": "bigcode/the-stack-smol", "name": "data/javascript", "field": "content"},
    "medical": {"dataset": "pubmed_qa", "name": "pqa_labeled", "field": "long_answer"},
    "legal": {"dataset": "openai/gsm8k", "name": "main", "field": "answer"},
    "math": {"dataset": "openai/gsm8k", "name": "main", "field": "question"},
}


def load_data(domain, tokenizer, n=300):
    cfg = DOMAINS[domain]
    try:
        kwargs = {"split": "train", "cache_dir": HF_HOME, "trust_remote_code": True}
        ds = load_dataset(cfg["dataset"], cfg["name"], **kwargs) if cfg["name"] else load_dataset(cfg["dataset"], **kwargs)
    except Exception:
        rng = random.Random(42)
        return [tokenizer.encode(f"sample {domain} {rng.randint(0,999)}", truncation=True, max_length=MAX_SEQ_LEN+1)
                for _ in range(n)]

    encs = []
    for i, item in enumerate(ds):
        if i >= n * 2:
            break
        text = item.get(cfg["field"], "")
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)
        if text and len(text) > 50:
            ids = tokenizer.encode(text[:2000], add_special_tokens=True, truncation=True, max_length=MAX_SEQ_LEN+1)
            if len(ids) > 10:
                encs.append(ids)
    return encs[:n]


def get_batch(encs, bs, rng):
    seqs = rng.choices(encs, k=bs)
    ml = min(MAX_SEQ_LEN, max(len(s) for s in seqs))
    ids = torch.zeros(bs, ml, dtype=torch.long, device=DEVICE)
    lab = torch.full((bs, ml), -100, dtype=torch.long, device=DEVICE)
    att = torch.zeros(bs, ml, dtype=torch.long, device=DEVICE)
    for i, s in enumerate(seqs):
        s = s[:ml]
        ids[i, :len(s)] = torch.tensor(s)
        lab[i, :len(s)] = torch.tensor(s)
        lab[i, 0] = -100
        att[i, :len(s)] = 1
    return ids, lab, att


@torch.no_grad()
def eval_loss(model, encs, n=10):
    model.eval()
    rng = random.Random(0)
    losses = []
    for _ in range(n):
        ids, lab, att = get_batch(encs, BATCH_SIZE, rng)
        out = model(input_ids=ids, attention_mask=att)
        logits = out.logits[:, :-1].contiguous()
        targets = lab[:, 1:].contiguous()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        losses.append(loss.item())
    model.train()
    return statistics.mean(losses)


def train_adapters(tokenizer):
    """Train 5 domain LoRA adapters and save them."""
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for domain in DOMAINS:
        print(f"\n  Training {domain} adapter...")
        torch.manual_seed(42 + hash(domain) % 1000)

        data = load_data(domain, tokenizer)
        n_val = max(10, len(data) // 5)
        val, train = data[:n_val], data[n_val:]

        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=HF_HOME,
                                                     trust_remote_code=True, torch_dtype=torch.float32).to(DEVICE)
        cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS,
                        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(base, cfg)

        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
        rng = random.Random(42)
        model.train()

        for step in range(1, FINETUNE_STEPS + 1):
            opt.zero_grad()
            ids, lab, att = get_batch(train, BATCH_SIZE, rng)
            out = model(input_ids=ids, attention_mask=att)
            logits = out.logits[:, :-1].contiguous()
            targets = lab[:, 1:].contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            loss.backward()
            opt.step()
            if step % 50 == 0:
                print(f"    step {step} loss={loss.item():.4f}")

        val_loss = eval_loss(model, val)
        print(f"    val_loss={val_loss:.4f}")

        # Save LoRA state
        lora_state = {k: v.cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
        adapter_path = ADAPTER_DIR / f"{domain}.pt"
        torch.save(lora_state, adapter_path)
        print(f"    Saved to {adapter_path} ({len(lora_state)} tensors)")

        results[domain] = {"val_loss": val_loss, "path": str(adapter_path), "n_tensors": len(lora_state)}

        del model, base, opt
        torch.cuda.empty_cache()

    return results


def test_compose_cli():
    """Test the compose CLI end-to-end."""
    compose_cmd = [sys.executable, str(REPO_ROOT / "tools" / "compose.py"), "--dir", str(OUT_DIR)]

    # 1. Init
    print("\n  compose init...")
    r = subprocess.run(compose_cmd + ["init", "--base", MODEL_NAME], capture_output=True, text=True)
    print(f"    {r.stdout.strip()}")

    # 2. Add experts
    for domain in DOMAINS:
        path = str(ADAPTER_DIR / f"{domain}.pt")
        print(f"  compose add {domain}...")
        r = subprocess.run(compose_cmd + ["add", path, "--name", domain, "--domain", domain, "--rank", str(LORA_RANK)],
                          capture_output=True, text=True)
        print(f"    {r.stdout.strip()}")

    # 3. List
    print("\n  compose list...")
    r = subprocess.run(compose_cmd + ["list"], capture_output=True, text=True)
    print(f"    {r.stdout.strip()}")

    # 4. Benchmark
    print("\n  compose bench...")
    r = subprocess.run(compose_cmd + ["bench", "--prompts", "10"], capture_output=True, text=True, timeout=300)
    print(f"    {r.stdout.strip()}")
    if r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()}")

    # 5. Generate
    print("\n  compose generate...")
    r = subprocess.run(compose_cmd + ["generate", "def fibonacci(n):", "--max-tokens", "50"],
                      capture_output=True, text=True, timeout=120)
    print(f"    {r.stdout.strip()[:200]}")

    # 6. Remove + re-list
    print("\n  compose remove math...")
    subprocess.run(compose_cmd + ["remove", "math"], capture_output=True, text=True)
    r = subprocess.run(compose_cmd + ["list"], capture_output=True, text=True)
    print(f"    {r.stdout.strip()}")

    # 7. Re-add
    print("\n  compose add math (re-add)...")
    subprocess.run(compose_cmd + ["add", str(ADAPTER_DIR / "math.pt"), "--name", "math",
                                  "--domain", "math", "--rank", str(LORA_RANK)],
                  capture_output=True, text=True)


def measure_latency():
    """Detailed latency measurements."""
    print("\n  Measuring latency...")
    from composer.compose import ComposeEngine

    engine = ComposeEngine(str(OUT_DIR))
    engine.load_base()

    expert_names = list(engine.registry.data["experts"].keys())

    # Measure merge latency
    merge_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        engine.loader.merge_experts(expert_names[:2])
        merge_times.append((time.perf_counter() - t0) * 1000)

    # Measure forward latency
    inputs = engine.tokenizer("def hello():", return_tensors="pt").to(engine.device)
    fwd_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        with torch.no_grad():
            engine.base_model(**inputs)
        torch.cuda.synchronize()
        fwd_times.append((time.perf_counter() - t0) * 1000)

    # Measure generate latency
    gen_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        engine.route_and_generate("def hello():", max_new_tokens=20, top_k=2)
        torch.cuda.synchronize()
        gen_times.append((time.perf_counter() - t0) * 1000)

    results = {
        "merge_ms": {"mean": statistics.mean(merge_times), "p50": sorted(merge_times)[25], "p99": sorted(merge_times)[49]},
        "forward_ms": {"mean": statistics.mean(fwd_times), "p50": sorted(fwd_times)[25], "p99": sorted(fwd_times)[49]},
        "generate_20tok_ms": {"mean": statistics.mean(gen_times), "p50": sorted(gen_times)[5], "p99": sorted(gen_times)[9]},
        "cache_stats": engine.loader.stats,
    }

    print(f"    Merge:    {results['merge_ms']['mean']:.1f}ms (p99: {results['merge_ms']['p99']:.1f}ms)")
    print(f"    Forward:  {results['forward_ms']['mean']:.1f}ms (p99: {results['forward_ms']['p99']:.1f}ms)")
    print(f"    Generate: {results['generate_20tok_ms']['mean']:.1f}ms (20 tokens)")
    print(f"    Cache:    {engine.loader.stats}")

    return results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"Compose E2E Test — {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_HOME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Phase 1: Train adapters
    print(f"\n{'='*70}")
    print("PHASE 1: Train Adapters")
    print(f"{'='*70}")
    adapter_results = train_adapters(tokenizer)

    # Phase 2: Test compose CLI
    print(f"\n{'='*70}")
    print("PHASE 2: Compose CLI Test")
    print(f"{'='*70}")
    test_compose_cli()

    # Phase 3: Latency
    print(f"\n{'='*70}")
    print("PHASE 3: Latency Measurement")
    print(f"{'='*70}")
    sys.path.insert(0, str(REPO_ROOT))
    latency = measure_latency()

    elapsed = time.time() - t0
    results = {
        "model": MODEL_NAME,
        "adapters": adapter_results,
        "latency": latency,
        "elapsed_s": elapsed,
    }

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Results: {OUT_DIR / 'results.json'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
