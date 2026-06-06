"""Consistent Hash Routing at N=20 — Qwen2.5-0.5B.

Tests consistent hash ring with 150 virtual nodes per expert, measures
displacement when adding expert #21, and compares quality vs uniform average.

Output: macro/hash_routing_scale/results.json
"""

import hashlib
import json
import math
import os
import random
import statistics
import time
from bisect import bisect_right
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_HOME = os.environ.get("HF_HOME", "/workspace/hf_cache")
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
N_EXPERTS = 20
VIRTUAL_NODES = 150
LORA_RANK = 16
LORA_ALPHA = 16
MAX_SEQ_LEN = 256
BATCH_SIZE = 4
N_EVAL_BATCHES = 20
LR = 2e-4
OUT_DIR = Path(__file__).parent / "hash_routing_scale"


class ConsistentHashRing:
    def __init__(self, n_experts, virtual_nodes=150):
        self.n_experts = n_experts
        self.virtual_nodes = virtual_nodes
        self.ring = []
        self._rebuild()

    def _hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def _rebuild(self):
        self.ring = []
        for expert_id in range(self.n_experts):
            for vn in range(self.virtual_nodes):
                key = f"expert_{expert_id}_vn_{vn}"
                h = self._hash(key)
                self.ring.append((h, expert_id))
        self.ring.sort()
        self._hashes = [h for h, _ in self.ring]
        self._experts = [e for _, e in self.ring]

    def route(self, token_hash):
        idx = bisect_right(self._hashes, token_hash) % len(self.ring)
        return self._experts[idx]

    def route_topk(self, token_hash, k=2):
        idx = bisect_right(self._hashes, token_hash) % len(self.ring)
        experts = []
        seen = set()
        for offset in range(len(self.ring)):
            e = self._experts[(idx + offset) % len(self.ring)]
            if e not in seen:
                seen.add(e)
                experts.append(e)
                if len(experts) >= k:
                    break
        return experts

    def add_expert(self):
        return ConsistentHashRing(self.n_experts + 1, self.virtual_nodes)

    def measure_displacement(self, new_ring, n_samples=100000):
        displaced = 0
        for i in range(n_samples):
            h = self._hash(f"sample_{i}")
            old_expert = self.route(h)
            new_expert = new_ring.route(h)
            if old_expert != new_expert:
                displaced += 1
        return displaced / n_samples


def create_domain_data(tokenizer, domain_id, n_samples=100, seed=42):
    rng = random.Random(seed + domain_id * 13)
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                         cache_dir=HF_HOME)
        texts = [t for t in ds["text"] if len(t) > 50]
        start = (domain_id * n_samples) % max(1, len(texts) - n_samples)
        texts = texts[start:start + n_samples]
    except Exception:
        words = "the a is was in of to and for on with".split()
        texts = [" ".join(rng.choices(words, k=rng.randint(30, 150)))
                 for _ in range(n_samples)]
    encodings = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True, truncation=True,
                              max_length=MAX_SEQ_LEN + 1)
        if len(ids) > 5:
            encodings.append(ids)
    return encodings


def get_batch(encodings, batch_size, rng, device=DEVICE):
    seqs = rng.choices(encodings, k=batch_size)
    max_len = min(MAX_SEQ_LEN, max(len(s) for s in seqs))
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    for i, seq in enumerate(seqs):
        seq = seq[:max_len]
        input_ids[i, :len(seq)] = torch.tensor(seq)
        labels[i, :len(seq)] = torch.tensor(seq)
        labels[i, 0] = -100
        attention_mask[i, :len(seq)] = 1
    return input_ids, labels, attention_mask


@torch.no_grad()
def evaluate_with_state(model, lora_state, encodings, n_batches=20, batch_size=4):
    """Evaluate model with a given LoRA state, reusing the same model."""
    model.load_state_dict(lora_state, strict=False)
    model.eval()
    rng = random.Random(0)
    total = 0.0
    for _ in range(n_batches):
        input_ids, labels, attention_mask = get_batch(encodings, batch_size, rng)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1].contiguous()
        targets = labels[:, 1:].contiguous()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                              targets.view(-1), ignore_index=-100)
        total += loss.item()
    return total / n_batches


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"Consistent Hash Routing N={N_EXPERTS} — {MODEL_NAME}")
    print(f"Virtual nodes: {VIRTUAL_NODES}, Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_HOME,
                                               trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === 1. Hash ring displacement test ===
    print(f"\n{'='*70}")
    print("HASH RING DISPLACEMENT")
    print(f"{'='*70}")

    ring_20 = ConsistentHashRing(N_EXPERTS, VIRTUAL_NODES)
    ring_21 = ring_20.add_expert()

    displacement = ring_20.measure_displacement(ring_21, n_samples=100000)
    expected_displacement = 1.0 / (N_EXPERTS + 1)
    print(f"  Displacement (N={N_EXPERTS} -> N={N_EXPERTS+1}): {displacement*100:.2f}%")
    print(f"  Expected (1/{N_EXPERTS+1}): {expected_displacement*100:.2f}%")
    print(f"  Ratio: {displacement/expected_displacement:.2f}")

    # Load balance
    counts = [0] * N_EXPERTS
    for i in range(100000):
        h = ring_20._hash(f"token_{i}")
        e = ring_20.route(h)
        counts[e] += 1
    ideal = 100000 / N_EXPERTS
    imbalance = max(abs(c - ideal) / ideal * 100 for c in counts)
    print(f"  Load balance: max imbalance={imbalance:.1f}%, "
          f"min/max={min(counts)}/{max(counts)} (ideal: {ideal:.0f})")

    # === 2. Create 20 LoRA adapters ===
    print(f"\n{'='*70}")
    print(f"CREATING {N_EXPERTS} LoRA ADAPTERS")
    print(f"{'='*70}")

    lora_states = []
    for i in range(N_EXPERTS):
        torch.manual_seed(42 + i)
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, cache_dir=HF_HOME, trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(DEVICE)
        lora_config = LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)

        domain_data = create_domain_data(tokenizer, i, n_samples=100)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=LR)
        rng = random.Random(42 + i)
        model.train()
        for step in range(30):
            input_ids, labels, attention_mask = get_batch(domain_data, BATCH_SIZE, rng)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1].contiguous()
            targets = labels[:, 1:].contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                  targets.view(-1), ignore_index=-100)
            loss.backward()
            optimizer.step()

        state = {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}
        lora_states.append(state)
        if i % 5 == 0:
            print(f"  Adapter {i}/{N_EXPERTS} done")

        del model, base_model, optimizer
        torch.cuda.empty_cache()

    # === 3. Routing comparison (reuse single model) ===
    print(f"\n{'='*70}")
    print("ROUTING COMPARISON")
    print(f"{'='*70}")

    eval_data = create_domain_data(tokenizer, 999, n_samples=200)

    # Create one model to reuse
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=HF_HOME, trust_remote_code=True,
        torch_dtype=torch.float32
    ).to(DEVICE)
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    eval_model = get_peft_model(base_model, lora_config)

    # Hash routing: select top-2 experts per batch, merge their LoRA
    print("\n  Hash routing (top-2 per batch)...")
    hash_losses = []
    rng = random.Random(0)
    for batch_idx in range(N_EVAL_BATCHES):
        input_ids, labels, attention_mask = get_batch(eval_data, BATCH_SIZE, rng)

        input_hash = hashlib.md5(str(input_ids[0, :5].tolist()).encode()).hexdigest()
        token_hash = int(input_hash, 16)
        selected = ring_20.route_topk(token_hash, k=2)

        merged = {}
        for key in lora_states[0]:
            merged[key] = sum(lora_states[e][key] for e in selected) / len(selected)

        eval_model.load_state_dict(merged, strict=False)
        eval_model.eval()
        with torch.no_grad():
            outputs = eval_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1].contiguous()
            targets = labels[:, 1:].contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                  targets.view(-1), ignore_index=-100)
            hash_losses.append(loss.item())

    hash_mean = statistics.mean(hash_losses)
    print(f"    Hash routing avg loss: {hash_mean:.4f}")

    # Uniform average (all 20)
    print("\n  Uniform average (all 20)...")
    uniform_merged = {}
    for key in lora_states[0]:
        uniform_merged[key] = sum(s[key] for s in lora_states) / N_EXPERTS
    uniform_loss = evaluate_with_state(eval_model, uniform_merged, eval_data)
    print(f"    Uniform avg loss: {uniform_loss:.4f}")

    # Base model (zero LoRA)
    print("\n  Base model (zero LoRA)...")
    zero_state = {k: torch.zeros_like(v) for k, v in lora_states[0].items()}
    base_loss = evaluate_with_state(eval_model, zero_state, eval_data)
    print(f"    Base loss: {base_loss:.4f}")

    # Single expert (best of 20)
    print("\n  Best single expert...")
    best_loss = float('inf')
    for i in range(N_EXPERTS):
        loss_i = evaluate_with_state(eval_model, lora_states[i], eval_data, n_batches=5)
        if loss_i < best_loss:
            best_loss = loss_i
    print(f"    Best single: {best_loss:.4f}")

    del eval_model, base_model
    torch.cuda.empty_cache()

    # === 4. Results ===
    results = {
        'model': MODEL_NAME,
        'n_experts': N_EXPERTS,
        'virtual_nodes': VIRTUAL_NODES,
        'lora_rank': LORA_RANK,
        'displacement': {
            'actual_pct': displacement * 100,
            'expected_pct': expected_displacement * 100,
            'ratio': displacement / expected_displacement,
        },
        'load_balance': {
            'max_imbalance_pct': imbalance,
            'min_count': min(counts),
            'max_count': max(counts),
        },
        'routing_quality': {
            'hash_top2_loss': hash_mean,
            'uniform_avg_loss': uniform_loss,
            'base_loss': base_loss,
            'best_single_loss': best_loss,
            'hash_vs_base_pct': (hash_mean - base_loss) / base_loss * 100,
            'hash_vs_uniform_pct': (hash_mean - uniform_loss) / uniform_loss * 100,
        },
    }

    # Kill criteria
    print(f"\n{'='*70}")
    print("KILL CRITERIA")
    print(f"{'='*70}")

    deg = (hash_mean - base_loss) / base_loss * 100
    if abs(deg) < 5:
        print(f"  [PASS] Hash routing degradation vs base: {deg:+.2f}% < 5%")
    else:
        print(f"  [WARN] Hash routing degradation vs base: {deg:+.2f}%")

    if displacement < 0.06:
        print(f"  [PASS] Displacement: {displacement*100:.2f}% (expected ~{expected_displacement*100:.1f}%)")
    else:
        print(f"  [KILL] Displacement too high: {displacement*100:.2f}%")

    elapsed = time.time() - t0
    results['elapsed_s'] = elapsed
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
