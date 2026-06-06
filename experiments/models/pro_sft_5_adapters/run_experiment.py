#!/usr/bin/env python3
"""Pierre Pro: Train 5 SFT domain adapters on Qwen3-4B-4bit with Grassmannian A.

Transfers the proven SFT recipe (Finding #206, sft_24_domain_adapters) from
BitNet-2B to Qwen3-4B-4bit. Uses frozen Grassmannian A-matrices from
pro_grassmannian_init (Finding #318).

Key difference from BitNet track: no BitLinear unpacking needed. QLoRA wraps
QuantizedLinear directly (Dettmers et al., 2305.14314).

Kill criteria:
  K812: Fewer than 4/5 converge -> FAIL
  K813: Mean behavioral < 0.3 -> FAIL
"""

import gc
import json
import math
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
ADAPTERS_OUT = EXPERIMENT_DIR / "adapters"

# Data from the BitNet SFT experiment (same instruction/response format)
DATA_DIR = EXPERIMENT_DIR.parent / "real_data_25_domain_adapters" / "data"
SKELETON_PATH = EXPERIMENT_DIR.parent / "pro_grassmannian_init" / "grassmannian_skeleton_n5.npz"

MODEL_ID = "mlx-community/Qwen3-4B-4bit"
LORA_RANK = 16
LORA_SCALE = 20.0
TRAIN_ITERS = 300
LEARNING_RATE = 1e-4
MAX_SEQ = 256
SEED = 42

RESPONSE_MARKER = "### Response:\n"

# 5 domains, matching skeleton indices 0-4
DOMAINS = ["medical", "code", "math", "legal", "finance"]

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)


def log(m): print(m, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# -- QLoRA with frozen Grassmannian A ------------------------------------------

class GrassmannianLoRALinear(nn.Module):
    """QLoRA with frozen Grassmannian A-matrix and trainable B-matrix.

    Works with QuantizedLinear base (no unpacking needed). Gradient flows
    through float LoRA path only (Dettmers et al., 2305.14314).
    """
    def __init__(self, base_linear, rank=16, scale=20.0, a_init=None):
        super().__init__()
        # Detect true dimensions from the base layer
        output_dims, packed_input = base_linear.weight.shape
        if isinstance(base_linear, nn.QuantizedLinear):
            input_dims = packed_input * (32 // base_linear.bits)
        else:
            input_dims = packed_input

        self.linear = base_linear

        # A-matrix: frozen Grassmannian init or random
        if a_init is not None:
            self.lora_a = a_init
        else:
            s = 1.0 / math.sqrt(input_dims)
            self.lora_a = mx.random.uniform(low=-s, high=s, shape=(input_dims, rank))

        # B-matrix: trainable, zero-init (standard LoRA)
        self.lora_b = mx.zeros((rank, output_dims))
        self.scale = scale

        # Freeze base weights and A-matrix
        self.linear.freeze()
        self.freeze(keys=["lora_a"], strict=False)

    def __call__(self, x):
        base_out = self.linear(x)
        z = (x @ self.lora_a) @ self.lora_b
        return base_out + (self.scale * z).astype(x.dtype)


# -- SFT data loading ----------------------------------------------------------

def load_domain_texts(domain, split="train", max_n=400):
    path = DATA_DIR / domain / f"{split}.jsonl"
    if not path.exists():
        log(f"  WARNING: data file not found: {path}")
        return []
    texts = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_n:
                break
            texts.append(json.loads(line)["text"])
    return texts


def tokenize_sft(text, tokenizer, max_len=256):
    """Tokenize with SFT mask: 1 for response tokens, 0 for instruction."""
    response_idx = text.find(RESPONSE_MARKER)
    if response_idx < 0:
        tokens = tokenizer.encode(text)[:max_len]
        return tokens, [1] * len(tokens)
    instruction_part = text[:response_idx + len(RESPONSE_MARKER)]
    instruction_len = len(tokenizer.encode(instruction_part))
    full_tokens = tokenizer.encode(text)[:max_len]
    mask = [0] * min(instruction_len, len(full_tokens))
    mask += [1] * (len(full_tokens) - len(mask))
    return full_tokens, mask


def sft_loss_fn(model, tokens, mask):
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    response_mask = mask[:, 1:]
    per_token_loss = nn.losses.cross_entropy(logits, targets, reduction="none")
    masked_loss = per_token_loss * response_mask
    n_response = mx.maximum(response_mask.sum(), mx.array(1.0))
    return masked_loss.sum() / n_response


# -- Behavioral evaluation -----------------------------------------------------

BEHAVIORAL_PROMPTS = {
    "medical": [
        "### Instruction:\nWhat are the common symptoms of type 2 diabetes?\n\n### Response:\n",
        "### Instruction:\nExplain the mechanism of action of metformin.\n\n### Response:\n",
        "### Instruction:\nWhat is the difference between a CT scan and an MRI?\n\n### Response:\n",
    ],
    "code": [
        "### Instruction:\nWrite a Python function to check if a string is a palindrome.\n\n### Response:\n",
        "### Instruction:\nExplain the difference between a list and a tuple in Python.\n\n### Response:\n",
        "### Instruction:\nWrite a function to find the factorial of a number using recursion.\n\n### Response:\n",
    ],
    "math": [
        "### Instruction:\nSolve: If a train travels 120 km in 2 hours, what is its average speed?\n\n### Response:\n",
        "### Instruction:\nWhat is the area of a circle with radius 7 cm?\n\n### Response:\n",
        "### Instruction:\nSimplify the expression: 3x + 5 - x + 2\n\n### Response:\n",
    ],
    "legal": [
        "### Instruction:\nWhat is the difference between civil law and criminal law?\n\n### Response:\n",
        "### Instruction:\nExplain what a tort is in legal terms.\n\n### Response:\n",
        "### Instruction:\nWhat does 'habeas corpus' mean?\n\n### Response:\n",
    ],
    "finance": [
        "### Instruction:\nWhat is compound interest and how does it work?\n\n### Response:\n",
        "### Instruction:\nExplain the difference between a stock and a bond.\n\n### Response:\n",
        "### Instruction:\nWhat is diversification in investing?\n\n### Response:\n",
    ],
}

# Domain-specific quality keywords for behavioral scoring
DOMAIN_KEYWORDS = {
    "medical": ["symptoms", "diagnosis", "treatment", "patient", "disease",
                "medication", "clinical", "therapy", "blood", "organ",
                "insulin", "glucose", "scan", "imaging", "doctor"],
    "code": ["def ", "return", "function", "class ", "import", "for ",
             "if ", "while", "print", "variable", "string", "list",
             "parameter", "argument", "recursion"],
    "math": ["=", "answer", "solve", "calculate", "total", "number",
             "equation", "formula", "area", "speed", "km", "radius",
             "pi", "multiply", "divide"],
    "legal": ["law", "court", "legal", "right", "contract", "statute",
              "jurisdiction", "defendant", "plaintiff", "case",
              "tort", "criminal", "civil", "liability", "damages"],
    "finance": ["invest", "money", "market", "stock", "interest", "return",
                "portfolio", "risk", "capital", "fund", "bond", "compound",
                "dividend", "diversif", "asset"],
}

REFUSAL_PHRASES = ["I don't know", "I cannot", "as an AI", "I'm sorry",
                   "I'm unable", "I am not able"]


def score_behavioral(text, domain):
    """Score domain-specific behavioral quality 0-1.

    Rubric:
    - 0.0 for empty or very short (<20 chars)
    - 0.2 base for any non-empty response
    - +0.06 per domain keyword found (up to 0.5 from keywords)
    - -0.15 per refusal phrase
    - +0.1 if substantive (>100 chars)
    - +0.1 if multi-sentence (>1 period)
    - +0.1 if response is coherent (no repetition of same phrase >3x)
    """
    if not text or len(text.strip()) < 20:
        return 0.0

    score = 0.2
    text_lower = text.lower()

    # Domain keyword matches
    keywords = DOMAIN_KEYWORDS.get(domain, [])
    keyword_hits = sum(1 for k in keywords if k.lower() in text_lower)
    score += min(keyword_hits * 0.06, 0.5)

    # Refusal penalty
    for phrase in REFUSAL_PHRASES:
        if phrase.lower() in text_lower:
            score -= 0.15

    # Substantive bonus
    if len(text.strip()) > 100:
        score += 0.1

    # Multi-sentence bonus
    if text.count('.') > 1 or text.count('\n') > 1:
        score += 0.1

    # Anti-repetition: penalize if any 10+ word phrase repeats >3 times
    words = text.split()
    if len(words) > 30:
        from collections import Counter
        ngram_counts = Counter()
        for i in range(len(words) - 9):
            ngram = " ".join(words[i:i+10])
            ngram_counts[ngram] += 1
        if ngram_counts and max(ngram_counts.values()) > 3:
            score -= 0.2

    return max(0.0, min(1.0, score))


def phase_evaluate_behavioral(model, tokenizer, domain):
    """Generate responses and score behavioral quality for a domain."""
    prompts = BEHAVIORAL_PROMPTS.get(domain, [])
    if not prompts:
        return {"score": 0.0, "responses": [], "n_prompts": 0}

    sampler = make_sampler(temp=0.0)
    scores = []
    responses = []

    for prompt in prompts:
        try:
            generated = mlx_generate(
                model, tokenizer, prompt=prompt,
                max_tokens=200, sampler=sampler, verbose=False,
            )
            score = score_behavioral(generated, domain)
            scores.append(score)
            responses.append({
                "prompt": prompt.split("### Response:")[0].strip()[-80:],
                "response_preview": generated[:200],
                "score": round(score, 2),
            })
        except Exception as e:
            log(f"  WARNING: generation failed for {domain}: {e}")
            scores.append(0.0)
            responses.append({"prompt": prompt[:80], "response_preview": str(e), "score": 0.0})

    mean_score = float(np.mean(scores)) if scores else 0.0
    return {"score": round(mean_score, 3), "responses": responses, "n_prompts": len(prompts)}


# -- Training phase (per domain, memory-isolated) ------------------------------

def phase_train_domain(di, domain, skeleton):
    """Train one SFT adapter on Qwen3-4B-4bit. Self-contained for memory isolation."""
    dt0 = time.time()
    log(f"\n{'='*60}")
    log(f"Domain {di+1}/{len(DOMAINS)}: {domain} (skeleton index {di})")
    log(f"{'='*60}")

    # Load model
    model, tokenizer = load(MODEL_ID)
    log_memory(f"loaded-{domain}")

    # Apply LoRA with Grassmannian A-matrices
    n_lora = 0
    n_missing = 0
    for li in range(len(model.model.layers)):
        layer = model.model.layers[li]
        updates = []
        for key in TARGET_KEYS:
            m = layer
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is None:
                continue
            if not isinstance(m, (nn.Linear, nn.QuantizedLinear)):
                continue

            skey = f"layer_{li}_{key}_domain_{di}"
            if skey in skeleton:
                a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)
            else:
                log(f"  WARNING: no skeleton key {skey}")
                a_mx = None
                n_missing += 1

            lora = GrassmannianLoRALinear(m, rank=LORA_RANK, scale=LORA_SCALE, a_init=a_mx)
            updates.append((key, lora))
            n_lora += 1

        if updates:
            layer.update_modules(tree_unflatten(updates))

    mx.eval(model.parameters())
    if n_missing > 0:
        log(f"  WARNING: {n_missing} skeleton keys missing")

    # Freeze everything, unfreeze only B-matrices
    model.freeze()
    model.unfreeze(keys=["lora_b"], strict=False)
    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    log(f"  LoRA modules: {n_lora}, trainable params: {trainable:,}")

    # Load SFT data
    train_texts = load_domain_texts(domain, "train", 400)
    val_texts = load_domain_texts(domain, "valid", 50)
    if not train_texts:
        log(f"  SKIP: no training data for {domain}")
        cleanup(model, tokenizer)
        return None

    train_batches = []
    for text in train_texts:
        tokens, mask = tokenize_sft(text, tokenizer, MAX_SEQ)
        if len(tokens) >= 4:
            train_batches.append((tokens, mask))

    val_batches = []
    for text in val_texts:
        tokens, mask = tokenize_sft(text, tokenizer, MAX_SEQ)
        if len(tokens) >= 4:
            val_batches.append((tokens, mask))

    log(f"  Data: {len(train_batches)} train, {len(val_batches)} val samples")

    # Baseline val loss (with LoRA at zero-init, matches base model)
    n_val = min(len(val_batches), 25)
    base_loss = 0.0
    for tokens, mask in val_batches[:n_val]:
        loss = sft_loss_fn(model, mx.array([tokens]), mx.array([mask]))
        mx.eval(loss)
        base_loss += loss.item()
        del loss
    base_loss /= max(n_val, 1)
    log(f"  Base val loss (zero-init LoRA): {base_loss:.4f}")

    # Train
    optimizer = opt.Adam(learning_rate=LEARNING_RATE)
    loss_and_grad = nn.value_and_grad(model, sft_loss_fn)
    initial_loss = None
    losses_log = []

    gc.disable()
    for step in range(TRAIN_ITERS):
        idx = step % len(train_batches)
        tokens, mask = train_batches[idx]
        loss, grads = loss_and_grad(model, mx.array([tokens]), mx.array([mask]))
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        lv = loss.item()
        if initial_loss is None:
            initial_loss = lv
        if (step + 1) % 50 == 0:
            losses_log.append({"step": step + 1, "loss": round(lv, 4)})
        if (step + 1) % 100 == 0:
            log(f"  Step {step+1}: loss={lv:.4f}")
    gc.enable()

    # Final val loss
    final_loss = 0.0
    for tokens, mask in val_batches[:n_val]:
        loss = sft_loss_fn(model, mx.array([tokens]), mx.array([mask]))
        mx.eval(loss)
        final_loss += loss.item()
        del loss
    final_loss /= max(n_val, 1)

    converged = final_loss < base_loss
    pct_change = (base_loss - final_loss) / base_loss * 100 if base_loss > 0 else 0
    log(f"  Val loss: {base_loss:.4f} -> {final_loss:.4f} ({pct_change:+.1f}%, "
        f"{'CONVERGED' if converged else 'FAILED'})")

    # Behavioral evaluation
    log(f"  Behavioral eval...")
    behavioral = phase_evaluate_behavioral(model, tokenizer, domain)
    log(f"  Behavioral score: {behavioral['score']:.3f}")
    for r in behavioral["responses"]:
        preview = r["response_preview"][:100].replace('\n', ' ')
        log(f"    [{r['score']:.2f}] {preview}...")

    # Save adapter B-matrices
    adapter_dir = ADAPTERS_OUT / domain
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_params = {}
    for name, param in tree_flatten(model.trainable_parameters()):
        if "lora_b" in name:
            adapter_params[name] = param
    mx.savez(str(adapter_dir / "adapter.npz"), **adapter_params)
    n_saved = len(adapter_params)
    adapter_size_kb = sum(p.size * 2 for p in adapter_params.values()) / 1024  # bf16
    log(f"  Saved {n_saved} B-matrices ({adapter_size_kb:.0f} KB) to {adapter_dir}/")

    elapsed = time.time() - dt0
    result = {
        "base_loss": round(base_loss, 4),
        "final_loss": round(final_loss, 4),
        "initial_train_loss": round(initial_loss, 4) if initial_loss else None,
        "converged": converged,
        "pct_improvement": round(pct_change, 2),
        "behavioral": behavioral,
        "train_time_s": round(elapsed, 1),
        "n_train": len(train_batches),
        "n_val": n_val,
        "n_lora": n_lora,
        "trainable_params": trainable,
        "adapter_size_kb": round(adapter_size_kb, 1),
        "losses_log": losses_log,
    }

    log_memory(f"post-{domain}")
    cleanup(model, tokenizer, optimizer)
    return result


# -- Baseline behavioral evaluation (no adapter) -------------------------------

def phase_baseline_behavioral():
    """Evaluate base model behavioral quality (no adapter) for comparison."""
    log(f"\n{'='*60}")
    log(f"Baseline Behavioral Evaluation (no adapter)")
    log(f"{'='*60}")

    model, tokenizer = load(MODEL_ID)
    log_memory("baseline-loaded")

    baseline_results = {}
    for domain in DOMAINS:
        behavioral = phase_evaluate_behavioral(model, tokenizer, domain)
        baseline_results[domain] = behavioral
        log(f"  {domain}: behavioral={behavioral['score']:.3f}")

    cleanup(model, tokenizer)
    return baseline_results


# -- Main orchestrator ----------------------------------------------------------

def main():
    t0 = time.time()
    log("Pierre Pro: SFT 5 Domain Adapters on Qwen3-4B-4bit")
    log("=" * 60)
    log(f"Model: {MODEL_ID}")
    log(f"Recipe: rank={LORA_RANK}, scale={LORA_SCALE}, lr={LEARNING_RATE}, steps={TRAIN_ITERS}")
    log(f"Domains: {DOMAINS}")
    mx.random.seed(SEED)
    log_memory("start")

    ADAPTERS_OUT.mkdir(parents=True, exist_ok=True)

    # Verify skeleton exists
    if not SKELETON_PATH.exists():
        log(f"ERROR: skeleton not found at {SKELETON_PATH}")
        log("Run exp_pro_grassmannian_init first.")
        return

    skeleton = dict(np.load(str(SKELETON_PATH)))
    log(f"Loaded skeleton: {len(skeleton)} keys from {SKELETON_PATH.name}")

    # Verify skeleton keys for our domains
    for di, domain in enumerate(DOMAINS):
        sample_key = f"layer_0_self_attn.q_proj_domain_{di}"
        if sample_key in skeleton:
            a = skeleton[sample_key]
            log(f"  {domain} (idx {di}): A shape={a.shape}, dtype={a.dtype}")
        else:
            log(f"  ERROR: {domain} (idx {di}): key {sample_key} NOT FOUND")

    # Verify data exists
    for domain in DOMAINS:
        train_path = DATA_DIR / domain / "train.jsonl"
        val_path = DATA_DIR / domain / "valid.jsonl"
        if train_path.exists():
            n_train = sum(1 for _ in open(train_path))
            n_val = sum(1 for _ in open(val_path)) if val_path.exists() else 0
            log(f"  {domain}: {n_train} train, {n_val} val samples")
        else:
            log(f"  WARNING: {domain}: no data at {train_path}")

    # Phase 0: Baseline behavioral (no adapter)
    baseline_behavioral = phase_baseline_behavioral()
    log_memory("after-baseline")

    # Phase 1: Train all 5 domains
    results = {
        "model_id": MODEL_ID,
        "recipe": {"rank": LORA_RANK, "scale": LORA_SCALE, "lr": LEARNING_RATE,
                    "steps": TRAIN_ITERS, "max_seq": MAX_SEQ},
        "per_domain": {},
        "baseline_behavioral": {d: v["score"] for d, v in baseline_behavioral.items()},
    }
    converged_count = 0

    for di, domain in enumerate(DOMAINS):
        domain_result = phase_train_domain(di, domain, skeleton)
        if domain_result is None:
            results["per_domain"][domain] = {"skipped": True, "converged": False}
            continue
        results["per_domain"][domain] = domain_result
        if domain_result["converged"]:
            converged_count += 1

    # Summary
    total_time = time.time() - t0

    behavioral_scores = []
    for domain in DOMAINS:
        d = results["per_domain"].get(domain, {})
        if d.get("behavioral"):
            behavioral_scores.append(d["behavioral"]["score"])

    mean_behavioral = float(np.mean(behavioral_scores)) if behavioral_scores else 0.0

    improvements = [d["pct_improvement"] for d in results["per_domain"].values()
                    if not d.get("skipped", False)]
    mean_improvement = float(np.mean(improvements)) if improvements else 0.0

    baseline_scores = [v for v in results["baseline_behavioral"].values()]
    mean_baseline = float(np.mean(baseline_scores)) if baseline_scores else 0.0

    results["summary"] = {
        "total_domains": len(DOMAINS),
        "converged": converged_count,
        "mean_pct_improvement": round(mean_improvement, 2),
        "mean_behavioral": round(mean_behavioral, 3),
        "mean_baseline_behavioral": round(mean_baseline, 3),
        "behavioral_delta": round(mean_behavioral - mean_baseline, 3),
        "total_time_s": round(total_time, 1),
        "total_time_min": round(total_time / 60, 1),
    }

    # Kill criteria
    k812 = converged_count >= 4
    k813 = mean_behavioral >= 0.3

    results["kill_criteria"] = {
        "K812": {
            "pass": k812,
            "converged": converged_count,
            "threshold": 4,
            "detail": f"{converged_count}/5 converged",
        },
        "K813": {
            "pass": k813,
            "mean_behavioral": round(mean_behavioral, 3),
            "threshold": 0.3,
            "detail": f"mean behavioral {mean_behavioral:.3f} (baseline {mean_baseline:.3f})",
        },
    }
    results["all_pass"] = k812 and k813

    # Print results table
    log(f"\n{'='*60}")
    log(f"RESULTS")
    log(f"{'='*60}")
    log(f"Converged: {converged_count}/{len(DOMAINS)}")
    log(f"Mean SFT loss improvement: {mean_improvement:.1f}%")
    log(f"Mean behavioral: {mean_behavioral:.3f} (baseline: {mean_baseline:.3f}, delta: {mean_behavioral - mean_baseline:+.3f})")

    log(f"\n{'Domain':<12s} {'BaseLoss':>9s} {'SFTLoss':>9s} {'Impr%':>8s} "
        f"{'BL_Beh':>7s} {'SFT_Beh':>7s} {'Time':>6s}")
    log("-" * 65)
    for domain in DOMAINS:
        d = results["per_domain"].get(domain, {})
        if d.get("skipped"):
            log(f"{domain:<12s} SKIPPED")
            continue
        bl_beh = results["baseline_behavioral"].get(domain, 0)
        beh = d.get("behavioral", {}).get("score", 0)
        log(f"{domain:<12s} {d['base_loss']:9.4f} {d['final_loss']:9.4f} "
            f"{d['pct_improvement']:+7.1f}% {bl_beh:7.3f} {beh:7.3f} {d['train_time_s']:5.0f}s")

    log(f"\nKill Criteria:")
    for k, v in results["kill_criteria"].items():
        status = "PASS" if v["pass"] else "FAIL"
        log(f"  {k}: {status} -- {v['detail']}")
    log(f"\n{'ALL PASS' if results['all_pass'] else 'KILLED'}")
    log(f"Total time: {total_time/60:.1f} min")

    log_memory("final")
    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
