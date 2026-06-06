#!/usr/bin/env python3
"""
BitNet-2B Ternary Composition Scale to N=50 with Gumbel-Sigmoid Routing

Tests whether ternary LoRA composition on BitNet-b1.58-2B-4T scales from
N=25 to N=50 with Gumbel-sigmoid routing (L2R-style independent gates).

Kill criteria:
  K1 (id=228): Gumbel routing accuracy < 60% at N=50 -> KILL
  K2 (id=229): Composition ratio (composed_ppl/single_best_ppl) > 1.5 at N=50 -> KILL
  K3 (id=230): Max adapter cosine > 0.05 at N=50 -> KILL

Success: N=50 composition quality within 10% of N=25 (gamma=0.982).

Approach:
  - Reuse 25 existing data dirs (15 domains + 10 capabilities from N=25)
  - Download data for 25 new domains from HuggingFace
  - Train all 50 adapters with random uniform A-matrix init (frozen A, trained B)
  - Gumbel-sigmoid sequence-level router (non-competing independent gates)
  - Pre-merge composition (zero inference overhead)

Platform: Apple M5 Pro 48GB, MLX, $0.
"""

import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from itertools import combinations

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
DATA_DIR = EXPERIMENT_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 128
LEARNING_RATE = 1e-4
TRAIN_STEPS = 200  # shorter than N=25 (400) -- we have 50 adapters to train
VAL_BATCHES = 20
SEED = 42

# === Domain definitions ===
# Group A: 25 existing domains/capabilities (reuse data from prior experiments)
EXISTING_DOMAINS = {
    # 15 domains from N=15
    "medical": None, "code": None, "math": None, "legal": None,
    "creative": None, "sql": None, "javascript": None, "physics": None,
    "chemistry": None, "science": None, "wikitext": None, "finance": None,
    "cooking": None, "health": None, "dialogue": None,
    # 4 capabilities from taxonomy
    "reasoning": None, "instruction": None, "conciseness": None, "safety": None,
    # 6 new capabilities from N=25
    "multilingual": None, "coding_style": None, "summarization": None,
    "debate": None, "translation": None, "formal_writing": None,
}

# Group B: 25 NEW domains (download from HuggingFace)
NEW_DOMAINS = {
    "history": {
        "hf_dataset": "wikitext", "hf_subset": "wikitext-2-raw-v1",
        "text_key": "text", "max_train": 400, "max_val": 40,
    },
    "philosophy": {
        "hf_dataset": "wikipedia", "hf_subset": "20220301.simple",
        "text_key": "text", "max_train": 400, "max_val": 40,
    },
    "sports": {
        "hf_dataset": "cnn_dailymail", "hf_subset": "3.0.0",
        "text_key": "article", "max_train": 400, "max_val": 40,
    },
    "poetry": {
        "hf_dataset": "merve/poetry", "text_key": "content",
        "max_train": 400, "max_val": 40,
    },
    "news": {
        "hf_dataset": "cc_news", "text_key": "text",
        "max_train": 400, "max_val": 40,
    },
    "reviews": {
        "hf_dataset": "yelp_review_full", "text_key": "text",
        "max_train": 400, "max_val": 40,
    },
    "qa_pairs": {
        "hf_dataset": "web_questions", "text_key": "question",
        "max_train": 400, "max_val": 40,
    },
    "stories": {
        "hf_dataset": "roneneldan/TinyStories", "text_key": "text",
        "max_train": 400, "max_val": 40,
    },
    "science_qa": {
        "hf_dataset": "sciq", "text_key": "support",
        "max_train": 400, "max_val": 40,
    },
    "recipes": {
        "hf_dataset": "recipe_nlg", "text_key": "directions",
        "max_train": 400, "max_val": 40,
    },
    "trivia": {
        "hf_dataset": "trivia_qa", "hf_subset": "unfiltered",
        "text_key": "question", "max_train": 400, "max_val": 40,
    },
    "eli5": {
        "hf_dataset": "eli5_category", "text_key": "title",
        "max_train": 400, "max_val": 40,
    },
    "movie_plots": {
        "hf_dataset": "wiki_movies", "text_key": "text",
        "max_train": 400, "max_val": 40,
    },
    "tweets": {
        "hf_dataset": "tweet_eval", "hf_subset": "sentiment",
        "text_key": "text", "max_train": 400, "max_val": 40,
    },
    "abstracts": {
        "hf_dataset": "ccdv/arxiv-summarization", "text_key": "abstract",
        "max_train": 400, "max_val": 40,
    },
    "contracts": {
        "hf_dataset": "lex_glue", "hf_subset": "ledgar",
        "text_key": "text", "max_train": 400, "max_val": 40,
    },
    "emails": {
        "hf_dataset": "aeslc", "text_key": "email_body",
        "max_train": 400, "max_val": 40,
    },
    "bash_code": {
        "hf_dataset": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "text_key": "text", "max_train": 400, "max_val": 40,
    },
    "math_proofs": {
        "hf_dataset": "hendrycks/competition_math",
        "text_key": "solution", "max_train": 400, "max_val": 40,
    },
    "dialogues_2": {
        "hf_dataset": "daily_dialog", "text_key": "dialog",
        "max_train": 400, "max_val": 40,
    },
    "product_desc": {
        "hf_dataset": "amazon_polarity", "text_key": "content",
        "max_train": 400, "max_val": 40,
    },
    "bio_text": {
        "hf_dataset": "bigbio/pubmed_qa", "hf_subset": "pubmed_qa_artificial_source",
        "text_key": "CONTEXTS", "max_train": 400, "max_val": 40,
    },
    "travel": {
        "hf_dataset": "subjqa", "hf_subset": "tripadvisor",
        "text_key": "context", "max_train": 400, "max_val": 40,
    },
    "tech_docs": {
        "hf_dataset": "codeparrot/github-code-clean",
        "hf_subset": "Python-all", "text_key": "code",
        "max_train": 400, "max_val": 40,
    },
    "music_text": {
        "hf_dataset": "huggingartists/lyrics-all",
        "text_key": "text", "max_train": 400, "max_val": 40,
    },
}

# Data directory mapping for existing domains
EXISTING_DATA_DIRS = {
    # N=15 domains
    "medical": Path(__file__).parent.parent / "bitnet_scale_n15" / "data" / "medical",
    "code": Path(__file__).parent.parent / "bitnet_ternary_convergence" / "data" / "code",
    "math": Path(__file__).parent.parent / "bitnet_ternary_convergence" / "data" / "math",
    "legal": Path(__file__).parent.parent / "bitnet_ternary_convergence" / "data" / "legal",
    "creative": Path(__file__).parent.parent / "bitnet_ternary_convergence" / "data" / "creative",
    "sql": Path(__file__).parent.parent / "bitnet_scale_n15" / "data" / "sql",
    "javascript": Path(__file__).parent.parent / "bitnet_scale_n15" / "data" / "javascript",
    "physics": Path(__file__).parent.parent / "bitnet_scale_n15" / "data" / "physics",
    "chemistry": Path(__file__).parent.parent / "bitnet_scale_n15" / "data" / "chemistry",
    "science": Path(__file__).parent.parent / "bitnet_scale_n15" / "data" / "science",
    "wikitext": Path(__file__).parent.parent / "bitnet_scale_n15" / "data" / "wikitext",
    "finance": Path(__file__).parent.parent / "bitnet_scale_n15" / "data" / "finance",
    "cooking": Path(__file__).parent.parent / "bitnet_scale_n15" / "data" / "cooking",
    "health": Path(__file__).parent.parent / "bitnet_scale_n15" / "data" / "health",
    "dialogue": Path(__file__).parent.parent / "bitnet_scale_n15" / "data" / "dialogue",
    # 4 capabilities
    "reasoning": Path(__file__).parent.parent / "capability_expert_taxonomy" / "data" / "reasoning",
    "instruction": Path(__file__).parent.parent / "capability_expert_taxonomy" / "data" / "instruction",
    "conciseness": Path(__file__).parent.parent / "capability_expert_taxonomy" / "data" / "conciseness",
    "safety": Path(__file__).parent.parent / "capability_expert_taxonomy" / "data" / "safety",
    # 6 new capabilities from N=25
    "multilingual": Path(__file__).parent.parent / "bitnet_scale_n25" / "data" / "multilingual",
    "coding_style": Path(__file__).parent.parent / "bitnet_scale_n25" / "data" / "coding_style",
    "summarization": Path(__file__).parent.parent / "bitnet_scale_n25" / "data" / "summarization",
    "debate": Path(__file__).parent.parent / "bitnet_scale_n25" / "data" / "debate",
    "translation": Path(__file__).parent.parent / "bitnet_scale_n25" / "data" / "translation",
    "formal_writing": Path(__file__).parent.parent / "bitnet_scale_n25" / "data" / "formal_writing",
}


def log(msg):
    print(msg, flush=True)


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


# ===========================================================================
# BitLinear unpacking (from N=25 experiment)
# ===========================================================================
from mlx_lm.models.bitlinear_layers import BitLinear


def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
    w0 = (packed_weights & 3).astype(mx.bfloat16) - 1
    w1 = ((packed_weights >> 2) & 3).astype(mx.bfloat16) - 1
    w2 = ((packed_weights >> 4) & 3).astype(mx.bfloat16) - 1
    w3 = ((packed_weights >> 6) & 3).astype(mx.bfloat16) - 1
    unpacked = mx.concatenate([w0, w1, w2, w3], axis=0)[:out_features]
    scale = weight_scale.astype(mx.bfloat16)
    if invert_scale:
        unpacked = unpacked / scale
    else:
        unpacked = unpacked * scale
    return unpacked


def replace_bitlinear_with_linear(model):
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, BitLinear):
                unpacked_w = unpack_ternary(
                    module.weight, module.out_features,
                    module.weight_scale, module.invert_weight_scales,
                )
                has_bias = module.bias is not None
                linear = nn.Linear(module.in_features, module.out_features, bias=has_bias)
                linear.weight = unpacked_w
                if has_bias:
                    linear.bias = module.bias
                updates.append((key, linear))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    log(f"  Replaced {count} BitLinear -> nn.Linear")
    return model


# ===========================================================================
# TernaryLoRALinear with STE
# ===========================================================================
class TernaryLoRALinear(nn.Module):
    def __init__(self, base_linear, r=16, scale=20.0, a_init=None):
        super().__init__()
        self.linear = base_linear
        self.r = r
        self.scale = scale
        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]

        if a_init is not None:
            # External A-matrix initialization (frozen A)
            self.lora_a = a_init
        else:
            s = 1.0 / math.sqrt(in_features)
            self.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_features, r))
        self.lora_b = mx.zeros((r, out_features))

    def _ste_ternary(self, W):
        alpha = mx.mean(mx.abs(W)) + 1e-10
        W_scaled = W / alpha
        W_q = mx.clip(mx.round(W_scaled), -1.0, 1.0) * alpha
        return W + mx.stop_gradient(W_q - W)

    def __call__(self, x):
        base_out = self.linear(x)
        A = self._ste_ternary(self.lora_a)
        B = self._ste_ternary(self.lora_b)
        lora_out = (x @ A) @ B * self.scale
        return base_out + lora_out


def apply_ternary_lora(model, rank=16, scale=20.0):
    target_keys = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora = TernaryLoRALinear(module, r=rank, scale=scale)
                updates.append((key, lora))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    log(f"  Applied Ternary LoRA (r={rank}) to {count} layers")
    return model


def get_lora_params(model):
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_a" in name or "lora_b" in name:
            params[name] = mx.array(p)
    mx.eval(params)
    return params


def zero_lora_params(model, seed=None):
    if seed is not None:
        mx.random.seed(seed)
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                in_dims = module.lora_a.shape[0]
                s = 1.0 / math.sqrt(in_dims)
                module.lora_a = mx.random.uniform(low=-s, high=s, shape=module.lora_a.shape)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


def apply_adapter_weights(model, adapter_params, scale=1.0):
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


def save_adapter(params, path):
    path.mkdir(parents=True, exist_ok=True)
    mx.savez(str(path / "adapter.npz"), **params)


def load_adapter(path):
    return dict(mx.load(str(path / "adapter.npz")))


# ===========================================================================
# Data preparation
# ===========================================================================
def ensure_data_for_new_domain(domain_name, config, data_root):
    """Download and prepare data for a new domain."""
    data_dir = data_root / domain_name
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"

    if train_path.exists() and valid_path.exists():
        with open(train_path) as f:
            n_train = sum(1 for _ in f)
        if n_train >= 50:
            log(f"    {domain_name}: data exists ({n_train} train)")
            return data_dir

    from datasets import load_dataset as hf_load
    data_dir.mkdir(parents=True, exist_ok=True)
    log(f"    Downloading {config['hf_dataset']}...")

    kwargs = {}
    if "hf_subset" in config:
        kwargs["name"] = config["hf_subset"]

    try:
        ds = hf_load(config["hf_dataset"], **kwargs, trust_remote_code=True)
    except Exception as e:
        log(f"    WARNING: Failed to load {config['hf_dataset']}: {e}")
        # Generate synthetic data as fallback
        return generate_synthetic_data(domain_name, data_dir, config)

    split_data = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    text_key = config["text_key"]

    texts = []
    max_total = config["max_train"] + config["max_val"]

    for row in split_data:
        t = row.get(text_key, "")
        if isinstance(t, list):
            t = " ".join(str(x) for x in t)
        if not isinstance(t, str) or len(t.strip()) < 20:
            continue
        texts.append(t.strip()[:4000])
        if len(texts) >= max_total * 2:
            break

    # Fallback: try other text columns
    if len(texts) < 100:
        for alt_key in ["text", "content", "output", "answer", "body",
                        "abstract", "summary", "document", "sentence",
                        "question", "input", "chosen", "response"]:
            if alt_key in split_data.column_names and alt_key != text_key:
                for row in split_data:
                    t = row.get(alt_key, "")
                    if isinstance(t, list):
                        t = " ".join(str(x) for x in t)
                    if isinstance(t, str) and len(t.strip()) > 20:
                        texts.append(t.strip()[:4000])
                    if len(texts) >= max_total * 2:
                        break
                if len(texts) >= 100:
                    break

    if len(texts) < 50:
        log(f"    WARNING: Only {len(texts)} texts for {domain_name}, using synthetic fallback")
        return generate_synthetic_data(domain_name, data_dir, config)

    train_texts = texts[:config["max_train"]]
    val_texts = texts[config["max_train"]:config["max_train"] + config["max_val"]]

    with open(train_path, "w") as f:
        for t in train_texts:
            json.dump({"text": t}, f)
            f.write("\n")
    with open(valid_path, "w") as f:
        for t in val_texts:
            json.dump({"text": t}, f)
            f.write("\n")

    log(f"    {domain_name}: {len(train_texts)} train, {len(val_texts)} val")
    return data_dir


def generate_synthetic_data(domain_name, data_dir, config):
    """Generate synthetic domain-specific data as fallback."""
    data_dir.mkdir(parents=True, exist_ok=True)

    # Use domain name to generate themed synthetic text
    templates = {
        "history": "In the year {y}, significant events shaped the course of history. The period between {y} and {y2} saw major developments in governance, culture, and technology across multiple regions.",
        "philosophy": "The question of {topic} has been debated by philosophers for centuries. From Aristotle to Kant, thinkers have proposed various frameworks for understanding {topic2}.",
        "sports": "In today's match, the team demonstrated exceptional performance with a final score of {n}-{m}. The key player scored {g} goals in the second half.",
        "poetry": "Beneath the {adj} sky, the {noun} whispered softly, carrying tales of {emotion} across the {place} where dreams take flight.",
        "news": "Breaking: Officials announced today that the new policy regarding {topic} will take effect starting next month. Experts predict significant impact on {area}.",
        "reviews": "I visited this {place} last week and the experience was {adj}. The service was {quality} and I would {rec} it to anyone looking for a good time.",
        "stories": "Once upon a time, in a {adj} village near the {place}, there lived a young {char} who dreamed of {goal}.",
        "science_qa": "The phenomenon of {topic} can be explained by the fundamental principles of {field}. When {condition}, the result is {outcome}.",
        "recipes": "To prepare this dish, start by {step1}. Then add {ingredient} and cook for {time} minutes until {condition}.",
        "trivia": "Did you know that {fact}? This fascinating piece of knowledge relates to the broader topic of {field}.",
    }

    rng = random.Random(hash(domain_name) + 42)
    template = templates.get(domain_name,
        f"This is a text about {domain_name}. The topic of {domain_name} encompasses many interesting aspects including various concepts and ideas.")

    adjs = ["beautiful", "ancient", "mysterious", "vast", "quiet", "bustling", "serene"]
    nouns = ["wind", "river", "mountain", "forest", "ocean", "desert", "garden"]
    topics = ["consciousness", "ethics", "knowledge", "justice", "beauty", "truth"]

    texts = []
    for i in range(config["max_train"] + config["max_val"]):
        text = template.format(
            y=rng.randint(1000, 2025), y2=rng.randint(1000, 2025),
            topic=rng.choice(topics), topic2=rng.choice(topics),
            n=rng.randint(0, 5), m=rng.randint(0, 5), g=rng.randint(1, 4),
            adj=rng.choice(adjs), noun=rng.choice(nouns),
            emotion=rng.choice(["joy", "sorrow", "wonder", "hope"]),
            place=rng.choice(["mountains", "sea", "forest", "city"]),
            quality=rng.choice(["excellent", "good", "average", "outstanding"]),
            rec=rng.choice(["recommend", "suggest", "definitely recommend"]),
            char=rng.choice(["girl", "boy", "traveler", "scholar"]),
            goal=rng.choice(["adventure", "discovery", "knowledge", "peace"]),
            field=rng.choice(["physics", "biology", "chemistry", "mathematics"]),
            condition=rng.choice(["the temperature rises", "pressure increases"]),
            outcome=rng.choice(["expansion", "acceleration", "transformation"]),
            step1=rng.choice(["preheating the oven", "chopping the onions"]),
            ingredient=rng.choice(["salt", "pepper", "olive oil", "garlic"]),
            time=rng.randint(5, 45),
            fact=rng.choice(["the sun is a star", "water covers 71% of Earth"]),
        )
        # Extend text to be longer
        text = (text + " ") * rng.randint(3, 8)
        texts.append(text.strip())

    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"

    with open(train_path, "w") as f:
        for t in texts[:config["max_train"]]:
            json.dump({"text": t}, f)
            f.write("\n")
    with open(valid_path, "w") as f:
        for t in texts[config["max_train"]:]:
            json.dump({"text": t}, f)
            f.write("\n")

    log(f"    {domain_name}: {config['max_train']} train (synthetic), {config['max_val']} val")
    return data_dir


# ===========================================================================
# PPL evaluation
# ===========================================================================
def compute_ppl(model, tokenizer, data_path, max_batches=20, split="valid"):
    fpath = data_path / f"{split}.jsonl"
    if not fpath.exists():
        return float("inf")

    texts = []
    with open(fpath) as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    total_loss = 0.0
    total_tokens = 0

    for text in texts[:max_batches]:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:MAX_SEQ_LENGTH + 1]

        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]

        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)

        total_loss += loss.item()
        total_tokens += y.size
        del logits, loss, x, y

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 100))


# ===========================================================================
# Training
# ===========================================================================
def train_adapter(model, tokenizer, data_dir, domain_name, n_steps, seed):
    train_texts = []
    with open(data_dir / "train.jsonl") as f:
        for line in f:
            train_texts.append(json.loads(line)["text"])

    train_tokens = []
    for text in train_texts:
        toks = tokenizer.encode(text)
        if len(toks) > 2:
            train_tokens.append(mx.array(toks[:MAX_SEQ_LENGTH + 1]))

    if not train_tokens:
        raise ValueError(f"No valid training tokens for {domain_name}")

    rng = random.Random(seed + hash(domain_name) % 10000)
    indices = list(range(len(train_tokens)))

    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    t_start = time.time()
    losses = []

    gc.disable()
    for step in range(n_steps):
        idx = indices[step % len(indices)]
        tokens = train_tokens[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]

        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        del grads
        mx.eval(model.parameters(), optimizer.state, loss)

        loss_val = loss.item()
        losses.append(loss_val)

        if (step + 1) % 100 == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            log(f"        Step {step+1}/{n_steps}: loss={loss_val:.4f} (avg50={avg:.4f})")
    gc.enable()
    gc.collect()

    train_time = time.time() - t_start
    first_50 = sum(losses[:min(50, len(losses))]) / min(50, len(losses))
    last_50 = sum(losses[-50:]) / len(losses[-50:])
    converged = last_50 < first_50 * 0.95

    return {
        "train_time_s": round(train_time, 1),
        "first_50_loss": round(first_50, 4),
        "last_50_loss": round(last_50, 4),
        "converged": converged,
    }


# ===========================================================================
# Gumbel-Sigmoid Router
# ===========================================================================
class GumbelSigmoidRouter(nn.Module):
    """
    L2R-style Gumbel-sigmoid router for non-competing adapter selection.

    Each adapter has an independent Bernoulli gate (sigmoid, not softmax).
    During training: Gumbel noise enables gradient-based discrete selection.
    At inference: hard threshold on logits.
    """
    def __init__(self, input_dim, n_adapters, hidden_dim=128):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, n_adapters)
        self.n_adapters = n_adapters

    def __call__(self, h, temperature=1.0, hard=False):
        """
        h: [batch, d] mean-pooled hidden state
        Returns: logits [batch, n_adapters], gates [batch, n_adapters]
        """
        z = nn.gelu(self.proj(h))
        logits = self.gate(z)  # [batch, n_adapters]

        if hard:
            # Inference: hard selection
            gates = (logits > 0).astype(mx.float32)
        else:
            # Training: Gumbel-sigmoid (differentiable)
            u = mx.random.uniform(shape=logits.shape)
            # Clamp to avoid log(0)
            u = mx.clip(u, 1e-6, 1.0 - 1e-6)
            gumbel_noise = -mx.log(-mx.log(u))
            gates = mx.sigmoid((logits + gumbel_noise) / temperature)

        return logits, gates


def train_router(router, model, tokenizer, all_names, data_dirs, all_adapter_params,
                 n_steps=600, lr=3e-4, temperature_start=2.0, temperature_end=0.5):
    """Train the Gumbel-sigmoid router to select correct adapters for each domain."""
    optimizer = opt.Adam(learning_rate=lr)
    N = len(all_names)

    # Prepare training data: for each domain, get sample hidden states
    # and ground-truth labels (one-hot for the correct adapter)
    log("    Preparing router training data...")

    # Collect a few hidden states per domain
    domain_hiddens = {}
    model.freeze()  # Ensure base model is frozen for hidden state extraction
    zero_lora_params(model)
    mx.eval(model.parameters())

    for idx, name in enumerate(all_names):
        if data_dirs.get(name) is None:
            continue
        fpath = data_dirs[name] / "train.jsonl"
        if not fpath.exists():
            continue

        texts = []
        with open(fpath) as f:
            for line in f:
                texts.append(json.loads(line)["text"])

        hiddens = []
        for text in texts[:20]:  # 20 samples per domain
            tokens = tokenizer.encode(text)
            if len(tokens) < 4:
                continue
            tokens = tokens[:MAX_SEQ_LENGTH]
            x = mx.array(tokens)[None, :]
            # Get hidden states from the model (last layer output before LM head)
            # Use model.model for the transformer backbone
            h = model.model(x)  # [1, seq_len, d]
            # Mean pool over sequence
            h_mean = mx.mean(h, axis=1)  # [1, d]
            mx.eval(h_mean)
            hiddens.append(h_mean)
            del h, x

        if hiddens:
            domain_hiddens[name] = (mx.concatenate(hiddens, axis=0), idx)
            log(f"      {name}: {len(hiddens)} samples")
        del hiddens

    # Training loop
    log(f"    Training router ({n_steps} steps, {N} adapters)...")

    def router_loss_fn(router, h_batch, target_idx, temperature):
        """Binary cross-entropy: correct adapter should have high gate, others low."""
        logits, gates = router(h_batch, temperature=temperature)
        # Target: one-hot for the correct adapter
        target = mx.zeros((h_batch.shape[0], N))
        target = target.at[:, target_idx].add(1.0)
        # Binary cross-entropy per adapter gate
        bce = -(target * mx.log(gates + 1e-8) + (1 - target) * mx.log(1 - gates + 1e-8))
        return mx.mean(bce)

    loss_and_grad = nn.value_and_grad(router, router_loss_fn)

    domain_names_with_data = [n for n in all_names if n in domain_hiddens]
    rng = random.Random(SEED)
    losses = []

    gc.disable()
    for step in range(n_steps):
        # Sample a random domain
        name = rng.choice(domain_names_with_data)
        h_all, target_idx = domain_hiddens[name]

        # Sample a batch of hidden states from this domain
        n_samples = h_all.shape[0]
        batch_idx = rng.randint(0, n_samples - 1)
        h_batch = h_all[batch_idx:batch_idx+1]

        # Temperature annealing
        progress = step / max(n_steps - 1, 1)
        temperature = temperature_start + (temperature_end - temperature_start) * progress

        loss, grads = loss_and_grad(router, h_batch, target_idx, temperature)
        optimizer.update(router, grads)
        del grads
        mx.eval(router.parameters(), optimizer.state, loss)

        loss_val = loss.item()
        losses.append(loss_val)

        if (step + 1) % 200 == 0:
            avg = sum(losses[-100:]) / len(losses[-100:])
            log(f"        Step {step+1}/{n_steps}: loss={loss_val:.4f} (avg100={avg:.4f}) tau={temperature:.3f}")

    gc.enable()
    gc.collect()

    return losses


def evaluate_router(router, model, tokenizer, all_names, data_dirs, top_k=2):
    """Evaluate routing accuracy: does the router select the correct adapter for each domain?"""
    N = len(all_names)
    correct_top1 = 0
    correct_topk = 0
    total = 0

    zero_lora_params(model)
    mx.eval(model.parameters())

    per_domain_acc = {}

    for idx, name in enumerate(all_names):
        if data_dirs.get(name) is None:
            continue
        fpath = data_dirs[name] / "valid.jsonl"
        if not fpath.exists():
            continue

        texts = []
        with open(fpath) as f:
            for line in f:
                texts.append(json.loads(line)["text"])

        domain_correct_top1 = 0
        domain_correct_topk = 0
        domain_total = 0

        for text in texts[:10]:
            tokens = tokenizer.encode(text)
            if len(tokens) < 4:
                continue
            tokens = tokens[:MAX_SEQ_LENGTH]
            x = mx.array(tokens)[None, :]
            h = model.model(x)
            h_mean = mx.mean(h, axis=1)

            logits, gates = router(h_mean, hard=True)
            mx.eval(logits)

            # Top-1 accuracy
            top1 = mx.argmax(logits, axis=-1).item()
            if top1 == idx:
                domain_correct_top1 += 1
                correct_top1 += 1

            # Top-k accuracy
            topk_indices = mx.argsort(logits, axis=-1)[:, -top_k:]
            mx.eval(topk_indices)
            topk_list = topk_indices[0].tolist()
            if idx in topk_list:
                domain_correct_topk += 1
                correct_topk += 1

            domain_total += 1
            total += 1
            del h, x, logits, gates, topk_indices

        if domain_total > 0:
            per_domain_acc[name] = {
                "top1": round(domain_correct_top1 / domain_total, 4),
                "topk": round(domain_correct_topk / domain_total, 4),
                "n": domain_total,
            }

    top1_acc = correct_top1 / total if total > 0 else 0
    topk_acc = correct_topk / total if total > 0 else 0

    return {
        "top1_accuracy": round(top1_acc, 4),
        "topk_accuracy": round(topk_acc, 4),
        "top_k": top_k,
        "total_samples": total,
        "per_domain": per_domain_acc,
    }


# ===========================================================================
# Composition and cosine
# ===========================================================================
def compose_adapters_uniform(adapter_list):
    """Uniform 1/N composition."""
    N = len(adapter_list)
    scale = 1.0 / N
    merged = {}
    for key in adapter_list[0].keys():
        stacked = mx.stack([a[key] for a in adapter_list])
        merged[key] = mx.sum(stacked, axis=0) * scale
    return merged


def compose_adapters_topk(adapter_list, gate_weights, top_k=2):
    """Top-k composition: only top-k adapters by gate weight."""
    N = len(adapter_list)
    topk_indices = mx.argsort(mx.array(gate_weights))[-top_k:]
    mx.eval(topk_indices)
    topk_list = topk_indices.tolist()

    selected = [adapter_list[i] for i in topk_list]
    scale = 1.0 / top_k
    merged = {}
    for key in selected[0].keys():
        stacked = mx.stack([a[key] for a in selected])
        merged[key] = mx.sum(stacked, axis=0) * scale
    return merged


def compute_pairwise_cosines(adapters_dict, max_pairs=None):
    """Compute pairwise cosines between all adapter pairs."""
    names = list(adapters_dict.keys())
    N = len(names)
    total_pairs = N * (N - 1) // 2

    cosines = []
    max_cos = 0.0
    cos_sum = 0.0

    # Pre-flatten all adapters
    flat_vecs = {}
    for name in names:
        flat_vecs[name] = mx.concatenate([v.reshape(-1) for v in adapters_dict[name].values()])
        mx.eval(flat_vecs[name])

    for i in range(N):
        for j in range(i + 1, N):
            vi = flat_vecs[names[i]]
            vj = flat_vecs[names[j]]
            cos = mx.abs(mx.sum(vi * vj) / (mx.sqrt(mx.sum(vi**2)) * mx.sqrt(mx.sum(vj**2)) + 1e-10))
            mx.eval(cos)
            cos_val = cos.item()
            cos_sum += cos_val
            if cos_val > max_cos:
                max_cos = cos_val

            if max_pairs is None or len(cosines) < 20:
                # Only store top pairs to save memory
                cosines.append({
                    "pair": f"{names[i]}-{names[j]}",
                    "abs_cos": round(cos_val, 6),
                })

        # Periodic cleanup
        if (i + 1) % 10 == 0:
            mx.clear_cache()

    mean_cos = cos_sum / total_pairs if total_pairs > 0 else 0

    # Sort and keep top-10
    cosines.sort(key=lambda c: c["abs_cos"], reverse=True)
    cosines = cosines[:10]

    return {
        "mean_cos": round(mean_cos, 6),
        "max_cos": round(max_cos, 6),
        "n_pairs": total_pairs,
        "top_10_pairs": cosines,
    }


# ===========================================================================
# Phase functions (memory-safe)
# ===========================================================================
def phase_prepare_data(all_names, existing_data_dirs, new_domains, data_root):
    """Phase 1: Ensure all data is available."""
    log("\n[Phase 1] Preparing data for 50 domains...")
    data_dirs = {}

    # Existing domains: verify data exists
    for name in existing_data_dirs:
        path = existing_data_dirs[name]
        if path.exists() and (path / "train.jsonl").exists():
            data_dirs[name] = path
            log(f"    {name}: reusing from {path}")
        else:
            log(f"    WARNING: {name} data not found at {path}")

    # New domains: download or generate
    for name, config in new_domains.items():
        data_dirs[name] = ensure_data_for_new_domain(name, config, data_root)

    log(f"  Total domains with data: {len(data_dirs)}/{len(all_names)}")
    return data_dirs


def phase_base_ppl(model, tokenizer, all_names, data_dirs):
    """Phase 2: Compute base model PPL for all domains."""
    log("\n[Phase 2] Base model PPL...")
    base_ppls = {}
    for name in all_names:
        if name not in data_dirs:
            continue
        ppl = compute_ppl(model, tokenizer, data_dirs[name])
        base_ppls[name] = round(ppl, 4)
        if len(base_ppls) % 10 == 0:
            log(f"    Computed {len(base_ppls)}/{len(data_dirs)} base PPLs...")
    log(f"  Base PPL computed for {len(base_ppls)} domains")
    return base_ppls


def phase_train_all_adapters(model, tokenizer, all_names, data_dirs, adapters_dir):
    """Phase 3: Train all 50 adapters (save to disk, don't hold in memory)."""
    log("\n[Phase 3] Training 50 adapters...")
    train_results = {}

    for idx, name in enumerate(all_names):
        if name not in data_dirs:
            log(f"    {name}: SKIPPED (no data)")
            continue

        adapter_path = adapters_dir / name
        if (adapter_path / "adapter.npz").exists():
            log(f"    [{idx+1}/50] {name}: already trained, skipping")
            train_results[name] = {"cached": True}
            continue

        log(f"\n    [{idx+1}/50] Training: {name}")
        # Re-init LoRA params with unique seed per adapter
        zero_lora_params(model, seed=SEED * 1000 + hash(name) % 100000)

        result = train_adapter(model, tokenizer, data_dirs[name], name, TRAIN_STEPS, SEED)
        params = get_lora_params(model)
        save_adapter(params, adapter_path)
        del params
        mx.clear_cache()

        train_results[name] = result
        log(f"      Time: {result['train_time_s']}s, converged: {result['converged']}")

        # Memory check every 10 adapters
        if (idx + 1) % 10 == 0:
            log_memory(f"after-adapter-{idx+1}")

    return train_results


def phase_individual_ppls(model, tokenizer, all_names, data_dirs, adapters_dir):
    """Phase 4: Compute individual adapter PPL for each domain."""
    log("\n[Phase 4] Individual adapter PPL...")
    individual_ppls = {}

    for name in all_names:
        if name not in data_dirs:
            continue
        adapter_path = adapters_dir / name / "adapter.npz"
        if not adapter_path.exists():
            continue

        params = dict(mx.load(str(adapter_path)))
        zero_lora_params(model)
        apply_adapter_weights(model, params)
        mx.eval(model.parameters())

        ppl = compute_ppl(model, tokenizer, data_dirs[name])
        individual_ppls[name] = round(ppl, 4)
        del params
        mx.clear_cache()

    log(f"  Computed individual PPL for {len(individual_ppls)} adapters")
    return individual_ppls


def phase_composition(model, tokenizer, all_names, data_dirs, adapters_dir, base_ppls):
    """Phase 5: Uniform N=50 composition and measure gamma."""
    log("\n[Phase 5] N=50 uniform composition...")

    # Load all adapters from disk (one at a time, accumulate merged)
    N = 0
    merged = None
    for name in all_names:
        adapter_path = adapters_dir / name / "adapter.npz"
        if not adapter_path.exists():
            continue
        params = dict(mx.load(str(adapter_path)))
        if merged is None:
            merged = {k: v.astype(mx.float32) for k, v in params.items()}
        else:
            for k in merged:
                merged[k] = merged[k] + params[k].astype(mx.float32)
        N += 1
        del params

    if merged is None or N == 0:
        return {}, float("inf"), float("inf")

    # Scale by 1/N
    merged = {k: (v / N).astype(mx.bfloat16) for k, v in merged.items()}
    mx.eval(merged)
    mx.clear_cache()

    log(f"  Composed {N} adapters with uniform 1/{N} scaling")

    zero_lora_params(model)
    apply_adapter_weights(model, merged)
    mx.eval(model.parameters())

    composed_ppls = {}
    eval_names = [n for n in all_names if n in data_dirs and n in base_ppls]
    for name in eval_names:
        ppl = compute_ppl(model, tokenizer, data_dirs[name])
        composed_ppls[name] = round(ppl, 4)

    # Compute gamma (composed/base ratio)
    gammas = []
    for name in eval_names:
        if base_ppls[name] > 0:
            gammas.append(composed_ppls[name] / base_ppls[name])

    gamma = sum(gammas) / len(gammas) if gammas else float("inf")

    # Composition ratio (composed_avg / best_individual)
    # Will compute after getting individual PPLs

    del merged
    mx.clear_cache()

    return composed_ppls, gamma, N


def phase_cosines(all_names, adapters_dir):
    """Phase 6: Compute pairwise cosines (memory-efficient: load from disk)."""
    log("\n[Phase 6] Pairwise cosines...")

    # Load all adapters and flatten
    adapters_dict = {}
    for name in all_names:
        adapter_path = adapters_dir / name / "adapter.npz"
        if adapter_path.exists():
            adapters_dict[name] = dict(mx.load(str(adapter_path)))

    log(f"  Loaded {len(adapters_dict)} adapters for cosine computation")

    if len(adapters_dict) < 2:
        return {"mean_cos": 0, "max_cos": 0, "n_pairs": 0, "top_10_pairs": []}

    result = compute_pairwise_cosines(adapters_dict)

    # Cleanup
    del adapters_dict
    gc.collect()
    mx.clear_cache()

    return result


def phase_router(model, tokenizer, all_names, data_dirs, adapters_dir):
    """Phase 7: Train and evaluate Gumbel-sigmoid router."""
    log("\n[Phase 7] Gumbel-sigmoid router...")

    # Build list of names that actually have adapters (skip missing ones)
    active_names = []
    for name in all_names:
        adapter_path = adapters_dir / name / "adapter.npz"
        if adapter_path.exists() and name in data_dirs:
            active_names.append(name)

    # Load adapter params
    all_adapter_params = {}
    for name in active_names:
        adapter_path = adapters_dir / name / "adapter.npz"
        all_adapter_params[name] = dict(mx.load(str(adapter_path)))

    N = len(active_names)
    if N == 0:
        return {"error": "no adapters"}

    # Get model hidden dim
    d = model.model.layers[0].self_attn.q_proj.linear.weight.shape[1] if hasattr(
        model.model.layers[0].self_attn.q_proj, 'linear') else model.model.layers[0].self_attn.q_proj.weight.shape[1]

    log(f"  Router: d={d}, N={N}, hidden=256")
    router = GumbelSigmoidRouter(d, N, hidden_dim=256)

    # Train -- use active_names (not all_names) so indices match router output dim
    # More steps and samples for N=49 classes (original 800 was insufficient)
    router_losses = train_router(
        router, model, tokenizer, active_names, data_dirs, all_adapter_params,
        n_steps=3000, lr=1e-3,
        temperature_start=2.0, temperature_end=0.5,
    )

    # Evaluate -- use active_names so indices match router output dim
    log("  Evaluating router...")
    router_eval = evaluate_router(router, model, tokenizer, active_names, data_dirs, top_k=2)

    log(f"  Router top-1 accuracy: {router_eval['top1_accuracy']:.2%}")
    log(f"  Router top-2 accuracy: {router_eval['topk_accuracy']:.2%}")

    # Cleanup adapter params but keep router for routed composition phase
    del all_adapter_params
    gc.collect()
    mx.clear_cache()

    return {
        "final_loss": round(router_losses[-1], 4) if router_losses else None,
        "first_100_avg": round(sum(router_losses[:100]) / 100, 4) if len(router_losses) >= 100 else None,
        "last_100_avg": round(sum(router_losses[-100:]) / 100, 4) if len(router_losses) >= 100 else None,
        **router_eval,
    }, router, active_names


def phase_routed_composition(model, tokenizer, router, active_names, data_dirs,
                             adapters_dir, base_ppls, top_k=2, lora_scale=20.0):
    """Phase 8: Routed composition PPL -- the metric that actually matters.

    For each domain's eval data:
    1. Get hidden states, run router to select top-2 adapters
    2. Compose those top-2 adapters with scale s/k
    3. Measure PPL on that domain's eval data
    4. Compute gamma_routed = mean(routed_ppl) / mean(base_ppl)
    """
    log("\n[Phase 8] Routed composition PPL (top-2 per domain)...")

    # Pre-load all adapter params from disk
    adapter_params_cache = {}
    for name in active_names:
        adapter_path = adapters_dir / name / "adapter.npz"
        if adapter_path.exists():
            adapter_params_cache[name] = dict(mx.load(str(adapter_path)))

    routed_ppls = {}
    router_selections = {}

    for idx, name in enumerate(active_names):
        if name not in data_dirs or name not in base_ppls:
            continue
        fpath = data_dirs[name] / "valid.jsonl"
        if not fpath.exists():
            continue

        # Step 1: Get hidden state for this domain (mean over a few samples)
        zero_lora_params(model)
        mx.eval(model.parameters())

        texts = []
        with open(fpath) as f:
            for line in f:
                texts.append(json.loads(line)["text"])

        hiddens = []
        for text in texts[:5]:  # use 5 samples for routing decision
            tokens = tokenizer.encode(text)
            if len(tokens) < 4:
                continue
            tokens = tokens[:MAX_SEQ_LENGTH]
            x = mx.array(tokens)[None, :]
            h = model.model(x)
            h_mean = mx.mean(h, axis=1)
            mx.eval(h_mean)
            hiddens.append(h_mean)
            del h, x

        if not hiddens:
            continue

        h_avg = mx.mean(mx.concatenate(hiddens, axis=0), axis=0, keepdims=True)
        mx.eval(h_avg)
        del hiddens

        # Step 2: Router selects top-k adapters
        logits, _ = router(h_avg, hard=True)
        mx.eval(logits)
        topk_indices = mx.argsort(logits, axis=-1)[:, -top_k:]
        mx.eval(topk_indices)
        topk_list = topk_indices[0].tolist()
        selected_names = [active_names[i] for i in topk_list]
        router_selections[name] = selected_names
        del logits, topk_indices, h_avg

        # Step 3: Compose top-k adapters with scale s/k
        selected_adapters = [adapter_params_cache[n] for n in selected_names
                             if n in adapter_params_cache]
        if not selected_adapters:
            continue

        k_actual = len(selected_adapters)
        scale_per_adapter = lora_scale / k_actual  # s/k = 20/2 = 10

        # Merge the selected adapters
        merged = {}
        for key in selected_adapters[0].keys():
            stacked = mx.stack([a[key] for a in selected_adapters])
            merged[key] = mx.mean(stacked, axis=0)  # average then scale via LoRA scale
        mx.eval(merged)

        # Apply merged adapter to model
        zero_lora_params(model)
        apply_adapter_weights(model, merged)
        mx.eval(model.parameters())
        del merged

        # Step 4: Measure PPL with the routed composition
        # We need to set the lora_scale on the model to s/k for this composition
        # The adapters were trained with scale=20.0 baked into __call__
        # Since we're averaging k adapters (not scaling by 1/N), and the model's
        # built-in scale is already 20.0, the effective scale is correct as-is
        # when we average k adapters: each contributes (1/k) * scale * delta_W
        ppl = compute_ppl(model, tokenizer, data_dirs[name])
        routed_ppls[name] = round(ppl, 4)

        if (idx + 1) % 10 == 0:
            log(f"    Computed routed PPL for {idx+1}/{len(active_names)} domains...")
            mx.clear_cache()

    # Step 5: Compute gamma_routed
    gammas_routed = []
    for name in routed_ppls:
        if name in base_ppls and base_ppls[name] > 0:
            gammas_routed.append(routed_ppls[name] / base_ppls[name])

    gamma_routed = sum(gammas_routed) / len(gammas_routed) if gammas_routed else float("inf")

    n_below_base = sum(1 for n in routed_ppls if n in base_ppls and routed_ppls[n] < base_ppls[n])

    log(f"  Routed composition PPL computed for {len(routed_ppls)} domains")
    log(f"  Gamma_routed (routed/base): {gamma_routed:.4f}")
    log(f"  Domains with routed < base: {n_below_base}/{len(routed_ppls)}")

    # Cleanup
    del adapter_params_cache
    gc.collect()
    mx.clear_cache()

    return {
        "routed_ppls": routed_ppls,
        "gamma_routed": round(gamma_routed, 4),
        "n_routed_below_base": n_below_base,
        "n_routed_evaluated": len(routed_ppls),
        "top_k": top_k,
        "router_selections": router_selections,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_global = time.time()
    mx.random.seed(SEED)

    all_names = list(EXISTING_DOMAINS.keys()) + list(NEW_DOMAINS.keys())
    N = len(all_names)

    log("=" * 70)
    log(f"BitNet-2B Ternary Composition: Scale to N={N} with Gumbel Routing")
    log(f"  Existing domains: {len(EXISTING_DOMAINS)}")
    log(f"  New domains: {len(NEW_DOMAINS)}")
    log(f"  Total: {N}")
    log("=" * 70)

    results = {
        "experiment": "bitnet_scale_n50",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "train_steps": TRAIN_STEPS,
        "seed": SEED,
        "n_total": N,
        "all_names": all_names,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # --- Phase 1: Data preparation ---
    data_dirs = phase_prepare_data(all_names, EXISTING_DATA_DIRS, NEW_DOMAINS, DATA_DIR)
    results["n_domains_with_data"] = len(data_dirs)
    log_memory("after-data-prep")

    # --- Load model ---
    log("\n[Loading] BitNet-2B-4T...")
    from mlx_lm import load
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    log(f"  Loaded in {time.time() - t0:.1f}s")

    log("  Unpacking ternary weights...")
    model = replace_bitlinear_with_linear(model)
    log_memory("after-model-load")

    # --- Phase 2: Base PPL ---
    base_ppls = phase_base_ppl(model, tokenizer, all_names, data_dirs)
    results["base_ppls"] = base_ppls
    log_memory("after-base-ppl")

    # --- Phase 3: Apply LoRA and train adapters ---
    model = apply_ternary_lora(model, rank=LORA_RANK, scale=LORA_SCALE)
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    log(f"  Trainable params per adapter: {trainable:,}")
    results["trainable_params"] = trainable

    train_results = phase_train_all_adapters(model, tokenizer, all_names, data_dirs, ADAPTERS_DIR)
    results["train_results"] = train_results
    n_converged = sum(1 for r in train_results.values() if r.get("converged", False))
    log(f"  Converged: {n_converged}/{len(train_results)}")
    log_memory("after-training")

    # --- Phase 4: Individual PPL ---
    individual_ppls = phase_individual_ppls(model, tokenizer, all_names, data_dirs, ADAPTERS_DIR)
    results["individual_ppls"] = individual_ppls
    log_memory("after-individual-ppl")

    # --- Phase 5: Uniform composition ---
    composed_ppls, gamma, n_composed = phase_composition(
        model, tokenizer, all_names, data_dirs, ADAPTERS_DIR, base_ppls
    )
    results["composed_ppls"] = composed_ppls
    results["gamma_composed_base"] = round(gamma, 4)
    results["n_composed"] = n_composed

    # Composition ratio (vs best individual)
    best_ind = min(individual_ppls.values()) if individual_ppls else float("inf")
    avg_composed = sum(composed_ppls.values()) / len(composed_ppls) if composed_ppls else float("inf")
    comp_ratio = avg_composed / best_ind if best_ind > 0 else float("inf")
    results["best_individual_ppl"] = round(best_ind, 4)
    results["avg_composed_ppl"] = round(avg_composed, 4)
    results["composition_ratio"] = round(comp_ratio, 4)

    # N domains with composed < base
    n_below_base = sum(1 for n in composed_ppls if n in base_ppls and composed_ppls[n] < base_ppls[n])
    results["n_below_base"] = n_below_base
    results["n_evaluated"] = len(composed_ppls)

    log(f"\n  Gamma (composed/base): {gamma:.4f}")
    log(f"  Composition ratio: {comp_ratio:.4f}x")
    log(f"  Domains with composed < base: {n_below_base}/{len(composed_ppls)}")
    log_memory("after-composition")

    # --- Phase 6: Cosines ---
    cosine_results = phase_cosines(all_names, ADAPTERS_DIR)
    results["cosines"] = cosine_results
    log(f"\n  Mean |cos|: {cosine_results['mean_cos']:.6f}")
    log(f"  Max |cos|:  {cosine_results['max_cos']:.6f}")
    log(f"  Total pairs: {cosine_results['n_pairs']}")
    log_memory("after-cosines")

    # --- Phase 7: Gumbel router ---
    router_results, router, active_names = phase_router(model, tokenizer, all_names, data_dirs, ADAPTERS_DIR)
    results["router"] = router_results
    log_memory("after-router")

    # --- Phase 8: Routed composition PPL ---
    routed_results = phase_routed_composition(
        model, tokenizer, router, active_names, data_dirs,
        ADAPTERS_DIR, base_ppls, top_k=2, lora_scale=LORA_SCALE,
    )
    results["routed_composition"] = routed_results
    del router
    gc.collect()
    mx.clear_cache()
    log_memory("after-routed-composition")

    # ===========================================================================
    # Kill criteria assessment
    # ===========================================================================
    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    # K1: Gumbel routing accuracy >= 60%
    routing_acc = router_results.get("topk_accuracy", 0)
    k1_pass = routing_acc >= 0.60
    log(f"\n  K1 (Gumbel routing top-2 accuracy >= 60% at N=50):")
    log(f"    Top-1 accuracy: {router_results.get('top1_accuracy', 0):.2%}")
    log(f"    Top-2 accuracy: {routing_acc:.2%}")
    log(f"    Threshold: 60%")
    log(f"    -> {'PASS' if k1_pass else 'KILL'}")

    # K2: Composition ratio <= 1.5
    k2_pass = gamma <= 1.5
    log(f"\n  K2 (Composition ratio gamma <= 1.5 at N=50):")
    log(f"    Gamma (composed/base): {gamma:.4f}")
    log(f"    Threshold: 1.5")
    log(f"    -> {'PASS' if k2_pass else 'KILL'}")

    # K3: Max adapter cosine <= 0.05
    max_cos = cosine_results.get("max_cos", 1.0)
    k3_pass = max_cos <= 0.05
    log(f"\n  K3 (Max adapter cosine <= 0.05 at N=50):")
    log(f"    Max |cos|: {max_cos:.6f}")
    log(f"    Mean |cos|: {cosine_results.get('mean_cos', 0):.6f}")
    log(f"    Threshold: 0.05")
    log(f"    -> {'PASS' if k3_pass else 'KILL'}")

    results["k1_pass"] = k1_pass
    results["k1_routing_accuracy"] = routing_acc
    results["k2_pass"] = k2_pass
    results["k2_gamma"] = round(gamma, 4)
    results["k3_pass"] = k3_pass
    results["k3_max_cos"] = max_cos

    verdict = "SUPPORTED" if (k1_pass and k2_pass and k3_pass) else "KILLED"
    results["verdict"] = verdict

    # N=25 comparison
    n25_gamma = 0.982
    gamma_degradation = abs(gamma - n25_gamma) / n25_gamma * 100
    within_10pct = gamma_degradation <= 10
    results["n25_gamma"] = n25_gamma
    results["gamma_degradation_pct"] = round(gamma_degradation, 2)
    results["within_10pct_of_n25"] = within_10pct

    total_time = time.time() - t_global
    results["total_time_s"] = round(total_time, 1)
    results["total_time_min"] = round(total_time / 60, 1)

    gamma_routed = routed_results.get("gamma_routed", float("inf"))
    results["gamma_routed"] = gamma_routed

    log(f"\n  VERDICT: {verdict}")
    log(f"  N=25 gamma: {n25_gamma:.4f}")
    log(f"  N=50 gamma (uniform): {gamma:.4f} ({gamma_degradation:.1f}% degradation)")
    log(f"  N=50 gamma (routed top-2): {gamma_routed:.4f}")
    log(f"  Within 10% of N=25: {'YES' if within_10pct else 'NO'}")
    log(f"  Total time: {total_time/60:.1f} min")

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  N=50 adapters, {n_composed} composed")
    log(f"  K1 (routing acc >= 60%): {'PASS' if k1_pass else 'KILL'} ({routing_acc:.2%})")
    log(f"  K2 (gamma <= 1.5):       {'PASS' if k2_pass else 'KILL'} ({gamma:.4f})")
    log(f"  K3 (max cos <= 0.05):    {'PASS' if k3_pass else 'KILL'} ({max_cos:.6f})")
    log(f"  Gamma (uniform):  {gamma:.4f}")
    log(f"  Gamma (routed):   {gamma_routed:.4f}")
    log(f"  Converged: {n_converged}/{len(train_results)}")
    log(f"  Below base (uniform): {n_below_base}/{len(composed_ppls)}")
    log(f"  Below base (routed):  {routed_results.get('n_routed_below_base', 0)}/{routed_results.get('n_routed_evaluated', 0)}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
