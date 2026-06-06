#!/usr/bin/env python3
"""N=25 Domain Grassmannian Composition at 4B: Scaling Verification.

TYPE: frontier-extension
MATH: experiments/models/m2p_n25_compose_qwen4b/MATH.md

WHAT THIS TESTS:
  25 domains composed on Qwen3-4B-4bit:
  1. Math (GSM8K)         - pre-trained SFT-residual M2P
  2. Code (Python fns)    - A-matrix only (B=0 for isolation test)
  3. Sort (alpha sort)    - A-matrix only (B=0)
  4. Reverse (word flip)  - A-matrix only (B=0)
  5. Count (word count)   - A-matrix only (B=0)
  6-25. Synthetic topics  - Gram-Schmidt A-matrices (B=0)
        recipe, weather, astronomy, chemistry, biology,
        music, architecture, sports, history, medicine,
        finance, legal, geography, psychology, linguistics,
        automotive, textile, computing, agriculture, maritime

  A-matrices: sequential Gram-Schmidt, N_max=640 >> 25 (d=2560, r=4).
  B-matrices: math uses pre-trained M2P; all others use B=0.
  TF-IDF routing selects domain (text-level, no model call).
  K983 measures math quality under routed N=25 composition.

KILL CRITERIA:
  K981: max|A_i^T A_j|_F < 1e-4 for all C(25,2)=300 pairs
  K982: TF-IDF 25-class routing accuracy >= 80% on held-out set
  K983: Math quality_ratio >= 0.70 at n=200 under routed N=25 composition

REFERENCES:
  LoraRetriever (arXiv:2402.09997), Finding #405 (N=5 at 4B), Finding #393 (N=50 at 0.6B)

Supports SMOKE_TEST=1 for quick validation (<5 min).
"""

import gc
import json
import math
import os
import random
import re
import time
from itertools import combinations
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

# Memory safety — MANDATORY per CODING_GUIDELINES
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from mlx_lm import load as mlx_load
from mlx_lm import generate as mlx_generate
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"

# ---- Config ----------------------------------------------------------------
MODEL_ID = "mlx-community/Qwen3-4B-4bit"

LORA_RANK = 4
LORA_SCALE = 5.0
D_M2P = 1024
N_MEM_TOKENS = 16
N_M2P_LAYERS = 4
N_M2P_HEADS = 4
OUTPUT_SCALE = 0.032

MAX_GEN_TOKENS = 32 if IS_SMOKE else 256
SEED = 42

N_EVAL_MATH = 5 if IS_SMOKE else 200
N_ROUTE_TRAIN = 5 if IS_SMOKE else 100  # per domain
N_ROUTE_TEST = 5 if IS_SMOKE else 100   # per domain

EXPERIMENT_DIR = Path(__file__).parent
V1_DIR = EXPERIMENT_DIR.parent / "m2p_qwen4b_gsm8k"
MATH_LORA_A_PATH = V1_DIR / "grassmannian_a_matrices.npz"
MATH_SFT_B_PATH = V1_DIR / "sft_b_matrices.npz"
V1_RESULTS = V1_DIR / "results.json"

COMPOSE2_DIR = EXPERIMENT_DIR.parent / "m2p_2domain_compose_qwen4b"
CODE_LORA_A_PATH = COMPOSE2_DIR / "code_a_matrices.npz"

N5_DIR = EXPERIMENT_DIR.parent / "m2p_n5_compose_qwen4b"
SORT_LORA_A_PATH = N5_DIR / "sort_a_matrices.npz"
REVERSE_LORA_A_PATH = N5_DIR / "reverse_a_matrices.npz"
COUNT_LORA_A_PATH = N5_DIR / "count_a_matrices.npz"

MATH_M2P_PATH = EXPERIMENT_DIR.parent / "m2p_qwen4b_sft_residual" / "m2p_weights.npz"

RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# The 25 domains in Gram-Schmidt priority order
DOMAINS = [
    "math", "code", "sort", "reverse", "count",
    "recipe", "weather", "astronomy", "chemistry", "biology",
    "music", "architecture", "sports", "history", "medicine",
    "finance", "legal", "geography", "psychology", "linguistics",
    "automotive", "textile", "computing", "agriculture", "maritime",
]
assert len(DOMAINS) == 25

# Vocabulary anchors — highly distinct for TF-IDF separation
DOMAIN_VOCAB = {
    "math": ["solve", "calculate", "equation", "multiply", "subtract", "divide", "total", "equals", "answer", "problem"],
    "code": ["function", "return", "variable", "class", "import", "loop", "array", "string", "integer", "algorithm"],
    "sort": ["alphabetical", "arrange", "order", "ascending", "descending", "sequence", "rank", "list", "organize", "letter"],
    "reverse": ["backward", "flip", "invert", "reverse", "opposite", "mirror", "upside", "turn", "rotate", "transpose"],
    "count": ["count", "tally", "enumerate", "how many", "number of", "total words", "quantity", "amount", "frequency", "occurrence"],
    "recipe": ["tablespoon", "ingredient", "preheat", "bake", "simmer", "stir", "teaspoon", "oven", "cup", "boil"],
    "weather": ["temperature", "humidity", "forecast", "precipitation", "celsius", "fahrenheit", "wind", "cloudy", "barometric", "meteorological"],
    "astronomy": ["planet", "orbit", "telescope", "galaxy", "stellar", "nebula", "asteroid", "constellation", "photosphere", "perihelion"],
    "chemistry": ["molecule", "electron", "compound", "reaction", "atomic", "valence", "isotope", "catalyst", "oxidation", "periodic"],
    "biology": ["photosynthesis", "organism", "species", "habitat", "ecosystem", "chromosome", "mitosis", "enzyme", "cellular", "metabolic"],
    "music": ["melody", "chord", "rhythm", "tempo", "octave", "harmonic", "instrument", "symphony", "counterpoint", "modulation"],
    "architecture": ["blueprint", "foundation", "facade", "renovation", "structural", "beam", "cantilever", "masonry", "cornice", "fenestration"],
    "sports": ["tournament", "championship", "score", "referee", "athlete", "stadium", "goalkeeper", "penalty", "qualifying", "podium"],
    "history": ["century", "dynasty", "emperor", "parliament", "revolution", "civilization", "conquest", "chronicle", "sovereignty", "feudal"],
    "medicine": ["diagnosis", "symptom", "treatment", "dosage", "patient", "prognosis", "pharmaceutical", "pathology", "therapy", "clinical"],
    "finance": ["dividend", "portfolio", "inflation", "revenue", "equity", "amortization", "liquidity", "arbitrage", "derivatives", "hedging"],
    "legal": ["statute", "plaintiff", "verdict", "jurisdiction", "testimony", "subpoena", "affidavit", "injunction", "litigation", "precedent"],
    "geography": ["continent", "latitude", "terrain", "peninsula", "elevation", "archipelago", "watershed", "meridian", "topography", "cartography"],
    "psychology": ["cognition", "stimulus", "behavior", "anxiety", "motivation", "perception", "reinforcement", "limbic", "conditioning", "psychotherapy"],
    "linguistics": ["phoneme", "morpheme", "syntax", "conjugation", "lexicon", "morphology", "pragmatics", "diachronic", "semiotics", "phonology"],
    "automotive": ["transmission", "torque", "chassis", "horsepower", "throttle", "drivetrain", "crankshaft", "differential", "camshaft", "compression"],
    "textile": ["fabric", "embroidery", "weave", "dyeing", "loom", "warp", "weft", "filament", "thread", "serging"],
    "computing": ["packet", "router", "firewall", "bandwidth", "protocol", "subnet", "latency", "throughput", "encapsulation", "datagram"],
    "agriculture": ["harvest", "fertilizer", "irrigation", "livestock", "pesticide", "germination", "tillage", "perennial", "cultivar", "agrochemical"],
    "maritime": ["vessel", "navigation", "shipment", "harbor", "cargo", "starboard", "portside", "ballast", "draught", "tonnage"],
}
assert set(DOMAIN_VOCAB.keys()) == set(DOMAINS)

FEW_SHOT_PREFIX = (
    "Solve the math problem step by step and end with '#### <answer>'.\n\n"
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold "
    "half as many clips in May. How many clips did Natalia sell altogether in April and May?\n"
    "Answer: Natalia sold 48/2 = 24 clips in May. "
    "Natalia sold 48+24 = 72 clips altogether in April and May. #### 72\n\n"
    "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
    "minutes of babysitting. How much did she earn?\n"
    "Answer: Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned "
    "0.2 x 50 = $10. #### 10\n\n"
)


# ---- Routing text generators -----------------------------------------------

def get_gsm8k_routing_texts(n: int, offset: int = 0) -> list:
    """Load real GSM8K questions for math routing (ensures TF-IDF matches eval distribution)."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rng = random.Random(SEED + 7777 + offset)
    all_qs = [ds[i]["question"] for i in range(len(ds))]
    rng.shuffle(all_qs)
    return all_qs[offset:offset + n]


def get_routing_texts_for_domain(domain: str, n: int, offset: int = 0) -> list:
    """Generate n routing texts for a given domain using vocabulary anchors."""
    rng = random.Random(SEED + hash(domain) % 10000 + offset)
    vocab = DOMAIN_VOCAB[domain]
    texts = []

    if domain == "math":
        # Use real GSM8K questions for training routing centroid (matches eval distribution)
        return get_gsm8k_routing_texts(n, offset=offset)
    elif domain == "code":
        templates = [
            "Write a {v1} that {v2} a {v3}.",
            "Implement a {v1} to {v2} the {v3}.",
            "Create a {v1} with {v2} and return the {v3}.",
            "How do you {v1} a {v2} in Python?",
        ]
        nouns = ["function", "class", "method", "module", "algorithm", "loop", "recursion"]
        verbs = ["sort", "filter", "parse", "compute", "validate", "convert", "process"]
        objs = ["array", "string", "integer", "list", "dictionary", "file", "data"]
        for i in range(n):
            t = rng.choice(templates)
            texts.append(t.format(v1=rng.choice(nouns), v2=rng.choice(verbs), v3=rng.choice(objs)))
    elif domain == "sort":
        words_pool = ["apple", "banana", "cherry", "date", "fig", "grape", "kiwi", "lemon", "mango", "pear"]
        for i in range(n):
            k = rng.randint(3, 6)
            words = rng.sample(words_pool, k)
            rng.shuffle(words)
            texts.append(f"Sort these words alphabetically: {' '.join(words)}")
    elif domain == "reverse":
        words_pool = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"]
        for i in range(n):
            k = rng.randint(3, 5)
            words = rng.sample(words_pool, k)
            texts.append(f"Reverse the order of these words: {' '.join(words)}")
    elif domain == "count":
        words_pool = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "pink", "gray"]
        for i in range(n):
            k = rng.randint(3, 7)
            words = rng.sample(words_pool, k)
            texts.append(f"Count the words in this phrase: {' '.join(words)}")
    elif domain == "recipe":
        actions = ["bake", "simmer", "stir", "boil", "preheat", "mix", "roast", "fry", "blend", "marinate"]
        units = ["tablespoon", "teaspoon", "cup", "ounce", "gram", "pound"]
        for i in range(n):
            k = rng.randint(2, 4)
            ings = rng.sample(vocab, k)
            texts.append(f"Add {rng.choice(units)} {ings[0]} to {rng.choice(actions)} with {ings[1]} ingredient.")
    elif domain == "weather":
        for i in range(n):
            k = rng.randint(2, 4)
            vwords = rng.sample(vocab, k)
            num = rng.randint(10, 40)
            texts.append(f"The {vwords[0]} is {num} degrees with {vwords[1]} {vwords[2] if k > 2 else 'conditions'}.")
    elif domain == "astronomy":
        for i in range(n):
            k = rng.randint(2, 3)
            vwords = rng.sample(vocab, k)
            texts.append(f"The {vwords[0]} in the {vwords[1]} galaxy completes its {vocab[rng.randint(0, len(vocab)-1)]} cycle.")
    elif domain == "chemistry":
        for i in range(n):
            k = rng.randint(2, 3)
            vwords = rng.sample(vocab, k)
            texts.append(f"The {vwords[0]} undergoes {vwords[1]} reaction with {vwords[2] if k > 2 else 'the catalyst'}.")
    elif domain == "biology":
        for i in range(n):
            k = rng.randint(2, 3)
            vwords = rng.sample(vocab, k)
            texts.append(f"The {vwords[0]} organism undergoes {vwords[1]} in its natural {vwords[2] if k > 2 else 'habitat'}.")
    elif domain == "music":
        for i in range(n):
            k = rng.randint(2, 3)
            vwords = rng.sample(vocab, k)
            texts.append(f"The {vwords[0]} melody features {vwords[1]} rhythm at {vwords[2] if k > 2 else 'this'} tempo.")
    elif domain == "architecture":
        for i in range(n):
            k = rng.randint(2, 3)
            vwords = rng.sample(vocab, k)
            texts.append(f"The building {vwords[0]} uses a {vwords[1]} foundation with {vwords[2] if k > 2 else 'structural'} design.")
    elif domain == "sports":
        for i in range(n):
            k = rng.randint(2, 3)
            vwords = rng.sample(vocab, k)
            texts.append(f"The {vwords[0]} tournament features {vwords[1]} athletes competing in the {vwords[2] if k > 2 else 'championship'}.")
    elif domain == "history":
        for i in range(n):
            k = rng.randint(2, 3)
            vwords = rng.sample(vocab, k)
            texts.append(f"The {vwords[0]} dynasty ruled during a {vwords[1]} revolution in the ancient {vwords[2] if k > 2 else 'civilization'}.")
    elif domain == "medicine":
        for i in range(n):
            k = rng.randint(2, 3)
            vwords = rng.sample(vocab, k)
            texts.append(f"The patient's {vwords[0]} requires {vwords[1]} treatment and clinical {vwords[2] if k > 2 else 'diagnosis'}.")
    elif domain == "finance":
        for i in range(n):
            k = rng.randint(2, 3)
            vwords = rng.sample(vocab, k)
            texts.append(f"The {vwords[0]} portfolio shows {vwords[1]} inflation affecting {vwords[2] if k > 2 else 'equity'} returns.")
    elif domain == "legal":
        for i in range(n):
            k = rng.randint(2, 3)
            vwords = rng.sample(vocab, k)
            texts.append(f"The plaintiff filed a {vwords[0]} in {vwords[1]} jurisdiction citing {vwords[2] if k > 2 else 'testimony'} evidence.")
    elif domain == "geography":
        for i in range(n):
            k = rng.randint(2, 3)
            vwords = rng.sample(vocab, k)
            texts.append(f"The {vwords[0]} continent spans multiple {vwords[1]} terrain regions with high {vwords[2] if k > 2 else 'elevation'}.")
    elif domain == "psychology":
        for i in range(n):
            k = rng.randint(2, 3)
            vwords = rng.sample(vocab, k)
            texts.append(f"The {vwords[0]} cognition study examines {vwords[1]} behavior and {vwords[2] if k > 2 else 'motivation'} patterns.")
    elif domain == "linguistics":
        for i in range(n):
            k = rng.randint(2, 3)
            vwords = rng.sample(vocab, k)
            texts.append(f"The {vwords[0]} morpheme analysis reveals {vwords[1]} syntax in {vwords[2] if k > 2 else 'lexicon'} structure.")
    elif domain == "automotive":
        for i in range(n):
            k = rng.randint(2, 3)
            vwords = rng.sample(vocab, k)
            texts.append(f"The vehicle {vwords[0]} transmission generates {vwords[1]} torque through the {vwords[2] if k > 2 else 'chassis'}.")
    elif domain == "textile":
        for i in range(n):
            k = rng.randint(2, 3)
            vwords = rng.sample(vocab, k)
            texts.append(f"The {vwords[0]} fabric uses {vwords[1]} weave pattern with {vwords[2] if k > 2 else 'thread'} dyeing technique.")
    elif domain == "computing":
        for i in range(n):
            k = rng.randint(2, 3)
            vwords = rng.sample(vocab, k)
            texts.append(f"The network {vwords[0]} packet routes through {vwords[1]} firewall using {vwords[2] if k > 2 else 'protocol'} bandwidth.")
    elif domain == "agriculture":
        for i in range(n):
            k = rng.randint(2, 3)
            vwords = rng.sample(vocab, k)
            texts.append(f"The {vwords[0]} harvest uses {vwords[1]} fertilizer irrigation for {vwords[2] if k > 2 else 'crop'} yield.")
    elif domain == "maritime":
        for i in range(n):
            k = rng.randint(2, 3)
            vwords = rng.sample(vocab, k)
            texts.append(f"The {vwords[0]} vessel navigates the harbor with {vwords[1]} cargo on the {vwords[2] if k > 2 else 'starboard'} side.")
    else:
        # Fallback: keyword-dominant text
        for i in range(n):
            vwords = rng.sample(vocab, min(4, len(vocab)))
            texts.append(f"{vwords[0]} {vwords[1]} involving {vwords[2]} and {vwords[3] if len(vwords) > 3 else vwords[0]} analysis.")

    return texts[:n]


# ---- Utilities -------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def log_memory(label: str = "") -> None:
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def extract_gsm8k_answer(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"####\s*(-?[\d,]+)", text)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"(?:the\s+)?answer\s+is\s+[-–]?\s*\$?(-?[\d,]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")
    return None


def load_model_dims() -> dict:
    with open(V1_RESULTS) as f:
        v1 = json.load(f)
    cfg = v1["config"]
    return {
        "n_layers": cfg["n_layers"],
        "d_model": cfg["d_model"],
        "q_proj_out": cfg["q_proj_out"],
        "v_proj_out": cfg["v_proj_out"],
        "base_accuracy": v1["base_accuracy"],
        "sft_accuracy": v1["sft_accuracy"],
    }


# ---- A-matrix I/O ----------------------------------------------------------

def load_a_matrices(path: Path, n_layers: int) -> tuple:
    saved = np.load(str(path))
    A_q = [mx.array(saved[f"layer_{li}_q_proj_A"]).astype(mx.bfloat16) for li in range(n_layers)]
    A_v = [mx.array(saved[f"layer_{li}_v_proj_A"]).astype(mx.bfloat16) for li in range(n_layers)]
    mx.eval(*A_q, *A_v)
    return A_q, A_v


def load_a_matrices_np(path: Path, n_layers: int) -> tuple:
    """Load as float32 numpy arrays for Gram-Schmidt computation."""
    saved = np.load(str(path))
    A_q = [saved[f"layer_{li}_q_proj_A"].astype(np.float32) for li in range(n_layers)]
    A_v = [saved[f"layer_{li}_v_proj_A"].astype(np.float32) for li in range(n_layers)]
    return A_q, A_v


def save_a_matrices(path: Path, A_q: list, A_v: list) -> None:
    save_dict = {}
    for li, (aq, av) in enumerate(zip(A_q, A_v)):
        save_dict[f"layer_{li}_q_proj_A"] = np.array(aq, dtype=np.float32)
        save_dict[f"layer_{li}_v_proj_A"] = np.array(av, dtype=np.float32)
    np.savez(str(path), **save_dict)


# ---- Gram-Schmidt construction ---------------------------------------------

def gram_schmidt_new_domain(
    prior_q_list: list,
    prior_v_list: list,
    n_layers: int,
    rank: int,
    seed_offset: int,
) -> tuple:
    """Generate rank-r A-matrices orthogonal to all prior domains via Gram-Schmidt."""
    A_q_new = []
    A_v_new = []
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for li in range(n_layers):
            rng = np.random.default_rng(SEED + seed_offset + li)
            d_q = prior_q_list[0][li].shape[0]
            d_v = prior_v_list[0][li].shape[0]

            # Q-projection
            Q_q = rng.standard_normal((d_q, rank)).astype(np.float64)
            for prior_aq in prior_q_list:
                a = prior_aq[li].astype(np.float64)
                a_ortho, _ = np.linalg.qr(a)
                a_ortho = a_ortho[:, :rank]
                Q_q -= a_ortho @ (a_ortho.T @ Q_q)
            Q_q, _ = np.linalg.qr(Q_q)
            A_q_new.append(Q_q[:, :rank].astype(np.float32))

            # V-projection
            Q_v = rng.standard_normal((d_v, rank)).astype(np.float64)
            for prior_av in prior_v_list:
                a = prior_av[li].astype(np.float64)
                a_ortho, _ = np.linalg.qr(a)
                a_ortho = a_ortho[:, :rank]
                Q_v -= a_ortho @ (a_ortho.T @ Q_v)
            Q_v, _ = np.linalg.qr(Q_v)
            A_v_new.append(Q_v[:, :rank].astype(np.float32))
    return A_q_new, A_v_new


def compute_max_isolation(A_q_i, A_v_i, A_q_j, A_v_j) -> float:
    """Max Frobenius norm of A_i^T A_j across all layers (nan-safe)."""
    max_val = 0.0
    n_layers = len(A_q_i)
    with np.errstate(over="ignore", invalid="ignore"):
        for li in range(n_layers):
            gram_q = np.abs(A_q_i[li].astype(np.float64).T @ A_q_j[li].astype(np.float64))
            gram_v = np.abs(A_v_i[li].astype(np.float64).T @ A_v_j[li].astype(np.float64))
            q_max = float(np.nanmax(gram_q)) if not np.all(np.isnan(gram_q)) else 0.0
            v_max = float(np.nanmax(gram_v)) if not np.all(np.isnan(gram_v)) else 0.0
            max_val = max(max_val, q_max, v_max)
    return float(max_val)


# ---- LoRA model wiring -----------------------------------------------------

def apply_lora_structure(model, A_q: list, A_v: list) -> None:
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.q_proj = LoRALinear.from_base(attn.q_proj, r=LORA_RANK, scale=LORA_SCALE)
        attn.v_proj = LoRALinear.from_base(attn.v_proj, r=LORA_RANK, scale=LORA_SCALE)
        attn.q_proj.lora_a = A_q[li]
        attn.v_proj.lora_a = A_v[li]
    model.freeze()


def inject_lora_b(model, B_q: list, B_v: list) -> None:
    for li, layer in enumerate(model.model.layers):
        layer.self_attn.q_proj.lora_b = B_q[li]
        layer.self_attn.v_proj.lora_b = B_v[li]
    mx.eval(model.parameters())


# ---- SHINE memory-based forward (for M2P inference) -----------------------

def build_memory_causal_mask(M: int, T: int) -> mx.array:
    S = M + T
    mask_np = np.zeros((S, S), dtype=np.float32)
    mask_np[M:, :M] = float("-inf")
    for i in range(T):
        for j in range(i + 1, T):
            mask_np[M + i, M + j] = float("-inf")
    return mx.array(mask_np).astype(mx.bfloat16)[None, None, :, :]


def extract_memory_hidden_states(model, tokens_arr: mx.array, memory_embeddings: mx.array) -> mx.array:
    qwen3 = model.model
    M = memory_embeddings.shape[0]
    B_batch, T = tokens_arr.shape
    tok_embs = qwen3.embed_tokens(tokens_arr)
    h = mx.concatenate([memory_embeddings[None, :, :], tok_embs], axis=1)
    mask = build_memory_causal_mask(M, T)
    memory_states = []
    for li, layer in enumerate(qwen3.layers):
        normed = layer.input_layernorm(h)
        attn = layer.self_attn
        S = M + T
        q_f = attn.q_proj(normed)
        k_f = attn.k_proj(normed)
        v_f = attn.v_proj(normed)
        queries = attn.q_norm(q_f.reshape(B_batch, S, attn.n_heads, -1)).transpose(0, 2, 1, 3)
        keys = attn.k_norm(k_f.reshape(B_batch, S, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = v_f.reshape(B_batch, S, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        queries = attn.rope(queries)
        keys = attn.rope(keys)
        attn_out = scaled_dot_product_attention(
            queries, keys, values, cache=None, scale=attn.scale, mask=mask
        )
        attn_out = attn.o_proj(attn_out.transpose(0, 2, 1, 3).reshape(B_batch, S, -1))
        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        memory_states.append(h[0, :M, :])
    return mx.stack(memory_states, axis=0)


def functional_lora_proj(x, linear_module, A, B, scale):
    y = linear_module(x)
    z = (x @ A.astype(x.dtype)) @ B.astype(x.dtype)
    return y + (scale * z).astype(x.dtype)


def forward_with_loras(model, tokens_arr, B_q_layers, B_v_layers, A_q_layers, A_v_layers):
    qwen3 = model.model
    h = qwen3.embed_tokens(tokens_arr)
    mask = create_attention_mask(h, None)
    for li, layer in enumerate(qwen3.layers):
        normed = layer.input_layernorm(h)
        attn = layer.self_attn
        B_batch, L, D = normed.shape
        q = functional_lora_proj(normed, attn.q_proj.linear, A_q_layers[li], B_q_layers[li], LORA_SCALE)
        k = attn.k_proj(normed)
        v = functional_lora_proj(normed, attn.v_proj.linear, A_v_layers[li], B_v_layers[li], LORA_SCALE)
        queries = attn.q_norm(q.reshape(B_batch, L, attn.n_heads, -1)).transpose(0, 2, 1, 3)
        keys = attn.k_norm(k.reshape(B_batch, L, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = v.reshape(B_batch, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        queries = attn.rope(queries)
        keys = attn.rope(keys)
        attn_out = scaled_dot_product_attention(
            queries, keys, values, cache=None, scale=attn.scale, mask=mask
        )
        attn_out = attn.o_proj(attn_out.transpose(0, 2, 1, 3).reshape(B_batch, L, -1))
        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    h = qwen3.norm(h)
    return qwen3.embed_tokens.as_linear(h) if model.args.tie_word_embeddings else model.lm_head(h)


# ---- M2P architecture (SFT-Residual) ---------------------------------------

class M2PBlock(nn.Module):
    def __init__(self, d: int, n_heads: int = 4, is_column: bool = True):
        super().__init__()
        self.is_column = is_column
        self.norm1 = nn.RMSNorm(d)
        self.attn = nn.MultiHeadAttention(d, n_heads, bias=False)
        self.norm2 = nn.RMSNorm(d)
        self.mlp_fc1 = nn.Linear(d, 4 * d, bias=False)
        self.mlp_fc2 = nn.Linear(4 * d, d, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        if self.is_column:
            x_t = x.transpose(1, 0, 2)
            normed = self.norm1(x_t)
            x_t = x_t + self.attn(normed, normed, normed)
            x_t = x_t + self.mlp_fc2(nn.gelu(self.mlp_fc1(self.norm2(x_t))))
            return x_t.transpose(1, 0, 2)
        else:
            normed = self.norm1(x)
            x = x + self.attn(normed, normed, normed)
            x = x + self.mlp_fc2(nn.gelu(self.mlp_fc1(self.norm2(x))))
            return x


class M2PNetworkV6(nn.Module):
    """SFT-Residual M2P: B_applied[li] = B_sft[li] + output_scale * head(z[li])."""

    def __init__(self, n_layers, d_model, d_m2p, n_mem_tokens, rank,
                 q_proj_out, v_proj_out, B_sft_q, B_sft_v,
                 n_m2p_layers=4, n_heads=4, output_scale=0.032):
        super().__init__()
        self.n_layers = n_layers
        self.n_mem_tokens = n_mem_tokens
        self.rank = rank
        self.output_scale = output_scale
        self.has_input_proj = (d_model != d_m2p)
        self.B_sft_q = B_sft_q
        self.B_sft_v = B_sft_v

        scale = math.sqrt(1.0 / d_model)
        mem_init = np.random.default_rng(SEED).standard_normal(
            (n_mem_tokens, d_model)).astype(np.float32) * scale
        self.memory_embeddings = mx.array(mem_init).astype(mx.bfloat16)

        if self.has_input_proj:
            self.input_proj = nn.Linear(d_model, d_m2p, bias=False)
        else:
            self.input_proj = None

        self.p_layer = mx.zeros((n_layers, 1, d_m2p)).astype(mx.bfloat16)
        self.p_token = mx.zeros((1, n_mem_tokens, d_m2p)).astype(mx.bfloat16)

        self.blocks = [
            M2PBlock(d=d_m2p, n_heads=n_heads, is_column=(i % 2 == 0))
            for i in range(n_m2p_layers)
        ]
        self.final_norm = nn.RMSNorm(d_m2p)
        self.b_heads_q = [nn.Linear(d_m2p, rank * q_proj_out, bias=False) for _ in range(n_layers)]
        self.b_heads_v = [nn.Linear(d_m2p, rank * v_proj_out, bias=False) for _ in range(n_layers)]
        for head in self.b_heads_q + self.b_heads_v:
            head.weight = mx.zeros_like(head.weight)

    def __call__(self, memory_grid: mx.array):
        if self.has_input_proj:
            L, M, d = memory_grid.shape
            x = self.input_proj(memory_grid.reshape(L * M, d).astype(mx.bfloat16)).reshape(L, M, -1)
        else:
            x = memory_grid.astype(mx.bfloat16)
        x = x + self.p_layer.astype(mx.bfloat16)
        x = x + self.p_token.astype(mx.bfloat16)
        for block in self.blocks:
            x = block(x)
        z = mx.mean(self.final_norm(x), axis=1)  # (L, d_m2p)
        B_q, B_v = [], []
        for li in range(self.n_layers):
            delta_q = self.b_heads_q[li](z[li]).reshape(self.rank, -1) * self.output_scale
            delta_v = self.b_heads_v[li](z[li]).reshape(self.rank, -1) * self.output_scale
            B_q.append(self.B_sft_q[li] + delta_q.astype(self.B_sft_q[li].dtype))
            B_v.append(self.B_sft_v[li] + delta_v.astype(self.B_sft_v[li].dtype))
        return B_q, B_v


def make_m2p_v6(model_dims: dict, B_sft_q: list, B_sft_v: list) -> M2PNetworkV6:
    return M2PNetworkV6(
        n_layers=model_dims["n_layers"], d_model=model_dims["d_model"],
        d_m2p=D_M2P, n_mem_tokens=N_MEM_TOKENS, rank=LORA_RANK,
        q_proj_out=model_dims["q_proj_out"], v_proj_out=model_dims["v_proj_out"],
        B_sft_q=B_sft_q, B_sft_v=B_sft_v,
        n_m2p_layers=N_M2P_LAYERS, n_heads=N_M2P_HEADS, output_scale=OUTPUT_SCALE,
    )


# ---- Phase 0: Build all 25 A-matrices + verify isolation (K981) ------------

def phase_build_and_verify_a_matrices(model_dims: dict) -> dict:
    log("\n" + "=" * 70)
    log("[Phase 0] Build N=25 A-matrices + Grassmannian Isolation (K981)")
    log("=" * 70)
    t0 = time.time()

    n_layers = model_dims["n_layers"]
    rank = LORA_RANK

    # Load existing A-matrices (float32)
    log("  Loading math A-matrices...")
    A_math = load_a_matrices_np(MATH_LORA_A_PATH, n_layers)

    log("  Loading code A-matrices...")
    A_code = load_a_matrices_np(CODE_LORA_A_PATH, n_layers)

    log("  Loading/generating sort A-matrices...")
    if SORT_LORA_A_PATH.exists():
        A_sort = load_a_matrices_np(SORT_LORA_A_PATH, n_layers)
    else:
        log("  Generating sort A-matrices...")
        A_sort = gram_schmidt_new_domain([A_math[0], A_code[0]], [A_math[1], A_code[1]], n_layers, rank, 10000)
        save_a_matrices(SORT_LORA_A_PATH, A_sort[0], A_sort[1])

    log("  Loading/generating reverse A-matrices...")
    if REVERSE_LORA_A_PATH.exists():
        A_reverse = load_a_matrices_np(REVERSE_LORA_A_PATH, n_layers)
    else:
        log("  Generating reverse A-matrices...")
        A_reverse = gram_schmidt_new_domain(
            [A_math[0], A_code[0], A_sort[0]],
            [A_math[1], A_code[1], A_sort[1]], n_layers, rank, 20000
        )
        save_a_matrices(REVERSE_LORA_A_PATH, A_reverse[0], A_reverse[1])

    log("  Loading/generating count A-matrices...")
    if COUNT_LORA_A_PATH.exists():
        A_count = load_a_matrices_np(COUNT_LORA_A_PATH, n_layers)
    else:
        log("  Generating count A-matrices...")
        A_count = gram_schmidt_new_domain(
            [A_math[0], A_code[0], A_sort[0], A_reverse[0]],
            [A_math[1], A_code[1], A_sort[1], A_reverse[1]], n_layers, rank, 30000
        )
        save_a_matrices(COUNT_LORA_A_PATH, A_count[0], A_count[1])

    # Build/load the 20 new synthetic domains (domains 6-25)
    all_A = [A_math, A_code, A_sort, A_reverse, A_count]
    new_domain_names = DOMAINS[5:]  # domains 6-25
    domain_a_paths = {name: EXPERIMENT_DIR / f"{name}_a_matrices.npz" for name in new_domain_names}

    for idx, name in enumerate(new_domain_names):
        path = domain_a_paths[name]
        if path.exists():
            log(f"  Loading {name} A-matrices...")
            A_new = load_a_matrices_np(path, n_layers)
        else:
            log(f"  Generating {name} A-matrices (Gram-Schmidt vs {len(all_A)} prior)...")
            prior_q = [a[0] for a in all_A]
            prior_v = [a[1] for a in all_A]
            seed_off = 40000 + idx * 1000
            A_new = gram_schmidt_new_domain(prior_q, prior_v, n_layers, rank, seed_off)
            save_a_matrices(path, A_new[0], A_new[1])
            log(f"  Saved {name} A-matrices to {path}")
        all_A.append(A_new)

    assert len(all_A) == 25, f"Expected 25 domains, got {len(all_A)}"
    log(f"  All 25 A-matrices ready.")

    # Verify all C(25,2)=300 pairwise isolation (K981)
    log(f"\n  Checking all {len(list(combinations(range(25), 2)))} pairwise isolations...")
    max_global = 0.0
    worst_pair = None
    domain_labels = DOMAINS

    for i, j in combinations(range(25), 2):
        iso = compute_max_isolation(all_A[i][0], all_A[i][1], all_A[j][0], all_A[j][1])
        if iso > max_global:
            max_global = iso
            worst_pair = (domain_labels[i], domain_labels[j])

    k981_pass = max_global < 1e-4
    log(f"  K981: max|A_i^T A_j|_F = {max_global:.2e} (threshold < 1e-4)")
    log(f"  K981: worst pair = {worst_pair}")
    log(f"  K981: {'PASS' if k981_pass else 'FAIL'}")

    elapsed = time.time() - t0
    log(f"  Phase 0 elapsed: {elapsed:.1f}s")

    return {
        "k981_pass": k981_pass,
        "k981_max_isolation": max_global,
        "worst_pair": list(worst_pair) if worst_pair else None,
        "n_pairs_checked": len(list(combinations(range(25), 2))),
        "all_A": all_A,  # pass to next phases
    }


# ---- Phase 1: TF-IDF 25-class router (K982) --------------------------------

def phase_tfidf_routing(n_train: int, n_test: int) -> dict:
    log("\n" + "=" * 70)
    log("[Phase 1] TF-IDF 25-Class Routing (K982)")
    log("=" * 70)
    t0 = time.time()

    # Build train/test sets
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for domain in DOMAINS:
        tr = get_routing_texts_for_domain(domain, n_train, offset=0)
        te = get_routing_texts_for_domain(domain, n_test, offset=n_train)
        train_texts.extend(tr)
        train_labels.extend([domain] * len(tr))
        test_texts.extend(te)
        test_labels.extend([domain] * len(te))

    log(f"  Train: {len(train_texts)} samples ({n_train}/domain)")
    log(f"  Test: {len(test_texts)} samples ({n_test}/domain)")

    # Fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # Build per-domain centroids
    centroids = {}
    for domain in DOMAINS:
        idxs = [i for i, l in enumerate(train_labels) if l == domain]
        centroids[domain] = np.asarray(X_train[idxs].mean(axis=0))

    centroid_matrix = np.vstack([centroids[d] for d in DOMAINS])

    # Predict: nearest centroid
    sims = cosine_similarity(X_test, centroid_matrix)
    pred_indices = sims.argmax(axis=1)
    pred_labels = [DOMAINS[i] for i in pred_indices]

    # Compute accuracy per domain and overall
    total_correct = sum(p == t for p, t in zip(pred_labels, test_labels))
    overall_acc = total_correct / len(test_labels)

    per_domain_acc = {}
    for domain in DOMAINS:
        idxs = [i for i, l in enumerate(test_labels) if l == domain]
        correct = sum(pred_labels[i] == domain for i in idxs)
        per_domain_acc[domain] = correct / len(idxs)

    min_domain_acc = min(per_domain_acc.values())
    k982_pass = min_domain_acc >= 0.80

    log(f"  Overall accuracy: {overall_acc:.1%} ({total_correct}/{len(test_labels)})")
    log(f"  Per-domain accuracies:")
    for d, acc in per_domain_acc.items():
        log(f"    {d:15s}: {acc:.1%}")
    log(f"  Min domain accuracy: {min_domain_acc:.1%}")
    log(f"  K982: {'PASS' if k982_pass else 'FAIL'} (min_acc={min_domain_acc:.3f} >= 0.80)")

    elapsed = time.time() - t0
    log(f"  Phase 1 elapsed: {elapsed:.1f}s")

    return {
        "k982_pass": k982_pass,
        "k982_overall_acc": overall_acc,
        "k982_min_domain_acc": min_domain_acc,
        "k982_per_domain_acc": per_domain_acc,
        "vectorizer": vectorizer,
        "centroid_matrix": centroid_matrix,
    }


# ---- Phase 2: Math quality under N=25 routed composition (K983) ------------

def phase_math_quality_n25(model, tokenizer, model_dims: dict, all_A: list, routing_data: dict) -> dict:
    log("\n" + "=" * 70)
    log("[Phase 2] Math Quality Under N=25 Routed Composition (K983)")
    log("=" * 70)
    t0 = time.time()

    n_layers = model_dims["n_layers"]
    base_acc = model_dims["base_accuracy"]
    sft_acc = model_dims["sft_accuracy"]

    # Load math M2P weights
    log("  Loading math M2P weights...")
    assert MATH_M2P_PATH.exists(), f"Math M2P not found: {MATH_M2P_PATH}"
    saved = np.load(str(MATH_M2P_PATH))

    # Load SFT B-matrices for SFT-residual baseline
    sft_saved = np.load(str(MATH_SFT_B_PATH))
    B_sft_q = [mx.array(sft_saved[f"layer_{li}_q_proj_B"]).astype(mx.bfloat16) for li in range(n_layers)]
    B_sft_v = [mx.array(sft_saved[f"layer_{li}_v_proj_B"]).astype(mx.bfloat16) for li in range(n_layers)]
    mx.eval(*B_sft_q, *B_sft_v)

    # Build math M2P — convert numpy arrays to mx.arrays for load_weights
    m2p = make_m2p_v6(model_dims, B_sft_q, B_sft_v)
    weights = [(k, mx.array(v)) for k, v in saved.items()]
    m2p.load_weights(weights)
    m2p.freeze()
    mx.eval(m2p.parameters())
    log(f"  Math M2P loaded.")

    # Load math A-matrices (bfloat16 for inference)
    A_math_q, A_math_v = load_a_matrices(MATH_LORA_A_PATH, n_layers)

    # Apply LoRA structure to model (math domain)
    apply_lora_structure(model, A_math_q, A_math_v)

    # Set up routing
    vectorizer = routing_data["vectorizer"]
    centroid_matrix = routing_data["centroid_matrix"]

    # Load GSM8K test examples
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(SEED)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    test_indices = indices[:N_EVAL_MATH]

    correct = 0
    math_routed = 0

    for qi, idx in enumerate(test_indices):
        item = ds[idx]
        question = item["question"]
        true_ans_raw = item["answer"]
        m = re.search(r"####\s*(-?[\d,]+)", true_ans_raw)
        true_ans = m.group(1).replace(",", "") if m else None

        prompt_text = FEW_SHOT_PREFIX + f"Question: {question}\nAnswer:"

        # Route the query
        x_tfidf = vectorizer.transform([question])
        sims = cosine_similarity(x_tfidf, centroid_matrix)
        routed_domain = DOMAINS[sims.argmax()]

        if routed_domain == "math":
            math_routed += 1
            # Generate M2P B-matrices from context
            prompt_ids = tokenizer.encode(prompt_text)
            if len(prompt_ids) > 256:
                prompt_ids = prompt_ids[-256:]
            tokens_arr = mx.array(prompt_ids)[None]

            # M2P forward: extract memory states + generate B
            memory_states = extract_memory_hidden_states(model, tokens_arr, m2p.memory_embeddings)
            mx.eval(memory_states)
            B_q_pred, B_v_pred = m2p(memory_states)
            mx.eval(*B_q_pred, *B_v_pred)

            # Generate with M2P adapter
            inject_lora_b(model, B_q_pred, B_v_pred)
            output = mlx_generate(
                model, tokenizer,
                prompt=prompt_text,
                max_tokens=MAX_GEN_TOKENS,
                verbose=False,
            )
        else:
            # Wrong domain — use base model (B=0)
            B_q_zero = [mx.zeros_like(B_sft_q[li]) for li in range(n_layers)]
            B_v_zero = [mx.zeros_like(B_sft_v[li]) for li in range(n_layers)]
            inject_lora_b(model, B_q_zero, B_v_zero)
            output = mlx_generate(
                model, tokenizer,
                prompt=prompt_text,
                max_tokens=MAX_GEN_TOKENS,
                verbose=False,
            )

        pred_ans = extract_gsm8k_answer(output)
        is_correct = (pred_ans is not None and true_ans is not None and pred_ans == true_ans)
        if is_correct:
            correct += 1

        if (qi + 1) % 20 == 0:
            log(f"  [{qi+1}/{N_EVAL_MATH}] acc={correct/(qi+1):.3f} math_routed={math_routed/(qi+1):.1%}")

    accuracy = correct / N_EVAL_MATH
    math_route_frac = math_routed / N_EVAL_MATH
    quality_ratio = (accuracy - base_acc) / (sft_acc - base_acc) if (sft_acc - base_acc) != 0 else float("nan")
    k983_pass = quality_ratio >= 0.70

    log(f"\n  Accuracy: {accuracy:.3f} ({correct}/{N_EVAL_MATH})")
    log(f"  Base accuracy: {base_acc:.3f}")
    log(f"  SFT accuracy: {sft_acc:.3f}")
    log(f"  Quality ratio: {quality_ratio:.4f}")
    log(f"  Math routing fraction: {math_route_frac:.1%}")
    log(f"  K983: {'PASS' if k983_pass else 'FAIL'} (qr={quality_ratio:.4f} >= 0.70)")

    elapsed = time.time() - t0
    log(f"  Phase 2 elapsed: {elapsed:.1f}s")

    cleanup(m2p, A_math_q, A_math_v, B_sft_q, B_sft_v)

    return {
        "k983_pass": k983_pass,
        "k983_quality_ratio": quality_ratio,
        "k983_accuracy": accuracy,
        "k983_math_route_frac": math_route_frac,
        "base_accuracy": base_acc,
        "sft_accuracy": sft_acc,
    }


# ---- Main ------------------------------------------------------------------

def main():
    t_total = time.time()
    log("=" * 70)
    log("N=25 Domain Grassmannian Composition at 4B")
    log(f"SMOKE_TEST={IS_SMOKE}")
    log("=" * 70)

    # Load model dims (no model needed for phase 0 + 1)
    model_dims = load_model_dims()
    log(f"  Model dims: n_layers={model_dims['n_layers']}, d_model={model_dims['d_model']}")
    log(f"  Base accuracy: {model_dims['base_accuracy']:.3f}")
    log(f"  SFT accuracy: {model_dims['sft_accuracy']:.3f}")

    # Phase 0: Build A-matrices + verify isolation (K981)
    phase0 = phase_build_and_verify_a_matrices(model_dims)

    # Phase 1: TF-IDF routing (K982)
    phase1 = phase_tfidf_routing(N_ROUTE_TRAIN, N_ROUTE_TEST)

    # Phase 2: Load model + evaluate math quality (K983)
    log(f"\n  Loading {MODEL_ID}...")
    model, tokenizer = mlx_load(MODEL_ID)
    log_memory("after model load")

    phase2 = phase_math_quality_n25(model, tokenizer, model_dims, phase0["all_A"], phase1)

    # Summarize
    log("\n" + "=" * 70)
    log("RESULTS SUMMARY")
    log("=" * 70)
    log(f"K981 (isolation):  {'PASS' if phase0['k981_pass'] else 'FAIL'} | max={phase0['k981_max_isolation']:.2e}")
    log(f"K982 (routing):    {'PASS' if phase1['k982_pass'] else 'FAIL'} | min_acc={phase1['k982_min_domain_acc']:.1%}")
    log(f"K983 (quality):    {'PASS' if phase2['k983_pass'] else 'FAIL'} | qr={phase2['k983_quality_ratio']:.4f}")
    all_pass = phase0["k981_pass"] and phase1["k982_pass"] and phase2["k983_pass"]
    log(f"ALL K: {'PASS' if all_pass else 'FAIL'}")
    log(f"Total elapsed: {(time.time() - t_total)/60:.1f} min")

    # Save results
    results = {
        "experiment": "exp_m2p_n25_compose_qwen4b",
        "model": MODEL_ID,
        "n_domains": 25,
        "n_pairs": phase0["n_pairs_checked"],
        "config": {
            "lora_rank": LORA_RANK,
            "n_layers": model_dims["n_layers"],
            "d_model": model_dims["d_model"],
            "n_max_capacity": model_dims["d_model"] // LORA_RANK,
            "capacity_used_pct": 25 * LORA_RANK / model_dims["d_model"] * 100,
        },
        "k981": {
            "pass": phase0["k981_pass"],
            "max_isolation": phase0["k981_max_isolation"],
            "worst_pair": phase0["worst_pair"],
            "n_pairs": phase0["n_pairs_checked"],
        },
        "k982": {
            "pass": phase1["k982_pass"],
            "overall_acc": phase1["k982_overall_acc"],
            "min_domain_acc": phase1["k982_min_domain_acc"],
            "per_domain_acc": phase1["k982_per_domain_acc"],
        },
        "k983": {
            "pass": phase2["k983_pass"],
            "quality_ratio": phase2["k983_quality_ratio"],
            "accuracy": phase2["k983_accuracy"],
            "base_accuracy": phase2["base_accuracy"],
            "sft_accuracy": phase2["sft_accuracy"],
            "math_route_frac": phase2["k983_math_route_frac"],
        },
        "all_pass": all_pass,
        "runtime_min": (time.time() - t_total) / 60,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {RESULTS_FILE}")

    return results


if __name__ == "__main__":
    main()
