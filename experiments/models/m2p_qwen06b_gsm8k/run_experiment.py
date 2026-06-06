#!/usr/bin/env python3
"""M2P on Qwen3-0.6B + GSM8K — First Real Language Test.

Kill criteria:
  K_real: M2P quality_ratio >= 70% of SFT accuracy improvement on GSM8K test.
  K_mmlu: MMLU accuracy degradation <= -3pp (not measured in this script).
  K_KILL: M2P quality_ratio < 30% => hypernetwork approach is toy-only.

Supports SMOKE_TEST=1 for <60s validation.
"""

import gc
import json
import math
import os
import random
import re
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from mlx_lm import load as mlx_load
from mlx_lm import generate as mlx_generate

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"

# ---- Config ----------------------------------------------------------------
MODEL_ID = "mlx-community/Qwen3-0.6B-4bit"
D_MODEL = 1024
N_LAYERS = 28

# LoRA: 2 modules per layer (q_proj and v_proj), output space = D_MODEL
LORA_RANK = 4
LORA_SCALE = 5.0
LORA_MODULES = ["q_proj", "v_proj"]
# modules_dims: (name, in_features, out_features) — all in D_MODEL space
MODULES_DIMS = [("q_proj", D_MODEL, D_MODEL), ("v_proj", D_MODEL, D_MODEL)]

# M2P config (proven recipe)
D_M2P = 64
L_M2P = 2
N_MEMORY = 32

# Training
N_TRAIN = 50 if IS_SMOKE else 2000
N_TEST = 10 if IS_SMOKE else 200
TRAIN_STEPS = 20 if IS_SMOKE else 300
LR = 1e-4
MAX_SEQ_LEN = 64 if IS_SMOKE else 256
MAX_GEN_TOKENS = 32 if IS_SMOKE else 128
SEED = 42

EXPERIMENT_DIR = Path(__file__).parent
ADAPTER_PATH = EXPERIMENT_DIR / "sft_adapter.npz"
M2P_PATH = EXPERIMENT_DIR / "m2p_weights.npz"
A_MATRIX_PATH = EXPERIMENT_DIR / "a_matrices.npz"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

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
    """Extract final numeric answer from GSM8K #### format."""
    match = re.search(r"####\s*(-?[\d,]+)", text)
    if match:
        return match.group(1).replace(",", "")
    return None


# ---- Grassmannian A-matrices (frozen) ----------------------------------------

def make_grassmannian_a(in_features: int, rank: int, seed: int) -> mx.array:
    """QR-based orthonormal A-matrix. Returns (in_features, rank)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(in_features, rank).astype(np.float32)
    Q, _ = np.linalg.qr(X)
    return mx.array(Q[:, :rank])


def phase_build_grassmannian() -> None:
    """Build and save Grassmannian A-matrices for all (layer, module) pairs."""
    log("\n" + "=" * 70)
    log("[Phase 3] Building Grassmannian A-matrices")
    log("=" * 70)
    t0 = time.time()
    save_dict = {}
    for li in range(N_LAYERS):
        for mod_name, in_f, out_f in MODULES_DIMS:
            seed = li * 100 + (hash(mod_name) % 97) + SEED
            A = make_grassmannian_a(in_f, LORA_RANK, seed)
            save_dict[f"layer_{li}_{mod_name}"] = A
    mx.savez(str(A_MATRIX_PATH), **save_dict)
    log(f"  Built {len(save_dict)} A-matrices in {time.time()-t0:.1f}s")


def load_a_matrices() -> dict:
    """Load A-matrices from disk. Returns dict[(li, mod_name)] -> mx.array."""
    saved = dict(mx.load(str(A_MATRIX_PATH)))
    return {(li, mod_name): saved[f"layer_{li}_{mod_name}"].astype(mx.bfloat16)
            for li in range(N_LAYERS)
            for mod_name, _, _ in MODULES_DIMS}


# ---- LoRA correction (trainable B, frozen A) --------------------------------

class LoRACorrection(nn.Module):
    """Additive LoRA correction: scale * (x @ A) @ B.

    A is frozen (Grassmannian orthonormal basis).
    B is the trainable matrix.
    """

    def __init__(self, in_features: int, out_features: int, rank: int,
                 scale: float, a_matrix: mx.array):
        super().__init__()
        self.scale = scale
        self.A = a_matrix        # (in_features, rank) — frozen
        self.B = mx.zeros((rank, out_features))  # trainable
        # Freeze A so gradients don't flow through it
        self.freeze(keys=["A"])

    def __call__(self, x: mx.array) -> mx.array:
        return self.scale * (x @ self.A) @ self.B


class LoRAModel(nn.Module):
    """Container for all LoRA corrections across layers and modules."""

    def __init__(self, corrections: dict):
        super().__init__()
        self.corrections = list(corrections.values())
        # key_list must NOT start with underscore (MLX treats those as private)
        self.key_list = list(corrections.keys())  # list of (li, mod_name) tuples

    def get_correction(self, key):
        idx = self.key_list.index(key)
        return self.corrections[idx]


# ---- M2P architecture -------------------------------------------------------

class M2PNetwork(nn.Module):
    """Hypernetwork: context hidden states -> LoRA B-matrices.

    Input:  (N_LAYERS, D_MODEL) — mean-pooled hidden state per layer.
    Output: list[layer_idx][mod_idx] = (rank, out_features)
    """

    def __init__(self, n_layers: int, d_model: int, d_m2p: int,
                 l_m2p: int, n_memory: int, rank: int):
        super().__init__()
        self.n_layers = n_layers
        self.rank = rank
        self.d_m2p = d_m2p
        n_modules = len(MODULES_DIMS)

        # Encoder: d_model -> d_m2p
        enc = []
        in_dim = d_model
        for i in range(l_m2p):
            out_dim = d_m2p if i == l_m2p - 1 else d_m2p * 2
            enc.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.encoder = enc

        # Learned memory bank: (n_memory, d_m2p)
        self.memory_bank = mx.random.normal(shape=(n_memory, d_m2p)) * 0.02

        # Attention query projection
        self.attn_q = nn.Linear(d_m2p, d_m2p)

        # B-matrix generator heads: one Linear per (layer, module)
        # head outputs: rank * out_features (= rank * D_MODEL)
        self.b_heads = [
            nn.Linear(d_m2p, rank * out_f)
            for _ in range(n_layers)
            for (_, in_f, out_f) in MODULES_DIMS
        ]
        self.n_modules = n_modules

    def __call__(self, layer_hs: mx.array):
        """layer_hs: (n_layers, d_model). Returns list[layer][mod] = (rank, out_f)."""
        # Mean-pool across layers
        h = mx.mean(layer_hs, axis=0)  # (d_model,)

        # Encode
        for i, layer in enumerate(self.encoder):
            h = layer(h)
            if i < len(self.encoder) - 1:
                h = nn.relu(h)

        # Memory attention
        q = self.attn_q(h)  # (d_m2p,)
        scores = (self.memory_bank @ q) / math.sqrt(self.d_m2p)  # (n_memory,)
        w = mx.softmax(scores, axis=0)
        context = (w[:, None] * self.memory_bank).sum(axis=0)  # (d_m2p,)
        z = h + context  # (d_m2p,)

        # Generate B-matrices
        b_matrices = []
        head_idx = 0
        for li in range(self.n_layers):
            layer_bs = []
            for (mod_name, in_f, out_f) in MODULES_DIMS:
                b_flat = self.b_heads[head_idx](z)  # (rank * out_f,)
                layer_bs.append(b_flat.reshape(self.rank, out_f))
                head_idx += 1
            b_matrices.append(layer_bs)
        return b_matrices


# ---- Forward pass with LoRA -------------------------------------------------

def forward_with_lora(model, tokens: mx.array, a_matrices: dict,
                      b_matrices_by_key: dict) -> mx.array:
    """Run Qwen3 forward with additive LoRA corrections.

    Applies LoRA as additive correction to each transformer layer's output.
    For each layer: h += sum_mod(scale * (h_norm @ A) @ B)

    a_matrices: dict[(li, mod_name)] -> mx.array (D_MODEL, rank)
    b_matrices_by_key: dict[(li, mod_name)] -> mx.array (rank, D_MODEL)
    """
    h = model.model.embed_tokens(tokens)  # (1, T, D_MODEL)

    for li, layer in enumerate(model.model.layers):
        h_norm = layer.input_layernorm(h)

        # Base attention
        attn_out = layer.self_attn(h_norm)

        # LoRA additive correction (applied to normed input, output in D_MODEL space)
        lora_add = mx.zeros_like(attn_out)
        for mod_name in LORA_MODULES:
            key = (li, mod_name)
            if key in b_matrices_by_key:
                A = a_matrices[key]      # (D_MODEL, rank)
                B = b_matrices_by_key[key]  # (rank, D_MODEL)
                delta = LORA_SCALE * (h_norm @ A) @ B  # (1, T, D_MODEL)
                lora_add = lora_add + delta

        h = h + attn_out + lora_add

        h_norm2 = layer.post_attention_layernorm(h)
        h = h + layer.mlp(h_norm2)

    h = model.model.norm(h)
    # Qwen3-0.6B uses tied embeddings (no separate lm_head)
    if model.args.tie_word_embeddings:
        return model.model.embed_tokens.as_linear(h)
    else:
        return model.lm_head(h)


def get_layer_hidden_states(model, tokens: mx.array) -> mx.array:
    """Extract mean-pooled hidden state from each layer.

    Returns: (N_LAYERS, D_MODEL)
    """
    h = model.model.embed_tokens(tokens)
    layer_states = []
    for layer in model.model.layers:
        h_norm = layer.input_layernorm(h)
        h = h + layer.self_attn(h_norm)
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        layer_states.append(mx.mean(h[0], axis=0))  # (D_MODEL,)
    return mx.stack(layer_states, axis=0)  # (N_LAYERS, D_MODEL)


def compute_ntp_loss(logits: mx.array, tokens: mx.array) -> mx.array:
    """Next-token prediction loss. logits: (1, T, V), tokens: (1, T+1)."""
    return nn.losses.cross_entropy(logits[0, :-1, :], tokens[0, 1:], reduction="mean")


# ---- Phase 1: Load data -----------------------------------------------------

def phase_load_data():
    """Load and format GSM8K examples."""
    log("\n" + "=" * 70)
    log("[Phase 1] Loading GSM8K data")
    log("=" * 70)
    t0 = time.time()

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")

    rng = random.Random(SEED)
    train_examples = list(ds["train"])
    rng.shuffle(train_examples)
    train_texts = [
        f"Question: {ex['question']}\nAnswer: {ex['answer']}"
        for ex in train_examples[:N_TRAIN]
    ]

    test_examples = list(ds["test"])
    rng.shuffle(test_examples)
    test_examples = test_examples[:N_TEST]
    test_prompts = [f"Question: {ex['question']}\nAnswer:" for ex in test_examples]
    test_answers = [extract_gsm8k_answer(ex["answer"]) for ex in test_examples]

    log(f"  Train: {len(train_texts)}, Test: {len(test_prompts)}")
    log(f"  Data loaded in {time.time()-t0:.1f}s")
    return train_texts, test_prompts, test_answers


# ---- Phase 2: Evaluate base model -------------------------------------------

def phase_eval_base(test_prompts: list, test_answers: list) -> dict:
    """Evaluate base Qwen3-0.6B-4bit on GSM8K."""
    log("\n" + "=" * 70)
    log("[Phase 2] Evaluating base model accuracy")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = mlx_load(MODEL_ID)
    model.eval()
    mx.eval(model.parameters())

    correct = 0
    total = len(test_prompts)
    for i, (prompt, gold) in enumerate(zip(test_prompts, test_answers)):
        generated = mlx_generate(
            model, tokenizer, prompt=prompt,
            max_tokens=MAX_GEN_TOKENS, verbose=False,
        )
        pred = extract_gsm8k_answer(generated)
        if pred is not None and gold is not None and pred == gold:
            correct += 1
        if (i + 1) % 20 == 0 or (i + 1) == total:
            log(f"  {i+1}/{total}: acc={correct/(i+1):.3f}")

    accuracy = correct / total if total > 0 else 0.0
    log(f"  Base accuracy: {accuracy:.4f} ({correct}/{total})")
    log(f"  Phase 2 time: {time.time()-t0:.1f}s")
    log_memory("post-base-eval")

    cleanup(model, tokenizer)
    return {"base_accuracy": accuracy, "base_correct": correct, "base_total": total}


# ---- Shared: tokenize training texts ----------------------------------------

def tokenize_texts(tokenizer, texts: list) -> list:
    """Tokenize and filter texts, returning list of token id lists."""
    result = []
    for text in texts:
        ids = tokenizer.encode(text)
        if len(ids) >= 2:
            ids = ids[:MAX_SEQ_LEN + 1]
            if len(ids) >= 2:
                result.append(ids)
    return result


# ---- Phase 4: SFT LoRA training ---------------------------------------------

def phase_sft_train(train_texts: list) -> dict:
    """Train SFT LoRA B-matrices on GSM8K using frozen Grassmannian A."""
    log("\n" + "=" * 70)
    log("[Phase 4] SFT LoRA Training")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = mlx_load(MODEL_ID)
    model.eval()
    mx.eval(model.parameters())

    a_matrices = load_a_matrices()

    # Build LoRA corrections (trainable B, frozen A)
    corrections = {}
    for li in range(N_LAYERS):
        for mod_name, in_f, out_f in MODULES_DIMS:
            key = (li, mod_name)
            corrections[key] = LoRACorrection(
                in_features=D_MODEL, out_features=D_MODEL,
                rank=LORA_RANK, scale=LORA_SCALE,
                a_matrix=a_matrices[key],
            )

    lora_model = LoRAModel(corrections)
    mx.eval(lora_model.parameters())

    log(f"  Tokenizing {len(train_texts)} texts...")
    tokenized = tokenize_texts(tokenizer, train_texts)
    log(f"  Tokenized: {len(tokenized)} sequences")

    rng = random.Random(SEED)
    optimizer = optim.Adam(learning_rate=LR)

    def loss_fn(lora_mod, tokens_arr):
        b_by_key = {k: lora_mod.corrections[i].B for i, k in enumerate(lora_mod.key_list)}
        logits = forward_with_lora(model, tokens_arr, a_matrices, b_by_key)
        return compute_ntp_loss(logits, tokens_arr)

    loss_and_grad = nn.value_and_grad(lora_model, loss_fn)

    gc.disable()
    losses = []
    for step in range(TRAIN_STEPS):
        seq = rng.choice(tokenized)
        tokens_arr = mx.array(seq)[None, :]
        loss, grads = loss_and_grad(lora_model, tokens_arr)
        optimizer.update(lora_model, grads)
        del grads, tokens_arr
        mx.eval(lora_model.parameters(), optimizer.state, loss)
        losses.append(loss.item())
        if (step + 1) % 50 == 0 or (step + 1) == TRAIN_STEPS:
            log(f"  Step {step+1}/{TRAIN_STEPS}: loss={sum(losses[-20:])/len(losses[-20:]):.4f}")
    gc.enable()
    gc.collect()

    final_loss = sum(losses[-20:]) / len(losses[-20:])
    log(f"  Final SFT loss: {final_loss:.4f}")

    # Save B-matrices to disk
    b_save = {f"layer_{li}_{mod_name}_B": lora_model.corrections[i].B.astype(mx.float32)
              for i, (li, mod_name) in enumerate(lora_model.key_list)}
    mx.savez(str(ADAPTER_PATH), **b_save)
    log(f"  Saved adapter to {ADAPTER_PATH}")
    log(f"  Phase 4 time: {time.time()-t0:.1f}s")
    log_memory("post-sft-train")

    cleanup(lora_model, optimizer, model, tokenizer)
    del a_matrices, corrections
    return {"sft_final_loss": final_loss}


# ---- Greedy decode with LoRA ------------------------------------------------

def greedy_decode_with_lora(model, tokenizer, prompt_ids: list,
                             a_matrices: dict, b_by_key: dict) -> str:
    """Greedy decode using LoRA-augmented forward pass."""
    generated_ids = list(prompt_ids)
    for _ in range(MAX_GEN_TOKENS):
        x = mx.array(generated_ids)[None, :]
        logits = forward_with_lora(model, x, a_matrices, b_by_key)
        mx.eval(logits)
        next_tok = mx.argmax(logits[0, -1, :]).item()
        del logits, x
        generated_ids.append(next_tok)
        if next_tok == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated_ids[len(prompt_ids):])


# ---- Phase 5: Evaluate SFT adapter ------------------------------------------

def phase_eval_sft(test_prompts: list, test_answers: list) -> dict:
    """Evaluate SFT LoRA adapter on GSM8K test."""
    log("\n" + "=" * 70)
    log("[Phase 5] Evaluating SFT adapter accuracy")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = mlx_load(MODEL_ID)
    model.eval()
    mx.eval(model.parameters())

    a_matrices = load_a_matrices()
    b_saved = dict(mx.load(str(ADAPTER_PATH)))
    b_by_key = {(li, mod_name): b_saved[f"layer_{li}_{mod_name}_B"].astype(mx.bfloat16)
                for li in range(N_LAYERS) for mod_name, _, _ in MODULES_DIMS}
    mx.eval(b_by_key)

    correct = 0
    total = len(test_prompts)
    for i, (prompt, gold) in enumerate(zip(test_prompts, test_answers)):
        prompt_ids = tokenizer.encode(prompt)
        text = greedy_decode_with_lora(model, tokenizer, prompt_ids, a_matrices, b_by_key)
        pred = extract_gsm8k_answer(text)
        if pred is not None and gold is not None and pred == gold:
            correct += 1
        if (i + 1) % 20 == 0 or (i + 1) == total:
            log(f"  [SFT] {i+1}/{total}: acc={correct/(i+1):.3f}")

    accuracy = correct / total if total > 0 else 0.0
    log(f"  SFT accuracy: {accuracy:.4f} ({correct}/{total})")
    log(f"  Phase 5 time: {time.time()-t0:.1f}s")
    log_memory("post-sft-eval")

    cleanup(model, tokenizer)
    del a_matrices, b_by_key, b_saved
    return {"sft_accuracy": accuracy, "sft_correct": correct}


# ---- Phase 6: M2P training --------------------------------------------------

def phase_m2p_train(train_texts: list) -> dict:
    """Train M2P hypernetwork to generate LoRA B-matrices from context."""
    log("\n" + "=" * 70)
    log("[Phase 6] M2P Hypernetwork Training")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = mlx_load(MODEL_ID)
    model.eval()
    mx.eval(model.parameters())

    a_matrices = load_a_matrices()
    # Pre-flatten A-matrices as a list matching MODULES_DIMS order
    a_flat = [a_matrices[(li, mod_name)]
              for li in range(N_LAYERS)
              for mod_name, _, _ in MODULES_DIMS]

    m2p = M2PNetwork(
        n_layers=N_LAYERS, d_model=D_MODEL, d_m2p=D_M2P,
        l_m2p=L_M2P, n_memory=N_MEMORY, rank=LORA_RANK,
    )
    mx.eval(m2p.parameters())
    n_params = sum(p.size for _, p in tree_flatten(m2p.parameters()))
    log(f"  M2P params: {n_params:,}")

    log(f"  Tokenizing {len(train_texts)} texts...")
    tokenized = tokenize_texts(tokenizer, train_texts)
    log(f"  Tokenized: {len(tokenized)} sequences")

    rng = random.Random(SEED + 1)
    optimizer = optim.Adam(learning_rate=LR)
    n_mods = len(MODULES_DIMS)

    def m2p_loss_fn(m2p_net, tokens_arr, a_mats):
        # Extract hidden states (stop-gradient through base model)
        layer_hs = get_layer_hidden_states(model, tokens_arr)

        # Generate B-matrices
        b_matrices = m2p_net(layer_hs)  # list[li][mi] = (rank, D_MODEL)

        # Build b_by_key for forward pass
        b_by_key = {}
        for li in range(N_LAYERS):
            for mi, (mod_name, _, _) in enumerate(MODULES_DIMS):
                b_by_key[(li, mod_name)] = b_matrices[li][mi]

        logits = forward_with_lora(model, tokens_arr, a_mats, b_by_key)
        return compute_ntp_loss(logits, tokens_arr)

    # Build a dict of a_matrices for the loss fn (consistent key format)
    a_matrices_for_fwd = a_matrices  # dict[(li, mod_name)] -> mx.array

    loss_and_grad = nn.value_and_grad(m2p, m2p_loss_fn)

    gc.disable()
    losses = []
    for step in range(TRAIN_STEPS):
        seq = rng.choice(tokenized)
        tokens_arr = mx.array(seq)[None, :]
        loss, grads = loss_and_grad(m2p, tokens_arr, a_matrices_for_fwd)
        optimizer.update(m2p, grads)
        del grads, tokens_arr
        mx.eval(m2p.parameters(), optimizer.state, loss)
        losses.append(loss.item())
        if (step + 1) % 50 == 0 or (step + 1) == TRAIN_STEPS:
            log(f"  Step {step+1}/{TRAIN_STEPS}: loss={sum(losses[-20:])/len(losses[-20:]):.4f}")
    gc.enable()
    gc.collect()

    final_loss = sum(losses[-20:]) / len(losses[-20:])
    log(f"  Final M2P loss: {final_loss:.4f}")

    m2p_params = dict(tree_flatten(m2p.parameters()))
    mx.savez(str(M2P_PATH), **m2p_params)
    log(f"  Saved M2P to {M2P_PATH}")
    log(f"  Phase 6 time: {time.time()-t0:.1f}s")
    log_memory("post-m2p-train")

    cleanup(m2p, optimizer, model, tokenizer)
    del a_matrices, a_flat, a_matrices_for_fwd
    return {"m2p_final_loss": final_loss, "m2p_params": n_params}


# ---- Phase 7: Evaluate M2P adapter ------------------------------------------

def phase_eval_m2p(test_prompts: list, test_answers: list) -> dict:
    """Evaluate M2P-generated adapter on GSM8K test.

    For each test example: extract hidden states from prompt, generate
    B-matrices via M2P, then greedy-decode with those B-matrices.
    """
    log("\n" + "=" * 70)
    log("[Phase 7] Evaluating M2P adapter accuracy")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = mlx_load(MODEL_ID)
    model.eval()
    mx.eval(model.parameters())

    m2p_params_saved = dict(mx.load(str(M2P_PATH)))
    m2p = M2PNetwork(
        n_layers=N_LAYERS, d_model=D_MODEL, d_m2p=D_M2P,
        l_m2p=L_M2P, n_memory=N_MEMORY, rank=LORA_RANK,
    )
    m2p.load_weights(list(m2p_params_saved.items()))
    m2p.eval()
    mx.eval(m2p.parameters())

    a_matrices = load_a_matrices()

    correct = 0
    total = len(test_prompts)
    for i, (prompt, gold) in enumerate(zip(test_prompts, test_answers)):
        prompt_ids = tokenizer.encode(prompt)
        tokens_arr = mx.array(prompt_ids)[None, :]

        # Generate B-matrices from prompt context
        layer_hs = get_layer_hidden_states(model, tokens_arr)
        mx.eval(layer_hs)
        b_matrices = m2p(layer_hs)
        mx.eval([b for row in b_matrices for b in row])

        b_by_key = {(li, mod_name): b_matrices[li][mi]
                    for li in range(N_LAYERS)
                    for mi, (mod_name, _, _) in enumerate(MODULES_DIMS)}

        text = greedy_decode_with_lora(model, tokenizer, prompt_ids, a_matrices, b_by_key)
        pred = extract_gsm8k_answer(text)
        if pred is not None and gold is not None and pred == gold:
            correct += 1

        del tokens_arr, layer_hs, b_matrices, b_by_key
        if (i + 1) % 20 == 0 or (i + 1) == total:
            log(f"  [M2P] {i+1}/{total}: acc={correct/(i+1):.3f}")

    accuracy = correct / total if total > 0 else 0.0
    log(f"  M2P accuracy: {accuracy:.4f} ({correct}/{total})")
    log(f"  Phase 7 time: {time.time()-t0:.1f}s")
    log_memory("post-m2p-eval")

    cleanup(m2p, model, tokenizer)
    del a_matrices
    return {"m2p_accuracy": accuracy, "m2p_correct": correct}


# ---- Main -------------------------------------------------------------------

def main():
    t_start = time.time()
    log("=" * 70)
    log("M2P on Qwen3-0.6B + GSM8K — First Real Language Test")
    log(f"SMOKE_TEST={IS_SMOKE} | N_TRAIN={N_TRAIN} | N_TEST={N_TEST} | STEPS={TRAIN_STEPS}")
    log("=" * 70)
    log_memory("start")

    # Phase 1: Load data (no model, just dataset)
    train_texts, test_prompts, test_answers = phase_load_data()
    log_memory("after-data")

    # Phase 2: Base model accuracy (load model, eval, unload)
    base_results = phase_eval_base(test_prompts, test_answers)
    log_memory("after-base-eval")

    # Phase 3: Build Grassmannian A-matrices (pure numpy, no model)
    phase_build_grassmannian()
    log_memory("after-grassmannian")

    # Phase 4: SFT LoRA training (load model, train B-matrices, unload)
    sft_train_results = phase_sft_train(train_texts)
    log_memory("after-sft-train")

    # Phase 5: Evaluate SFT adapter (load model, eval, unload)
    sft_eval_results = phase_eval_sft(test_prompts, test_answers)
    log_memory("after-sft-eval")

    # Phase 6: M2P training (load model, train M2P, unload)
    m2p_train_results = phase_m2p_train(train_texts)
    log_memory("after-m2p-train")

    # Phase 7: Evaluate M2P adapter (load model, eval, unload)
    m2p_eval_results = phase_eval_m2p(test_prompts, test_answers)
    log_memory("after-m2p-eval")

    # Kill criteria assessment
    base_acc = base_results["base_accuracy"]
    sft_acc = sft_eval_results["sft_accuracy"]
    m2p_acc = m2p_eval_results["m2p_accuracy"]

    sft_improvement = sft_acc - base_acc
    m2p_improvement = m2p_acc - base_acc
    quality_ratio = (m2p_improvement / sft_improvement
                     if abs(sft_improvement) > 1e-9 else 0.0)

    k_real_pass = quality_ratio >= 0.70
    k_kill_triggered = quality_ratio < 0.30
    peak_gb = mx.get_peak_memory() / 1e9

    results = {
        "experiment": "m2p_qwen06b_gsm8k",
        "model": MODEL_ID,
        "is_smoke": IS_SMOKE,
        "config": {
            "lora_rank": LORA_RANK,
            "lora_scale": LORA_SCALE,
            "lora_modules": LORA_MODULES,
            "d_m2p": D_M2P,
            "l_m2p": L_M2P,
            "n_memory": N_MEMORY,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "train_steps": TRAIN_STEPS,
            "lr": LR,
            "max_seq_len": MAX_SEQ_LEN,
            "max_gen_tokens": MAX_GEN_TOKENS,
        },
        **base_results,
        **sft_train_results,
        **sft_eval_results,
        **m2p_train_results,
        **m2p_eval_results,
        "sft_improvement": round(sft_improvement, 4),
        "m2p_improvement": round(m2p_improvement, 4),
        "quality_ratio": round(quality_ratio, 4),
        "kill_criteria": {
            "K_real_quality_ratio_ge_70pct": "PASS" if k_real_pass else "FAIL",
            "K_kill_quality_ratio_lt_30pct": "TRIGGERED" if k_kill_triggered else "not triggered",
            "K_mmlu": "N/A (not measured in this script)",
            "quality_ratio": round(quality_ratio, 4),
            "base_accuracy": round(base_acc, 4),
            "sft_accuracy": round(sft_acc, 4),
            "m2p_accuracy": round(m2p_acc, 4),
        },
        "peak_memory_gb": round(peak_gb, 2),
        "total_time_s": round(time.time() - t_start, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")

    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  Base accuracy:   {base_acc:.4f}")
    log(f"  SFT accuracy:    {sft_acc:.4f} ({sft_improvement:+.4f})")
    log(f"  M2P accuracy:    {m2p_acc:.4f} ({m2p_improvement:+.4f})")
    log(f"  Quality ratio:   {quality_ratio:.4f} ({quality_ratio*100:.1f}%)")
    log(f"  Peak memory:     {peak_gb:.1f} GB")
    log(f"  Total time:      {results['total_time_s']:.0f}s")
    log(f"")
    log(f"  K_real  (ratio >= 70%): {'PASS' if k_real_pass else 'FAIL'}")
    log(f"  K_KILL  (ratio <  30%): {'TRIGGERED' if k_kill_triggered else 'not triggered'}")


if __name__ == "__main__":
    main()
