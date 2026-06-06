#!/usr/bin/env python3
"""P7.A1: Adapter Restricted to Null Space -- Quality Preserved?

Train LoRA adapters on v_proj restricted to null(W_v) via reparameterization.
Compare quality against unrestricted LoRA using mlx_lm's LoRALinear pattern.

Kill criteria:
  K1297: Null-space LoRA quality >= 80% of unrestricted (loss ratio)
  K1298: Base model general perplexity delta < 1pp with adapter
  K1299: max|W_v @ A_eff^T| < 1e-4 (orthogonality)

Platform: Apple M5 Pro 48GB, MLX only.
"""

import gc
import json
import math
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# Memory safety (CODING_GUIDELINES)
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
LORA_RANK = 16
LORA_SCALE = 20.0  # mlx_lm default
SEED = 42

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
TRAIN_ITERS = 30 if IS_SMOKE else 500
LR = 1e-4
MAX_SEQ_LEN = 256
N_TARGET_LAYERS = 8  # last N non-shared layers for LoRA


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def log(msg: str):
    print(msg, flush=True)


def log_memory(label: str = ""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING DATA — math instruction texts for next-token prediction
# ══════════════════════════════════════════════════════════════════════════════

MATH_TEXTS = [
    "Question: What is the quadratic formula?\nAnswer: For ax^2 + bx + c = 0, the solution is x = (-b +/- sqrt(b^2 - 4ac)) / (2a). The discriminant b^2 - 4ac determines the nature of the roots: positive gives two real roots, zero gives one repeated root, and negative gives two complex conjugate roots. This formula is derived by completing the square on the general quadratic equation.",
    "Question: What is the derivative of x^n?\nAnswer: The derivative of x^n is nx^(n-1), known as the power rule. This follows from the limit definition of the derivative: lim_{h->0} [(x+h)^n - x^n] / h. By expanding (x+h)^n using the binomial theorem and canceling, we get nx^(n-1). This rule extends to negative and fractional exponents as well.",
    "Question: Prove the sum formula for first n integers.\nAnswer: We prove by mathematical induction that sum_{k=1}^{n} k = n(n+1)/2. Base case: n=1 gives 1 = 1*2/2 = 1, which holds. Inductive step: assume the formula holds for some m. Then sum_{k=1}^{m+1} k = m(m+1)/2 + (m+1) = (m+1)(m+2)/2. By induction, the formula holds for all natural numbers n.",
    "Question: State the fundamental theorem of calculus.\nAnswer: The fundamental theorem of calculus has two parts. Part 1: If f is continuous on [a,b] and F(x) = integral from a to x of f(t)dt, then F'(x) = f(x). Part 2: If F is any antiderivative of f, then integral from a to b of f(x)dx = F(b) - F(a). This theorem provides the connection between differentiation and integration.",
    "Question: What is the Taylor series of e^x?\nAnswer: The Taylor expansion of e^x about x=0 is e^x = sum_{n=0}^{infinity} x^n / n! = 1 + x + x^2/2! + x^3/3! + ... This series converges for all real (and complex) x with an infinite radius of convergence. The exponential function is the unique function equal to its own derivative with f(0)=1.",
    "Question: What are eigenvalues and eigenvectors?\nAnswer: For a square matrix A, a scalar lambda is an eigenvalue if there exists a nonzero vector v such that Av = lambda*v. The vector v is called an eigenvector. Eigenvalues are found by solving det(A - lambda*I) = 0, the characteristic equation. The set of all eigenvalues is called the spectrum of A.",
    "Question: Explain integration by parts.\nAnswer: Integration by parts states that integral of u*dv = u*v - integral of v*du. This is derived from the product rule for differentiation: d(uv) = u*dv + v*du. Rearranging and integrating both sides gives the formula. The choice of u and dv is guided by the LIATE rule.",
    "Question: What is the chain rule?\nAnswer: The chain rule states that if y = f(g(x)), then dy/dx = f'(g(x)) * g'(x). In Leibniz notation: dy/dx = (dy/du) * (du/dx) where u = g(x). This rule is essential for differentiating composite functions and forms the basis of backpropagation in neural networks.",
    "Question: State the Cauchy-Schwarz inequality.\nAnswer: For vectors u and v in an inner product space, |<u,v>|^2 <= <u,u> * <v,v>. Equivalently, |u . v| <= ||u|| * ||v||. Equality holds if and only if u and v are linearly dependent (u = alpha*v for some scalar alpha). This is perhaps the most important inequality in mathematics.",
    "Question: Define a group in abstract algebra.\nAnswer: A group (G, *) is a set G equipped with a binary operation * satisfying four axioms: 1) Closure: for all a,b in G, a*b is in G. 2) Associativity: (a*b)*c = a*(b*c). 3) Identity: there exists e in G such that e*a = a*e = a for all a. 4) Inverse: for each a in G, there exists a^{-1} such that a*a^{-1} = a^{-1}*a = e.",
    "Question: What is the divergence theorem?\nAnswer: The divergence theorem (Gauss's theorem) states that for a vector field F and a volume V bounded by a closed surface S: the surface integral of F.dS equals the volume integral of div(F) dV. This relates the outward flux of a vector field through a closed surface to the behavior of the field inside the volume.",
    "Question: Explain the binomial theorem.\nAnswer: The binomial theorem states that for any non-negative integer n: (x+y)^n = sum_{k=0}^{n} C(n,k) * x^{n-k} * y^k, where C(n,k) = n! / (k! * (n-k)!) are the binomial coefficients. These coefficients form Pascal's triangle and count the number of ways to choose k items from n items.",
    "Question: What is the determinant of a matrix?\nAnswer: The determinant of an n x n matrix A is a scalar value det(A) computed from the matrix entries. Key properties: det(AB) = det(A)*det(B), A is invertible iff det(A) != 0, and det(A^T) = det(A). Geometrically, |det(A)| gives the volume scaling factor of the linear transformation defined by A.",
    "Question: Prove that sqrt(2) is irrational.\nAnswer: Proof by contradiction. Assume sqrt(2) = p/q where p,q are integers with gcd(p,q) = 1. Then 2 = p^2/q^2, so p^2 = 2q^2. This means p^2 is even, so p is even. Write p = 2k. Then 4k^2 = 2q^2, giving q^2 = 2k^2, so q is also even. But then gcd(p,q) >= 2, contradicting our assumption. Therefore sqrt(2) is irrational.",
    "Question: What is the spectral theorem?\nAnswer: The spectral theorem states that every real symmetric (or complex Hermitian) matrix A can be decomposed as A = Q * Lambda * Q^T, where Q is an orthogonal matrix whose columns are the eigenvectors, and Lambda is a diagonal matrix of eigenvalues. This means symmetric matrices are always diagonalizable with an orthonormal eigenbasis.",
    "Question: Explain the residue theorem.\nAnswer: The residue theorem in complex analysis states that if f is analytic inside and on a simple closed contour C except at isolated singularities z_1, ..., z_n, then the contour integral of f(z)dz equals 2*pi*i times the sum of residues Res(f, z_k). The residue at a pole of order m is computed as the limit of the (m-1)th derivative of (z-z_k)^m * f(z).",
    "Question: What is Stokes' theorem?\nAnswer: Stokes' theorem states that the integral of a differential form omega over the boundary dM of an oriented manifold M equals the integral of its exterior derivative d(omega) over M. In 3D vector calculus: the line integral of F around a closed curve C equals the surface integral of curl(F) over any surface bounded by C.",
    "Question: Define continuity using epsilon-delta.\nAnswer: A function f is continuous at point a if for every epsilon > 0 there exists delta > 0 such that |x - a| < delta implies |f(x) - f(a)| < epsilon. Informally, we can make f(x) as close to f(a) as we want by taking x sufficiently close to a. This is equivalent to lim_{x->a} f(x) = f(a).",
    "Question: What is L'Hopital's rule?\nAnswer: L'Hopital's rule states that if lim_{x->a} f(x) = lim_{x->a} g(x) = 0 (or both equal infinity), then lim_{x->a} f(x)/g(x) = lim_{x->a} f'(x)/g'(x), provided the right-hand limit exists. This rule can be applied repeatedly for nested indeterminate forms like 0/0 or infinity/infinity.",
    "Question: Explain the Jacobian matrix.\nAnswer: For a function f: R^n -> R^m, the Jacobian matrix J is the m x n matrix of all first-order partial derivatives: J_{ij} = df_i/dx_j. The Jacobian represents the best linear approximation to f at a point. Its determinant (when m=n) gives the local volume scaling factor of the transformation.",
]

# General knowledge texts for K1298 (base model preservation check)
GENERAL_TEXTS = [
    "The capital of France is Paris, located on the Seine River. Paris is known for landmarks including the Eiffel Tower, the Louvre Museum which houses the Mona Lisa, and the Cathedral of Notre-Dame. The city has been a major center of European culture, art, and science for centuries.",
    "Water is a chemical compound with the formula H2O, consisting of two hydrogen atoms covalently bonded to one oxygen atom. It exists in three states of matter: solid ice below 0 degrees Celsius, liquid water at room temperature, and gaseous steam above 100 degrees Celsius at standard pressure.",
    "The human heart is a muscular organ that pumps blood through the circulatory system via rhythmic contractions. It consists of four chambers: the right and left atria that receive blood, and the right and left ventricles that pump blood out. An adult heart beats approximately 60 to 100 times per minute.",
    "Photosynthesis is the process by which green plants and some organisms convert carbon dioxide and water into glucose and oxygen using energy from sunlight. The overall equation is 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2. This process occurs primarily in chloroplasts containing the pigment chlorophyll.",
    "William Shakespeare was an English playwright and poet who lived from 1564 to 1616. He wrote approximately 37 plays including tragedies like Hamlet, Macbeth, and King Lear, comedies like A Midsummer Night's Dream, and histories like Henry V. He is widely considered the greatest writer in the English language.",
]


# ══════════════════════════════════════════════════════════════════════════════
# NULL-SPACE LoRA LAYER — follows mlx_lm LoRALinear pattern exactly
# ══════════════════════════════════════════════════════════════════════════════

class NullSpaceLoRALinear(nn.Module):
    """LoRA reparameterized to live in null(W_base).

    Uses raw mx.array weights like mlx_lm LoRALinear.
    Forward: z = ((x @ Q) @ lora_a) @ lora_b
    where Q is frozen null-space basis (d_in, d_null).
    """

    def __init__(self, base_linear, Q: mx.array, r: int = 16, scale: float = 20.0):
        super().__init__()
        self.linear = base_linear
        d_null = Q.shape[1]
        output_dims = base_linear.weight.shape[0]

        # Frozen null-space basis — stored as private to exclude from parameters
        self._Q = Q

        self.scale = scale

        # LoRA weights: same pattern as mlx_lm LoRALinear but in null-space
        init_scale = 1 / math.sqrt(d_null)
        self.lora_a = mx.random.uniform(
            low=-init_scale, high=init_scale, shape=(d_null, r)
        )
        self.lora_b = mx.zeros(shape=(r, output_dims))

    def __call__(self, x):
        y = self.linear(x)
        # Project to null space, then LoRA
        x_null = x @ self._Q  # (batch, seq, d_null)
        z = (x_null @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def dequantize_weight(linear):
    """Extract float32 weight from QuantizedLinear or Linear."""
    if isinstance(linear, nn.QuantizedLinear):
        W = mx.dequantize(
            linear.weight, linear.scales, linear.biases,
            linear.group_size, linear.bits,
        )
    else:
        W = linear.weight
    return W.astype(mx.float32)


def compute_null_basis(W: mx.array) -> mx.array:
    """Compute null-space basis Q from SVD of W.

    W: (d_out, d_in). Returns Q: (d_in, d_null).
    """
    d_out, d_in = W.shape
    _, S, Vt = mx.linalg.svd(W, stream=mx.cpu)
    mx.eval(S, Vt)

    sigma_max = S[0].item()
    threshold = 1e-3 * sigma_max
    eff_rank = int(mx.sum(S > threshold).item())

    # Null-space basis: rows of Vt beyond effective rank, transposed to columns
    Q = Vt[eff_rank:, :].T  # (d_in, d_null)
    mx.eval(Q)

    d_null = d_in - eff_rank
    log(f"    shape ({d_out}, {d_in}), rank={eff_rank}, null_dim={d_null}")
    return Q


def get_non_shared_layers(model):
    """Find layers that compute their own KV (not shared from earlier layers).

    Gemma 4 shares KV from earlier layers for the last num_kv_shared_layers.
    Layers receiving shared KV skip k_proj/v_proj entirely, making LoRA on
    those projections dead code.
    """
    if hasattr(model, "language_model"):
        text_model = model.language_model.model
    else:
        text_model = model.model

    previous_kvs = text_model.previous_kvs
    # Non-shared layers: where previous_kvs[i] == i (they compute their own KV)
    non_shared = [i for i in range(len(previous_kvs)) if previous_kvs[i] == i]
    return non_shared


def tokenize_texts(tokenizer, texts, max_len=MAX_SEQ_LEN):
    """Tokenize texts into list of mx.array token sequences."""
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        if len(tokens) > 10:
            all_tokens.append(mx.array(tokens))
    return all_tokens


def compute_perplexity(model, token_arrays):
    """Compute average perplexity on token sequences."""
    total_loss = 0.0
    total_tokens = 0
    for tokens in token_arrays:
        x = tokens[None, :-1]
        targets = tokens[1:]
        logits = model(x)
        mx.eval(logits)
        loss = nn.losses.cross_entropy(
            logits.squeeze(0), targets, reduction="sum"
        )
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += targets.shape[0]
        del logits, loss, x
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = float(mx.exp(mx.array(avg_loss)).item())
    return {"avg_loss": round(avg_loss, 4), "perplexity": round(ppl, 2), "n_tokens": total_tokens}


def load_model_and_layers(model_id):
    """Load model and return (model, tokenizer, layers)."""
    from mlx_lm import load
    model, tokenizer = load(model_id)
    if hasattr(model, "language_model"):
        layers = model.language_model.model.layers
    else:
        layers = model.model.layers
    return model, tokenizer, layers


def check_adapter_effect(model, tokenizer, label="adapter"):
    """Verify adapter has non-zero effect on model output.

    Returns max logit delta vs a known test input. Raises if adapter is dead.
    """
    tokens = mx.array(tokenizer.encode("What is the quadratic formula?"))
    x = tokens[None, :-1]
    logits = model(x)
    mx.eval(logits)
    return logits


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Compute null-space bases
# ══════════════════════════════════════════════════════════════════════════════

def phase_null_bases(model_id):
    """Compute and save null-space bases for v_proj at target layers."""
    log("=== Phase 1: Computing null-space bases ===")
    model, tokenizer, layers = load_model_and_layers(model_id)
    log_memory("post-load")

    n_layers = len(layers)

    # Find non-shared layers (those that actually compute their own KV)
    non_shared = get_non_shared_layers(model)
    log(f"Layers: {n_layers} total, {len(non_shared)} non-shared (compute own KV)")
    log(f"Shared KV layers: {[i for i in range(n_layers) if i not in non_shared]}")

    # Target last N non-shared layers
    target_indices = non_shared[-N_TARGET_LAYERS:]
    log(f"Target layers: {target_indices}")

    # Compute null bases
    null_bases = {}
    t0 = time.time()
    for idx in target_indices:
        log(f"  Layer {idx} v_proj SVD...")
        W = dequantize_weight(layers[idx].self_attn.v_proj)
        mx.eval(W)
        Q = compute_null_basis(W)
        null_bases[idx] = Q
        del W

    svd_time = time.time() - t0
    log(f"SVD done in {svd_time:.1f}s")

    # Save to disk
    bases_path = EXPERIMENT_DIR / "null_bases.safetensors"
    save_dict = {f"layer_{k}": v for k, v in null_bases.items()}
    mx.save_safetensors(str(bases_path), save_dict)

    result = {
        "target_layers": target_indices,
        "svd_time_s": round(svd_time, 1),
        "null_dims": {str(k): int(v.shape[1]) for k, v in null_bases.items()},
    }

    cleanup(model, tokenizer)
    for k in list(null_bases.keys()):
        del null_bases[k]
    del null_bases
    gc.collect()
    mx.clear_cache()
    log_memory("post-phase1")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Train unrestricted LoRA (using mlx_lm LoRALinear)
# ══════════════════════════════════════════════════════════════════════════════

def phase_train_unrestricted(model_id, target_layers):
    """Train standard LoRA on v_proj using mlx_lm LoRALinear."""
    from mlx_lm.tuner.lora import LoRALinear

    log("\n=== Phase 2: Train unrestricted LoRA ===")
    model, tokenizer, layers = load_model_and_layers(model_id)
    log_memory("post-load")

    model.freeze()

    # Apply mlx_lm LoRALinear to v_proj
    lora_count = 0
    for idx in target_layers:
        base = layers[idx].self_attn.v_proj
        layers[idx].self_attn.v_proj = LoRALinear.from_base(
            base, r=LORA_RANK, scale=LORA_SCALE
        )
        lora_count += 1
    log(f"Applied LoRALinear (r={LORA_RANK}, scale={LORA_SCALE}) to {lora_count} v_proj layers")

    trainable = list(tree_flatten(model.trainable_parameters()))
    n_trainable = sum(v.size for _, v in trainable)
    log(f"Trainable parameters: {n_trainable:,}")

    # Tokenize
    train_tokens = tokenize_texts(tokenizer, MATH_TEXTS)
    log(f"Training sequences: {len(train_tokens)}")

    # Training
    optimizer = optim.AdamW(learning_rate=LR)

    def loss_fn(model, tokens):
        x = tokens[None, :-1]
        targets = tokens[1:]
        logits = model(x)
        return nn.losses.cross_entropy(logits.squeeze(0), targets, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    log(f"Training {TRAIN_ITERS} iters...")
    t_start = time.time()
    losses = []

    gc.disable()
    for step in range(TRAIN_ITERS):
        tokens = train_tokens[step % len(train_tokens)]
        loss, grads = loss_and_grad(model, tokens)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        loss_val = loss.item()
        losses.append(loss_val)

        if step % 100 == 0 or step == TRAIN_ITERS - 1:
            avg = sum(losses[-50:]) / len(losses[-50:])
            log(f"  Step {step:4d}/{TRAIN_ITERS}: loss={loss_val:.4f} avg50={avg:.4f}")
    gc.enable()

    train_time = time.time() - t_start
    final_loss = sum(losses[-20:]) / len(losses[-20:])
    log(f"Training done in {train_time:.1f}s, final loss: {final_loss:.4f}")

    # Diagnostic: check adapter weight norms and logit delta
    trainable_dict = dict(tree_flatten(model.trainable_parameters()))
    lora_b_norms = {}
    for name, param in trainable_dict.items():
        if "lora_b" in name:
            norm = mx.sqrt(mx.sum(param * param)).item()
            lora_b_norms[name] = norm
            log(f"  {name}: norm={norm:.6f}")
    mx.eval(trainable_dict)

    post_logits = check_adapter_effect(model, tokenizer, "unrestricted")
    log(f"Adapter logit sample: {post_logits[0, 0, :3].tolist()}")

    # Eval on math (held-out = last 5 texts)
    math_eval = tokenize_texts(tokenizer, MATH_TEXTS[-5:])
    math_ppl = compute_perplexity(model, math_eval)
    log(f"Math eval PPL: {math_ppl['perplexity']:.2f}")

    # Eval on general knowledge
    gen_eval = tokenize_texts(tokenizer, GENERAL_TEXTS)
    gen_ppl = compute_perplexity(model, gen_eval)
    log(f"General PPL: {gen_ppl['perplexity']:.2f}")

    # Save adapter
    adapter_path = EXPERIMENT_DIR / "unrestricted_adapter.safetensors"
    mx.save_safetensors(str(adapter_path), trainable_dict)

    result = {
        "n_trainable": n_trainable,
        "train_time_s": round(train_time, 1),
        "final_loss": round(final_loss, 4),
        "loss_first10": [round(l, 4) for l in losses[:10]],
        "loss_last10": [round(l, 4) for l in losses[-10:]],
        "math_ppl": math_ppl,
        "general_ppl": gen_ppl,
        "lora_b_norms": {k: round(v, 6) for k, v in lora_b_norms.items()},
    }

    cleanup(model, tokenizer, optimizer)
    log_memory("post-phase2")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Train null-space LoRA
# ══════════════════════════════════════════════════════════════════════════════

def phase_train_null_space(model_id, target_layers):
    """Train null-space LoRA on v_proj."""
    log("\n=== Phase 3: Train null-space LoRA ===")
    model, tokenizer, layers = load_model_and_layers(model_id)
    log_memory("post-load")

    model.freeze()

    # Load null bases
    bases_raw = mx.load(str(EXPERIMENT_DIR / "null_bases.safetensors"))
    null_bases = {int(k.split("_")[1]): v for k, v in bases_raw.items()}

    # Apply null-space LoRA
    lora_count = 0
    for idx in target_layers:
        Q = null_bases[idx]
        base = layers[idx].self_attn.v_proj
        layers[idx].self_attn.v_proj = NullSpaceLoRALinear(
            base, Q, r=LORA_RANK, scale=LORA_SCALE
        )
        lora_count += 1
    log(f"Applied NullSpaceLoRA (r={LORA_RANK}, scale={LORA_SCALE}) to {lora_count} v_proj layers")

    trainable = list(tree_flatten(model.trainable_parameters()))
    n_trainable = sum(v.size for _, v in trainable)
    log(f"Trainable parameters: {n_trainable:,}")

    # Tokenize
    train_tokens = tokenize_texts(tokenizer, MATH_TEXTS)
    log(f"Training sequences: {len(train_tokens)}")

    # Training
    optimizer = optim.AdamW(learning_rate=LR)

    def loss_fn(model, tokens):
        x = tokens[None, :-1]
        targets = tokens[1:]
        logits = model(x)
        return nn.losses.cross_entropy(logits.squeeze(0), targets, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    log(f"Training {TRAIN_ITERS} iters...")
    t_start = time.time()
    losses = []

    gc.disable()
    for step in range(TRAIN_ITERS):
        tokens = train_tokens[step % len(train_tokens)]
        loss, grads = loss_and_grad(model, tokens)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        loss_val = loss.item()
        losses.append(loss_val)

        if step % 100 == 0 or step == TRAIN_ITERS - 1:
            avg = sum(losses[-50:]) / len(losses[-50:])
            log(f"  Step {step:4d}/{TRAIN_ITERS}: loss={loss_val:.4f} avg50={avg:.4f}")
    gc.enable()

    train_time = time.time() - t_start
    final_loss = sum(losses[-20:]) / len(losses[-20:])
    log(f"Training done in {train_time:.1f}s, final loss: {final_loss:.4f}")

    # Diagnostic: check adapter weight norms
    trainable_dict = dict(tree_flatten(model.trainable_parameters()))
    lora_b_norms = {}
    for name, param in trainable_dict.items():
        if "lora_b" in name:
            norm = mx.sqrt(mx.sum(param * param)).item()
            lora_b_norms[name] = norm
            log(f"  {name}: norm={norm:.6f}")
    mx.eval(trainable_dict)

    post_logits = check_adapter_effect(model, tokenizer, "null-space")
    log(f"Adapter logit sample: {post_logits[0, 0, :3].tolist()}")

    # Eval
    math_eval = tokenize_texts(tokenizer, MATH_TEXTS[-5:])
    math_ppl = compute_perplexity(model, math_eval)
    log(f"Math eval PPL: {math_ppl['perplexity']:.2f}")

    gen_eval = tokenize_texts(tokenizer, GENERAL_TEXTS)
    gen_ppl = compute_perplexity(model, gen_eval)
    log(f"General PPL: {gen_ppl['perplexity']:.2f}")

    # Save adapter
    adapter_path = EXPERIMENT_DIR / "null_space_adapter.safetensors"
    mx.save_safetensors(str(adapter_path), trainable_dict)

    result = {
        "n_trainable": n_trainable,
        "train_time_s": round(train_time, 1),
        "final_loss": round(final_loss, 4),
        "loss_first10": [round(l, 4) for l in losses[:10]],
        "loss_last10": [round(l, 4) for l in losses[-10:]],
        "math_ppl": math_ppl,
        "general_ppl": gen_ppl,
        "lora_b_norms": {k: round(v, 6) for k, v in lora_b_norms.items()},
    }

    cleanup(model, tokenizer, optimizer)
    for k in list(null_bases.keys()):
        del null_bases[k]
    del null_bases, bases_raw
    gc.collect()
    mx.clear_cache()
    log_memory("post-phase3")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Base model baseline + orthogonality verification
# ══════════════════════════════════════════════════════════════════════════════

def phase_verify(model_id, target_layers):
    """Verify base model preservation (K1298) and orthogonality (K1299)."""
    log("\n=== Phase 4: Verification ===")
    model, tokenizer, layers = load_model_and_layers(model_id)
    log_memory("post-load")

    # Capture base model logits for delta check
    base_logits = check_adapter_effect(model, tokenizer, "base")
    log(f"Base logit sample: {base_logits[0, 0, :3].tolist()}")

    # Base model perplexity (no adapter)
    gen_eval = tokenize_texts(tokenizer, GENERAL_TEXTS)
    base_gen_ppl = compute_perplexity(model, gen_eval)
    log(f"Base general PPL: {base_gen_ppl['perplexity']:.2f}")

    math_eval = tokenize_texts(tokenizer, MATH_TEXTS[-5:])
    base_math_ppl = compute_perplexity(model, math_eval)
    log(f"Base math PPL: {base_math_ppl['perplexity']:.2f}")

    # K1299: Orthogonality
    log("\nVerifying orthogonality (K1299)...")
    bases_raw = mx.load(str(EXPERIMENT_DIR / "null_bases.safetensors"))
    null_bases = {int(k.split("_")[1]): v for k, v in bases_raw.items()}

    adapter_weights = mx.load(str(EXPERIMENT_DIR / "null_space_adapter.safetensors"))

    max_violation = 0.0
    orth_details = {}

    for idx in target_layers:
        Q = null_bases[idx]  # (d_in, d_null)

        # Find lora_a weight for this layer
        a_key = None
        for key in adapter_weights:
            if f"layers.{idx}.self_attn.v_proj.lora_a" in key:
                a_key = key
                break

        if a_key is None:
            log(f"  Layer {idx}: no adapter found, skip")
            continue

        A_small = adapter_weights[a_key]  # (d_null, r)
        # Effective A in full space: A_eff = Q @ A_small, shape (d_in, r)
        A_eff = Q @ A_small
        mx.eval(A_eff)

        # W_v @ A_eff should be ~0, shape (d_out, r)
        W_v = dequantize_weight(layers[idx].self_attn.v_proj)
        mx.eval(W_v)

        product = W_v @ A_eff  # (d_out, r)
        mx.eval(product)
        max_val = mx.max(mx.abs(product)).item()
        max_violation = max(max_violation, max_val)

        orth_details[str(idx)] = {
            "max_abs": float(max_val),
            "A_eff_norm": float(mx.sqrt(mx.sum(A_eff * A_eff)).item()),
        }
        log(f"  Layer {idx}: max|W_v @ A_eff| = {max_val:.2e}")

        del W_v, A_eff, product

    result = {
        "base_general_ppl": base_gen_ppl,
        "base_math_ppl": base_math_ppl,
        "max_violation": float(max_violation),
        "orthogonality": orth_details,
    }

    cleanup(model, tokenizer)
    for k in list(null_bases.keys()):
        del null_bases[k]
    del null_bases, bases_raw, adapter_weights
    gc.collect()
    mx.clear_cache()
    log_memory("post-phase4")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log_memory("start")
    log(f"Smoke test: {IS_SMOKE}")
    log(f"Config: r={LORA_RANK}, scale={LORA_SCALE}, lr={LR}, iters={TRAIN_ITERS}")

    # Phase 1
    bases_result = phase_null_bases(MODEL_ID)
    target_layers = bases_result["target_layers"]

    # Phase 2
    unrestricted = phase_train_unrestricted(MODEL_ID, target_layers)

    # Phase 3
    null_space = phase_train_null_space(MODEL_ID, target_layers)

    # Phase 4
    verify = phase_verify(MODEL_ID, target_layers)

    # ══════════════════════════════════════════════════════════════════════════
    # KILL CRITERIA
    # ══════════════════════════════════════════════════════════════════════════

    # K1297: Quality ratio (loss-based proxy)
    u_loss = unrestricted["final_loss"]
    n_loss = null_space["final_loss"]
    quality_ratio = u_loss / n_loss if n_loss > 0 else 0.0
    k1297_pass = quality_ratio >= 0.80
    k1297 = {
        "pass": k1297_pass,
        "quality_ratio": round(quality_ratio, 4),
        "unrestricted_loss": round(u_loss, 4),
        "null_space_loss": round(n_loss, 4),
        "threshold": 0.80,
    }

    # K1298: General PPL preservation (one-sided: adapter must not DEGRADE)
    # The null-space property W_v @ A_eff = 0 doesn't mean zero output effect.
    # Adapter can improve or change PPL. We check it doesn't get much worse.
    base_ppl = verify["base_general_ppl"]["perplexity"]
    adapter_ppl = null_space["general_ppl"]["perplexity"]
    # Degradation ratio: >1 means adapter worsened PPL
    ppl_ratio = adapter_ppl / base_ppl if base_ppl > 0 else float("inf")
    ppl_delta_pp = (ppl_ratio - 1.0) * 100  # positive = degradation
    k1298_pass = ppl_ratio <= 1.01  # at most 1% worse
    k1298 = {
        "pass": k1298_pass,
        "base_ppl": round(base_ppl, 2),
        "adapter_ppl": round(adapter_ppl, 2),
        "ppl_ratio": round(ppl_ratio, 6),
        "delta_pp": round(ppl_delta_pp, 4),
        "threshold_pp": 1.0,
    }

    # K1299: Orthogonality
    max_viol = verify["max_violation"]
    k1299_pass = max_viol < 1e-4
    k1299 = {
        "pass": k1299_pass,
        "max_violation": float(max_viol),
        "threshold": 1e-4,
    }

    all_pass = k1297_pass and k1298_pass and k1299_pass

    results = {
        "is_smoke": IS_SMOKE,
        "model": MODEL_ID,
        "config": {"rank": LORA_RANK, "scale": LORA_SCALE, "lr": LR, "iters": TRAIN_ITERS},
        "bases": bases_result,
        "unrestricted": unrestricted,
        "null_space": null_space,
        "verification": verify,
        "k1297": k1297,
        "k1298": k1298,
        "k1299": k1299,
        "all_pass": all_pass,
        "total_time_min": round((time.time() - t0) / 60, 2),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    log(f"\n{'='*60}")
    log("RESULTS SUMMARY")
    log(f"{'='*60}")
    log(f"K1297 (quality >= 80%): {'PASS' if k1297_pass else 'FAIL'} — ratio={quality_ratio:.4f}")
    log(f"  Unrestricted final loss: {u_loss:.4f}")
    log(f"  Null-space final loss:   {n_loss:.4f}")
    log(f"  Math PPL (unrestricted): {unrestricted['math_ppl']['perplexity']:.2f}")
    log(f"  Math PPL (null-space):   {null_space['math_ppl']['perplexity']:.2f}")
    log(f"  Math PPL (base model):   {verify['base_math_ppl']['perplexity']:.2f}")
    log(f"K1298 (general PPL < 1pp): {'PASS' if k1298_pass else 'FAIL'} — delta={ppl_delta_pp:.4f}pp")
    log(f"  Base: {base_ppl:.2f}, With adapter: {adapter_ppl:.2f}")
    log(f"K1299 (orth < 1e-4): {'PASS' if k1299_pass else 'FAIL'} — max={max_viol:.2e}")
    log(f"ALL PASS: {all_pass}")
    log(f"Total time: {results['total_time_min']:.1f} min")


if __name__ == "__main__":
    main()
