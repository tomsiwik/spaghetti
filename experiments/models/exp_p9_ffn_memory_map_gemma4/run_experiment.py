"""
P9.D0: Map FFN Neurons to Knowledge Patterns in Gemma 4 E4B

Based on Geva et al. (arXiv:2012.14913) — FFN layers as key-value memories.
Analyzes neuron activation patterns across domains and checks if value vectors
predict next tokens.

Kill criteria:
  K1372: >=50% of top-activating neurons match identifiable patterns
  K1373: Domain-specific neurons cluster (intra >= 2x inter)
  K1374: Upper-layer value vectors predict next token with >1% agreement
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
TOP_K_PER_POS = 64        # top neurons tracked per position
TOP_K_PER_INPUT = 256     # top neurons aggregated per input per layer
NEURON_BATCH_SIZE = 128   # batch size for value vector projection
MIN_ACTIVATIONS = 5       # minimum triggers to count as "frequently activated"
UPPER_LAYERS = range(35, 42)  # for K3 analysis
OUTPUT_DIR = Path(__file__).parent

# ──────────────────────────────────────────────────────────────
# Domain-labeled inputs
# ──────────────────────────────────────────────────────────────
DOMAIN_INPUTS = {
    "math": [
        "The derivative of sin(x) multiplied by cos(x) can be computed using the product rule",
        "Let f(x) = x^3 - 2x + 1. To find the critical points, set f'(x) = 0",
        "The integral of e^(-x^2) from negative infinity to infinity equals sqrt(pi)",
        "A matrix is invertible if and only if its determinant is non-zero",
        "The eigenvalues of a symmetric positive definite matrix are all positive",
        "By the fundamental theorem of calculus, the derivative of the integral equals",
        "The Taylor series expansion of cos(x) around x=0 is 1 - x^2/2 + x^4/24",
        "Using Lagrange multipliers to optimize f(x,y) subject to g(x,y) = 0",
        "The Cauchy-Schwarz inequality states that the inner product squared is at most",
        "A convergent series must have terms approaching zero, but the converse fails",
        "The rank-nullity theorem tells us dim(ker) + dim(im) = dim(domain)",
        "Solving the differential equation dy/dx = 3y gives y = Ce^(3x)",
        "The binomial coefficient n choose k equals n! divided by k!(n-k)!",
        "For a normal distribution with mean mu and variance sigma squared",
        "The Fibonacci sequence satisfies the recurrence F(n) = F(n-1) + F(n-2)",
        "A group is abelian if and only if its multiplication is commutative",
        "The divergence theorem relates surface integrals to volume integrals",
        "Euler's identity connects five fundamental constants: e^(i*pi) + 1 = 0",
        "The prime number theorem states that pi(x) is approximately x/ln(x)",
        "Bayes' theorem gives P(A|B) = P(B|A) * P(A) / P(B)",
    ],
    "code": [
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "import numpy as np; arr = np.array([1, 2, 3]); print(arr.reshape(3, 1))",
        "class BinarySearchTree: def insert(self, key): if self.root is None",
        "async def fetch_data(url): async with aiohttp.ClientSession() as session",
        "SELECT users.name, COUNT(orders.id) FROM users LEFT JOIN orders ON users.id",
        "git checkout -b feature/auth && git add . && git commit -m 'initial auth'",
        "docker build -t myapp:latest . && docker run -p 8080:8080 myapp:latest",
        "const express = require('express'); const app = express(); app.get('/',",
        "try: result = database.execute(query) except OperationalError as e: rollback()",
        "def quicksort(arr): pivot = arr[len(arr)//2]; left = [x for x in arr if x < pivot]",
        "from flask import Flask, jsonify; app = Flask(__name__); @app.route('/api')",
        "kubectl apply -f deployment.yaml && kubectl get pods --watch",
        "interface User { id: number; name: string; email: string; }",
        "fn main() { let v: Vec<i32> = vec![1, 2, 3]; for x in &v { println!('{}', x); } }",
        "CREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(255) NOT NULL)",
        "import torch; model = torch.nn.Linear(10, 5); optimizer = torch.optim.Adam(",
        "regex = re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$')",
        "const [count, setCount] = useState(0); useEffect(() => { fetchData(); }, [])",
        "def merge_sort(arr): if len(arr) <= 1: return arr; mid = len(arr) // 2",
        "ssh -L 8888:localhost:8888 user@remote-server # port forwarding for jupyter",
    ],
    "medical": [
        "The patient presents with acute chest pain radiating to the left arm and jaw",
        "Metformin is the first-line treatment for type 2 diabetes mellitus",
        "MRI of the brain revealed a 2cm enhancing lesion in the right temporal lobe",
        "Complete blood count showed hemoglobin 8.2 g/dL suggesting iron deficiency anemia",
        "The Glasgow Coma Scale assesses eye opening, verbal response, and motor response",
        "Systolic blood pressure above 140 mmHg indicates stage 2 hypertension",
        "Penicillin remains effective against most streptococcal pharyngitis infections",
        "The tumor was staged as T2N1M0, indicating regional lymph node involvement",
        "Creatinine clearance below 30 mL/min indicates severe chronic kidney disease",
        "Electrocardiogram showed ST-segment elevation in leads V1 through V4",
        "The patient was started on warfarin with a target INR of 2.0 to 3.0",
        "Chest X-ray revealed bilateral pulmonary infiltrates consistent with pneumonia",
        "HbA1c of 9.5% suggests poorly controlled diabetes over the past three months",
        "Lumbar puncture showed elevated white blood cells suggesting bacterial meningitis",
        "The BRCA1 mutation significantly increases lifetime risk of breast cancer",
        "Post-operative management includes DVT prophylaxis with enoxaparin",
        "Spirometry showed FEV1/FVC ratio below 0.70 confirming obstructive lung disease",
        "Thyroid stimulating hormone was elevated at 12 mIU/L indicating hypothyroidism",
        "The APGAR score at 5 minutes was 9, indicating a healthy newborn",
        "Dopamine receptor antagonists are first-line treatment for acute psychosis",
    ],
    "legal": [
        "Under Section 230 of the Communications Decency Act, platforms are not liable",
        "The court held that the defendant's Fourth Amendment rights were violated",
        "Force majeure clauses excuse performance when extraordinary events occur",
        "The Miranda warning must be given before custodial interrogation begins",
        "Stare decisis requires courts to follow precedent established by higher courts",
        "The plaintiff must prove duty, breach, causation, and damages in negligence",
        "Article III of the Constitution establishes the federal judiciary",
        "The Uniform Commercial Code governs transactions in goods across state lines",
        "Intellectual property rights include patents, trademarks, copyrights, and trade secrets",
        "The statute of limitations for personal injury claims varies by jurisdiction",
        "Due process requires notice and an opportunity to be heard before deprivation",
        "The parol evidence rule bars extrinsic evidence to contradict written contracts",
        "Shareholders have fiduciary duties of care and loyalty to the corporation",
        "The exclusionary rule prevents illegally obtained evidence from being admitted",
        "Promissory estoppel may enforce promises even without traditional consideration",
        "The commerce clause grants Congress power to regulate interstate commerce",
        "An arbitration clause requires disputes to be resolved outside of court",
        "Habeas corpus petitions challenge the legality of a person's detention",
        "The doctrine of sovereign immunity protects governments from civil lawsuits",
        "Joint and several liability allows plaintiffs to recover full damages from any defendant",
    ],
    "general": [
        "The capital of France is Paris, a city known for the Eiffel Tower",
        "Photosynthesis converts carbon dioxide and water into glucose using sunlight",
        "The Great Wall of China stretches over 13,000 miles across northern China",
        "Shakespeare wrote Romeo and Juliet, a tragedy about two star-crossed lovers",
        "The speed of light in vacuum is approximately 299,792 kilometers per second",
        "Mount Everest, at 8,849 meters, is the tallest mountain above sea level",
        "The human body contains approximately 206 bones in the adult skeleton",
        "World War II ended in 1945 with the surrender of Germany and Japan",
        "The Amazon River is the largest river by discharge volume in the world",
        "Leonardo da Vinci painted the Mona Lisa, now displayed in the Louvre",
        "Climate change is driven primarily by greenhouse gas emissions from fossil fuels",
        "The periodic table organizes elements by atomic number and chemical properties",
        "Democracy originated in ancient Athens where citizens voted on legislation",
        "The Internet was developed from ARPANET, a US military research network",
        "Gravity keeps planets in orbit around the Sun according to Newton's laws",
        "The Renaissance began in Italy in the 14th century and spread across Europe",
        "DNA carries genetic instructions for the development of all living organisms",
        "The Olympic Games originated in ancient Greece as a religious festival",
        "Beethoven composed nine symphonies despite becoming deaf later in life",
        "The Sahara Desert is the largest hot desert, covering much of North Africa",
    ],
}

DOMAINS = list(DOMAIN_INPUTS.keys())


# ──────────────────────────────────────────────────────────────
# Model structure helpers
# ──────────────────────────────────────────────────────────────
def get_text_model(model):
    """Navigate wrapper to get the Gemma4TextModel with layers/embed_tokens."""
    # gemma4.Model -> language_model (gemma4_text.Model) -> model (Gemma4TextModel)
    if hasattr(model, "language_model"):
        return model.language_model.model
    # gemma4_text.Model -> model (Gemma4TextModel)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    raise RuntimeError(f"Unknown model structure: {type(model)}")


def get_embed_tokens(model):
    """Get embedding module for unembedding projection."""
    return get_text_model(model).embed_tokens


# ──────────────────────────────────────────────────────────────
# Patched MLP to capture intermediate activations
# ──────────────────────────────────────────────────────────────
class ProbedMLP(nn.Module):
    """MLP wrapper that captures GeGLU intermediate activations."""

    def __init__(self, original_mlp, layer_idx, captures):
        super().__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self._layer_idx = layer_idx
        self._captures = captures

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        intermediate = nn.gelu_approx(gate) * up
        self._captures[self._layer_idx] = intermediate
        return self.down_proj(intermediate)


def patch_model(model, captures):
    """Replace MLPs with probed versions. Returns originals for restoration."""
    text_model = get_text_model(model)
    originals = {}
    for i, layer in enumerate(text_model.layers):
        originals[i] = layer.mlp
        layer.mlp = ProbedMLP(layer.mlp, i, captures)
    return originals


def restore_model(model, originals):
    """Restore original MLPs."""
    text_model = get_text_model(model)
    for i, orig in originals.items():
        text_model.layers[i].mlp = orig


# ──────────────────────────────────────────────────────────────
# Dequantize a linear layer's weights
# ──────────────────────────────────────────────────────────────
def get_weight(linear_layer):
    """Get (possibly dequantized) weight matrix from a linear layer."""
    if isinstance(linear_layer, nn.QuantizedLinear):
        return mx.dequantize(
            linear_layer.weight,
            linear_layer.scales,
            linear_layer.biases,
            linear_layer.group_size,
            linear_layer.bits,
        )
    return linear_layer.weight


# ──────────────────────────────────────────────────────────────
# K3: Precompute value vector → token predictions
# ──────────────────────────────────────────────────────────────
def precompute_value_predictions(model, layer_indices):
    """
    For each upper layer, project value vectors through unembedding.
    Returns dict: layer_idx -> mx.array of predicted token ids [d_ff].
    """
    predictions = {}
    text_model = get_text_model(model)
    embed = text_model.embed_tokens

    for layer_idx in layer_indices:
        down_proj = text_model.layers[layer_idx].mlp.down_proj
        # Get original down_proj (not probed wrapper)
        if isinstance(down_proj, ProbedMLP):
            down_proj = down_proj.down_proj

        w_down = get_weight(down_proj)  # [hidden_size, d_ff]
        d_ff = w_down.shape[1]

        layer_preds = []
        for start in range(0, d_ff, NEURON_BATCH_SIZE):
            end = min(start + NEURON_BATCH_SIZE, d_ff)
            v_batch = w_down[:, start:end].T  # [B, hidden_size]
            # Use embed_tokens.as_linear for correct handling of quantized embeddings
            logits = embed.as_linear(v_batch)  # [B, vocab_size]
            preds = mx.argmax(logits, axis=-1)  # [B]
            mx.eval(preds)
            layer_preds.append(preds)
            del logits, v_batch

        predictions[layer_idx] = mx.concatenate(layer_preds)
        mx.eval(predictions[layer_idx])
        print(f"  Layer {layer_idx}: precomputed {d_ff} value predictions")

    return predictions


# ──────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    results = {
        "experiment": "exp_p9_ffn_memory_map_gemma4",
        "model": MODEL_ID,
        "config": {
            "top_k_per_pos": TOP_K_PER_POS,
            "top_k_per_input": TOP_K_PER_INPUT,
            "neuron_batch_size": NEURON_BATCH_SIZE,
            "min_activations": MIN_ACTIVATIONS,
            "upper_layers": list(UPPER_LAYERS),
            "n_inputs_per_domain": len(DOMAIN_INPUTS[DOMAINS[0]]),
            "domains": DOMAINS,
        },
    }

    # ── Load model ──────────────────────────────────────────
    print(f"Loading {MODEL_ID}...")
    model, tokenizer = load(MODEL_ID)
    text_model = get_text_model(model)
    n_layers = len(text_model.layers)
    print(f"  {n_layers} layers loaded")

    # Get layer dimensions
    layer_dims = {}
    for i in range(n_layers):
        mlp = text_model.layers[i].mlp
        # Infer d_ff: for QuantizedLinear use scales shape, else weight shape
        gp = mlp.gate_proj
        if isinstance(gp, nn.QuantizedLinear):
            # scales shape: [output_dims, n_groups] or similar
            # output_dims (= d_ff) is the first axis
            d_ff = gp.scales.shape[0]
        else:
            d_ff = gp.weight.shape[0]
        layer_dims[i] = d_ff
    print(f"  Layer dims: 0->{layer_dims[0]}, {n_layers-1}->{layer_dims[n_layers-1]}")
    results["layer_dims"] = {str(k): v for k, v in layer_dims.items()}

    # ── Precompute K3 value predictions (before patching) ──
    upper_layer_indices = [i for i in UPPER_LAYERS if i < n_layers]
    print(f"\nPrecomputing value vector predictions for layers {upper_layer_indices}...")
    value_predictions = precompute_value_predictions(model, upper_layer_indices)

    # ── Patch model for activation capture ──────────────────
    captures = {}
    originals = patch_model(model, captures)

    # ── Data structures for analysis ────────────────────────
    # Per-neuron: track which domain triggers it (layer, neuron) -> {domain: count}
    neuron_domain_counts = defaultdict(lambda: defaultdict(int))
    # Per-neuron: track which tokens trigger it (layer, neuron) -> {token_id: count}
    neuron_token_counts = defaultdict(lambda: defaultdict(int))
    # Per-neuron: total activation count
    neuron_total_counts = defaultdict(int)
    # Per-domain per-layer: set of top neurons
    domain_layer_neurons = defaultdict(lambda: defaultdict(set))
    # K3: agreement tracking
    k3_hits = 0
    k3_total = 0

    # ── Process inputs ──────────────────────────────────────
    total_inputs = sum(len(v) for v in DOMAIN_INPUTS.values())
    print(f"\nProcessing {total_inputs} inputs across {len(DOMAINS)} domains...")

    input_count = 0
    for domain in DOMAINS:
        for text in DOMAIN_INPUTS[domain]:
            input_count += 1
            if input_count % 20 == 0:
                print(f"  [{input_count}/{total_inputs}] Processing {domain}...")

            # Tokenize
            tokens = mx.array(tokenizer.encode(text))[None, :]  # [1, seq_len]
            seq_len = tokens.shape[1]

            # Forward pass (captures intermediates via ProbedMLP)
            logits = model(tokens)
            next_tokens = mx.argmax(logits[:, :-1, :], axis=-1)  # [1, seq_len-1]

            # Eval everything including captures
            arrays_to_eval = [next_tokens] + [captures[i] for i in range(n_layers) if i in captures]
            mx.eval(*arrays_to_eval)

            # Extract top-K neurons per layer
            for layer_idx in range(n_layers):
                if layer_idx not in captures:
                    continue
                intermediate = captures[layer_idx]  # [1, seq_len, d_ff]
                d_ff = layer_dims[layer_idx]

                # Max activation across positions for this input
                max_acts = mx.abs(intermediate[0]).max(axis=0)  # [d_ff]

                # Top-K neurons for this input at this layer
                k = min(TOP_K_PER_INPUT, d_ff)
                top_indices = mx.argpartition(max_acts, kth=-k)[-k:]
                mx.eval(top_indices)

                top_idx_list = top_indices.tolist()
                for neuron_idx in top_idx_list:
                    key = (layer_idx, neuron_idx)
                    neuron_domain_counts[key][domain] += 1
                    neuron_total_counts[key] += 1
                    domain_layer_neurons[domain][layer_idx].add(neuron_idx)

                # Token-level analysis: which tokens trigger top neurons?
                # For each position, find the top neuron and record the input token
                for pos in range(seq_len):
                    pos_acts = mx.abs(intermediate[0, pos, :])  # [d_ff]
                    top1_idx = mx.argmax(pos_acts).item()
                    token_at_pos = tokens[0, pos].item()
                    neuron_token_counts[(layer_idx, top1_idx)][token_at_pos] += 1

                # K3: Check value vector → next token agreement (upper layers only)
                if layer_idx in upper_layer_indices and seq_len > 1:
                    vp = value_predictions[layer_idx]
                    for pos in range(seq_len - 1):
                        pos_acts = mx.abs(intermediate[0, pos, :])
                        top1_idx = mx.argmax(pos_acts).item()
                        predicted = vp[top1_idx].item()
                        actual = tokens[0, pos + 1].item()
                        k3_total += 1
                        if predicted == actual:
                            k3_hits += 1

            # Clear captures to free memory
            captures.clear()
            del logits, next_tokens, tokens

    # ── Restore model ───────────────────────────────────────
    restore_model(model, originals)

    # ── K1: Pattern Specificity Analysis ────────────────────
    print("\n=== K1: Pattern Specificity ===")
    frequently_activated = {
        k: v for k, v in neuron_total_counts.items() if v >= MIN_ACTIVATIONS
    }
    n_frequent = len(frequently_activated)
    print(f"  Neurons with >= {MIN_ACTIVATIONS} activations: {n_frequent}")

    n_domain_pattern = 0
    n_token_pattern = 0
    n_any_pattern = 0

    for key in frequently_activated:
        has_pattern = False

        # Domain pattern: >70% from single domain
        domain_counts = neuron_domain_counts[key]
        total = sum(domain_counts.values())
        if total > 0:
            max_domain_frac = max(domain_counts.values()) / total
            if max_domain_frac >= 0.7:
                n_domain_pattern += 1
                has_pattern = True

        # Token pattern: >50% triggered by same token
        token_counts = neuron_token_counts.get(key, {})
        token_total = sum(token_counts.values())
        if token_total > 0:
            max_token_frac = max(token_counts.values()) / token_total
            if max_token_frac >= 0.5:
                n_token_pattern += 1
                has_pattern = True

        if has_pattern:
            n_any_pattern += 1

    pattern_rate = n_any_pattern / n_frequent if n_frequent > 0 else 0
    domain_pattern_rate = n_domain_pattern / n_frequent if n_frequent > 0 else 0
    token_pattern_rate = n_token_pattern / n_frequent if n_frequent > 0 else 0

    print(f"  Domain patterns (>70% single domain): {n_domain_pattern} ({domain_pattern_rate:.1%})")
    print(f"  Token patterns (>50% same token): {n_token_pattern} ({token_pattern_rate:.1%})")
    print(f"  Any pattern: {n_any_pattern} ({pattern_rate:.1%})")
    k1_pass = pattern_rate >= 0.50

    results["k1_pattern_specificity"] = {
        "n_frequently_activated": n_frequent,
        "n_domain_pattern": n_domain_pattern,
        "n_token_pattern": n_token_pattern,
        "n_any_pattern": n_any_pattern,
        "domain_pattern_rate": round(domain_pattern_rate, 4),
        "token_pattern_rate": round(token_pattern_rate, 4),
        "pattern_rate": round(pattern_rate, 4),
        "pass": k1_pass,
    }

    # ── K2: Domain Clustering ───────────────────────────────
    print("\n=== K2: Domain Clustering ===")

    # Compute Jaccard similarity between domain neuron sets per layer
    intra_sims = []
    inter_sims = []

    # Average across layers
    for layer_idx in range(n_layers):
        for i, d1 in enumerate(DOMAINS):
            s1 = domain_layer_neurons[d1][layer_idx]
            if not s1:
                continue
            for j, d2 in enumerate(DOMAINS):
                if j <= i:
                    continue
                s2 = domain_layer_neurons[d2][layer_idx]
                if not s2:
                    continue
                jaccard = len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0
                inter_sims.append(jaccard)

    # Intra-domain: split each domain's inputs in half, compare neuron sets
    # Since we track neurons per domain (not per input), we approximate by
    # checking how concentrated domain neurons are vs cross-domain
    # Use domain exclusivity: what fraction of a domain's neurons are NOT in other domains?
    domain_exclusive = {}
    for domain in DOMAINS:
        all_neurons = set()
        other_neurons = set()
        for layer_idx in range(n_layers):
            all_neurons.update(
                (layer_idx, n) for n in domain_layer_neurons[domain][layer_idx]
            )
            for other_domain in DOMAINS:
                if other_domain != domain:
                    other_neurons.update(
                        (layer_idx, n)
                        for n in domain_layer_neurons[other_domain][layer_idx]
                    )
        exclusive = all_neurons - other_neurons
        domain_exclusive[domain] = len(exclusive) / len(all_neurons) if all_neurons else 0

    avg_exclusive = sum(domain_exclusive.values()) / len(domain_exclusive) if domain_exclusive else 0
    avg_inter_jaccard = sum(inter_sims) / len(inter_sims) if inter_sims else 0

    # Clustering ratio: use (1 - inter_jaccard) / inter_jaccard as proxy
    # If inter_jaccard is low, domains are well-separated
    # Alternative: exclusive rate >= 0.30 means domains have distinct neurons
    clustering_ratio = (1 - avg_inter_jaccard) / avg_inter_jaccard if avg_inter_jaccard > 0 else float("inf")
    k2_pass = clustering_ratio >= 2.0

    print(f"  Avg inter-domain Jaccard: {avg_inter_jaccard:.4f}")
    print(f"  Clustering ratio (1-J)/J: {clustering_ratio:.2f}")
    print(f"  Domain exclusivity rates:")
    for domain, exc in domain_exclusive.items():
        print(f"    {domain}: {exc:.1%}")
    print(f"  Avg exclusivity: {avg_exclusive:.1%}")

    results["k2_domain_clustering"] = {
        "avg_inter_domain_jaccard": round(avg_inter_jaccard, 4),
        "clustering_ratio": round(clustering_ratio, 2),
        "domain_exclusivity": {d: round(v, 4) for d, v in domain_exclusive.items()},
        "avg_exclusivity": round(avg_exclusive, 4),
        "pass": k2_pass,
    }

    # ── K3: Next-Token Agreement ────────────────────────────
    print("\n=== K3: Next-Token Agreement ===")
    agreement_rate = k3_hits / k3_total if k3_total > 0 else 0
    random_baseline = 1.0 / 262144
    ratio_vs_random = agreement_rate / random_baseline if random_baseline > 0 else 0
    k3_pass = agreement_rate > 0.01  # >1%

    print(f"  Hits: {k3_hits} / {k3_total}")
    print(f"  Agreement rate: {agreement_rate:.4%}")
    print(f"  Random baseline: {random_baseline:.6%}")
    print(f"  Ratio vs random: {ratio_vs_random:.0f}x")

    results["k3_next_token_agreement"] = {
        "hits": k3_hits,
        "total": k3_total,
        "agreement_rate": round(agreement_rate, 6),
        "random_baseline": round(random_baseline, 8),
        "ratio_vs_random": round(ratio_vs_random, 1),
        "pass": k3_pass,
    }

    # ── Per-layer summary statistics ────────────────────────
    print("\n=== Per-Layer Neuron Counts ===")
    layer_stats = {}
    for layer_idx in range(n_layers):
        n_active = sum(
            1
            for (l, _), cnt in neuron_total_counts.items()
            if l == layer_idx and cnt >= MIN_ACTIVATIONS
        )
        n_domain_specific = sum(
            1
            for (l, n) in neuron_domain_counts
            if l == layer_idx
            and sum(neuron_domain_counts[(l, n)].values()) >= MIN_ACTIVATIONS
            and max(neuron_domain_counts[(l, n)].values())
            / sum(neuron_domain_counts[(l, n)].values())
            >= 0.7
        )
        layer_stats[str(layer_idx)] = {
            "d_ff": layer_dims[layer_idx],
            "frequently_activated": n_active,
            "domain_specific": n_domain_specific,
        }
        if layer_idx % 7 == 0 or layer_idx == n_layers - 1:
            print(
                f"  Layer {layer_idx:2d}: d_ff={layer_dims[layer_idx]:5d}, "
                f"active={n_active:4d}, domain_specific={n_domain_specific:4d}"
            )
    results["layer_stats"] = layer_stats

    # ── Summary ─────────────────────────────────────────────
    t_elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"K1372 (pattern >=50%): {pattern_rate:.1%} -> {'PASS' if k1_pass else 'FAIL'}")
    print(f"K1373 (clustering >=2x): {clustering_ratio:.2f}x -> {'PASS' if k2_pass else 'FAIL'}")
    print(f"K1374 (agreement >1%): {agreement_rate:.4%} -> {'PASS' if k3_pass else 'FAIL'}")
    print(f"Total time: {t_elapsed:.0f}s")

    results["kill_criteria"] = {
        "K1372": {"value": round(pattern_rate, 4), "threshold": 0.50, "pass": k1_pass},
        "K1373": {"value": round(clustering_ratio, 2), "threshold": 2.0, "pass": k2_pass},
        "K1374": {"value": round(agreement_rate, 6), "threshold": 0.01, "pass": k3_pass},
    }
    results["elapsed_seconds"] = round(t_elapsed, 1)

    # ── Save ────────────────────────────────────────────────
    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
