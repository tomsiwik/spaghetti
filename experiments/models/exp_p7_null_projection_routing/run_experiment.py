#!/usr/bin/env python3
"""P7.B0: Null-Space Projection as Natural Router (Geometry = Relevance)

Test whether |A_null_i @ Q^T @ x|^2 discriminates domains for routing.
The A-matrix's projection magnitude IS the routing signal — no separate router needed.

Kill criteria:
  K1300: Projection-based routing accuracy >= 80% on 5 domains (vs TF-IDF 97%)
  K1301: Projection magnitude correlates with response quality (Spearman r >= 0.3)
  K1302: Routing latency < 0.5ms (just matrix projection, no TF-IDF)

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
LORA_SCALE = 20.0
SEED = 42

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
TRAIN_ITERS = 30 if IS_SMOKE else 300
LR = 1e-4
MAX_SEQ_LEN = 256
N_TARGET_LAYERS = 8

DOMAINS = ["medical", "code", "math", "legal", "finance"]


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN DATA — 8 train + 4 test texts per domain (clearly separable)
# ══════════════════════════════════════════════════════════════════════════════

DOMAIN_DATA = {
    "medical": {
        "train": [
            "The patient presents with acute myocardial infarction characterized by ST-segment elevation in leads V1-V4. Troponin levels are elevated at 2.5 ng/mL. Immediate percutaneous coronary intervention is indicated. Dual antiplatelet therapy with aspirin and clopidogrel should be initiated.",
            "Type 2 diabetes mellitus is managed through lifestyle modifications and pharmacotherapy. First-line treatment is metformin, which reduces hepatic glucose production. HbA1c targets below 7% are recommended for most adults. Regular monitoring of renal function is essential due to risk of nephropathy.",
            "Pneumonia caused by Streptococcus pneumoniae typically presents with fever, productive cough, and consolidation on chest radiograph. Empiric antibiotic therapy with amoxicillin is first-line for community-acquired pneumonia. Blood cultures should be obtained before initiating antibiotics in hospitalized patients.",
            "The hypothalamic-pituitary-adrenal axis regulates cortisol secretion through a negative feedback loop. Cushing syndrome results from chronic cortisol excess, presenting with central obesity, moon facies, and striae. Diagnosis requires dexamethasone suppression testing and 24-hour urinary free cortisol measurement.",
            "Rheumatoid arthritis is an autoimmune inflammatory condition affecting synovial joints symmetrically. Disease-modifying antirheumatic drugs such as methotrexate are the cornerstone of treatment. Early aggressive therapy prevents joint erosion and disability progression. Anti-CCP antibodies have high specificity for diagnosis.",
            "Chronic kidney disease staging is based on glomerular filtration rate estimated from serum creatinine. Stage 3 CKD corresponds to GFR 30-59 mL/min. Management includes blood pressure control with ACE inhibitors, phosphate restriction, and erythropoietin supplementation for anemia of chronic kidney disease.",
            "Atrial fibrillation increases stroke risk by fivefold due to thrombus formation in the left atrial appendage. CHA2DS2-VASc scoring guides anticoagulation decisions. Direct oral anticoagulants such as apixaban are preferred over warfarin for stroke prevention in non-valvular atrial fibrillation.",
            "The Glasgow Coma Scale assesses consciousness by scoring eye opening, verbal response, and motor response. A score below 8 indicates severe brain injury requiring intubation. Pupil reactivity and brainstem reflexes provide additional prognostic information in comatose patients.",
        ],
        "test": [
            "Sepsis is defined as life-threatening organ dysfunction caused by dysregulated host response to infection. The qSOFA criteria include respiratory rate above 22, altered mental status, and systolic blood pressure below 100 mmHg. Early goal-directed therapy with fluid resuscitation and broad-spectrum antibiotics improves survival.",
            "Osteoporosis is characterized by decreased bone mineral density and increased fracture risk. Dual-energy X-ray absorptiometry measures T-scores for diagnosis. Bisphosphonates such as alendronate inhibit osteoclast-mediated bone resorption and reduce vertebral fracture risk by approximately 50 percent.",
            "Hepatitis C virus infection causes chronic liver inflammation leading to cirrhosis in 20-30% of cases. Direct-acting antivirals achieve sustained virological response rates exceeding 95%. Screening is recommended for all adults born between 1945 and 1965 due to high prevalence in this cohort.",
            "Parkinson disease results from dopaminergic neuron loss in the substantia nigra pars compacta. Cardinal motor features include bradykinesia, rigidity, resting tremor, and postural instability. Levodopa combined with carbidopa remains the most effective symptomatic treatment for motor symptoms.",
        ],
    },
    "code": {
        "train": [
            "Implement a binary search tree insertion function that maintains the BST invariant. For each new node, traverse left if the value is less than the current node, right if greater. After insertion, the tree depth increases by at most one. Time complexity is O(h) where h is the tree height.",
            "A hash map resolves collisions using either chaining with linked lists or open addressing with linear probing. The load factor determines when to resize the underlying array. Amortized insertion is O(1) but worst case is O(n) when all keys hash to the same bucket.",
            "Implement merge sort by recursively dividing the array into halves until single elements remain, then merging sorted subarrays. The merge operation compares elements from both subarrays and places them in order. Time complexity is O(n log n) in all cases with O(n) auxiliary space.",
            "Design a REST API endpoint that accepts JSON POST requests for user registration. Validate the email format using a regex pattern and hash the password with bcrypt before storing. Return HTTP 201 on success with the user ID, or HTTP 400 with validation error details.",
            "Write a Python decorator that caches function results using an LRU eviction policy. The cache stores key-value pairs where the key is the function arguments tuple. When capacity is exceeded, remove the least recently used entry. Use a doubly linked list and dictionary for O(1) operations.",
            "Implement depth-first search on a directed graph represented as an adjacency list. Track visited nodes to avoid infinite loops in cyclic graphs. The algorithm explores each branch to its deepest node before backtracking. DFS can detect cycles by checking if a neighbor is in the current recursion stack.",
            "Configure a CI/CD pipeline that runs unit tests on every pull request. Use parallel job execution to split the test suite across multiple workers. Cache dependencies between runs to reduce build times. Deploy to staging automatically when tests pass on the main branch.",
            "Implement a thread-safe producer-consumer queue using mutex locks and condition variables. The producer adds items to the buffer and signals the consumer. The consumer waits on the condition variable when the buffer is empty. Use RAII lock guards to prevent deadlocks from exceptions.",
        ],
        "test": [
            "Write a function to detect a cycle in a linked list using Floyd's tortoise and hare algorithm. Initialize two pointers at the head. Move the slow pointer one step and the fast pointer two steps per iteration. If they meet, a cycle exists. Time complexity is O(n) with O(1) space.",
            "Implement a trie data structure for efficient prefix matching of strings. Each node contains a dictionary mapping characters to child nodes and a boolean flag indicating word completion. Insert and search operations run in O(m) time where m is the string length.",
            "Build a rate limiter using the token bucket algorithm. Initialize the bucket with a maximum capacity and refill rate. Each request consumes one token. If the bucket is empty, reject the request with HTTP 429. Refill tokens based on elapsed time since last refill.",
            "Design a database connection pool that reuses existing connections instead of creating new ones. Maintain a queue of idle connections and a counter of active ones. When a connection is requested, dequeue an idle connection or create a new one if below the maximum pool size.",
        ],
    },
    "math": {
        "train": [
            "Question: What is the quadratic formula?\nAnswer: For ax^2 + bx + c = 0, the solution is x = (-b +/- sqrt(b^2 - 4ac)) / (2a). The discriminant b^2 - 4ac determines the nature of the roots: positive gives two real roots, zero gives one repeated root, and negative gives two complex conjugate roots.",
            "Question: What is the derivative of x^n?\nAnswer: The derivative of x^n is nx^(n-1), known as the power rule. This follows from the limit definition of the derivative: lim_{h->0} [(x+h)^n - x^n] / h. By expanding (x+h)^n using the binomial theorem, we get nx^(n-1).",
            "Question: Prove the sum formula for first n integers.\nAnswer: We prove by mathematical induction that sum_{k=1}^{n} k = n(n+1)/2. Base case: n=1 gives 1 = 1*2/2 = 1. Inductive step: assume for m, then sum_{k=1}^{m+1} k = m(m+1)/2 + (m+1) = (m+1)(m+2)/2. QED.",
            "Question: State the fundamental theorem of calculus.\nAnswer: Part 1: If f is continuous on [a,b] and F(x) = integral from a to x of f(t)dt, then F'(x) = f(x). Part 2: If F is any antiderivative of f, then integral from a to b of f(x)dx = F(b) - F(a).",
            "Question: What are eigenvalues and eigenvectors?\nAnswer: For a square matrix A, a scalar lambda is an eigenvalue if there exists nonzero v such that Av = lambda*v. Eigenvalues are found by solving det(A - lambda*I) = 0. The set of all eigenvalues is the spectrum of A.",
            "Question: What is the Taylor series of e^x?\nAnswer: e^x = sum_{n=0}^{infinity} x^n / n! = 1 + x + x^2/2! + x^3/3! + ... This series converges for all real and complex x. The exponential function is the unique function equal to its own derivative with f(0)=1.",
            "Question: State the Cauchy-Schwarz inequality.\nAnswer: For vectors u and v in an inner product space, |<u,v>|^2 <= <u,u>*<v,v>. Equality holds iff u and v are linearly dependent. This is the most important inequality in mathematics, underlying Minkowski, triangle, and Bessel inequalities.",
            "Question: Define a group in abstract algebra.\nAnswer: A group (G,*) satisfies: closure (a*b in G), associativity ((a*b)*c = a*(b*c)), identity (exists e: e*a = a*e = a), and inverse (for each a, exists a^{-1}: a*a^{-1} = e). Examples include integers under addition and permutations under composition.",
        ],
        "test": [
            "Question: What is the divergence theorem?\nAnswer: The divergence theorem states that for a vector field F and volume V bounded by closed surface S: the surface integral of F dot dS equals the volume integral of div(F) dV. This relates outward flux through a surface to field behavior inside.",
            "Question: Prove that sqrt(2) is irrational.\nAnswer: By contradiction. Assume sqrt(2) = p/q with gcd(p,q)=1. Then p^2 = 2q^2, so p is even. Write p=2k, giving q^2 = 2k^2, so q is even. But gcd(p,q) >= 2, contradiction. Therefore sqrt(2) is irrational.",
            "Question: What is the spectral theorem?\nAnswer: Every real symmetric matrix A can be decomposed as A = Q Lambda Q^T where Q is orthogonal with eigenvector columns and Lambda is diagonal with eigenvalues. Symmetric matrices are always diagonalizable with orthonormal eigenbasis.",
            "Question: Explain the residue theorem.\nAnswer: If f is analytic inside and on closed contour C except at isolated singularities z_k, then the contour integral of f(z)dz equals 2*pi*i times the sum of residues Res(f, z_k). Residues at poles of order m are computed from derivatives.",
        ],
    },
    "legal": {
        "train": [
            "Under Article 2 of the Uniform Commercial Code, a contract for the sale of goods valued at $500 or more must be evidenced by a written memorandum signed by the party to be charged. This statute of frauds requirement has exceptions for specially manufactured goods and partial performance.",
            "The doctrine of stare decisis requires courts to follow precedent established by higher courts in the same jurisdiction. A court may distinguish a prior case on its facts but cannot directly overrule a binding precedent from a superior court. This promotes consistency and predictability in legal outcomes.",
            "In tort law, negligence requires four elements: duty of care, breach of that duty, causation both actual and proximate, and damages. The reasonable person standard is used to determine whether the defendant's conduct fell below the expected standard of care in the circumstances.",
            "Miranda rights must be administered before custodial interrogation. The suspect must be informed of the right to remain silent, that statements may be used against them, the right to an attorney, and that an attorney will be appointed if they cannot afford one.",
            "A limited liability company provides its members with protection from personal liability for business debts while allowing pass-through taxation. The operating agreement governs member rights, profit distribution, and management structure. Members may participate in management without losing limited liability protection.",
            "The Federal Rules of Civil Procedure govern discovery in federal litigation. Parties may obtain discovery regarding any nonprivileged matter relevant to any party's claim or defense. Rule 26 requires initial disclosures including identification of witnesses, documents, and computation of damages.",
            "Copyright protection attaches automatically to original works of authorship fixed in a tangible medium of expression. Registration with the Copyright Office is not required for protection but is necessary to bring an infringement action. Fair use permits limited copying for criticism, commentary, and education.",
            "A fiduciary duty requires the fiduciary to act in the best interest of the beneficiary with loyalty and care. Corporate directors owe fiduciary duties to the corporation and its shareholders. The business judgment rule protects informed, good-faith decisions from judicial second-guessing.",
        ],
        "test": [
            "The Fourth Amendment protects against unreasonable searches and seizures by requiring probable cause for warrants. The exclusionary rule bars illegally obtained evidence from trial. Exceptions include consent searches, plain view doctrine, exigent circumstances, and searches incident to lawful arrest.",
            "A trust is created when a settlor transfers property to a trustee for the benefit of designated beneficiaries. The trustee holds legal title and manages the trust corpus according to the trust instrument. Revocable trusts may be modified during the settlor's lifetime.",
            "Under contract law, promissory estoppel prevents a party from reneging on a promise when the promisee has reasonably relied on that promise to their detriment. Unlike traditional contract formation, promissory estoppel does not require consideration but does require reasonable and foreseeable reliance.",
            "The doctrine of res judicata bars relitigation of claims that were or could have been raised in a prior action between the same parties. Collateral estoppel prevents relitigating specific issues that were actually decided in a prior proceeding. Both promote judicial economy.",
        ],
    },
    "finance": {
        "train": [
            "The discounted cash flow model values a company by projecting future free cash flows and discounting them to present value using the weighted average cost of capital. Terminal value accounts for cash flows beyond the projection period, typically calculated using the Gordon growth model.",
            "The Black-Scholes option pricing model derives the fair value of European call and put options. Key inputs include the underlying asset price, strike price, time to expiration, risk-free rate, and implied volatility. The model assumes log-normal price distribution and continuous trading.",
            "A bond's yield to maturity is the internal rate of return earned if held to maturity with all coupon payments reinvested at the same rate. Duration measures price sensitivity to interest rate changes. Convexity captures the curvature of the price-yield relationship for large rate movements.",
            "Portfolio diversification reduces unsystematic risk by combining assets with low correlation. The efficient frontier represents portfolios offering maximum expected return for each level of risk. The Capital Asset Pricing Model derives expected return as the risk-free rate plus beta times the market risk premium.",
            "Earnings per share is calculated by dividing net income minus preferred dividends by the weighted average shares outstanding. The price-to-earnings ratio compares share price to EPS, indicating how much investors pay per dollar of earnings. Forward PE uses projected earnings for valuation.",
            "The Federal Reserve implements monetary policy through open market operations, adjusting the federal funds rate target. Quantitative easing involves purchasing Treasury securities and mortgage-backed securities to increase the money supply. These actions influence lending rates, inflation expectations, and economic activity.",
            "Financial statement analysis uses ratio analysis to evaluate liquidity, profitability, and solvency. The current ratio measures short-term liquidity as current assets divided by current liabilities. Return on equity measures profitability as net income divided by shareholders' equity.",
            "Mergers and acquisitions involve due diligence review of the target's financial statements, contracts, and liabilities. Synergy valuation estimates cost savings and revenue enhancements from the combination. The acquisition premium represents the amount paid above the target's current market valuation.",
        ],
        "test": [
            "Value at Risk quantifies the maximum expected loss over a specified time horizon at a given confidence level. A 95% one-day VaR of $1 million means there is a 5% probability of losing more than $1 million in a single day. Monte Carlo simulation is one common VaR methodology.",
            "The balance sheet reports a company's assets, liabilities, and equity at a point in time. Assets equal liabilities plus shareholders' equity. Working capital is current assets minus current liabilities, measuring the firm's ability to meet short-term obligations.",
            "Venture capital funds invest in early-stage companies in exchange for equity ownership. Due diligence evaluates the founding team, market size, competitive moat, and unit economics. Term sheets specify valuation, liquidation preferences, anti-dilution protections, and board composition.",
            "Derivative instruments include futures, options, and swaps used for hedging or speculation. An interest rate swap exchanges fixed-rate payments for floating-rate payments. Credit default swaps transfer credit risk from the protection buyer to the seller in exchange for periodic premium payments.",
        ],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# NULL-SPACE LoRA LAYER (reused from exp_p7_null_space_adapter_quality)
# ══════════════════════════════════════════════════════════════════════════════

class NullSpaceLoRALinear(nn.Module):
    """LoRA reparameterized to live in null(W_base).

    Forward: z = ((x @ Q) @ lora_a) @ lora_b
    where Q is frozen null-space basis (d_in, d_null).
    """

    def __init__(self, base_linear, Q: mx.array, r: int = 16, scale: float = 20.0):
        super().__init__()
        self.linear = base_linear
        d_null = Q.shape[1]
        output_dims = base_linear.weight.shape[0]
        self._Q = Q
        self.scale = scale
        init_scale = 1 / math.sqrt(d_null)
        self.lora_a = mx.random.uniform(
            low=-init_scale, high=init_scale, shape=(d_null, r)
        )
        self.lora_b = mx.zeros(shape=(r, output_dims))

    def __call__(self, x):
        y = self.linear(x)
        x_null = x @ self._Q
        z = (x_null @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)


class CaptureLinear(nn.Module):
    """Wraps a linear layer to capture its input activations."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.last_input = None

    def __call__(self, x):
        self.last_input = x
        return self.base(x)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

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


def dequantize_weight(linear):
    if isinstance(linear, nn.QuantizedLinear):
        W = mx.dequantize(
            linear.weight, linear.scales, linear.biases,
            linear.group_size, linear.bits,
        )
    else:
        W = linear.weight
    return W.astype(mx.float32)


def compute_null_basis(W: mx.array) -> mx.array:
    d_out, d_in = W.shape
    _, S, Vt = mx.linalg.svd(W, stream=mx.cpu)
    mx.eval(S, Vt)
    sigma_max = S[0].item()
    threshold = 1e-3 * sigma_max
    eff_rank = int(mx.sum(S > threshold).item())
    Q = Vt[eff_rank:, :].T  # (d_in, d_null)
    mx.eval(Q)
    d_null = d_in - eff_rank
    log(f"    shape ({d_out}, {d_in}), rank={eff_rank}, null_dim={d_null}")
    return Q


def get_non_shared_layers(model):
    if hasattr(model, "language_model"):
        text_model = model.language_model.model
    else:
        text_model = model.model
    previous_kvs = text_model.previous_kvs
    return [i for i in range(len(previous_kvs)) if previous_kvs[i] == i]


def load_model_and_layers(model_id):
    from mlx_lm import load
    model, tokenizer = load(model_id)
    if hasattr(model, "language_model"):
        layers = model.language_model.model.layers
    else:
        layers = model.model.layers
    return model, tokenizer, layers


def tokenize_texts(tokenizer, texts, max_len=MAX_SEQ_LEN):
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        if len(tokens) > 10:
            all_tokens.append(mx.array(tokens))
    return all_tokens


def compute_ntp_loss(model, token_arrays):
    """Compute average NTP loss on token sequences."""
    total_loss = 0.0
    total_tokens = 0
    for tokens in token_arrays:
        x = tokens[None, :-1]
        targets = tokens[1:]
        logits = model(x)
        mx.eval(logits)
        loss = nn.losses.cross_entropy(logits.squeeze(0), targets, reduction="sum")
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += targets.shape[0]
        del logits, loss, x
    return total_loss / max(total_tokens, 1)


def spearman_r(x, y):
    """Spearman rank correlation (no scipy dependency)."""
    n = len(x)
    if n < 3:
        return 0.0
    # Compute ranks (average rank for ties)
    def rank(vals):
        indexed = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[indexed[j + 1]] == vals[indexed[j]]:
                j += 1
            avg_rank = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg_rank
            i = j + 1
        return ranks

    rx = rank(x)
    ry = rank(y)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - 6.0 * d_sq / (n * (n * n - 1))


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Compute or reuse null-space bases
# ══════════════════════════════════════════════════════════════════════════════

def phase_null_bases(model_id):
    """Compute null-space bases for v_proj at target layers."""
    log("=== Phase 1: Null-space bases ===")

    # Try to reuse from prior experiment
    prior_bases = EXPERIMENT_DIR.parent / "exp_p7_null_space_adapter_quality" / "null_bases.safetensors"
    bases_path = EXPERIMENT_DIR / "null_bases.safetensors"

    if prior_bases.exists():
        log(f"Reusing null bases from {prior_bases.name}")
        import shutil
        shutil.copy2(prior_bases, bases_path)
        bases_raw = mx.load(str(bases_path))
        null_bases = {int(k.split("_")[1]): v for k, v in bases_raw.items()}
        target_layers = sorted(null_bases.keys())
        null_dims = {str(k): int(v.shape[1]) for k, v in null_bases.items()}
        del bases_raw, null_bases
        gc.collect()
        mx.clear_cache()
        return {"target_layers": target_layers, "null_dims": null_dims, "reused": True}

    # Compute fresh
    model, tokenizer, layers = load_model_and_layers(model_id)
    log_memory("post-load")
    non_shared = get_non_shared_layers(model)
    target_layers = non_shared[-N_TARGET_LAYERS:]
    log(f"Target layers: {target_layers}")

    null_bases = {}
    for idx in target_layers:
        log(f"  Layer {idx} v_proj SVD...")
        W = dequantize_weight(layers[idx].self_attn.v_proj)
        mx.eval(W)
        Q = compute_null_basis(W)
        null_bases[idx] = Q
        del W

    save_dict = {f"layer_{k}": v for k, v in null_bases.items()}
    mx.save_safetensors(str(bases_path), save_dict)
    null_dims = {str(k): int(v.shape[1]) for k, v in null_bases.items()}

    cleanup(model, tokenizer)
    for k in list(null_bases.keys()):
        del null_bases[k]
    del null_bases
    gc.collect()
    mx.clear_cache()
    log_memory("post-phase1")
    return {"target_layers": target_layers, "null_dims": null_dims, "reused": False}


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Train 5 domain null-space LoRA adapters
# ══════════════════════════════════════════════════════════════════════════════

def phase_train_adapter(model_id, domain, target_layers):
    """Train one null-space LoRA adapter for a single domain."""
    log(f"\n--- Training adapter: {domain} ---")
    model, tokenizer, layers = load_model_and_layers(model_id)
    model.freeze()
    log_memory(f"post-load-{domain}")

    # Load null bases
    bases_raw = mx.load(str(EXPERIMENT_DIR / "null_bases.safetensors"))
    null_bases = {int(k.split("_")[1]): v for k, v in bases_raw.items()}

    # Apply NullSpaceLoRALinear
    for idx in target_layers:
        Q = null_bases[idx]
        base = layers[idx].self_attn.v_proj
        layers[idx].self_attn.v_proj = NullSpaceLoRALinear(
            base, Q, r=LORA_RANK, scale=LORA_SCALE
        )

    trainable = list(tree_flatten(model.trainable_parameters()))
    n_trainable = sum(v.size for _, v in trainable)
    log(f"  Trainable params: {n_trainable:,}")

    # Tokenize domain training data
    train_tokens = tokenize_texts(tokenizer, DOMAIN_DATA[domain]["train"])
    log(f"  Training sequences: {len(train_tokens)}")

    # Train
    optimizer = optim.AdamW(learning_rate=LR)

    def loss_fn(model, tokens):
        x = tokens[None, :-1]
        targets = tokens[1:]
        logits = model(x)
        return nn.losses.cross_entropy(logits.squeeze(0), targets, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    t_start = time.time()
    losses = []
    gc.disable()
    for step in range(TRAIN_ITERS):
        tokens = train_tokens[step % len(train_tokens)]
        loss, grads = loss_and_grad(model, tokens)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(loss.item())
        if step % 100 == 0 or step == TRAIN_ITERS - 1:
            avg = sum(losses[-50:]) / len(losses[-50:])
            log(f"  Step {step:4d}/{TRAIN_ITERS}: loss={losses[-1]:.4f} avg50={avg:.4f}")
    gc.enable()

    train_time = time.time() - t_start
    final_loss = sum(losses[-20:]) / len(losses[-20:])
    log(f"  Done in {train_time:.1f}s, final loss: {final_loss:.4f}")

    # Save adapter A and B weights per layer
    adapter_dict = {}
    for idx in target_layers:
        lora_mod = layers[idx].self_attn.v_proj
        adapter_dict[f"layer_{idx}_lora_a"] = lora_mod.lora_a
        adapter_dict[f"layer_{idx}_lora_b"] = lora_mod.lora_b
    mx.eval(adapter_dict)
    adapter_path = EXPERIMENT_DIR / f"{domain}_adapter.safetensors"
    mx.save_safetensors(str(adapter_path), adapter_dict)

    cleanup(model, tokenizer, optimizer)
    for k in list(null_bases.keys()):
        del null_bases[k]
    del null_bases, bases_raw, adapter_dict
    gc.collect()
    mx.clear_cache()
    log_memory(f"post-train-{domain}")

    return {
        "domain": domain,
        "n_trainable": n_trainable,
        "train_time_s": round(train_time, 1),
        "final_loss": round(final_loss, 4),
    }


def phase_train_all(model_id, target_layers):
    """Train all domain adapters sequentially."""
    log("\n=== Phase 2: Training 5 domain adapters ===")
    results = {}
    for domain in DOMAINS:
        mx.random.seed(SEED)  # Same init for fairness
        results[domain] = phase_train_adapter(model_id, domain, target_layers)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Routing evaluation via projection magnitude
# ══════════════════════════════════════════════════════════════════════════════

def phase_routing_eval(model_id, target_layers):
    """Evaluate routing by projecting v_proj inputs onto adapter A-matrices."""
    log("\n=== Phase 3: Routing evaluation ===")
    model, tokenizer, layers = load_model_and_layers(model_id)
    log_memory("post-load-routing")

    # Load null bases
    bases_raw = mx.load(str(EXPERIMENT_DIR / "null_bases.safetensors"))
    null_bases = {int(k.split("_")[1]): v for k, v in bases_raw.items()}

    # Load all adapter A-matrices (only lora_a needed for routing)
    adapter_A = {}
    for domain in DOMAINS:
        path = EXPERIMENT_DIR / f"{domain}_adapter.safetensors"
        weights = mx.load(str(path))
        adapter_A[domain] = {
            int(k.split("_")[1]): v
            for k, v in weights.items() if "lora_a" in k
        }
        del weights

    # Wrap v_proj with CaptureLinear to get input activations
    originals = {}
    for idx in target_layers:
        originals[idx] = layers[idx].self_attn.v_proj
        layers[idx].self_attn.v_proj = CaptureLinear(originals[idx])

    # Collect test texts with domain labels
    test_items = []
    for domain in DOMAINS:
        for text in DOMAIN_DATA[domain]["test"]:
            test_items.append((domain, text))

    # Compute projections for each test text
    projection_matrix = []  # [n_test x n_domains]
    per_layer_correct = {idx: 0 for idx in target_layers}
    per_layer_total = {idx: 0 for idx in target_layers}

    for true_domain, text in test_items:
        tokens = tokenizer.encode(text)
        if len(tokens) > MAX_SEQ_LEN:
            tokens = tokens[:MAX_SEQ_LEN]
        token_arr = mx.array(tokens)[None, :-1]

        # Forward pass to capture activations
        logits = model(token_arr)
        mx.eval(logits)
        del logits

        # Compute projection magnitudes across layers
        domain_scores = {d: 0.0 for d in DOMAINS}
        per_layer_scores = {idx: {} for idx in target_layers}

        for idx in target_layers:
            x = layers[idx].self_attn.v_proj.last_input  # (1, seq, d_in)
            Q = null_bases[idx]  # (d_in, d_null)
            x_null = x @ Q  # (1, seq, d_null)
            mx.eval(x_null)

            # Normalize A-matrices for fair comparison
            for domain in DOMAINS:
                A_null = adapter_A[domain][idx]  # (d_null, r)
                proj = x_null @ A_null  # (1, seq, r)
                # Mean squared projection across tokens and rank dims
                magnitude = mx.mean(mx.sum(proj * proj, axis=-1)).item()
                domain_scores[domain] += magnitude
                per_layer_scores[idx][domain] = magnitude

            del x_null
            # Per-layer routing accuracy
            layer_pred = max(DOMAINS, key=lambda d: per_layer_scores[idx][d])
            if layer_pred == true_domain:
                per_layer_correct[idx] += 1
            per_layer_total[idx] += 1

        # Store projection vector for this text
        proj_vec = [domain_scores[d] for d in DOMAINS]
        projection_matrix.append({
            "true_domain": true_domain,
            "projections": {d: round(domain_scores[d], 6) for d in DOMAINS},
            "predicted": max(DOMAINS, key=lambda d: domain_scores[d]),
        })

    # Restore original v_proj
    for idx in target_layers:
        layers[idx].self_attn.v_proj = originals[idx]

    # Compute routing accuracy
    correct = sum(1 for item in projection_matrix if item["predicted"] == item["true_domain"])
    total = len(projection_matrix)
    accuracy = correct / total if total > 0 else 0.0

    # Per-layer accuracy
    layer_accuracy = {
        str(idx): round(per_layer_correct[idx] / max(per_layer_total[idx], 1), 4)
        for idx in target_layers
    }

    log(f"Routing accuracy: {correct}/{total} = {accuracy:.1%}")
    log(f"Per-layer accuracy: {layer_accuracy}")

    # Confusion matrix
    confusion = {d: {d2: 0 for d2 in DOMAINS} for d in DOMAINS}
    for item in projection_matrix:
        confusion[item["true_domain"]][item["predicted"]] += 1
    log(f"Confusion matrix:")
    for d in DOMAINS:
        log(f"  {d:10s}: {confusion[d]}")

    cleanup(model, tokenizer)
    for k in list(null_bases.keys()):
        del null_bases[k]
    del null_bases, bases_raw, adapter_A, originals
    gc.collect()
    mx.clear_cache()
    log_memory("post-phase3")

    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "per_item": projection_matrix,
        "per_layer_accuracy": layer_accuracy,
        "confusion": confusion,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Quality correlation (projection magnitude vs NTP loss)
# ══════════════════════════════════════════════════════════════════════════════

def phase_quality_correlation(model_id, target_layers):
    """Correlate projection magnitude with adapter quality (NTP loss)."""
    log("\n=== Phase 4: Quality correlation ===")
    model, tokenizer, layers = load_model_and_layers(model_id)
    log_memory("post-load-quality")

    # Load null bases
    bases_raw = mx.load(str(EXPERIMENT_DIR / "null_bases.safetensors"))
    null_bases = {int(k.split("_")[1]): v for k, v in bases_raw.items()}

    # Collect all test texts
    test_items = []
    for domain in DOMAINS:
        for text in DOMAIN_DATA[domain]["test"]:
            test_items.append((domain, text))

    test_tokens_all = [tokenize_texts(tokenizer, [text])[0] for _, text in test_items]

    # Save original v_proj references
    originals = {}
    for idx in target_layers:
        originals[idx] = layers[idx].self_attn.v_proj

    # For each adapter, compute NTP loss on all test texts
    all_magnitudes = []
    all_qualities = []

    for adapter_domain in DOMAINS:
        log(f"  Evaluating adapter: {adapter_domain}")

        # Apply null-space LoRA with saved weights
        adapter_weights = mx.load(str(EXPERIMENT_DIR / f"{adapter_domain}_adapter.safetensors"))
        for idx in target_layers:
            Q = null_bases[idx]
            lora = NullSpaceLoRALinear(originals[idx], Q, r=LORA_RANK, scale=LORA_SCALE)
            lora.lora_a = adapter_weights[f"layer_{idx}_lora_a"]
            lora.lora_b = adapter_weights[f"layer_{idx}_lora_b"]
            layers[idx].self_attn.v_proj = lora
        mx.eval(model.parameters())

        # Compute NTP loss on each test text
        for i, tokens in enumerate(test_tokens_all):
            ntp_loss = compute_ntp_loss(model, [tokens])
            all_qualities.append(-ntp_loss)  # negative loss = higher quality

        del adapter_weights
        # Restore originals
        for idx in target_layers:
            layers[idx].self_attn.v_proj = originals[idx]

    # Load projection magnitudes from routing eval results
    routing_results = json.loads((EXPERIMENT_DIR / "routing_eval.json").read_text())
    for adapter_idx, adapter_domain in enumerate(DOMAINS):
        for item in routing_results["per_item"]:
            all_magnitudes.append(item["projections"][adapter_domain])

    # Compute Spearman correlation
    r = spearman_r(all_magnitudes, all_qualities)
    log(f"Spearman r (projection vs quality): {r:.4f}")
    log(f"  N pairs: {len(all_magnitudes)}")

    cleanup(model, tokenizer)
    for k in list(null_bases.keys()):
        del null_bases[k]
    del null_bases, bases_raw, originals
    gc.collect()
    mx.clear_cache()
    log_memory("post-phase4")

    return {
        "spearman_r": round(r, 4),
        "n_pairs": len(all_magnitudes),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Routing latency benchmark
# ══════════════════════════════════════════════════════════════════════════════

def phase_latency(target_layers):
    """Benchmark routing latency (projection computation only)."""
    log("\n=== Phase 5: Latency benchmark ===")

    # Load null bases and adapter A-matrices
    bases_raw = mx.load(str(EXPERIMENT_DIR / "null_bases.safetensors"))
    null_bases = {int(k.split("_")[1]): v for k, v in bases_raw.items()}

    adapter_A = {}
    for domain in DOMAINS:
        weights = mx.load(str(EXPERIMENT_DIR / f"{domain}_adapter.safetensors"))
        adapter_A[domain] = {
            int(k.split("_")[1]): v
            for k, v in weights.items() if "lora_a" in k
        }
        del weights

    # Simulate input: (1, seq_len, d_in)
    d_in = null_bases[target_layers[0]].shape[0]
    x = mx.random.normal((1, 32, d_in))
    mx.eval(x)

    # Warmup
    for _ in range(5):
        for domain in DOMAINS:
            for idx in target_layers:
                Q = null_bases[idx]
                A_null = adapter_A[domain][idx]
                x_null = x @ Q
                proj = x_null @ A_null
                score = mx.mean(mx.sum(proj * proj, axis=-1))
        mx.eval(score)

    # Benchmark
    n_runs = 100
    t0 = time.perf_counter()
    for _ in range(n_runs):
        scores = {}
        for domain in DOMAINS:
            total = mx.array(0.0)
            for idx in target_layers:
                Q = null_bases[idx]
                A_null = adapter_A[domain][idx]
                x_null = x @ Q
                proj = x_null @ A_null
                total = total + mx.mean(mx.sum(proj * proj, axis=-1))
            scores[domain] = total
        # Force eval of all scores
        mx.eval(*scores.values())
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) / n_runs * 1000
    log(f"Routing latency: {latency_ms:.3f}ms (avg over {n_runs} runs, seq_len=32, 5 adapters, {len(target_layers)} layers)")

    del bases_raw, null_bases, adapter_A, x
    gc.collect()
    mx.clear_cache()

    return {"latency_ms": round(latency_ms, 4), "n_runs": n_runs, "seq_len": 32}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log_memory("start")
    log(f"Smoke test: {IS_SMOKE}")
    log(f"Config: r={LORA_RANK}, scale={LORA_SCALE}, lr={LR}, iters={TRAIN_ITERS}")
    log(f"Domains: {DOMAINS}")

    # Phase 1: Null bases
    bases_result = phase_null_bases(MODEL_ID)
    target_layers = bases_result["target_layers"]
    log(f"Target layers: {target_layers}, null dims: {bases_result['null_dims']}")

    # Phase 2: Train 5 domain adapters
    train_results = phase_train_all(MODEL_ID, target_layers)

    # Phase 3: Routing evaluation
    routing = phase_routing_eval(MODEL_ID, target_layers)

    # Save routing results for phase 4 to read
    (EXPERIMENT_DIR / "routing_eval.json").write_text(json.dumps(routing, indent=2))

    # Phase 4: Quality correlation
    quality = phase_quality_correlation(MODEL_ID, target_layers)

    # Phase 5: Latency
    latency = phase_latency(target_layers)

    # ══════════════════════════════════════════════════════════════════════════
    # KILL CRITERIA
    # ══════════════════════════════════════════════════════════════════════════

    # K1300: Routing accuracy >= 80%
    k1300_pass = routing["accuracy"] >= 0.80
    k1300 = {
        "pass": k1300_pass,
        "accuracy": routing["accuracy"],
        "correct": routing["correct"],
        "total": routing["total"],
        "threshold": 0.80,
    }

    # K1301: Spearman r >= 0.3
    k1301_pass = quality["spearman_r"] >= 0.30
    k1301 = {
        "pass": k1301_pass,
        "spearman_r": quality["spearman_r"],
        "n_pairs": quality["n_pairs"],
        "threshold": 0.30,
    }

    # K1302: Latency < 0.5ms
    k1302_pass = latency["latency_ms"] < 0.5
    k1302 = {
        "pass": k1302_pass,
        "latency_ms": latency["latency_ms"],
        "threshold_ms": 0.5,
    }

    all_pass = k1300_pass and k1301_pass and k1302_pass

    results = {
        "is_smoke": IS_SMOKE,
        "model": MODEL_ID,
        "config": {
            "rank": LORA_RANK,
            "scale": LORA_SCALE,
            "lr": LR,
            "iters": TRAIN_ITERS,
            "domains": DOMAINS,
        },
        "bases": bases_result,
        "training": train_results,
        "routing": routing,
        "quality_correlation": quality,
        "latency": latency,
        "k1300": k1300,
        "k1301": k1301,
        "k1302": k1302,
        "all_pass": all_pass,
        "total_time_min": round((time.time() - t0) / 60, 2),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    log(f"\n{'='*60}")
    log("RESULTS SUMMARY")
    log(f"{'='*60}")
    log(f"K1300 (routing >= 80%): {'PASS' if k1300_pass else 'FAIL'} — {routing['accuracy']:.1%} ({routing['correct']}/{routing['total']})")
    log(f"K1301 (Spearman >= 0.3): {'PASS' if k1301_pass else 'FAIL'} — r={quality['spearman_r']:.4f}")
    log(f"K1302 (latency < 0.5ms): {'PASS' if k1302_pass else 'FAIL'} — {latency['latency_ms']:.3f}ms")
    log(f"ALL PASS: {all_pass}")
    log(f"Total time: {results['total_time_min']:.1f} min")


if __name__ == "__main__":
    main()
