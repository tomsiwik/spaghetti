#!/usr/bin/env python3
"""P7.B1: Weighted Multi-Adapter Composition via Null-Space Projections

Reuses 5 null-space adapters from exp_p7_null_projection_routing (Finding #495 killed
routing, but adapters trained fine). Tests whether TF-IDF-weighted composition of
null-space adapters outperforms exclusive (argmax) routing on mixed-domain queries.

Kill criteria:
  K1303: Weighted > exclusive by >= 3pp on mixed-domain queries
  K1304: Weighted does not degrade single-domain (< 2pp vs exclusive)
  K1305: Cross-domain queries show measurable benefit from multi-adapter

Prior: Finding #494 (null-space quality), #495 (routing killed), #354 (TF-IDF 95%)
Platform: Apple M5 Pro 48GB, MLX only.
"""

import gc
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# Memory safety (CODING_GUIDELINES)
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
SOURCE_DIR = EXPERIMENT_DIR.parent / "exp_p7_null_projection_routing"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
LORA_SCALE = 20.0
SEED = 42

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
MAX_SEQ_LEN = 256

DOMAINS = ["medical", "code", "math", "legal", "finance"]


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
# DOMAIN DATA — reused from exp_p7_null_projection_routing
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
        ],
    },
}

# Mixed-domain test queries — each spans exactly two domains
MIXED_DOMAIN_TEXTS = {
    "medical+legal": [
        "The medical malpractice case requires expert testimony on the standard of care for acute myocardial infarction treatment. The defendant physician's failure to order troponin levels within 30 minutes constitutes a breach of duty under the reasonable physician standard. Damages include the patient's subsequent cardiac arrest and extended ICU stay.",
        "Clinical trial informed consent must satisfy both FDA regulatory requirements and common law standards of voluntary agreement. The institutional review board ensures that risks are minimized and reasonable in relation to anticipated benefits. Failure to disclose material risks may constitute negligence per se.",
    ],
    "code+finance": [
        "Implement a Monte Carlo simulation for Black-Scholes option pricing using parallel random number generation. The function accepts strike price, current price, risk-free rate, volatility, and time to expiration, returning the expected option value averaged across 10000 simulated geometric Brownian motion paths.",
        "Design a REST API for portfolio rebalancing that computes optimal asset weights using the mean-variance optimization algorithm. The endpoint accepts current holdings, expected returns vector, and covariance matrix, then returns target allocations that maximize the Sharpe ratio subject to constraints.",
    ],
    "math+medical": [
        "Bayesian posterior probability for a diagnostic test with sensitivity 0.95 and specificity 0.90 given disease prevalence 0.01. Using Bayes theorem: P(disease|positive) = P(positive|disease)*P(disease) / P(positive). The positive predictive value is surprisingly low at 8.7 percent, illustrating the base rate fallacy in clinical screening.",
        "Survival analysis using the Kaplan-Meier estimator computes the probability of surviving beyond time t as the product of conditional survival probabilities at each event time. The log-rank test compares survival curves between treatment and control groups with the null hypothesis of no difference.",
    ],
    "legal+finance": [
        "Securities fraud under Rule 10b-5 requires proof of material misstatement, scienter, connection to purchase or sale, reliance, economic loss, and loss causation. The plaintiff must demonstrate that the defendant's earnings per share misstatement affected the stock price through event study methodology analyzing abnormal returns.",
        "The Dodd-Frank Act requires systemically important financial institutions to maintain higher capital reserves and submit to annual stress testing. Living wills must demonstrate that the institution can be resolved under the bankruptcy code without requiring taxpayer-funded bailouts or causing systemic risk.",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# TF-IDF ROUTER (pure Python, no external deps)
# ══════════════════════════════════════════════════════════════════════════════

def tokenize_words(text):
    return re.findall(r"\b\w+\b", text.lower())


class TFIDFRouter:
    """Minimal TF-IDF router for domain classification."""

    def __init__(self, domain_data):
        # Collect all training texts per domain
        all_docs = []
        doc_domains = []
        for domain in DOMAINS:
            for text in domain_data[domain]["train"]:
                all_docs.append(tokenize_words(text))
                doc_domains.append(domain)

        # Build IDF
        n_docs = len(all_docs)
        df = Counter()
        for doc in all_docs:
            for term in set(doc):
                df[term] += 1
        self.idf = {t: math.log(n_docs / c) for t, c in df.items()}
        self.vocab = sorted(self.idf.keys())
        self.term_idx = {t: i for i, t in enumerate(self.vocab)}

        # Domain centroids
        self.centroids = {}
        for domain in DOMAINS:
            domain_docs = [
                all_docs[i] for i, d in enumerate(doc_domains) if d == domain
            ]
            vecs = [self._tfidf_vec(doc) for doc in domain_docs]
            centroid = [
                sum(v[j] for v in vecs) / len(vecs) for j in range(len(self.vocab))
            ]
            self.centroids[domain] = centroid

    def _tfidf_vec(self, tokens):
        tf = Counter(tokens)
        n = max(len(tokens), 1)
        vec = [0.0] * len(self.vocab)
        for term, count in tf.items():
            if term in self.term_idx:
                vec[self.term_idx[term]] = (count / n) * self.idf.get(term, 0)
        return vec

    @staticmethod
    def _cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def route(self, text, temperature=1.0):
        """Return (weights_dict, similarities_dict) for a query text."""
        qvec = self._tfidf_vec(tokenize_words(text))
        sims = {d: self._cosine(qvec, c) for d, c in self.centroids.items()}

        # Softmax with temperature
        max_s = max(sims.values())
        exps = {d: math.exp((s - max_s) / temperature) for d, s in sims.items()}
        total = sum(exps.values())
        weights = {d: v / total for d, v in exps.items()}
        return weights, sims


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHTED ADAPTER MODULE
# ══════════════════════════════════════════════════════════════════════════════

class MergedDeltaLinear(nn.Module):
    """Base linear + precomputed adapter delta. Delta is set externally per query."""

    def __init__(self, base_linear, scale=20.0):
        super().__init__()
        self.linear = base_linear
        self.scale = scale
        self._delta = None

    def set_delta(self, D):
        """Set the merged delta matrix. None = no adapter."""
        self._delta = D

    def __call__(self, x):
        y = self.linear(x)
        if self._delta is not None:
            z = x @ self._delta
            y = y + (self.scale * z).astype(x.dtype)
        return y


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def dequantize_weight(linear):
    if isinstance(linear, nn.QuantizedLinear):
        W = mx.dequantize(
            linear.weight, linear.scales, linear.biases,
            linear.group_size, linear.bits,
        )
    else:
        W = linear.weight
    return W.astype(mx.float32)


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


def tokenize_text(tokenizer, text, max_len=MAX_SEQ_LEN):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    return mx.array(tokens)


def compute_ntp_loss(model, tokens):
    """Compute next-token prediction loss for a single token sequence."""
    x = tokens[None, :-1]
    targets = tokens[1:]
    logits = model(x)
    mx.eval(logits)
    loss = nn.losses.cross_entropy(logits.squeeze(0), targets, reduction="mean")
    mx.eval(loss)
    val = loss.item()
    del logits, loss, x
    return val


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Precompute delta matrices from saved adapters
# ══════════════════════════════════════════════════════════════════════════════

def phase_precompute_deltas():
    """Load null bases + adapters, compute full delta matrices D_i^l = Q^l @ A_i^l @ B_i^l."""
    log("=== Phase 1: Precompute delta matrices ===")
    t0 = time.time()

    # Load null bases
    bases_raw = mx.load(str(SOURCE_DIR / "null_bases.safetensors"))
    null_bases = {int(k.split("_")[1]): v for k, v in bases_raw.items()}
    target_layers = sorted(null_bases.keys())
    log(f"Target layers: {target_layers}")

    # Compute delta matrices for each adapter
    # deltas[domain][layer_idx] = Q @ A @ B, shape (d_in, d_out)
    deltas = {}
    for domain in DOMAINS:
        adapter_path = SOURCE_DIR / f"{domain}_adapter.safetensors"
        adapter = mx.load(str(adapter_path))
        deltas[domain] = {}

        for idx in target_layers:
            Q = null_bases[idx]  # (d_in, d_null)
            A = adapter[f"layer_{idx}_lora_a"]  # (d_null, r)
            B = adapter[f"layer_{idx}_lora_b"]  # (r, d_out)
            D = Q @ A @ B  # (d_in, d_out)
            mx.eval(D)
            deltas[domain][idx] = D

        del adapter
        log(f"  {domain}: computed {len(target_layers)} deltas")

    # Report sizes
    sample_key = target_layers[0]
    sample_shape = deltas[DOMAINS[0]][sample_key].shape
    total_bytes = sum(
        deltas[d][l].size * 4
        for d in DOMAINS
        for l in target_layers
    )
    log(f"Delta shape sample: {sample_shape}, total memory: {total_bytes/1e6:.1f}MB")

    elapsed = time.time() - t0
    log(f"Phase 1 done in {elapsed:.1f}s")

    del bases_raw, null_bases
    gc.collect()
    mx.clear_cache()
    log_memory("post-phase1")
    return deltas, target_layers


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Evaluate all conditions
# ══════════════════════════════════════════════════════════════════════════════

def phase_evaluate(deltas, target_layers):
    """Evaluate no-adapter, exclusive, and weighted routing on single + mixed domain."""
    log("\n=== Phase 2: Evaluate routing strategies ===")
    t0 = time.time()

    model, tokenizer, layers = load_model_and_layers(MODEL_ID)
    log_memory("post-load")

    # Wrap v_proj with MergedDeltaLinear
    wrapped_layers = {}
    for idx in target_layers:
        wrapper = MergedDeltaLinear(layers[idx].self_attn.v_proj, scale=LORA_SCALE)
        layers[idx].self_attn.v_proj = wrapper
        wrapped_layers[idx] = wrapper
    log(f"Wrapped {len(wrapped_layers)} v_proj layers")

    # Build TF-IDF router
    router = TFIDFRouter(DOMAIN_DATA)
    log("TF-IDF router built")

    def set_no_adapter():
        for idx in target_layers:
            wrapped_layers[idx].set_delta(None)

    def set_exclusive_adapter(domain):
        for idx in target_layers:
            wrapped_layers[idx].set_delta(deltas[domain][idx])

    def set_weighted_adapter(weights):
        for idx in target_layers:
            D = None
            for d in DOMAINS:
                w = weights[d]
                if w < 1e-6:
                    continue
                contrib = w * deltas[d][idx]
                D = contrib if D is None else D + contrib
            mx.eval(D)
            wrapped_layers[idx].set_delta(D)

    # --- Single-domain evaluation ---
    log("\n--- Single-domain evaluation ---")
    single_results = {}

    n_test = 1 if IS_SMOKE else 2  # texts per domain
    for domain in DOMAINS:
        domain_results = []
        test_texts = DOMAIN_DATA[domain]["test"][:n_test]

        for i, text in enumerate(test_texts):
            tokens = tokenize_text(tokenizer, text)
            if tokens.shape[0] < 5:
                continue

            # Route
            weights, sims = router.route(text)
            argmax_domain = max(weights, key=weights.get)

            # No adapter
            set_no_adapter()
            loss_none = compute_ntp_loss(model, tokens)

            # Exclusive (TF-IDF argmax)
            set_exclusive_adapter(argmax_domain)
            loss_exclusive = compute_ntp_loss(model, tokens)

            # Weighted
            set_weighted_adapter(weights)
            loss_weighted = compute_ntp_loss(model, tokens)

            # Oracle: try each adapter, pick best
            best_oracle_loss = float("inf")
            best_oracle_domain = None
            for d in DOMAINS:
                set_exclusive_adapter(d)
                l = compute_ntp_loss(model, tokens)
                if l < best_oracle_loss:
                    best_oracle_loss = l
                    best_oracle_domain = d

            result = {
                "text_idx": i,
                "true_domain": domain,
                "routed_domain": argmax_domain,
                "routing_correct": argmax_domain == domain,
                "weights": {d: round(w, 4) for d, w in weights.items()},
                "similarities": {d: round(s, 4) for d, s in sims.items()},
                "loss_none": round(loss_none, 4),
                "loss_exclusive": round(loss_exclusive, 4),
                "loss_weighted": round(loss_weighted, 4),
                "loss_oracle": round(best_oracle_loss, 4),
                "oracle_domain": best_oracle_domain,
            }
            domain_results.append(result)
            log(f"  {domain}[{i}]: none={loss_none:.4f} excl={loss_exclusive:.4f} "
                f"wt={loss_weighted:.4f} oracle={best_oracle_loss:.4f} "
                f"route={argmax_domain} wt_top={max(weights.values()):.3f}")

        single_results[domain] = domain_results

    # --- Mixed-domain evaluation ---
    log("\n--- Mixed-domain evaluation ---")
    mixed_results = {}

    for mix_key, texts in MIXED_DOMAIN_TEXTS.items():
        mix_texts = texts[:1] if IS_SMOKE else texts
        mix_results = []

        for i, text in enumerate(mix_texts):
            tokens = tokenize_text(tokenizer, text)
            if tokens.shape[0] < 5:
                continue

            weights, sims = router.route(text)
            argmax_domain = max(weights, key=weights.get)

            # Weight entropy (higher = more mixed)
            entropy = -sum(
                w * math.log(w + 1e-10) for w in weights.values()
            ) / math.log(len(DOMAINS))

            set_no_adapter()
            loss_none = compute_ntp_loss(model, tokens)

            set_exclusive_adapter(argmax_domain)
            loss_exclusive = compute_ntp_loss(model, tokens)

            set_weighted_adapter(weights)
            loss_weighted = compute_ntp_loss(model, tokens)

            # Oracle
            best_oracle_loss = float("inf")
            best_oracle_domain = None
            for d in DOMAINS:
                set_exclusive_adapter(d)
                l = compute_ntp_loss(model, tokens)
                if l < best_oracle_loss:
                    best_oracle_loss = l
                    best_oracle_domain = d

            result = {
                "text_idx": i,
                "mix_key": mix_key,
                "domains": mix_key.split("+"),
                "routed_domain": argmax_domain,
                "weights": {d: round(w, 4) for d, w in weights.items()},
                "weight_entropy": round(entropy, 4),
                "loss_none": round(loss_none, 4),
                "loss_exclusive": round(loss_exclusive, 4),
                "loss_weighted": round(loss_weighted, 4),
                "loss_oracle": round(best_oracle_loss, 4),
                "oracle_domain": best_oracle_domain,
            }
            mix_results.append(result)
            log(f"  {mix_key}[{i}]: none={loss_none:.4f} excl={loss_exclusive:.4f} "
                f"wt={loss_weighted:.4f} oracle={best_oracle_loss:.4f} "
                f"route={argmax_domain} entropy={entropy:.3f}")

        mixed_results[mix_key] = mix_results

    elapsed = time.time() - t0
    log(f"\nPhase 2 done in {elapsed:.1f}s")

    # Clean up wrapped layers
    set_no_adapter()
    cleanup(model, tokenizer)
    log_memory("post-phase2")
    return single_results, mixed_results


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Verify orthogonality under weighted composition
# ══════════════════════════════════════════════════════════════════════════════

def phase_verify_orthogonality(deltas, target_layers):
    """Check W_v @ D = 0 for representative weight vectors."""
    log("\n=== Phase 3: Verify orthogonality ===")
    t0 = time.time()

    model, tokenizer, layers = load_model_and_layers(MODEL_ID)

    # Test weight vectors: uniform, peaked, and a realistic mixed one
    test_weights = [
        ("uniform", {d: 1.0 / len(DOMAINS) for d in DOMAINS}),
        ("peaked_medical", {"medical": 0.8, "code": 0.05, "math": 0.05, "legal": 0.05, "finance": 0.05}),
        ("mixed_med_legal", {"medical": 0.4, "code": 0.05, "math": 0.05, "legal": 0.4, "finance": 0.1}),
    ]

    max_violation = 0.0
    orth_results = []

    for name, weights in test_weights:
        layer_violations = {}
        for idx in target_layers:
            # Merge delta
            D = sum(weights[d] * deltas[d][idx] for d in DOMAINS)
            mx.eval(D)

            # Get W_v
            W_v = dequantize_weight(layers[idx].self_attn.v_proj)
            mx.eval(W_v)

            # Check W_v @ D should be 0
            product = W_v @ D  # (d_out, d_out)
            mx.eval(product)
            violation = mx.max(mx.abs(product)).item()
            layer_violations[str(idx)] = violation
            max_violation = max(max_violation, violation)

            del W_v, product, D

        orth_results.append({
            "name": name,
            "weights": {d: round(w, 4) for d, w in weights.items()},
            "layer_violations": {k: float(v) for k, v in layer_violations.items()},
            "max_violation": float(max(layer_violations.values())),
        })
        log(f"  {name}: max|W_v @ D| = {max(layer_violations.values()):.2e}")

    elapsed = time.time() - t0
    log(f"Phase 3 done in {elapsed:.1f}s")

    cleanup(model, tokenizer)
    log_memory("post-phase3")
    return {"max_violation": float(max_violation), "details": orth_results}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log_memory("start")
    log(f"Smoke test: {IS_SMOKE}")
    log(f"Model: {MODEL_ID}")
    log(f"Source adapters: {SOURCE_DIR}")

    # Check source adapters exist
    for domain in DOMAINS:
        p = SOURCE_DIR / f"{domain}_adapter.safetensors"
        assert p.exists(), f"Missing adapter: {p}"
    assert (SOURCE_DIR / "null_bases.safetensors").exists(), "Missing null bases"
    log("All source adapters verified")

    # Phase 1: Precompute
    deltas, target_layers = phase_precompute_deltas()

    # Phase 2: Evaluate
    single_results, mixed_results = phase_evaluate(deltas, target_layers)

    # Phase 3: Orthogonality
    orth = phase_verify_orthogonality(deltas, target_layers)

    # Free deltas
    for d in list(deltas.keys()):
        for l in list(deltas[d].keys()):
            del deltas[d][l]
        del deltas[d]
    del deltas
    gc.collect()
    mx.clear_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # AGGREGATE & KILL CRITERIA
    # ══════════════════════════════════════════════════════════════════════════

    # Aggregate single-domain
    single_losses = {"none": [], "exclusive": [], "weighted": [], "oracle": []}
    for domain, results in single_results.items():
        for r in results:
            single_losses["none"].append(r["loss_none"])
            single_losses["exclusive"].append(r["loss_exclusive"])
            single_losses["weighted"].append(r["loss_weighted"])
            single_losses["oracle"].append(r["loss_oracle"])

    single_avg = {k: sum(v) / max(len(v), 1) for k, v in single_losses.items()}

    # Aggregate mixed-domain
    mixed_losses = {"none": [], "exclusive": [], "weighted": [], "oracle": []}
    for mix_key, results in mixed_results.items():
        for r in results:
            mixed_losses["none"].append(r["loss_none"])
            mixed_losses["exclusive"].append(r["loss_exclusive"])
            mixed_losses["weighted"].append(r["loss_weighted"])
            mixed_losses["oracle"].append(r["loss_oracle"])

    mixed_avg = {k: sum(v) / max(len(v), 1) for k, v in mixed_losses.items()}

    # K1303: Weighted > exclusive by >= 3pp on mixed-domain
    # Improvement = (exclusive - weighted) / exclusive * 100
    if mixed_avg["exclusive"] > 0:
        mixed_improvement_pp = (
            (mixed_avg["exclusive"] - mixed_avg["weighted"]) / mixed_avg["exclusive"] * 100
        )
    else:
        mixed_improvement_pp = 0.0
    k1303_pass = mixed_improvement_pp >= 3.0
    k1303 = {
        "pass": k1303_pass,
        "mixed_improvement_pp": round(mixed_improvement_pp, 2),
        "mixed_avg_exclusive": round(mixed_avg["exclusive"], 4),
        "mixed_avg_weighted": round(mixed_avg["weighted"], 4),
        "threshold_pp": 3.0,
    }

    # K1304: Weighted does not degrade single-domain (< 2pp vs exclusive)
    if single_avg["exclusive"] > 0:
        single_degradation_pp = (
            (single_avg["weighted"] - single_avg["exclusive"]) / single_avg["exclusive"] * 100
        )
    else:
        single_degradation_pp = 0.0
    k1304_pass = single_degradation_pp < 2.0
    k1304 = {
        "pass": k1304_pass,
        "single_degradation_pp": round(single_degradation_pp, 2),
        "single_avg_exclusive": round(single_avg["exclusive"], 4),
        "single_avg_weighted": round(single_avg["weighted"], 4),
        "threshold_pp": 2.0,
    }

    # K1305: Cross-domain queries show measurable benefit from multi-adapter
    # weighted < no_adapter on mixed-domain (adapters help)
    if mixed_avg["none"] > 0:
        mixed_benefit_pp = (
            (mixed_avg["none"] - mixed_avg["weighted"]) / mixed_avg["none"] * 100
        )
    else:
        mixed_benefit_pp = 0.0
    k1305_pass = mixed_benefit_pp > 0
    k1305 = {
        "pass": k1305_pass,
        "mixed_benefit_pp": round(mixed_benefit_pp, 2),
        "mixed_avg_none": round(mixed_avg["none"], 4),
        "mixed_avg_weighted": round(mixed_avg["weighted"], 4),
    }

    all_pass = k1303_pass and k1304_pass and k1305_pass

    results = {
        "is_smoke": IS_SMOKE,
        "model": MODEL_ID,
        "source_dir": str(SOURCE_DIR),
        "target_layers": target_layers,
        "single_domain": single_results,
        "mixed_domain": mixed_results,
        "single_avg": {k: round(v, 4) for k, v in single_avg.items()},
        "mixed_avg": {k: round(v, 4) for k, v in mixed_avg.items()},
        "orthogonality": orth,
        "k1303": k1303,
        "k1304": k1304,
        "k1305": k1305,
        "all_pass": all_pass,
        "total_time_min": round((time.time() - t0) / 60, 2),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    log(f"\n{'='*60}")
    log("RESULTS SUMMARY")
    log(f"{'='*60}")
    log(f"\nSingle-domain averages:")
    log(f"  No adapter:  {single_avg['none']:.4f}")
    log(f"  Exclusive:   {single_avg['exclusive']:.4f}")
    log(f"  Weighted:    {single_avg['weighted']:.4f}")
    log(f"  Oracle:      {single_avg['oracle']:.4f}")
    log(f"\nMixed-domain averages:")
    log(f"  No adapter:  {mixed_avg['none']:.4f}")
    log(f"  Exclusive:   {mixed_avg['exclusive']:.4f}")
    log(f"  Weighted:    {mixed_avg['weighted']:.4f}")
    log(f"  Oracle:      {mixed_avg['oracle']:.4f}")
    log(f"\nOrthogonality: max|W_v @ D| = {orth['max_violation']:.2e}")
    log(f"\nK1303 (mixed weighted > excl by >=3pp): {'PASS' if k1303_pass else 'FAIL'} — {mixed_improvement_pp:.2f}pp")
    log(f"K1304 (single degradation < 2pp):        {'PASS' if k1304_pass else 'FAIL'} — {single_degradation_pp:.2f}pp")
    log(f"K1305 (mixed benefit from adapters):      {'PASS' if k1305_pass else 'FAIL'} — {mixed_benefit_pp:.2f}pp")
    log(f"ALL PASS: {all_pass}")
    log(f"Total time: {results['total_time_min']:.1f} min")


if __name__ == "__main__":
    main()
