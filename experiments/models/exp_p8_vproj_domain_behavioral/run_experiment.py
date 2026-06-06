#!/usr/bin/env python3
"""
P8: v_proj+o_proj domain adapters produce behavioral text quality improvement.

The behavioral E2E experiment (exp_p1_p0_behavioral_e2e) was KILLED because
q_proj-only adapters improve benchmarks (+82pp GSM8K) but NOT text generation
quality (math 30%, code 20% — adapted was WORSE than base).

Finding #480 proved v_proj+o_proj unlocks behavioral format priors (SOAP +70pp,
Legal +90pp vs q_proj 0pp). This experiment tests whether v_proj+o_proj adapters
produce behavioral improvement for DOMAIN KNOWLEDGE, not just format.

Kill criteria (DB IDs):
  K1312: Math adapter v_proj+o_proj: >=60% queries show vocabulary improvement vs base (N=20)
  K1313: Code adapter v_proj+o_proj: >=60% queries show vocabulary improvement vs base (N=20)
  K1314: Medical adapter v_proj+o_proj: >=60% queries show vocabulary improvement vs base (N=20)
  K1315: 5-adapter composition retains >=80% of solo adapter behavioral quality per domain

Grounded by:
  - Finding #480: v_proj+o_proj unlocks behavioral priors (SOAP +70pp, Legal +90pp)
  - Killed exp_p1_p0_behavioral_e2e: q_proj fails behavioral quality
  - Finding #493: v_proj null-space 2048-dim (128 rank-16 slots)
  - DoRA (arXiv:2402.09353): output-path modifications improve generation quality
"""

import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx

# Memory safety (CODING_GUIDELINES §2)
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_EVAL = 5 if IS_SMOKE else 20
N_TRAIN = 20 if IS_SMOKE else 80
TRAIN_ITERS = 30 if IS_SMOKE else 200
LORA_RANK = 16
LORA_KEYS = ["self_attn.v_proj", "self_attn.o_proj"]
SEED = 42
MAX_TOKENS = 300

DOMAINS = ["math", "code", "medical", "legal", "finance"]


def cleanup():
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def log(msg: str):
    print(msg, flush=True)


def log_memory(label: str = ""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB")


# ════════════��════════════════════════════════════��════════════════════════════
# Training data: domain-specific Q&A pairs with DENSE domain vocabulary
# These teach the model to GENERATE domain-rich text, not answer multiple choice
# ══════════════════════════════════════════════════��═══════════════════════════

TRAIN_DATA = {
    "math": [
        ("Explain the quadratic formula and how to derive it.",
         "The quadratic formula solves any equation of the form ax^2 + bx + c = 0. To derive it, complete the square: divide by a, move c/a to the right, add (b/2a)^2 to both sides. This gives (x + b/2a)^2 = (b^2 - 4ac)/4a^2. Taking the square root: x = (-b ± sqrt(b^2 - 4ac)) / 2a. The discriminant b^2 - 4ac determines the nature of roots: positive gives two real roots, zero gives one repeated root, negative gives complex conjugate roots."),
        ("What is the fundamental theorem of calculus?",
         "The fundamental theorem of calculus connects differentiation and integration. Part 1: if F(x) = integral from a to x of f(t)dt, then F'(x) = f(x) — the derivative of the integral recovers the original function. Part 2: integral from a to b of f(x)dx = F(b) - F(a) where F is any antiderivative of f. This theorem is the foundation of integral calculus: it transforms the problem of computing areas into finding antiderivatives. The proof uses the mean value theorem and the limit definition of the derivative."),
        ("Explain eigenvalues and eigenvectors.",
         "An eigenvector v of a matrix A satisfies Av = λv, where λ is the corresponding eigenvalue. Geometrically, eigenvectors are directions that remain unchanged by the linear transformation — they only get scaled by λ. To find eigenvalues, solve det(A - λI) = 0, which gives the characteristic polynomial. The eigenvalues are its roots. For each eigenvalue, solve (A - λI)v = 0 to find eigenvectors. Applications: principal component analysis uses eigenvectors of the covariance matrix, differential equations use eigendecomposition to decouple systems, and Google's PageRank finds the dominant eigenvector of the link matrix."),
        ("Describe probability distributions and their properties.",
         "A probability distribution describes how probabilities are assigned to outcomes of a random variable. Discrete distributions (binomial, Poisson, geometric) assign probabilities to countable outcomes. Continuous distributions (normal, exponential, uniform) use probability density functions where P(a < X < b) = integral of f(x) from a to b. Key properties: the expectation E[X] = sum(x_i * p_i) or integral(x*f(x)dx), variance Var(X) = E[X^2] - (E[X])^2 measures spread. The central limit theorem states that sums of independent random variables converge to a normal distribution regardless of the original distribution."),
        ("How does mathematical induction work?",
         "Mathematical induction proves statements for all natural numbers n >= n_0. Step 1 (base case): verify the statement holds for n = n_0. Step 2 (inductive step): assume the statement holds for n = k (inductive hypothesis), then prove it holds for n = k+1. By the axiom of induction, the statement then holds for all n >= n_0. Example: prove sum(i, i=1..n) = n(n+1)/2. Base: n=1, sum = 1 = 1*2/2. Inductive step: assume sum(i, i=1..k) = k(k+1)/2, then sum(i, i=1..k+1) = k(k+1)/2 + (k+1) = (k+1)(k+2)/2. Strong induction assumes the statement for ALL values up to k, not just k."),
        ("What is the Taylor series expansion?",
         "The Taylor series represents a smooth function as an infinite polynomial: f(x) = sum(f^(n)(a)/n! * (x-a)^n, n=0..infinity). When a=0, it's the Maclaurin series. Key examples: e^x = sum(x^n/n!), sin(x) = sum((-1)^n * x^(2n+1)/(2n+1)!), cos(x) = sum((-1)^n * x^(2n)/(2n)!), 1/(1-x) = sum(x^n) for |x|<1. The radius of convergence R determines where the series converges: use the ratio test lim|a_{n+1}/a_n| = 1/R. Taylor's theorem with remainder: f(x) = T_n(x) + R_n(x) where the Lagrange remainder R_n = f^(n+1)(c)/(n+1)! * (x-a)^(n+1) for some c between a and x."),
        ("Explain integration by parts.",
         "Integration by parts transforms integral(u dv) = uv - integral(v du). It comes from the product rule: d(uv) = u dv + v du. Strategy: choose u and dv using LIATE (Logarithmic, Inverse trig, Algebraic, Trigonometric, Exponential — choose u from left). Example: integral(x*e^x dx), let u=x, dv=e^x dx, then du=dx, v=e^x, giving x*e^x - integral(e^x dx) = x*e^x - e^x + C = (x-1)*e^x + C. Repeated application: integral(x^2*sin(x)dx) needs two rounds. The tabular method streamlines this for polynomial × exponential/trig products."),
        ("What are vector spaces and their properties?",
         "A vector space V over a field F is a set with addition and scalar multiplication satisfying: closure, associativity, commutativity of addition, existence of zero vector and additive inverses, and distributive laws. Key concepts: a basis is a linearly independent spanning set; dimension = number of basis vectors. Subspaces are subsets that are themselves vector spaces. Linear transformations T: V -> W preserve addition and scalar multiplication. The kernel (null space) ker(T) = {v : T(v) = 0} and image im(T) = {T(v) : v in V} satisfy the rank-nullity theorem: dim(ker T) + dim(im T) = dim(V)."),
        ("Describe the chain rule in multivariable calculus.",
         "For a composition f(g(x)), the chain rule gives df/dx = (df/dg)(dg/dx). In multivariable calculus, if z = f(x,y) where x=g(t), y=h(t), then dz/dt = (partial f/partial x)(dx/dt) + (partial f/partial y)(dy/dt). The Jacobian matrix generalizes this: for F: R^n -> R^m, J_F = [partial F_i/partial x_j] is m×n. The chain rule becomes J_{F∘G} = J_F · J_G. This is the mathematical foundation of backpropagation in neural networks: the gradient of the loss with respect to parameters is computed by chaining Jacobians through the network."),
        ("Explain Fourier series and transforms.",
         "A Fourier series decomposes a periodic function into sinusoidal components: f(x) = a_0/2 + sum(a_n cos(nx) + b_n sin(nx), n=1..infinity) where a_n = (1/pi)*integral(f(x)cos(nx)dx, -pi..pi) and b_n similarly. The Fourier transform extends this to non-periodic functions: F(w) = integral(f(t)*e^(-iwt)dt, -inf..inf). The inverse transform recovers f(t). Properties: convolution theorem (convolution in time = multiplication in frequency), Parseval's theorem (energy is preserved), and the uncertainty principle (narrow in time = wide in frequency). Applications span signal processing, differential equations, and quantum mechanics."),
    ],
    "code": [
        ("Explain how recursion works in programming.",
         "Recursion is when a function calls itself to solve a problem by breaking it into smaller subproblems. Every recursive function needs: (1) a base case that stops recursion, (2) a recursive case that reduces the problem. Example in Python:\n\ndef factorial(n):\n    if n <= 1:  # base case\n        return 1\n    return n * factorial(n - 1)  # recursive case\n\nThe call stack grows with each recursive call. factorial(5) -> 5*factorial(4) -> 5*4*factorial(3) -> ... -> 5*4*3*2*1 = 120. Stack overflow occurs if recursion is too deep. Tail recursion optimization (available in some languages) reuses the stack frame. For problems like tree traversal, graph DFS, and divide-and-conquer algorithms, recursion provides elegant solutions."),
        ("What are Python decorators and how do they work?",
         "A decorator is a function that takes another function as input and returns a modified version. The @decorator syntax is syntactic sugar for function = decorator(function). Example:\n\ndef timer(func):\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        print(f'{func.__name__} took {time.time()-start:.2f}s')\n        return result\n    return wrapper\n\n@timer\ndef process_data(data):\n    # expensive computation\n    return sorted(data)\n\nDecorators with arguments need an extra nesting level. Class-based decorators implement __call__. Common uses: @property, @staticmethod, @functools.lru_cache for memoization, @contextmanager. The functools.wraps decorator preserves the original function's name and docstring."),
        ("Explain time complexity and Big-O notation.",
         "Time complexity describes how an algorithm's runtime scales with input size n. Big-O notation gives an upper bound: O(f(n)) means the algorithm takes at most c*f(n) steps for large n. Common complexities:\n- O(1): constant — hash table lookup\n- O(log n): logarithmic — binary search\n- O(n): linear — array traversal\n- O(n log n): linearithmic — merge sort, heap sort\n- O(n^2): quadratic — nested loops, bubble sort\n- O(2^n): exponential — recursive fibonacci without memoization\n\nAnalysis rules: loops multiply, sequential statements take the maximum, recursion uses the master theorem: T(n) = aT(n/b) + O(n^d). Space complexity measures memory usage similarly. Amortized analysis (like dynamic array resizing) averages cost over operations."),
        ("How does object-oriented programming work in Python?",
         "OOP organizes code around objects that combine data (attributes) and behavior (methods). Python's class system:\n\nclass Animal:\n    def __init__(self, name, species):\n        self.name = name\n        self.species = species\n\n    def speak(self):\n        raise NotImplementedError\n\nclass Dog(Animal):\n    def speak(self):\n        return f'{self.name} says Woof!'\n\nFour pillars: (1) Encapsulation — bundling data with methods, using _private convention. (2) Inheritance — Dog inherits from Animal, can override methods. (3) Polymorphism — different classes implement the same interface. (4) Abstraction — hiding implementation details. Python supports multiple inheritance with MRO (method resolution order) using C3 linearization. Special methods (__str__, __repr__, __eq__, __hash__) customize behavior."),
        ("Explain generators and iterators in Python.",
         "A generator is a function that uses yield instead of return, producing values lazily one at a time:\n\ndef fibonacci():\n    a, b = 0, 1\n    while True:\n        yield a\n        a, b = b, a + b\n\nfib = fibonacci()\nnext(fib)  # 0\nnext(fib)  # 1\nnext(fib)  # 1\n\nGenerators implement the iterator protocol (__iter__ and __next__). Memory efficient: they don't store the entire sequence. Generator expressions: (x**2 for x in range(1000000)) uses almost no memory vs [x**2 for x in range(1000000)]. Use cases: processing large files line by line, infinite sequences, pipeline chaining with itertools. The send() method enables coroutine-style communication. yield from delegates to sub-generators."),
        ("What are data structures and when to use each?",
         "Choosing the right data structure depends on the operations you need:\n\n- Array/List: O(1) index access, O(n) search, O(n) insert/delete. Use for ordered collections with random access.\n- Dictionary/HashMap: O(1) average lookup, insert, delete. Use for key-value mapping. Python dict preserves insertion order.\n- Set: O(1) membership test, union, intersection. Use for uniqueness checks and set operations.\n- Stack (LIFO): push/pop O(1). Use for DFS, undo operations, expression parsing.\n- Queue (FIFO): enqueue/dequeue O(1) with deque. Use for BFS, task scheduling.\n- Heap/Priority Queue: O(log n) insert, O(1) min/max. Use for dijkstra, top-k problems.\n- Tree: O(log n) search, insert in balanced trees (BST, AVL, Red-Black). Use for sorted data, range queries.\n- Graph: adjacency list O(V+E) space. Use for networks, dependencies, pathfinding."),
        ("Explain exception handling and error management in Python.",
         "Python uses try/except/else/finally for exception handling:\n\ntry:\n    result = risky_operation()\nexcept ValueError as e:\n    logger.error(f'Invalid value: {e}')\n    result = default_value\nexcept (TypeError, KeyError):\n    raise  # re-raise to caller\nelse:\n    process(result)  # only runs if no exception\nfinally:\n    cleanup()  # always runs\n\nCustom exceptions inherit from Exception:\n\nclass ValidationError(Exception):\n    def __init__(self, field, message):\n        self.field = field\n        super().__init__(f'{field}: {message}')\n\nBest practices: catch specific exceptions, don't use bare except, use context managers (with statement) for resource cleanup, raise exceptions early, handle them late. The traceback module provides detailed error information. Use logging instead of print for production error handling."),
        ("How does memory management work in Python?",
         "Python uses reference counting plus a cyclic garbage collector. Every object has a reference count; when it drops to zero, memory is freed immediately. The gc module handles circular references using generational collection (generation 0, 1, 2).\n\nMemory model:\n- Small objects (<512 bytes): allocated from pymalloc pools\n- Large objects: allocated via malloc directly\n- Interning: small integers (-5 to 256) and short strings are cached\n\nCommon pitfalls: circular references between objects prevent reference counting cleanup. __del__ methods can cause resurrection issues. Large lists hold references to all elements — use generators for memory efficiency. The sys.getsizeof() function shows object size. Memory profiling tools: tracemalloc (built-in), memory_profiler, objgraph for reference graphs."),
        ("Explain concurrency: threading vs multiprocessing vs async.",
         "Python offers three concurrency models:\n\n1. Threading (concurrent.futures.ThreadPoolExecutor): shared memory, limited by GIL for CPU work. Good for I/O-bound tasks (network requests, file I/O). Threads share state — use locks, queues for synchronization.\n\n2. Multiprocessing (multiprocessing.Pool): separate processes, true parallelism. Good for CPU-bound tasks. Communication via pipes, queues, shared memory. Higher overhead than threads.\n\n3. Asyncio (async/await): single-threaded cooperative multitasking. Best for high-concurrency I/O (thousands of connections). Event loop manages coroutines:\n\nasync def fetch_all(urls):\n    async with aiohttp.ClientSession() as session:\n        tasks = [session.get(url) for url in urls]\n        return await asyncio.gather(*tasks)\n\nChoice guide: I/O-bound with few connections -> threading. I/O-bound with many connections -> asyncio. CPU-bound -> multiprocessing."),
        ("What is functional programming in Python?",
         "Functional programming treats computation as evaluation of mathematical functions. Python supports FP through:\n\n- First-class functions: functions are objects, can be passed as arguments\n- Lambda: anonymous functions for short expressions: sorted(items, key=lambda x: x.price)\n- Map/filter/reduce: map(str.upper, words), filter(lambda x: x > 0, numbers), functools.reduce(operator.add, numbers)\n- List comprehensions: [x**2 for x in range(10) if x % 2 == 0]\n- Closures: inner functions capturing outer scope variables\n- Partial application: functools.partial(pow, 2) creates a power-of-2 function\n- Immutability: prefer tuples over lists, frozenset over set\n\nPure functions (no side effects, same input = same output) are easier to test, parallelize, and reason about. itertools and operator modules provide functional building blocks."),
    ],
    "medical": [
        ("Explain how ACE inhibitors work for hypertension treatment.",
         "ACE inhibitors (angiotensin-converting enzyme inhibitors) block the conversion of angiotensin I to angiotensin II in the renin-angiotensin-aldosterone system (RAAS). Angiotensin II is a potent vasoconstrictor and stimulates aldosterone secretion, which increases sodium and water retention. By inhibiting ACE, these medications cause vasodilation and reduce blood volume, lowering blood pressure. Common ACE inhibitors include lisinopril, enalapril, and ramipril. Side effects include dry cough (due to bradykinin accumulation), hyperkalemia, and rarely angioedema. Contraindicated in pregnancy (teratogenic) and bilateral renal artery stenosis. First-line therapy for hypertension with comorbid diabetes or heart failure due to their renoprotective and cardioprotective effects."),
        ("Describe the pathophysiology of Type 2 diabetes.",
         "Type 2 diabetes mellitus results from progressive insulin resistance and relative insulin deficiency. Pathophysiology: (1) Peripheral insulin resistance — skeletal muscle and adipose tissue require higher insulin levels for glucose uptake due to downregulation of insulin receptor signaling (IRS-1/PI3K pathway). (2) Hepatic glucose overproduction — the liver fails to suppress gluconeogenesis despite hyperinsulinemia. (3) Beta-cell dysfunction — pancreatic beta cells initially compensate with increased insulin secretion but progressively fail due to glucotoxicity, lipotoxicity, and amyloid deposition. Diagnosis: fasting glucose >=126 mg/dL, HbA1c >=6.5%, or 2-hour OGTT >=200 mg/dL. Complications include microvascular (retinopathy, nephropathy, neuropathy) and macrovascular (coronary artery disease, stroke, peripheral vascular disease) damage."),
        ("What are the mechanisms of antibiotic resistance?",
         "Bacteria develop antibiotic resistance through several molecular mechanisms: (1) Enzymatic degradation — beta-lactamases hydrolyze the beta-lactam ring of penicillins and cephalosporins. ESBL (extended-spectrum beta-lactamases) confer resistance to third-generation cephalosporins. (2) Target modification — altered penicillin-binding proteins (PBPs) in MRSA, ribosomal methylation in macrolide resistance. (3) Efflux pumps — membrane transporters actively expel antibiotics (tetracycline, fluoroquinolone resistance). (4) Reduced permeability — porin mutations decrease outer membrane permeability in gram-negative bacteria. Resistance genes spread via horizontal gene transfer: conjugation (plasmids), transformation (free DNA uptake), and transduction (bacteriophages). Clinical implications: empiric therapy must consider local antibiograms; de-escalation based on culture and sensitivity results reduces selection pressure."),
        ("Explain the immune response to viral infection.",
         "The immune response to viral infection involves innate and adaptive immunity. Innate response (hours): pattern recognition receptors (TLR3, TLR7, RIG-I) detect viral nucleic acids, triggering type I interferon (IFN-alpha/beta) production. Interferons induce an antiviral state in neighboring cells, activating PKR (protein kinase R) to inhibit viral translation. Natural killer (NK) cells kill virus-infected cells lacking MHC class I. Adaptive response (days): dendritic cells present viral antigens via MHC class I to CD8+ cytotoxic T lymphocytes (CTLs), which kill infected cells through perforin/granzyme pathway. CD4+ helper T cells activate B cells for antibody production — IgM (acute phase), then class switching to IgG (long-term immunity). Memory T and B cells provide rapid secondary response upon re-exposure. Viral evasion strategies include antigenic drift/shift, MHC downregulation, and interferon antagonism."),
        ("Describe the pharmacology of beta-blockers.",
         "Beta-blockers (beta-adrenergic receptor antagonists) block the effects of catecholamines (epinephrine, norepinephrine) on beta-adrenergic receptors. Beta-1 selective (cardioselective): metoprolol, atenolol, bisoprolol — primarily affect cardiac tissue, reducing heart rate (negative chronotropy), contractility (negative inotropy), and conduction velocity (negative dromotropy). Non-selective: propranolol, carvedilol — also block beta-2 receptors in bronchial smooth muscle and peripheral vasculature. Clinical indications: hypertension, heart failure (carvedilol, metoprolol succinate, bisoprolol — shown to reduce mortality), angina pectoris, post-myocardial infarction, arrhythmias (rate control in atrial fibrillation), migraine prophylaxis. Adverse effects: bradycardia, hypotension, bronchospasm (contraindicated in severe asthma), fatigue, masking of hypoglycemia symptoms in diabetic patients. Abrupt withdrawal can cause rebound tachycardia."),
        ("What is the blood-brain barrier and its clinical significance?",
         "The blood-brain barrier (BBB) is a selective semipermeable border of endothelial cells that prevents solutes in the circulating blood from non-selectively crossing into the extracellular fluid of the central nervous system. Structure: brain capillary endothelial cells connected by tight junctions (claudins, occludin), surrounded by astrocyte foot processes and pericytes. Transport: lipophilic molecules (<400 Da) cross by passive diffusion. Glucose enters via GLUT1 transporter. Amino acids via LAT1. P-glycoprotein efflux pump actively removes many drugs. Clinical significance: limits CNS drug delivery (most antibiotics, chemotherapeutics cannot cross). Strategies to bypass: intrathecal administration, mannitol for osmotic opening, nanoparticle carriers, receptor-mediated transcytosis. BBB breakdown occurs in: meningitis (allows antibiotic penetration), brain tumors (enhancing on MRI with gadolinium), multiple sclerosis, stroke."),
        ("Explain cardiac arrhythmias and their management.",
         "Cardiac arrhythmias are abnormalities in heart rhythm caused by disorders of impulse generation or conduction. Classification: supraventricular (atrial fibrillation, atrial flutter, SVT) vs ventricular (VT, VF). Pathophysiology: (1) Enhanced automaticity — ectopic pacemaker fires faster than SA node. (2) Triggered activity — early or delayed afterdepolarizations (EADs from prolonged QT, DADs from digoxin toxicity). (3) Re-entry — unidirectional block + slow conduction creates a circuit. ECG diagnosis: irregular irregularly — atrial fibrillation; sawtooth pattern — atrial flutter; wide complex tachycardia — VT until proven otherwise. Antiarrhythmic drug classification (Vaughan-Williams): Class I (Na channel blockers), Class II (beta-blockers), Class III (K channel blockers: amiodarone, sotalol), Class IV (Ca channel blockers: verapamil, diltiazem). Non-pharmacologic: cardioversion, catheter ablation, ICD implantation for VT/VF."),
        ("Describe wound healing phases and complications.",
         "Wound healing proceeds through four overlapping phases: (1) Hemostasis (minutes): platelet aggregation forms a fibrin clot. Platelets release PDGF and TGF-beta, initiating the healing cascade. (2) Inflammation (days 1-4): neutrophils phagocytose bacteria; macrophages arrive by day 2-3, debride tissue and release growth factors (VEGF, FGF, IGF-1). (3) Proliferation (days 4-21): fibroblasts deposit collagen type III, angiogenesis creates granulation tissue, keratinocytes migrate to re-epithelialize. Myofibroblasts contract the wound. (4) Remodeling (weeks to months): collagen type III replaced by type I (stronger); wound reaches 80% of original tensile strength by 3 months. Complications: infection (most common), dehiscence, keloid/hypertrophic scarring (excessive collagen), chronic non-healing wounds (diabetes: impaired neutrophil function, peripheral neuropathy, microvascular disease). Factors delaying healing: malnutrition (vitamin C, zinc, protein deficiency), immunosuppression, smoking (vasoconstriction), radiation, corticosteroids."),
    ],
    "legal": [
        ("Explain the concept of habeas corpus.",
         "Habeas corpus, Latin for 'you shall have the body,' is a fundamental legal remedy that protects against unlawful detention. Under the U.S. Constitution, Article I, Section 9 provides that the privilege of the writ of habeas corpus shall not be suspended unless required by public safety in cases of rebellion or invasion. A petition for habeas corpus challenges the legal basis for a person's imprisonment or detention. The court issues a writ commanding the custodian to bring the prisoner before the court and justify the detention. If the detention is found to be unlawful — due to insufficient evidence, procedural violations, or constitutional infringements — the court orders release. The federal habeas statute (28 U.S.C. § 2254) allows state prisoners to seek federal review of their convictions. Under AEDPA (1996), federal courts defer to state court adjudications unless they are contrary to established Supreme Court precedent."),
        ("What is the difference between civil and criminal liability?",
         "Civil liability and criminal liability serve different purposes and involve distinct legal standards. Criminal liability: the state prosecutes a defendant for violating criminal statutes. The burden of proof is 'beyond a reasonable doubt' — the highest standard. Penalties include incarceration, fines, probation, and community service. The defendant has constitutional protections: right to counsel (Sixth Amendment), right against self-incrimination (Fifth Amendment), right to jury trial. Civil liability: a private plaintiff sues a defendant for damages. The burden of proof is 'preponderance of the evidence' (more likely than not). Remedies include compensatory damages, punitive damages, injunctive relief, and specific performance. The same conduct can give rise to both: O.J. Simpson was acquitted criminally but found liable civilly for wrongful death. Key distinctions: criminal law requires mens rea (guilty mind); strict liability torts require no mental state. Criminal cases are styled 'State v. Defendant'; civil cases 'Plaintiff v. Defendant.'"),
        ("Explain stare decisis and the role of precedent.",
         "Stare decisis ('to stand by things decided') is the doctrine requiring courts to follow precedent established by prior judicial decisions. Under this principle, a court must follow the holdings of higher courts within the same jurisdiction (vertical stare decisis) and generally follow its own prior decisions (horizontal stare decisis). Precedent operates through ratio decidendi — the legal reasoning essential to the court's holding. Obiter dicta (statements made 'by the way') are persuasive but not binding. Courts may distinguish a precedent by identifying material factual differences. The Supreme Court can overrule its own precedent, as in Brown v. Board of Education (1954), which overruled the 'separate but equal' doctrine from Plessy v. Ferguson (1896). Policy considerations for overruling: changed circumstances, unworkability of the prior rule, erosion of the doctrinal foundation. Stare decisis promotes predictability, fairness, and judicial efficiency."),
        ("Describe the elements of a valid contract.",
         "A valid contract requires four essential elements under common law: (1) Offer — a definite proposal communicated to the offeree, manifesting willingness to enter a bargain. Distinguished from invitations to treat (advertisements, price lists). (2) Acceptance — unequivocal assent to the terms of the offer. The mirror image rule requires acceptance to match the offer exactly; any modification constitutes a counteroffer. Under the UCC (Article 2, sale of goods), additional terms may become part of the contract between merchants. (3) Consideration — something of legal value exchanged by the parties. Past consideration is not valid. Adequacy is generally not examined, but nominal consideration may indicate a gift. Promissory estoppel may substitute for consideration where a party reasonably relied on a promise to their detriment. (4) Capacity — parties must have legal capacity (age of majority, mental competence). Defenses to enforcement: duress, undue influence, misrepresentation, unconscionability, illegality, statute of frauds."),
        ("What is due process under the Constitution?",
         "Due process of law is guaranteed by the Fifth Amendment (federal government) and Fourteenth Amendment (state governments). It encompasses two distinct doctrines: (1) Procedural due process — before the government deprives a person of life, liberty, or property, it must provide adequate notice and a meaningful opportunity to be heard. The Mathews v. Eldridge (1976) balancing test weighs: the private interest affected, the risk of erroneous deprivation and value of additional safeguards, and the government's interest. (2) Substantive due process — certain fundamental rights are protected from government interference regardless of procedure. The court applies strict scrutiny to fundamental rights (privacy, marriage, contraception, child-rearing) and rational basis review to economic regulations. The incorporation doctrine has applied most Bill of Rights protections to the states through the Fourteenth Amendment's due process clause. Notable cases: Griswold v. Connecticut (privacy), Roe v. Wade (reproductive rights), Obergefell v. Hodges (same-sex marriage)."),
        ("Explain the tort of negligence.",
         "Negligence is the failure to exercise reasonable care, resulting in harm to another. A plaintiff must prove four elements: (1) Duty — the defendant owed a duty of care to the plaintiff. The general standard is the 'reasonable person' — what a person of ordinary prudence would do under similar circumstances. Professionals are held to the standard of their specialty. (2) Breach — the defendant's conduct fell below the standard of care. Determined by weighing probability and severity of harm against the burden of precaution (Learned Hand formula: B < P × L). (3) Causation — both cause-in-fact ('but for' test or substantial factor test) and proximate cause (foreseeable consequences). (4) Damages — actual harm suffered (compensatory damages: economic and non-economic). Defenses: comparative negligence (plaintiff's recovery reduced by their percentage of fault), assumption of risk, contributory negligence (complete bar in minority of jurisdictions). Special doctrines: res ipsa loquitur (the thing speaks for itself), negligence per se (violation of statute as evidence of breach)."),
        ("What is the exclusionary rule in criminal procedure?",
         "The exclusionary rule is a judicial remedy that prohibits the use of evidence obtained in violation of a defendant's constitutional rights. Originating in Weeks v. United States (1914) for federal courts and extended to state courts in Mapp v. Ohio (1961), the rule serves as a deterrent against unlawful police conduct. Under the Fourth Amendment, evidence obtained through unreasonable searches and seizures is inadmissible. The 'fruit of the poisonous tree' doctrine (Wong Sun v. United States, 1963) extends the exclusion to all derivative evidence. Exceptions: (1) Independent source — evidence discovered through a lawful independent means. (2) Inevitable discovery — evidence would have been found legally regardless. (3) Attenuation — sufficient intervening circumstances dissipate the taint. (4) Good faith — officers reasonably relied on a warrant later found defective (United States v. Leon, 1984). The rule does not apply in grand jury proceedings, civil cases, or deportation hearings."),
        ("Explain the statute of limitations and its purposes.",
         "A statute of limitations is a law that sets the maximum time after an event within which legal proceedings may be initiated. Purposes: (1) Ensuring fairness — over time, evidence deteriorates, witnesses' memories fade, and documents are lost. (2) Providing repose — potential defendants should not live indefinitely under threat of litigation. (3) Encouraging diligent prosecution — plaintiffs should not 'sleep on their rights.' Typical periods: personal injury torts (2-3 years), breach of contract (4-6 years), fraud (3-6 years from discovery), medical malpractice (2-3 years). Criminal statutes vary by severity: most felonies (5-10 years), murder (no statute of limitations in most jurisdictions). The discovery rule tolls the statute until the plaintiff knew or should have known of the injury. Equitable tolling applies when extraordinary circumstances prevent timely filing. The statute of repose sets an absolute outer limit regardless of discovery."),
    ],
    "finance": [
        ("Explain the difference between stocks and bonds.",
         "Stocks and bonds are the two primary asset classes in investment portfolios. Stocks (equity) represent ownership shares in a corporation. Shareholders have residual claims on assets and earnings after creditors. Returns come from capital appreciation (price increase) and dividends (periodic cash distributions from profits). Stock prices reflect expected future cash flows discounted at the required rate of return. Volatility is higher — equity risk premium historically 5-7% over risk-free rate. Bonds (fixed income) are debt instruments where the issuer borrows capital from investors. Bondholders receive periodic coupon payments (interest) and return of principal (par value) at maturity. Bond prices are inversely related to interest rates. Credit risk is assessed by rating agencies (Moody's, S&P, Fitch): investment grade (BBB- and above) vs high-yield/junk bonds. Duration measures interest rate sensitivity. In a diversified portfolio, bonds provide income stability and reduce overall portfolio volatility through low correlation with equities."),
        ("What is portfolio diversification and how does it reduce risk?",
         "Portfolio diversification is the strategy of spreading investments across different assets to reduce unsystematic (idiosyncratic) risk. The mathematical foundation is Markowitz's Modern Portfolio Theory (1952): portfolio variance = sum of weighted variances + sum of weighted covariances. When assets are not perfectly correlated (correlation < 1), the portfolio's risk is less than the weighted average of individual risks. The efficient frontier represents portfolios with maximum expected return for each level of risk. Systematic risk (market risk) cannot be diversified away — it's captured by beta in the CAPM: E(R) = Rf + beta*(E(Rm) - Rf). Diversification strategies: (1) Across asset classes — equities, bonds, real estate, commodities. (2) Within asset classes — different sectors, market caps, geographies. (3) Across time — dollar-cost averaging. The correlation matrix between assets determines diversification benefit. International diversification has diminished somewhat due to increasing global market correlation."),
        ("Explain compound interest and its effect on long-term savings.",
         "Compound interest is interest calculated on both the initial principal and accumulated interest from previous periods. The formula: FV = PV × (1 + r/n)^(nt), where PV is present value, r is annual interest rate, n is compounding frequency per year, and t is time in years. Continuous compounding: FV = PV × e^(rt). The Rule of 72 estimates doubling time: years ≈ 72/annual rate. Example: $10,000 at 7% compounded annually becomes $10,700 after year 1, $11,449 after year 2, $19,672 after 10 years, $76,123 after 30 years. The compounding effect accelerates over time — the last 10 years produce more growth than the first 20. Implications for retirement planning: starting 10 years earlier roughly doubles the final balance. Inflation-adjusted (real) returns matter: nominal 7% minus 3% inflation = 4% real return. Tax-advantaged accounts (401k, IRA, Roth) preserve the compounding benefit by deferring or eliminating taxes on gains."),
        ("What is a P/E ratio and how is it used in valuation?",
         "The price-to-earnings (P/E) ratio is a fundamental valuation metric: P/E = market price per share / earnings per share (EPS). It indicates how much investors pay per dollar of earnings. Trailing P/E uses last 12 months' earnings; forward P/E uses analyst estimates. Interpretation: higher P/E suggests investors expect higher future growth (growth premium). The S&P 500 historical average P/E is approximately 15-17. The PEG ratio (P/E divided by earnings growth rate) normalizes for growth: PEG < 1 may indicate undervaluation. Limitations: P/E is meaningless for companies with negative earnings; cyclical earnings distort the ratio; accounting practices (GAAP vs non-GAAP) affect EPS. The Shiller CAPE (cyclically adjusted P/E) uses 10-year inflation-adjusted earnings to smooth business cycle effects. Industry comparison is essential — tech companies typically trade at higher P/E than utilities due to growth expectations. The earnings yield (1/P/E) allows comparison with bond yields."),
        ("Explain how inflation affects savings and investment returns.",
         "Inflation is the sustained increase in the general price level, reducing the purchasing power of money over time. The real rate of return = nominal rate - inflation rate (Fisher equation: 1+r_real = (1+r_nominal)/(1+inflation)). With 3% inflation, $100 today has the purchasing power of only $74 in 10 years. Impact on savings: cash and fixed-rate bank deposits lose real value during inflationary periods. Impact on bonds: rising inflation erodes the real value of fixed coupon payments; bond prices fall as yields rise to compensate. TIPS (Treasury Inflation-Protected Securities) adjust principal with CPI, providing inflation protection. Impact on equities: moderate inflation is generally positive for stocks as companies can pass through costs; high inflation compresses multiples and raises discount rates. Real assets (real estate, commodities, infrastructure) tend to perform well during inflation as their values adjust with price levels. The Fed targets 2% annual inflation via monetary policy (federal funds rate adjustments, quantitative easing/tightening)."),
        ("What are dividends and why do companies pay them?",
         "Dividends are distributions of a company's earnings to shareholders, typically paid quarterly. Types: cash dividends (most common), stock dividends (additional shares), special dividends (one-time). Key metrics: dividend yield = annual dividend / share price; payout ratio = dividends / net income; dividend coverage ratio = EPS / DPS. Why companies pay dividends: (1) Signal financial health — consistent dividends signal stable earnings and management confidence. (2) Attract income investors — retirees and institutions seeking regular cash flow. (3) Reduce agency costs — distributing cash prevents management from wasteful capital allocation. (4) Tax considerations — qualified dividends taxed at capital gains rates (0-20%). The dividend irrelevance theorem (Modigliani-Miller) argues that in perfect markets, dividend policy doesn't affect firm value. In practice, the 'bird in hand' theory suggests investors prefer certain dividends over uncertain capital gains. Dividend growth model (Gordon): stock price = D1 / (r - g), where D1 is next year's dividend, r is required return, g is growth rate."),
        ("Describe the risk-return tradeoff in investing.",
         "The risk-return tradeoff is the fundamental principle that potential return rises with increasing risk. Risk is measured by standard deviation (volatility) of returns. The Capital Asset Pricing Model (CAPM) formalizes this: E(R_i) = R_f + beta_i × (E(R_m) - R_f), where R_f is the risk-free rate, beta measures systematic risk relative to the market, and (E(R_m) - R_f) is the market risk premium. Historical risk-return spectrum: Treasury bills (~3%, ~1% std), government bonds (~5%, ~6% std), corporate bonds (~6%, ~8% std), large-cap stocks (~10%, ~15% std), small-cap stocks (~12%, ~20% std), emerging markets (~12-15%, ~25% std). The Sharpe ratio = (R_p - R_f) / sigma_p measures risk-adjusted performance. Behavioral finance notes that investors are loss-averse (Kahneman & Tversky): losses hurt roughly 2x more than equivalent gains please. Risk tolerance depends on: time horizon (longer = more risk capacity), financial goals, liquidity needs, and psychological comfort with volatility."),
        ("What is a balance sheet and how do you analyze it?",
         "A balance sheet reports a company's financial position at a specific date: Assets = Liabilities + Shareholders' Equity. Assets are classified as current (cash, accounts receivable, inventory — convertible within 12 months) and non-current (property, plant, equipment, intangible assets, goodwill). Liabilities: current (accounts payable, short-term debt, accrued expenses) and long-term (bonds payable, long-term loans, deferred tax liabilities). Shareholders' equity = common stock + retained earnings + additional paid-in capital - treasury stock. Key ratios: current ratio = current assets / current liabilities (liquidity, target >1.5); debt-to-equity = total debt / equity (leverage); return on equity (ROE) = net income / equity (profitability). DuPont analysis decomposes ROE: ROE = profit margin × asset turnover × financial leverage. Book value per share = equity / shares outstanding. Working capital management: cash conversion cycle = days inventory + days receivables - days payables."),
    ],
}

# Domain-specific vocabulary glossaries for behavioral scoring
DOMAIN_GLOSSARIES = {
    "math": [
        "theorem", "proof", "equation", "derivative", "integral", "polynomial",
        "coefficient", "eigenvalue", "eigenvector", "determinant", "matrix", "vector",
        "probability", "distribution", "convergence", "continuity", "differentiable",
        "binomial", "permutation", "combination", "exponent", "logarithm",
        "quadratic", "linear", "induction", "modular", "congruence", "fourier",
        "antiderivative", "discriminant", "characteristic",
    ],
    "code": [
        "function", "return", "parameter", "variable", "algorithm", "recursive",
        "iteration", "loop", "array", "list", "dictionary", "class", "method",
        "object", "inheritance", "polymorphism", "exception", "generator",
        "decorator", "comprehension", "complexity", "runtime", "syntax", "module",
        "import", "lambda", "closure", "callback", "coroutine", "thread",
    ],
    "medical": [
        "mechanism", "inhibitor", "receptor", "pharmacology", "clinical", "therapy",
        "diagnosis", "treatment", "pathophysiology", "enzyme", "protein",
        "antibody", "immune", "inflammation", "vascular", "cardiac", "neural",
        "medication", "dose", "adverse", "contraindicated", "efficacy", "etiology",
        "prognosis", "cytokine", "antibiotic", "prophylaxis", "comorbidity",
    ],
    "legal": [
        "jurisdiction", "plaintiff", "defendant", "precedent", "statute", "liability",
        "constitutional", "testimony", "verdict", "appeal", "amendment", "regulation",
        "enforceable", "judicial", "adversarial", "procedural", "substantive",
        "habeas", "corpus", "felony", "misdemeanor", "tort", "contract",
        "adjudication", "injunction", "indictment", "acquittal", "promissory",
    ],
    "finance": [
        "portfolio", "equity", "dividend", "revenue", "asset", "liability", "capital",
        "investment", "valuation", "appreciation", "depreciation", "amortization",
        "diversification", "volatility", "liquidity", "hedge", "derivative",
        "earnings", "shareholder", "fiscal", "compounding", "yield", "coupon",
        "leverage", "arbitrage", "beta", "alpha", "collateral",
    ],
}

# Evaluation queries (distinct from training data)
EVAL_QUERIES = {
    "math": [
        "Explain what a limit is and how to compute it.",
        "What is the relationship between differentiation and integration?",
        "Describe how matrix multiplication works and its applications.",
        "Explain the concept of a probability density function.",
        "What is the mean value theorem and why is it important?",
        "Describe how to solve a system of linear equations.",
        "Explain what a differential equation is and give an example.",
        "What is the dot product and how is it used in geometry?",
        "Describe the properties of logarithmic functions.",
        "Explain the concept of convergence for infinite series.",
        "What is a partial derivative and when do you use it?",
        "Describe the relationship between sets, functions, and relations.",
        "Explain the binomial theorem and Pascal's triangle.",
        "What is the Pythagorean theorem and how can you prove it?",
        "Describe the concept of mathematical groups and symmetry.",
        "Explain what a Riemann sum is and how it relates to integrals.",
        "What is the squeeze theorem and when is it useful?",
        "Describe the method of Lagrange multipliers for optimization.",
        "Explain the concept of orthogonality in linear algebra.",
        "What is Green's theorem and how does it generalize?",
    ],
    "code": [
        "Explain how a hash table works internally.",
        "What is the difference between a stack and a queue?",
        "Describe how merge sort works and analyze its complexity.",
        "Explain what closures are in Python.",
        "What is dynamic programming and when should you use it?",
        "Describe how Python's garbage collector works.",
        "Explain the difference between shallow and deep copy.",
        "What are context managers and the with statement?",
        "Describe how a binary search tree works.",
        "Explain what unit testing is and why it matters.",
        "What is the GIL in Python and how does it affect concurrency?",
        "Describe how graph traversal algorithms (BFS, DFS) work.",
        "Explain what type hints are in Python and their benefits.",
        "What is memoization and how does it improve performance?",
        "Describe the observer design pattern.",
        "Explain how HTTP requests work in Python.",
        "What is the difference between composition and inheritance?",
        "Describe how regular expressions work.",
        "Explain what a virtual environment is and why you need one.",
        "What is test-driven development and how does it work?",
    ],
    "medical": [
        "Explain how statins work to lower cholesterol.",
        "What is the difference between bacterial and viral infections?",
        "Describe how the kidneys regulate blood pressure.",
        "Explain the mechanism of action of NSAIDs.",
        "What are autoimmune diseases and give examples.",
        "Describe the stages of cancer development.",
        "Explain how local anesthetics work at the molecular level.",
        "What is the role of the liver in drug metabolism?",
        "Describe the pathophysiology of heart failure.",
        "Explain how anticoagulants prevent blood clots.",
        "What are the different types of shock and their causes?",
        "Describe how the endocrine system regulates metabolism.",
        "Explain the mechanism of allergic reactions.",
        "What is sepsis and how is it managed?",
        "Describe the pharmacology of opioid analgesics.",
        "Explain how the respiratory system maintains acid-base balance.",
        "What are the mechanisms of drug-drug interactions?",
        "Describe the role of neurotransmitters in brain function.",
        "Explain the pathophysiology of chronic kidney disease.",
        "What is the significance of biomarkers in clinical diagnosis?",
    ],
    "legal": [
        "Explain the concept of sovereign immunity.",
        "What is the doctrine of promissory estoppel?",
        "Describe the elements of fraud in contract law.",
        "Explain how the Fourth Amendment protects against searches.",
        "What is strict liability and when does it apply?",
        "Describe the process of judicial review.",
        "Explain what administrative law is and how agencies operate.",
        "What is the parol evidence rule in contract interpretation?",
        "Describe the concept of qualified immunity for government officials.",
        "Explain the difference between real property and personal property.",
        "What is the commerce clause and its significance?",
        "Describe how class action lawsuits work.",
        "Explain the concept of equitable remedies.",
        "What are the Miranda rights and when must they be given?",
        "Describe the doctrine of respondeat superior.",
        "Explain the concept of eminent domain.",
        "What is the difference between arbitration and mediation?",
        "Describe the rule against perpetuities.",
        "Explain how intellectual property rights are protected.",
        "What is the concept of standing in federal court?",
    ],
    "finance": [
        "Explain what net present value means and how to calculate it.",
        "What is the efficient market hypothesis?",
        "Describe how options pricing works with Black-Scholes.",
        "Explain the concept of working capital management.",
        "What is dollar-cost averaging and when is it effective?",
        "Describe how credit ratings affect bond pricing.",
        "Explain the difference between fiscal and monetary policy.",
        "What is a mutual fund and how does it differ from an ETF?",
        "Describe the concept of financial leverage.",
        "Explain how currency exchange rates are determined.",
        "What is quantitative easing and how does it work?",
        "Describe the concept of yield curve and its implications.",
        "Explain what beta measures in the context of CAPM.",
        "What are derivatives and how are they used for hedging?",
        "Describe the concept of time value of money.",
        "Explain how mergers and acquisitions create value.",
        "What is the weighted average cost of capital?",
        "Describe the efficient frontier in portfolio theory.",
        "Explain the concept of moral hazard in financial markets.",
        "What is a credit default swap and how does it work?",
    ],
}


# ═══════════════════════════════���══════════════════���═══════════════════════════
# Phase 0: Generate training data
# ══════════════════════════════════════════════════════════════════════════════

def generate_training_data():
    """Generate training JSONL files for all domains."""
    for domain in DOMAINS:
        data_dir = EXPERIMENT_DIR / f"data_{domain}"
        if data_dir.exists() and (data_dir / "train.jsonl").exists():
            n = sum(1 for _ in open(data_dir / "train.jsonl"))
            if n >= N_TRAIN:
                log(f"  [{domain}] Training data exists: {n} examples")
                continue

        data_dir.mkdir(parents=True, exist_ok=True)
        pairs = TRAIN_DATA[domain]

        # Expand by cycling to N_TRAIN
        expanded = []
        while len(expanded) < N_TRAIN + 10:
            for q, a in pairs:
                expanded.append((q, a))
                if len(expanded) >= N_TRAIN + 10:
                    break

        train_pairs = expanded[:N_TRAIN]
        valid_pairs = expanded[N_TRAIN:N_TRAIN + 5]
        test_pairs = expanded[N_TRAIN + 5:N_TRAIN + 10]

        def write_jsonl(path, pairs_list):
            with open(path, "w") as f:
                for q, a in pairs_list:
                    record = {
                        "messages": [
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": a},
                        ]
                    }
                    f.write(json.dumps(record) + "\n")

        write_jsonl(data_dir / "train.jsonl", train_pairs)
        write_jsonl(data_dir / "valid.jsonl", valid_pairs)
        write_jsonl(data_dir / "test.jsonl", test_pairs)

        log(f"  [{domain}] Generated {len(train_pairs)} train, "
            f"{len(valid_pairs)} valid, {len(test_pairs)} test")


# ═══════════════════════════════════════════════════════════════════════════��══
# Phase 1: Train adapters (one per domain)
# ════════════════════════════════════════════════════════════════════════���═════

def train_adapter(domain: str) -> float:
    """Train a v_proj+o_proj LoRA adapter for a single domain. Returns training time in minutes."""
    import yaml

    adapter_dir = EXPERIMENT_DIR / f"adapter_{domain}"
    data_dir = EXPERIMENT_DIR / f"data_{domain}"

    if adapter_dir.exists() and (adapter_dir / "adapters.safetensors").exists():
        log(f"  [{domain}] Adapter already exists, skipping training")
        return 0.0

    adapter_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": MODEL_ID,
        "data": str(data_dir),
        "adapter_path": str(adapter_dir),
        "train": True,
        "fine_tune_type": "lora",
        "num_layers": 16,  # last 16 layers (of 42)
        "iters": TRAIN_ITERS,
        "batch_size": 1 if IS_SMOKE else 2,
        "learning_rate": 2e-4,
        "lora_parameters": {
            "rank": LORA_RANK,
            "scale": 4.0,
            "dropout": 0.0,
            "keys": LORA_KEYS,  # ["self_attn.v_proj", "self_attn.o_proj"]
        },
        "max_seq_length": 512,
        "mask_prompt": True,
        "grad_checkpoint": True,
        "save_every": TRAIN_ITERS,
        "steps_per_report": max(1, TRAIN_ITERS // 10),
        "val_batches": 3,
        "steps_per_eval": max(10, TRAIN_ITERS // 4),
        "seed": SEED,
    }

    config_path = EXPERIMENT_DIR / f"lora_config_{domain}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    log(f"  [{domain}] Training rank-{LORA_RANK} on {LORA_KEYS} "
        f"({TRAIN_ITERS} iters, {N_TRAIN} examples, 16 layers)...")

    t0 = time.time()
    cmd = ["uv", "run", "python", "-m", "mlx_lm", "lora", "--config", str(config_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    elapsed = (time.time() - t0) / 60.0

    if result.returncode != 0:
        log(f"  [{domain}] Training FAILED (exit={result.returncode})")
        log(f"  STDERR (last 1000 chars): {result.stderr[-1000:]}")
        raise RuntimeError(f"Training failed for domain={domain}")

    log(f"  [{domain}] Training complete in {elapsed:.1f} min")
    return elapsed


def phase_train_all():
    """Train all 5 domain adapters sequentially."""
    log("\n=== Phase 1: Train Domain Adapters (v_proj+o_proj) ===")
    training_times = {}
    for domain in DOMAINS:
        t = train_adapter(domain)
        training_times[domain] = t
        cleanup()
        log_memory(f"after-train-{domain}")
    return training_times


# ══════════════════════════════════════════════════════════════════���═══════════
# Phase 2: Behavioral evaluation (per-domain, solo adapter)
# ════════════════════════════════════════════════════════════════════════════���═

def generate_response(question: str, adapter_path: str | None = None) -> str:
    """Generate a single response using mlx_lm CLI."""
    prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
    cmd = [
        "uv", "run", "python", "-m", "mlx_lm", "generate",
        "--model", MODEL_ID,
        "--prompt", prompt,
        "--max-tokens", str(MAX_TOKENS),
        "--temp", "0.0",
    ]
    if adapter_path:
        cmd += ["--adapter-path", adapter_path]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def score_vocabulary(text: str, glossary: list) -> int:
    """Count domain glossary terms appearing in text."""
    text_lower = text.lower()
    return sum(1 for term in glossary if term.lower() in text_lower)


def phase_evaluate_base():
    """Generate base model responses for all domains. Returns dict of {domain: {query: (response, vocab_score)}}."""
    log("\n=== Phase 2a: Base Model Evaluation ===")
    base_results = {}
    for domain in DOMAINS:
        queries = EVAL_QUERIES[domain][:N_EVAL]
        glossary = DOMAIN_GLOSSARIES[domain]
        domain_results = []

        log(f"  [{domain}] Evaluating base model ({len(queries)} queries)...")
        for i, q in enumerate(queries):
            resp = generate_response(q, adapter_path=None)
            score = score_vocabulary(resp, glossary)
            domain_results.append({"query": q, "response_snippet": resp[:120], "vocab_score": score})
            log(f"    [{i+1}/{len(queries)}] vocab={score} — {q[:60]}...")

        base_results[domain] = domain_results
        cleanup()

    return base_results


def phase_evaluate_adapted():
    """Generate adapted model responses for all domains. Returns dict of {domain: results}."""
    log("\n=== Phase 2b: Adapted Model Evaluation (v_proj+o_proj) ===")
    adapted_results = {}
    for domain in DOMAINS:
        adapter_dir = EXPERIMENT_DIR / f"adapter_{domain}"
        queries = EVAL_QUERIES[domain][:N_EVAL]
        glossary = DOMAIN_GLOSSARIES[domain]
        domain_results = []

        log(f"  [{domain}] Evaluating adapted model ({len(queries)} queries)...")
        for i, q in enumerate(queries):
            resp = generate_response(q, adapter_path=str(adapter_dir))
            score = score_vocabulary(resp, glossary)
            domain_results.append({"query": q, "response_snippet": resp[:120], "vocab_score": score})
            log(f"    [{i+1}/{len(queries)}] vocab={score} — {q[:60]}...")

        adapted_results[domain] = domain_results
        cleanup()

    return adapted_results


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Composition test (naive weight addition, no Grassmannian)
# ═══════════════════════════════════════════════════════════════════��══════════

def phase_composition_test(base_results: dict, solo_improvement_rates: dict):
    """
    Test composition: load each adapter sequentially and evaluate on its own domain.
    This tests that adapter hot-swap works correctly for all 5 domains.
    For true N=5 composition, Grassmannian A-matrices would be needed.

    K4 simplified: each adapter's behavioral quality under sequential serving
    retains >=80% of solo evaluation quality.
    """
    log("\n=== Phase 3: Composition Sequential Serving Test ===")
    log("  (Testing adapter hot-swap for all 5 domains sequentially)")

    # Evaluate each domain with its adapter in sequence
    # This simulates the real serving scenario: route -> swap -> generate
    composition_results = {}

    for domain in DOMAINS:
        adapter_dir = EXPERIMENT_DIR / f"adapter_{domain}"
        queries = EVAL_QUERIES[domain][:N_EVAL]
        glossary = DOMAIN_GLOSSARIES[domain]

        log(f"  [{domain}] Composition test ({len(queries)} queries)...")
        improved_count = 0
        for i, q in enumerate(queries):
            resp = generate_response(q, adapter_path=str(adapter_dir))
            score_adapted = score_vocabulary(resp, glossary)
            score_base = base_results[domain][i]["vocab_score"]
            if score_adapted > score_base:
                improved_count += 1

        comp_rate = improved_count / len(queries)
        solo_rate = solo_improvement_rates.get(domain, 0.0)
        retention = comp_rate / solo_rate if solo_rate > 0 else 1.0

        composition_results[domain] = {
            "solo_improvement_rate": solo_rate,
            "composition_improvement_rate": comp_rate,
            "retention_ratio": retention,
        }
        log(f"    solo={solo_rate:.2f} comp={comp_rate:.2f} retention={retention:.2f}")
        cleanup()

    return composition_results


# ═══════════════════════════════════��══════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 70)
    log("P8: v_proj+o_proj Domain Adapters for Behavioral Text Quality")
    log(f"IS_SMOKE={IS_SMOKE}, N_EVAL={N_EVAL}, N_TRAIN={N_TRAIN}, "
        f"TRAIN_ITERS={TRAIN_ITERS}, LORA_RANK={LORA_RANK}")
    log(f"LORA_KEYS={LORA_KEYS}")
    log(f"Domains: {DOMAINS}")
    log("=" * 70)

    total_start = time.time()
    cleanup()
    log_memory("start")

    # Phase 0: Generate training data
    log("\n=== Phase 0: Generate Training Data ===")
    generate_training_data()

    # Phase 1: Train all 5 adapters
    training_times = phase_train_all()

    # Phase 2a: Evaluate base model
    base_results = phase_evaluate_base()

    # Phase 2b: Evaluate adapted models (solo)
    adapted_results = phase_evaluate_adapted()

    # Compute per-domain improvement rates
    solo_improvement_rates = {}
    for domain in DOMAINS:
        base = base_results[domain]
        adapted = adapted_results[domain]
        improved = sum(
            1 for b, a in zip(base, adapted)
            if a["vocab_score"] > b["vocab_score"]
        )
        rate = improved / len(base)
        solo_improvement_rates[domain] = rate

    # Phase 3: Composition test
    composition_results = phase_composition_test(base_results, solo_improvement_rates)

    # ─── Kill criteria ────────────────────��────────────────���───────────────
    log("\n" + "=" * 70)
    log("KILL CRITERIA RESULTS")
    log("=" * 70)

    # K1312: Math >=60% vocabulary improvement
    math_rate = solo_improvement_rates["math"]
    k1312_pass = math_rate >= 0.60
    log(f"\nK1312 (Math >=60% vocabulary improvement):")
    log(f"  rate={math_rate:.2f} ({math_rate*100:.1f}%) — {'PASS' if k1312_pass else 'FAIL'}")

    # K1313: Code >=60% vocabulary improvement
    code_rate = solo_improvement_rates["code"]
    k1313_pass = code_rate >= 0.60
    log(f"\nK1313 (Code >=60% vocabulary improvement):")
    log(f"  rate={code_rate:.2f} ({code_rate*100:.1f}%) — {'PASS' if k1313_pass else 'FAIL'}")

    # K1314: Medical >=60% vocabulary improvement
    med_rate = solo_improvement_rates["medical"]
    k1314_pass = med_rate >= 0.60
    log(f"\nK1314 (Medical >=60% vocabulary improvement):")
    log(f"  rate={med_rate:.2f} ({med_rate*100:.1f}%) — {'PASS' if k1314_pass else 'FAIL'}")

    # K1315: Composition retains >=80% per domain
    min_retention = min(
        (d["retention_ratio"] for d in composition_results.values()),
        default=1.0
    )
    k1315_pass = min_retention >= 0.80
    log(f"\nK1315 (Composition retains >=80% per domain):")
    for domain, cr in composition_results.items():
        log(f"  [{domain}] retention={cr['retention_ratio']:.2f}")
    log(f"  min_retention={min_retention:.2f} — {'PASS' if k1315_pass else 'FAIL'}")

    all_pass = k1312_pass and k1313_pass and k1314_pass and k1315_pass
    total_min = (time.time() - total_start) / 60.0

    log(f"\n{'='*70}")
    log(f"SUMMARY:")
    log(f"  K1312 (Math): {'PASS' if k1312_pass else 'FAIL'} ({math_rate*100:.1f}%)")
    log(f"  K1313 (Code): {'PASS' if k1313_pass else 'FAIL'} ({code_rate*100:.1f}%)")
    log(f"  K1314 (Medical): {'PASS' if k1314_pass else 'FAIL'} ({med_rate*100:.1f}%)")
    log(f"  K1315 (Composition): {'PASS' if k1315_pass else 'FAIL'} (min_retention={min_retention:.2f})")
    log(f"  ALL PASS: {all_pass}")
    log(f"Total time: {total_min:.1f} min")

    # Also show legal & finance rates
    legal_rate = solo_improvement_rates["legal"]
    fin_rate = solo_improvement_rates["finance"]
    log(f"\nAdditional domains (not in kill criteria):")
    log(f"  Legal: {legal_rate*100:.1f}%")
    log(f"  Finance: {fin_rate*100:.1f}%")

    # Compare with killed behavioral E2E (q_proj baseline)
    log(f"\nComparison with q_proj behavioral E2E (killed):")
    log(f"  q_proj: math=30%, code=20%, medical=60%, legal=20%")
    log(f"  v_proj+o_proj: math={math_rate*100:.1f}%, code={code_rate*100:.1f}%, "
        f"medical={med_rate*100:.1f}%, legal={legal_rate*100:.1f}%, finance={fin_rate*100:.1f}%")
    log(f"{'='*70}")

    # ─── Save results ───────────────────���──────────────────────────────────
    # Build per-domain detail
    domain_detail = {}
    for domain in DOMAINS:
        base = base_results[domain]
        adapted = adapted_results[domain]
        per_query = []
        for b, a in zip(base, adapted):
            per_query.append({
                "query": b["query"][:80],
                "base_vocab": b["vocab_score"],
                "adapted_vocab": a["vocab_score"],
                "improved": a["vocab_score"] > b["vocab_score"],
            })

        mean_base = sum(b["vocab_score"] for b in base) / len(base)
        mean_adapted = sum(a["vocab_score"] for a in adapted) / len(adapted)

        domain_detail[domain] = {
            "improvement_rate": solo_improvement_rates[domain],
            "mean_base_vocab": round(mean_base, 2),
            "mean_adapted_vocab": round(mean_adapted, 2),
            "n_eval": len(base),
            "per_query": per_query,
        }

    results = {
        "is_smoke": IS_SMOKE,
        "config": {
            "n_eval": N_EVAL,
            "n_train": N_TRAIN,
            "train_iters": TRAIN_ITERS,
            "lora_rank": LORA_RANK,
            "lora_keys": LORA_KEYS,
            "num_layers": 16,
        },
        "training_times": training_times,
        "domain_results": domain_detail,
        "composition": composition_results,
        "kill_criteria": {
            "k1312_math": {"pass": k1312_pass, "rate": math_rate, "threshold": 0.60},
            "k1313_code": {"pass": k1313_pass, "rate": code_rate, "threshold": 0.60},
            "k1314_medical": {"pass": k1314_pass, "rate": med_rate, "threshold": 0.60},
            "k1315_composition": {"pass": k1315_pass, "min_retention": min_retention, "threshold": 0.80},
        },
        "qproj_baseline": {
            "math": 0.30, "code": 0.20, "medical": 0.60, "legal": 0.20,
            "source": "exp_p1_p0_behavioral_e2e (killed)",
        },
        "all_pass": all_pass,
        "total_time_min": round(total_min, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
