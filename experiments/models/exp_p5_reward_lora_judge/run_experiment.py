#!/usr/bin/env python3
"""
P5.B1: Per-Domain Reward LoRA Judge — rank-16 reward adapter per domain.

arXiv:2506.05748 shows reward LoRA (0.8% params) achieves 96.2% RewardBench.
We train per-domain reward adapters on Gemma 4 E4B to score whether outputs
match domain-specific quality criteria using Bradley-Terry preference loss.

Kill criteria:
  K1273: >= 80% agreement (reward correctly ranks preferred > rejected)
  K1274: < 10MB per adapter
  K1275: < 100ms scoring latency on M5 Pro
"""

import gc
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# Memory safety (CODING_GUIDELINES §2)
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
LORA_RANK = 16
N_LORA_LAYERS = 8  # last 8 layers
SEED = 42

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
TRAIN_ITERS = 20 if IS_SMOKE else 200
LR = 1e-4
MAX_SEQ_LEN = 256

LORA_MODULES = ["self_attn.q_proj", "self_attn.o_proj"]
LORA_SCALE = 1.0  # conservative scaling for reward signal


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
# PREFERENCE DATA — synthetic pairs per domain
# Format: (prompt, chosen_response, rejected_response)
# Chosen = domain-appropriate formatting/content
# Rejected = casual/generic response
# ══════════════════════════════════════════════════════════════════════════════

MATH_PAIRS = [
    ("Derive the quadratic formula.",
     r"Starting from $ax^2 + bx + c = 0$, divide by $a$: $x^2 + \frac{b}{a}x + \frac{c}{a} = 0$. Complete the square: $\left(x + \frac{b}{2a}\right)^2 = \frac{b^2 - 4ac}{4a^2}$. Therefore: $$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$",
     "To solve a quadratic equation, you take the coefficients and plug them into a formula. The answer gives you two possible x values by using addition and subtraction."),
    ("What is the derivative of x^n?",
     r"By the power rule: $$\frac{d}{dx}[x^n] = nx^{n-1}$$. This follows from the limit definition: $\lim_{h \to 0} \frac{(x+h)^n - x^n}{h} = nx^{n-1}$ via the binomial theorem.",
     "The derivative of x to the n is n times x to the n minus 1. You just bring down the exponent and subtract one."),
    ("Prove that the sum of first n integers is n(n+1)/2.",
     r"**Proof by induction.** Base: $n=1$: $1 = \frac{1 \cdot 2}{2} = 1$ \checkmark. Inductive step: assume $\sum_{k=1}^{m} k = \frac{m(m+1)}{2}$. Then $\sum_{k=1}^{m+1} k = \frac{m(m+1)}{2} + (m+1) = \frac{(m+1)(m+2)}{2}$. $\square$",
     "You can show this by pairing numbers: 1 pairs with n, 2 pairs with n-1, and so on. Each pair adds up to n+1, and there are n/2 pairs."),
    ("State the fundamental theorem of calculus.",
     r"If $f$ is continuous on $[a,b]$ and $F(x) = \int_a^x f(t)\,dt$, then $F'(x) = f(x)$ for all $x \in (a,b)$. Consequently: $$\int_a^b f(x)\,dx = F(b) - F(a)$$",
     "The fundamental theorem says that integration and differentiation are inverse operations. If you integrate and then differentiate, you get back what you started with."),
    ("What is the Taylor series of e^x?",
     r"The Taylor expansion of $e^x$ about $x=0$: $$e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!} = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$$ Convergent $\forall x \in \mathbb{R}$ with radius of convergence $R = \infty$.",
     "e to the x can be written as an infinite sum. You keep adding terms with higher powers of x divided by factorials. It works for all values of x."),
    ("Explain eigenvalues.",
     r"For a square matrix $A \in \mathbb{R}^{n \times n}$, scalar $\lambda$ is an eigenvalue if $\exists v \neq 0$ such that $Av = \lambda v$. Equivalently, $\det(A - \lambda I) = 0$. The set $\sigma(A) = \{\lambda_1, \ldots, \lambda_n\}$ is the spectrum of $A$.",
     "Eigenvalues are special numbers associated with a matrix. When you multiply the matrix by certain vectors, the result is just a scaled version of that vector."),
    ("What is the chain rule?",
     r"If $f$ and $g$ are differentiable, then $(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$. In Leibniz notation: $$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$",
     "The chain rule is for taking derivatives of composite functions. You differentiate the outer function and multiply by the derivative of the inner function."),
    ("Explain the Cauchy-Schwarz inequality.",
     r"For vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$: $$|\langle \mathbf{u}, \mathbf{v} \rangle|^2 \leq \langle \mathbf{u}, \mathbf{u} \rangle \cdot \langle \mathbf{v}, \mathbf{v} \rangle$$ Equality iff $\mathbf{u} = \alpha \mathbf{v}$ for some $\alpha \in \mathbb{R}$.",
     "The Cauchy-Schwarz inequality says the dot product of two vectors is at most the product of their lengths. Equality happens when the vectors point in the same direction."),
    ("What is a Fourier transform?",
     r"The Fourier transform of $f: \mathbb{R} \to \mathbb{C}$: $$\hat{f}(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t}\,dt$$ with inverse $f(t) = \frac{1}{2\pi}\int_{-\infty}^{\infty} \hat{f}(\omega) e^{i\omega t}\,d\omega$.",
     "A Fourier transform converts a signal from the time domain to the frequency domain. It tells you what frequencies are present in a signal."),
    ("Prove sqrt(2) is irrational.",
     r"Assume $\sqrt{2} = \frac{p}{q}$ with $\gcd(p,q) = 1$. Then $2q^2 = p^2$, so $p^2$ is even, hence $p$ is even: $p = 2k$. Then $2q^2 = 4k^2 \Rightarrow q^2 = 2k^2$, so $q$ is even. Contradiction with $\gcd(p,q) = 1$. $\square$",
     "You assume it's rational, write it as a fraction in lowest terms, and then show both numerator and denominator must be even, which is a contradiction."),
    ("What is L'Hopital's rule?",
     r"If $\lim_{x \to a} f(x) = \lim_{x \to a} g(x) = 0$ (or $\pm\infty$), then $$\lim_{x \to a} \frac{f(x)}{g(x)} = \lim_{x \to a} \frac{f'(x)}{g'(x)}$$ provided the right-hand limit exists.",
     "L'Hopital's rule lets you evaluate limits of indeterminate forms. When you get 0/0 or infinity/infinity, you can take derivatives of top and bottom."),
    ("Define a group in abstract algebra.",
     r"A group $(G, \cdot)$ is a set $G$ with binary operation $\cdot$ satisfying: (1) Closure: $\forall a,b \in G: a \cdot b \in G$; (2) Associativity: $(a \cdot b) \cdot c = a \cdot (b \cdot c)$; (3) Identity: $\exists e \in G: e \cdot a = a \cdot e = a$; (4) Inverse: $\forall a \in G, \exists a^{-1}: a \cdot a^{-1} = e$.",
     "A group is a mathematical structure with a set and an operation that follows certain rules: closure, associativity, having an identity element, and every element having an inverse."),
    ("What is the Jacobian matrix?",
     r"For $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian is: $$J = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{pmatrix} \in \mathbb{R}^{m \times n}$$",
     "The Jacobian is a matrix of partial derivatives. It shows how a vector function changes with respect to each input variable."),
    ("Explain the divergence theorem.",
     r"For a vector field $\mathbf{F}$ and volume $V$ bounded by surface $S$: $$\oint_S \mathbf{F} \cdot d\mathbf{S} = \iiint_V (\nabla \cdot \mathbf{F})\,dV$$ This relates the flux through a closed surface to the divergence within the enclosed volume.",
     "The divergence theorem connects a surface integral to a volume integral. The total flow out through a surface equals the total source strength inside."),
    ("What is a metric space?",
     r"A metric space $(X, d)$ is a set $X$ with distance function $d: X \times X \to [0, \infty)$ satisfying: (1) $d(x,y) = 0 \iff x = y$; (2) $d(x,y) = d(y,x)$; (3) $d(x,z) \leq d(x,y) + d(y,z)$ (triangle inequality).",
     "A metric space is a set where you can measure distances between points. The distance function has to follow certain common-sense rules like the triangle inequality."),
    ("Explain integration by parts.",
     r"For differentiable $u$ and $v$: $$\int u\,dv = uv - \int v\,du$$ Derived from the product rule: $(uv)' = u'v + uv'$, so $uv' = (uv)' - u'v$. Integrating both sides yields the formula.",
     "Integration by parts is a technique where you split the integrand into two parts. One part you differentiate and the other you integrate."),
    ("What is the binomial theorem?",
     r"For $n \in \mathbb{N}$: $$(x + y)^n = \sum_{k=0}^{n} \binom{n}{k} x^{n-k} y^k$$ where $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ are the binomial coefficients.",
     "The binomial theorem tells you how to expand expressions like (x+y) raised to a power. Each term involves a binomial coefficient."),
    ("Define continuity.",
     r"Function $f: \mathbb{R} \to \mathbb{R}$ is continuous at $a$ if $\forall \epsilon > 0, \exists \delta > 0$ such that $|x - a| < \delta \Rightarrow |f(x) - f(a)| < \epsilon$. Equivalently, $\lim_{x \to a} f(x) = f(a)$.",
     "A function is continuous if you can draw it without lifting your pen. More precisely, small changes in input produce small changes in output."),
    ("What is a determinant?",
     r"For $A \in \mathbb{R}^{n \times n}$: $\det(A) = \sum_{\sigma \in S_n} \text{sgn}(\sigma) \prod_{i=1}^{n} a_{i,\sigma(i)}$. Key properties: $\det(AB) = \det(A)\det(B)$, $A$ is invertible iff $\det(A) \neq 0$.",
     "The determinant is a number you can calculate from a square matrix. It tells you whether the matrix is invertible and how it scales area or volume."),
    ("Explain the residue theorem.",
     r"If $f$ is analytic in a region except at isolated singularities $z_1, \ldots, z_n$ inside a closed contour $C$: $$\oint_C f(z)\,dz = 2\pi i \sum_{k=1}^{n} \text{Res}(f, z_k)$$",
     "The residue theorem in complex analysis lets you evaluate contour integrals by summing up residues at poles inside the contour."),
]

LEGAL_PAIRS = [
    ("Analyze liability for a slip-and-fall in a grocery store.",
     "Under premises liability doctrine (Restatement (Second) of Torts § 343), the store as business invitor owes a duty of reasonable care to discover and remedy dangerous conditions. The plaintiff must establish: (1) the defendant owned/controlled the premises; (2) a dangerous condition existed; (3) defendant knew or should have known of the condition; (4) defendant failed to exercise reasonable care. The constructive notice standard (Anjou v. Boston Elevated Railway Co., 94 N.E. 386 (Mass. 1911)) applies if the hazard existed long enough that a reasonable inspection would have discovered it.",
     "If someone slips and falls in a grocery store, the store might be liable. They need to keep the floors safe and clean up spills promptly. The injured person could potentially sue for damages."),
    ("What constitutes breach of fiduciary duty?",
     "A fiduciary duty breach requires: (1) existence of a fiduciary relationship (Meinhard v. Salmon, 249 N.Y. 458 (1928)); (2) breach of duty of loyalty or care; (3) causation; (4) damages. The duty of loyalty prohibits self-dealing, corporate opportunity usurpation, and conflicts of interest (In re Walt Disney Co. Derivative Litigation, 906 A.2d 27 (Del. 2006)). Under the business judgment rule (Aronson v. Lewis, 473 A.2d 805 (Del. 1984)), directors' decisions receive deference absent gross negligence or bad faith.",
     "Fiduciary duty means someone in a position of trust has to act in the best interest of the person they serve. If they put their own interests first or are negligent, they breach this duty."),
    ("Explain the doctrine of promissory estoppel.",
     "Promissory estoppel (Restatement (Second) of Contracts § 90) requires: (1) a clear and definite promise; (2) the promisor reasonably expected the promise to induce action or forbearance; (3) the promisee actually relied on the promise; (4) injustice can only be avoided by enforcement. As held in Ricketts v. Scothorn, 57 Neb. 51 (1898), the doctrine substitutes for consideration when detrimental reliance occurs. Damages are typically limited to reliance interest rather than expectation interest (Hoffman v. Red Owl Stores, 26 Wis. 2d 683 (1965)).",
     "Promissory estoppel is when someone makes a promise and the other person relies on it. Even without a formal contract, the promise can be enforced if the other person changed their position because of it."),
    ("What are the elements of negligence?",
     "Negligence requires four elements per Palsgraf v. Long Island Railroad Co., 248 N.Y. 339 (1928): (1) Duty — defendant owed plaintiff a duty of care (determined by foreseeability of harm); (2) Breach — defendant's conduct fell below the applicable standard of care (reasonable person standard, T.J. Hooper, 60 F.2d 737 (2d Cir. 1932)); (3) Causation — both cause-in-fact (but-for test) and proximate cause; (4) Damages — actual, compensable harm. Comparative negligence (pure vs. modified) may reduce recovery proportionally.",
     "Negligence is when someone doesn't take reasonable care and someone else gets hurt. You need to show they had a duty, broke it, caused your injury, and you suffered actual damages."),
    ("Explain the parol evidence rule.",
     "The parol evidence rule (UCC § 2-202; Restatement (Second) of Contracts § 213) bars introduction of prior or contemporaneous oral agreements that contradict a fully integrated written contract. Per Thompson v. Libby, 34 Minn. 374 (1885), a merger/integration clause creates a presumption of complete integration. Exceptions include: (1) fraud, duress, or mistake; (2) ambiguity requiring interpretation (Pacific Gas & Electric Co. v. G.W. Thomas Drayage, 69 Cal.2d 33 (1968)); (3) conditions precedent; (4) subsequent modifications.",
     "The parol evidence rule says that if you have a written contract, you generally can't bring in outside conversations or agreements to change what the contract says. There are some exceptions though."),
    ("What is strict liability in tort?",
     "Strict liability imposes liability without fault for: (1) abnormally dangerous activities (Rylands v. Fletcher [1868] UKHL 1; Restatement (Second) of Torts § 519-520); (2) defective products (Greenman v. Yuba Power Products, 59 Cal.2d 57 (1963)); (3) wild animal keeping. For products liability under Restatement (Third) § 2, a product is defective if: (a) manufacturing defect — deviation from intended design; (b) design defect — foreseeable risks exceed benefits (risk-utility test); (c) inadequate warnings or instructions.",
     "Strict liability means you're responsible for harm even if you weren't negligent. It applies to things like dangerous activities, defective products, and keeping wild animals."),
    ("Analyze the Fourth Amendment search and seizure doctrine.",
     "The Fourth Amendment protects against unreasonable searches and seizures, requiring probable cause and a warrant (Katz v. United States, 389 U.S. 347 (1967)). The exclusionary rule (Mapp v. Ohio, 367 U.S. 643 (1961)) bars illegally obtained evidence. Key exceptions: (1) search incident to arrest (Chimel v. California, 395 U.S. 752 (1969)); (2) plain view; (3) consent; (4) exigent circumstances; (5) automobile exception (Carroll v. United States, 267 U.S. 132 (1925)); (6) stop and frisk (Terry v. Ohio, 392 U.S. 1 (1968)).",
     "The Fourth Amendment protects you from unreasonable searches by police. Generally they need a warrant, but there are several exceptions like when you consent or they see something in plain view."),
    ("Explain consideration in contract law.",
     "Consideration is a bargained-for exchange of legal value (Hamer v. Sidway, 124 N.Y. 538 (1891)). It requires: (1) a legal detriment to the promisee or legal benefit to the promisor; (2) the detriment must induce the promise and vice versa. Past consideration is insufficient (Mills v. Wyman, 20 Mass. 207 (1825)). Pre-existing duty rule: performing an existing obligation is not consideration (Stilk v. Myrick [1809] EWHC KB J58), though UCC § 2-209 permits modification without consideration for goods contracts.",
     "Consideration in contracts means each side has to give something of value. You can't just have a one-sided promise — both parties need to exchange something for the contract to be enforceable."),
    ("What constitutes defamation?",
     "Defamation requires: (1) a false statement of fact (not opinion — Milkovich v. Lorain Journal Co., 497 U.S. 1 (1990)); (2) published to a third party; (3) at least negligence regarding falsity; (4) damages. For public figures, actual malice (knowledge of falsity or reckless disregard) is required per New York Times Co. v. Sullivan, 376 U.S. 254 (1964). Libel (written) is actionable per se; slander (oral) generally requires proof of special damages, except for slander per se categories.",
     "Defamation is when someone makes false statements that damage your reputation. If it's written it's called libel, if spoken it's called slander. Public figures have a harder time winning defamation cases."),
    ("Explain the doctrine of res judicata.",
     "Res judicata (claim preclusion) bars re-litigation of claims that were or could have been raised in a prior action. Per Cromwell v. County of Sac, 94 U.S. 351 (1876), requirements are: (1) final judgment on the merits; (2) same parties or those in privity; (3) same cause of action (transactional test per Restatement (Second) of Judgments § 24). Collateral estoppel (issue preclusion) bars re-litigation of specific issues actually litigated and necessarily decided (Blonder-Tongue Laboratories v. University of Illinois Foundation, 402 U.S. 313 (1971)).",
     "Res judicata means you can't sue someone twice for the same thing. Once a court has made a final decision, you can't bring the same claim again."),
    ("What is the duty to mitigate damages?",
     "The mitigation doctrine requires injured parties to take reasonable steps to minimize losses (Rockingham County v. Luten Bridge Co., 35 F.2d 301 (4th Cir. 1929)). In employment: the terminated employee must seek substantially similar employment (Parker v. Twentieth Century-Fox Film Corp., 3 Cal.3d 176 (1970)). The burden of proving failure to mitigate falls on the breaching party. Mitigation costs are recoverable as damages even if mitigation efforts ultimately fail.",
     "If someone wrongs you, you have a duty to try to minimize your losses. You can't just sit back and let damages pile up — you need to make reasonable efforts to reduce the harm."),
    ("Explain adverse possession.",
     "Adverse possession requires continuous possession for the statutory period that is: (1) actual and exclusive; (2) open and notorious; (3) adverse/hostile — without owner's permission; (4) continuous for the statutory period (varies by jurisdiction: 5-20 years). Per Marengo Cave Co. v. Ross, 10 N.E.2d 917 (Ind. 1937), possession must be visible enough to put a reasonable owner on notice. Color of title may reduce the required period. Tax payment requirements exist in some jurisdictions.",
     "Adverse possession is when someone can claim ownership of land by occupying it for a long time without the owner's permission. They have to use it openly and continuously for a number of years."),
    ("What is the implied warranty of merchantability?",
     "Under UCC § 2-314, goods sold by a merchant must be: (1) fit for ordinary purposes; (2) adequately packaged and labeled; (3) conforming to any promises on the label; (4) of even quality within each unit; (5) of fair average quality. This warranty arises automatically in sales by merchants (Henningsen v. Bloomfield Motors, 32 N.J. 358 (1960)). It can be disclaimed with conspicuous language mentioning 'merchantability' (UCC § 2-316), subject to unconscionability limitations.",
     "The implied warranty of merchantability means products should work for their basic intended purpose. When you buy something, there's an automatic guarantee it's reasonably fit for what it's supposed to do."),
    ("Explain the statute of frauds.",
     "The Statute of Frauds (Restatement (Second) of Contracts § 110) requires a signed writing for: (1) contracts not performable within one year; (2) land sale contracts; (3) contracts for goods ≥$500 (UCC § 2-201); (4) suretyship agreements; (5) executor contracts to pay estate debts personally; (6) marriage contracts. Part performance (Restatement § 129) and promissory estoppel (§ 139) may satisfy the writing requirement in equity.",
     "The statute of frauds requires certain types of contracts to be in writing. Things like real estate deals, contracts that take more than a year, and big sales need to be written down to be enforceable."),
    ("Analyze the doctrine of unconscionability.",
     "Unconscionability (UCC § 2-302; Restatement (Second) of Contracts § 208) renders a contract unenforceable when terms are unreasonably favorable to one party. Per Williams v. Walker-Thomas Furniture Co., 350 F.2d 445 (D.C. Cir. 1965), courts examine: (1) procedural unconscionability — absence of meaningful choice (adhesion, unequal bargaining power, deception); (2) substantive unconscionability — unreasonably favorable terms. Both elements are typically required on a sliding scale.",
     "Unconscionability is when a contract is so unfair that a court won't enforce it. This can happen when one side has all the power and the other side had no real choice."),
]

SOAP_PAIRS = [
    ("Document a patient visit for acute lower back pain.",
     "S: 45-year-old male presents with acute onset lower back pain × 3 days. Pain rated 7/10, radiating to left buttock. Denies numbness, tingling, or bowel/bladder changes. Aggravated by bending and prolonged sitting. History of similar episode 2 years ago, resolved with physical therapy.\n\nO: VS: BP 132/84, HR 78, T 98.6°F. Musculoskeletal: Paravertebral muscle spasm L3-L5 bilateral. ROM limited in flexion to 40°. Negative SLR bilaterally. DTRs 2+ symmetric. Strength 5/5 bilateral lower extremities.\n\nA: Acute mechanical low back pain (M54.5), likely musculoligamentous strain. No red flags for cauda equina or radiculopathy.\n\nP: 1. Naproxen 500mg PO BID × 10 days with food. 2. Cyclobenzaprine 10mg PO QHS PRN × 7 days. 3. Physical therapy referral, 2×/week × 4 weeks. 4. Activity modification: avoid heavy lifting >20 lbs. 5. Return in 2 weeks, sooner if neurological symptoms develop.",
     "The patient came in with back pain that started a few days ago. He says it hurts a lot, especially when bending. I examined him and his back muscles are tight. I think it's just a muscle strain, nothing serious. I prescribed some anti-inflammatory medication and muscle relaxants, and recommended physical therapy. He should come back if it gets worse."),
    ("Document a diabetic follow-up visit.",
     "S: 58-year-old female with Type 2 DM presents for 3-month follow-up. Reports good medication adherence. Occasional fasting glucose readings 140-160 mg/dL. Denies polyuria, polydipsia, or vision changes. Diet compliance approximately 80%. Walking 30 min/day, 4 days/week.\n\nO: VS: BP 128/78, HR 72, BMI 31.2. Labs: HbA1c 7.4% (prev 7.8%), FPG 148 mg/dL, total cholesterol 198, LDL 112, HDL 48, TG 190, Cr 0.9, eGFR >60. Foot exam: intact sensation monofilament, pedal pulses 2+ bilateral, no lesions.\n\nA: Type 2 diabetes mellitus (E11.65) with improving glycemic control. Dyslipidemia, mixed. Obesity class I.\n\nP: 1. Continue metformin 1000mg BID. 2. Increase atorvastatin 20mg → 40mg daily (LDL goal <100). 3. Continue lifestyle modifications. 4. Ophthalmology referral for annual diabetic eye exam. 5. Repeat HbA1c and lipid panel in 3 months. 6. Discussed dietary counseling referral.",
     "Patient is here for diabetes checkup. Her blood sugar is better than last time but still a bit high. Cholesterol could be better too. I'm keeping her on the same diabetes medication and increasing her cholesterol medicine. She needs to keep exercising and eating healthy. Follow up in 3 months."),
    ("Document a pediatric well-child visit for a 2-year-old.",
     "S: 2-year-old male presents for well-child visit. Mother reports normal developmental milestones: speaks 50+ words, combines 2-word phrases, runs well, climbs stairs with support. Eating well, varied diet including fruits, vegetables, proteins. Sleeps 11-12 hours nightly with one 2-hour nap. No behavioral concerns. Immunizations up to date.\n\nO: VS: Wt 13.2 kg (65th %ile), Ht 88 cm (70th %ile), HC 49 cm (55th %ile). HEENT: TMs clear bilaterally, oropharynx normal, no dental caries. Lungs: CTA bilaterally. CV: RRR, no murmur. Abdomen: soft, non-tender, no organomegaly. GU: normal male, testes descended bilaterally. Neuro: appropriate gait, good muscle tone. Developmental: stacks 6 blocks, kicks ball, turns book pages.\n\nA: Healthy 2-year-old male. Growth and development appropriate for age.\n\nP: 1. Hepatitis A vaccine dose 2 today. 2. Anticipatory guidance: car seat safety, water safety, poison prevention. 3. Discuss transition to cup from bottle if not completed. 4. Return for 30-month well-child visit. 5. Dental referral if not yet established.",
     "Cute 2-year-old boy came in for his regular checkup. He's growing well and meeting all his milestones. He's talking a lot and running around. Everything looks good on the exam. I gave him his next vaccine and talked to mom about safety. He should come back in 6 months."),
    ("Document an initial evaluation for chest pain.",
     "S: 62-year-old male presents with substernal chest pressure × 2 hours. Pain rated 8/10, radiating to left arm and jaw. Associated diaphoresis and nausea. PMH: HTN, hyperlipidemia, 30 pack-year smoking history (quit 5 years ago). Family history: father MI at age 55.\n\nO: VS: BP 158/92, HR 96, RR 20, SpO2 96% RA, T 98.2°F. General: diaphoretic, moderate distress. CV: tachycardic, regular, no murmur/gallop. Lungs: CTA bilaterally, no crackles. ECG: 2mm ST elevation V2-V5. Troponin I: 0.8 ng/mL (normal <0.04). CBC, BMP within normal limits.\n\nA: Acute ST-elevation myocardial infarction (STEMI) — anterior wall (I21.0). High-risk presentation with multiple risk factors.\n\nP: 1. STAT cardiology consult for emergent cardiac catheterization. 2. Aspirin 325mg PO chewed and swallowed. 3. Heparin bolus 60 units/kg IV, then 12 units/kg/hr drip. 4. Nitroglycerin 0.4mg SL × 3 PRN. 5. Morphine 2-4mg IV PRN pain. 6. NPO status. 7. Continuous cardiac monitoring. 8. Serial troponins q6h. 9. Informed patient of diagnosis and need for emergent intervention.",
     "A 62-year-old man came in with severe chest pain for the last 2 hours. He was sweating and nauseous. His EKG showed a heart attack and his cardiac enzymes were elevated. I called cardiology immediately and started treatment with aspirin and blood thinners. He needs emergency catheterization right away."),
    ("Document a mental health assessment for depression.",
     "S: 34-year-old female presents with persistent low mood × 6 weeks. PHQ-9 score: 18 (moderately severe). Reports anhedonia, insomnia (difficulty initiating sleep, 2-3 hours to fall asleep), decreased appetite with 8-lb weight loss, poor concentration affecting work performance, fatigue, and feelings of worthlessness. Denies suicidal ideation, homicidal ideation, or psychotic symptoms. No prior psychiatric history. Recent stressor: job loss 2 months ago.\n\nO: Mental Status Exam: Appearance appropriate, poor eye contact, psychomotor retardation noted. Mood: 'empty.' Affect: constricted, tearful. Speech: soft, decreased rate. Thought process: linear, goal-directed. Thought content: negative self-referential themes, no SI/HI/AH/VH. Cognition: oriented ×4, concentration impaired on serial 7s. Insight: fair. Judgment: intact.\n\nA: Major depressive disorder, single episode, moderate-severe (F32.2). Rule out adjustment disorder with depressed mood given temporal relationship to job loss.\n\nP: 1. Start sertraline 50mg PO daily, titrate to 100mg in 2 weeks if tolerated. 2. CBT referral — weekly sessions recommended. 3. Sleep hygiene education handout provided. 4. Columbia Suicide Severity Rating Scale: negative — low acute risk. 5. Safety plan reviewed and documented. 6. Follow-up in 2 weeks for medication assessment. 7. Return precautions: if SI develops, present to ED or call 988.",
     "Young woman came in feeling really down for the past month and a half. She can't sleep well, lost weight, and is having trouble at work. She scored high on the depression questionnaire. I started her on an antidepressant and referred her for therapy. She's not suicidal. I'll see her again in two weeks to check on the medication."),
    ("Document a follow-up for hypertension management.",
     "S: 52-year-old male with HTN presents for BP check. Home BP readings averaging 142/88 on current regimen (lisinopril 10mg daily). Denies headaches, visual changes, chest pain, or dyspnea. Reports moderate sodium intake, sedentary lifestyle. Medication adherence: 90%.\n\nO: VS: BP 146/90 (right arm, seated), 144/88 (repeat), HR 76. BMI 28.4. CV: RRR, no murmur. Lungs: clear. Extremities: no edema. Labs (2 weeks ago): Cr 1.0, K 4.2, lipid panel within target.\n\nA: Essential hypertension (I10), uncontrolled on current regimen. Goal BP <130/80 per ACC/AHA 2017 guidelines.\n\nP: 1. Increase lisinopril 10mg → 20mg daily. 2. Add amlodipine 5mg daily if BP remains >130/80 at next visit. 3. DASH diet counseling. 4. Exercise prescription: 150 min/week moderate-intensity aerobic activity. 5. Recheck BMP in 4 weeks (monitor K and Cr with lisinopril increase). 6. Follow-up 4 weeks.",
     "Patient's blood pressure is still too high on his current medication. I'm going to increase his dose and told him to eat less salt and exercise more. He needs to come back in a month so we can check if the higher dose is working."),
    ("Document assessment of a child with ear infection.",
     "S: 4-year-old female brought by mother for right ear pain × 2 days. Fever to 101.2°F at home last night. Tugging at right ear, irritable, decreased appetite. No vomiting or diarrhea. No recent URI symptoms. No prior ear infections. Allergies: NKDA.\n\nO: VS: T 100.8°F, HR 110, RR 22. HEENT: Right TM erythematous, bulging, with decreased mobility on insufflation. Purulent effusion noted behind TM. Left TM normal landmarks, mobile. Oropharynx: no erythema. Cervical lymphadenopathy: 1cm right anterior chain node, mobile, non-tender.\n\nA: Acute otitis media, right ear (H66.91). Meets criteria for antibiotic treatment per AAP 2013 guidelines (moderate-severe otalgia, fever ≥ 39°C or otalgia ≥48h).\n\nP: 1. Amoxicillin 90 mg/kg/day divided BID × 10 days (weight-based: 720mg BID). 2. Ibuprofen 10 mg/kg PO q6h PRN fever/pain. 3. Acetaminophen 15 mg/kg PO q4h PRN alternating with ibuprofen. 4. Return in 48-72 hours if no improvement. 5. Recheck ears at 2 weeks. 6. Discussed watchful waiting option — parents prefer treatment given fever and severity.",
     "Little girl has an ear infection. Her right ear is red and bulging. She has a fever. I'm putting her on antibiotics for 10 days and told mom to give her Tylenol for the pain. Come back if she's not better in a few days."),
    ("Document a prenatal visit at 28 weeks.",
     "S: 30-year-old G2P1 at 28w2d by LMP (consistent with first trimester US). Fetal movement present and active. Denies vaginal bleeding, leakage of fluid, contractions, or headaches. Mild bilateral ankle edema at end of day. No dysuria. Current medications: prenatal vitamin, iron supplement.\n\nO: VS: BP 118/72, Wt 168 lbs (pre-pregnancy 148, gain 20 lbs). Fundal height: 28 cm (appropriate for dates). FHR: 148 bpm by doppler. Leopold's: cephalic presentation. Edema: trace bilateral pedal, non-pitting. Urine dip: protein negative, glucose negative. Labs: 1-hour GCT 118 mg/dL (normal <140). CBC: Hgb 11.2, Hct 33.6. RhoGAM eligibility: Rh-negative, antibody screen negative.\n\nA: Intrauterine pregnancy at 28 weeks, uncomplicated (Z34.82). Appropriate fetal growth. Gestational diabetes screen normal.\n\nP: 1. RhoGAM 300mcg IM administered today (Rh-negative). 2. Tdap vaccine administered. 3. Begin kick counts: 10 movements in 2 hours. 4. Continue prenatal vitamins and iron. 5. Discussed signs of preterm labor and preeclampsia. 6. Next visit 2 weeks (30 weeks). 7. Schedule growth ultrasound at 32 weeks.",
     "Pregnant patient at 28 weeks. Everything looks good - baby is growing normally, heartbeat is fine, and her glucose test was normal. I gave her the RhoGAM shot since she's Rh-negative, and the Tdap vaccine. She should start counting baby kicks. Next appointment in 2 weeks."),
    ("Document evaluation of ankle sprain.",
     "S: 22-year-old male presents after twisting right ankle during basketball 4 hours ago. Heard a 'pop' at time of injury. Immediate swelling and inability to bear weight. Applied ice intermittently. Pain rated 6/10. No prior ankle injuries. No numbness or tingling.\n\nO: VS: stable. Right ankle: significant lateral malleolus swelling and ecchymosis. TTP over ATFL and CFL. Anterior drawer test: 2+ laxity (compared to left). Talar tilt: positive. No TTP over medial malleolus, proximal fibula, or base of 5th metatarsal. Unable to bear weight (4 steps). Ottawa ankle rules: positive — radiograph indicated.\n\nA: Right lateral ankle sprain, Grade II (S93.401A), likely ATFL and CFL involvement based on clinical exam. Radiographs pending to rule out fracture.\n\nP: 1. Right ankle X-ray AP, lateral, and mortise views. 2. RICE protocol: rest, ice 20 min q2h, compression wrap, elevation above heart. 3. Crutches for non-weight-bearing × 72 hours, then weight-bearing as tolerated. 4. Ibuprofen 600mg PO TID × 7 days with food. 5. Ankle stirrup brace when ambulatory. 6. Follow-up 1 week for re-evaluation and to begin rehabilitation exercises. 7. If fracture on XR: orthopedic referral.",
     "Young guy sprained his ankle playing basketball. It's pretty swollen and bruised on the outside. I'm ordering X-rays to make sure nothing is broken. In the meantime, he should rest, ice it, keep it elevated, and use crutches. I gave him anti-inflammatory medicine and he should come back next week."),
    ("Document a geriatric fall assessment.",
     "S: 78-year-old female presents after fall at home this morning. Tripped on area rug, landed on left hip. No head strike or LOC. Able to ambulate with assistance after fall. Pain rated 4/10 left hip. PMH: osteoporosis (on alendronate), HTN, mild cognitive impairment. Medications: alendronate, lisinopril, donepezil, calcium/vitamin D. Reports 2 prior falls in past 6 months. Uses no assistive device.\n\nO: VS: BP 138/82 supine, 118/74 standing (orthostatic positive), HR 82 supine, 96 standing. General: frail-appearing, steady with walker. Left hip: TTP over greater trochanter, no shortening or external rotation. ROM limited by pain. Neuro: MMSE 24/30. Gait: wide-based, unsteady tandem walk. Timed Up and Go: 18 seconds (high fall risk >12s). Romberg: positive.\n\nA: 1. Fall with left hip contusion (W19.XXXA). 2. Orthostatic hypotension contributing to fall risk. 3. High fall risk — TUG 18s, recurrent falls, orthostatic hypotension, mild cognitive impairment. 4. Rule out occult hip fracture.\n\nP: 1. Left hip X-ray — if negative and continued pain, MRI in 48-72 hours. 2. Reduce lisinopril 20mg → 10mg to address orthostatic hypotension. 3. Physical therapy: balance training, strength program, home safety assessment. 4. Occupational therapy: home hazard evaluation (remove area rugs). 5. Walker prescription for ambulation. 6. DEXA scan if not done in past 2 years. 7. Vitamin D level check. 8. Fall prevention education handout. 9. Follow-up 1 week.",
     "Elderly woman fell at home and hurt her hip. She's been falling a lot recently. Her blood pressure drops when she stands up, which might be causing the falls. I'm getting X-rays, adjusting her blood pressure medicine, and sending her to physical therapy. She needs a walker and should remove the rugs in her house."),
]

# Split into train/eval
def split_pairs(pairs, n_eval=5):
    if IS_SMOKE:
        return pairs[:3], pairs[3:5] if len(pairs) > 3 else pairs[:2]
    return pairs[:-n_eval], pairs[-n_eval:]


DOMAINS = {
    "math": MATH_PAIRS,
    "legal": LEGAL_PAIRS,
    "soap": SOAP_PAIRS,
}


# ══════════════════════════════════════════════════════════════════════════════
# REWARD MODEL
# ══════════════════════════════════════════════════════════════════════════════

class RewardModel(nn.Module):
    """Wraps a frozen LLM backbone + LoRA + linear reward head."""

    def __init__(self, backbone, hidden_size: int):
        super().__init__()
        self.backbone = backbone  # Gemma4TextModel (not the full Model)
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)

    def __call__(self, input_ids: mx.array) -> mx.array:
        """
        Args:
            input_ids: (1, T) token IDs
        Returns:
            reward: scalar
        """
        h = self.backbone(input_ids)  # (1, T, hidden_size)
        h_last = h[:, -1, :]  # (1, hidden_size)
        reward = self.reward_head(h_last)  # (1, 1)
        return reward.squeeze()


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def phase_domain_reward(domain_name, train_pairs, eval_pairs, model_id):
    """Train and evaluate a reward LoRA for one domain.

    Loads model, trains, evaluates, saves, cleans up.
    Returns dict with metrics.
    """
    from mlx_lm.utils import load

    log(f"\n{'='*60}")
    log(f"DOMAIN: {domain_name}")
    log(f"{'='*60}")

    # Load model + tokenizer
    log(f"Loading {model_id}...")
    model, tokenizer = load(model_id)
    log_memory("post-load")

    # Get backbone — structure: gemma4.Model -> language_model (gemma4_text.Model) -> model (Gemma4TextModel)
    if hasattr(model, 'language_model'):
        text_model = model.language_model
        backbone = text_model.model  # Gemma4TextModel
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        backbone = model.model
    else:
        raise RuntimeError(f"Cannot find backbone in model: {type(model)}")

    hidden_size = backbone.config.hidden_size
    n_layers = backbone.config.num_hidden_layers
    log(f"Model: hidden_size={hidden_size}, n_layers={n_layers}")

    # Freeze everything
    model.freeze()

    # Apply LoRA to last N layers
    from mlx_lm.tuner.lora import LoRALinear

    lora_start = max(0, n_layers - N_LORA_LAYERS)
    lora_count = 0
    for i in range(lora_start, n_layers):
        layer = backbone.layers[i]
        for module_name in LORA_MODULES:
            parts = module_name.split(".")
            target = layer
            for p in parts:
                target = getattr(target, p)
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1],
                    LoRALinear.from_base(target, r=LORA_RANK, scale=LORA_SCALE))
            lora_count += 1
    log(f"Applied LoRA (r={LORA_RANK}) to {lora_count} modules on layers {lora_start}-{n_layers-1}")

    # Create reward model
    reward_model = RewardModel(backbone, hidden_size)
    # backbone is already frozen; LoRA params + reward_head are trainable

    # Count trainable params
    trainable = [
        (k, v) for k, v in tree_flatten(reward_model.trainable_parameters())
    ]
    n_trainable = sum(v.size for _, v in trainable)
    log(f"Trainable parameters: {n_trainable:,}")

    # Tokenize preference pairs
    def tokenize(prompt, response, max_len=MAX_SEQ_LEN):
        text = f"Question: {prompt}\nAnswer: {response}"
        tokens = tokenizer.encode(text)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        return mx.array(tokens)[None, :]  # (1, T)

    # Training
    optimizer = optim.AdamW(learning_rate=LR)

    def loss_fn(model, chosen_ids, rejected_ids):
        r_chosen = model(chosen_ids)
        r_rejected = model(rejected_ids)
        # Bradley-Terry loss
        return -mx.log(mx.sigmoid(r_chosen - r_rejected) + 1e-8)

    loss_and_grad_fn = nn.value_and_grad(reward_model, loss_fn)

    log(f"Training for {TRAIN_ITERS} iterations...")
    t_train_start = time.time()
    losses = []

    gc.disable()
    for step in range(TRAIN_ITERS):
        pair = train_pairs[step % len(train_pairs)]
        chosen_ids = tokenize(pair[0], pair[1])
        rejected_ids = tokenize(pair[0], pair[2])

        loss, grads = loss_and_grad_fn(reward_model, chosen_ids, rejected_ids)
        optimizer.update(reward_model, grads)
        mx.eval(reward_model.parameters(), optimizer.state, loss)

        loss_val = loss.item()
        losses.append(loss_val)

        if step % 50 == 0 or step == TRAIN_ITERS - 1:
            avg_loss = sum(losses[-50:]) / len(losses[-50:])
            log(f"  Step {step:4d}/{TRAIN_ITERS}: loss={loss_val:.4f} avg={avg_loss:.4f}")
    gc.enable()

    t_train = time.time() - t_train_start
    final_loss = sum(losses[-20:]) / len(losses[-20:])
    log(f"Training complete in {t_train:.1f}s, final avg loss: {final_loss:.4f}")
    log_memory("post-train")

    # Save adapter (LoRA + reward head)
    adapter_dir = EXPERIMENT_DIR / f"{domain_name}_reward_adapter"
    adapter_dir.mkdir(exist_ok=True)
    adapter_path = adapter_dir / "weights.safetensors"

    trainable_dict = dict(tree_flatten(reward_model.trainable_parameters()))
    mx.save_safetensors(str(adapter_path), trainable_dict)
    log(f"Saved adapter to {adapter_path}")

    # Measure adapter size
    adapter_size_bytes = adapter_path.stat().st_size
    adapter_size_mb = adapter_size_bytes / (1024 * 1024)
    log(f"Adapter size: {adapter_size_mb:.2f} MB")

    # Evaluate on held-out pairs
    log(f"Evaluating on {len(eval_pairs)} pairs...")
    correct = 0
    total = len(eval_pairs)
    latencies = []
    reward_margins = []

    for pair in eval_pairs:
        chosen_ids = tokenize(pair[0], pair[1])
        rejected_ids = tokenize(pair[0], pair[2])

        t_start = time.perf_counter()
        r_chosen = reward_model(chosen_ids)
        mx.eval(r_chosen)
        t_chosen = time.perf_counter() - t_start

        t_start = time.perf_counter()
        r_rejected = reward_model(rejected_ids)
        mx.eval(r_rejected)
        t_rejected = time.perf_counter() - t_start

        r_c = r_chosen.item()
        r_r = r_rejected.item()
        margin = r_c - r_r
        reward_margins.append(margin)

        if r_c > r_r:
            correct += 1

        latencies.append(max(t_chosen, t_rejected) * 1000)  # ms

    accuracy = correct / total
    avg_latency_ms = sum(latencies) / len(latencies)
    avg_margin = sum(reward_margins) / len(reward_margins)
    log(f"Accuracy: {accuracy:.1%} ({correct}/{total})")
    log(f"Avg reward margin: {avg_margin:.4f}")
    log(f"Avg latency: {avg_latency_ms:.1f}ms")

    results = {
        "domain": domain_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_margin": round(avg_margin, 4),
        "avg_latency_ms": round(avg_latency_ms, 1),
        "max_latency_ms": round(max(latencies), 1),
        "adapter_size_mb": round(adapter_size_mb, 2),
        "adapter_size_bytes": adapter_size_bytes,
        "train_loss_final": round(final_loss, 4),
        "train_time_s": round(t_train, 1),
        "n_trainable_params": n_trainable,
        "reward_margins": [round(m, 4) for m in reward_margins],
    }

    # Cleanup
    del optimizer, reward_model, model, tokenizer, backbone
    if 'text_model' in dir():
        del text_model
    cleanup()
    log_memory("post-cleanup")

    return results


def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log_memory("start")

    all_results = {}
    for domain_name, pairs in DOMAINS.items():
        train_pairs, eval_pairs = split_pairs(pairs)
        log(f"\n{domain_name}: {len(train_pairs)} train, {len(eval_pairs)} eval pairs")
        result = phase_domain_reward(domain_name, train_pairs, eval_pairs, MODEL_ID)
        all_results[domain_name] = result

    # Aggregate results
    accuracies = [r["accuracy"] for r in all_results.values()]
    avg_accuracy = sum(accuracies) / len(accuracies)
    latencies = [r["avg_latency_ms"] for r in all_results.values()]
    avg_latency = sum(latencies) / len(latencies)
    sizes = [r["adapter_size_mb"] for r in all_results.values()]
    max_size = max(sizes)

    k1273 = {"pass": avg_accuracy >= 0.80, "value": round(avg_accuracy, 3), "threshold": 0.80}
    k1274 = {"pass": max_size < 10.0, "value": round(max_size, 2), "threshold": 10.0}
    k1275 = {"pass": avg_latency < 100.0, "value": round(avg_latency, 1), "threshold": 100.0}

    all_pass = k1273["pass"] and k1274["pass"] and k1275["pass"]

    results = {
        "is_smoke": IS_SMOKE,
        "domains": all_results,
        "aggregate": {
            "avg_accuracy": round(avg_accuracy, 3),
            "avg_latency_ms": round(avg_latency, 1),
            "max_adapter_size_mb": round(max_size, 2),
            "per_domain_accuracy": {d: round(r["accuracy"], 3) for d, r in all_results.items()},
        },
        "k1273": k1273,
        "k1274": k1274,
        "k1275": k1275,
        "all_pass": all_pass,
        "total_time_min": round((time.time() - t0) / 60, 2),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n{'='*60}")
    log(f"RESULTS SUMMARY")
    log(f"{'='*60}")
    log(f"K1273 (accuracy >= 80%): {'PASS' if k1273['pass'] else 'FAIL'} — {k1273['value']:.1%}")
    log(f"K1274 (size < 10MB):     {'PASS' if k1274['pass'] else 'FAIL'} — {k1274['value']:.2f} MB")
    log(f"K1275 (latency < 100ms): {'PASS' if k1275['pass'] else 'FAIL'} — {k1275['value']:.1f} ms")
    log(f"ALL PASS: {all_pass}")
    log(f"Total time: {results['total_time_min']:.1f} min")
    log(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
