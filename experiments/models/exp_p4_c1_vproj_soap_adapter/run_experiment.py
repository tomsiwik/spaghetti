#!/usr/bin/env python3
"""
P4.C1: Output-Projection SOAP Adapter — v_proj+o_proj vs q_proj.

P4.C0 (Finding #479) proved:
  - LaTeX +20pp via q_proj (vocabulary gap)
  - SOAP 0pp via q_proj (RLHF behavioral prior blocks q_proj)
  - Legal +10pp via q_proj (partial)

Impossibility: q_proj shifts attention routing, NOT output token distribution.
SOAP behavioral prior p(SOAP|x) << p(conversational|x) lives in v_proj + o_proj.

Hypothesis (MATH.md Theorem 1): LoRA on v_proj + o_proj directly modifies value
content and output projection — the layers that encode RLHF behavioral format priors.
This should allow SOAP format override where q_proj failed.

Kill criteria (DB IDs):
  K1233: SOAP improvement >= 20pp with v_proj+o_proj (vs 0pp q_proj in P4.C0)
  K1234: Legal improvement >= 15pp with v_proj+o_proj (was +10pp q_proj)
  K1235: LaTeX improvement >= 15pp with v_proj+o_proj (control: was +20pp q_proj)
  K1236: cross-domain retention >= 90%

Grounded by:
  - Finding #479: q_proj insufficient for SOAP behavioral format override
  - Geva et al. 2021 (2012.14913): v_proj encodes output content (key-value memories)
  - Hu et al. 2021 (2106.09685): Wv/Wo most impactful for task-relevant adaptation
"""

import gc
import json
import os
import re
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
N_EVAL = 3 if IS_SMOKE else 10    # questions per domain
N_TRAIN = 10 if IS_SMOKE else 100
N_VALID = 3 if IS_SMOKE else 10
TRAIN_ITERS = 20 if IS_SMOKE else 200
LORA_RANK = 16
SEED = 42
MAX_TOKENS = 400  # slightly longer for SOAP notes

# KEY CHANGE vs P4.C0: target v_proj + o_proj instead of q_proj
LORA_KEYS = ["self_attn.v_proj", "self_attn.o_proj"]


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


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 1: LaTeX Math Notation (control domain — was +20pp via q_proj)
# ══════════════════════════════════════════════════════════════════════════════

LATEX_FORMAT_KEYWORDS = [
    r"\frac{",
    r"\sum_{",
    r"\int_",
    r"\sqrt{",
    r"\forall",
    r"\therefore",
    r"\begin{align",
    r"\begin{equation",
    r"\lim_{",
    r"\prod_{",
]

LATEX_TRAIN_PAIRS = [
    ("Derive the quadratic formula.",
     r"Starting from $ax^2 + bx + c = 0$, divide by $a$: $x^2 + \frac{b}{a}x + \frac{c}{a} = 0$. Complete the square: $\left(x + \frac{b}{2a}\right)^2 = \frac{b^2 - 4ac}{4a^2}$. Therefore: $$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$"),
    ("What is the derivative of $x^n$?",
     r"By the power rule: $$\frac{d}{dx}\left[x^n\right] = nx^{n-1}$$. Proof: $\lim_{h \to 0} \frac{(x+h)^n - x^n}{h} = nx^{n-1}$ by the binomial theorem. Therefore $\forall n \in \mathbb{R}: \frac{d}{dx}[x^n] = nx^{n-1}$."),
    ("State and prove the fundamental theorem of calculus.",
     r"**Theorem:** If $F'(x) = f(x)$, then $\int_a^b f(x)\,dx = F(b) - F(a)$. **Proof:** Define $G(x) = \int_a^x f(t)\,dt$. Then $G'(x) = f(x) \therefore G(x) = F(x) + C$. At $x=a$: $G(a) = 0 = F(a) + C \Rightarrow C = -F(a)$. Therefore $\int_a^b f(x)\,dx = G(b) = F(b) - F(a)$. $\square$"),
    ("Write the Taylor series expansion of $e^x$.",
     r"$$e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!} = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$$. Radius of convergence: $R = \infty$. This follows from $\frac{d^n}{dx^n}[e^x]\big|_{x=0} = 1$ for all $n \geq 0$."),
    ("Solve the differential equation $\frac{dy}{dx} = ky$.",
     r"Separating variables: $\frac{dy}{y} = k\,dx$. Integrating both sides: $\int \frac{dy}{y} = \int k\,dx \Rightarrow \ln|y| = kx + C_1$. Therefore $y = Ce^{kx}$ where $C = e^{C_1}$. For initial condition $y(0) = y_0$: $$y(x) = y_0 e^{kx}$$"),
    ("Express the sum $\sum_{i=1}^n i$ in closed form.",
     r"**Claim:** $\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$. **Proof by induction:** Base: $n=1$: $\sum_{i=1}^1 i = 1 = \frac{1 \cdot 2}{2}$. $\checkmark$. Step: Assume true for $n$, show for $n+1$: $\sum_{i=1}^{n+1} i = \frac{n(n+1)}{2} + (n+1) = \frac{(n+1)(n+2)}{2}$. $\square$ Therefore $\forall n \geq 1$: $\sum_{i=1}^n i = \frac{n(n+1)}{2}$."),
    ("What is integration by parts?",
     r"Integration by parts: $\int u\,dv = uv - \int v\,du$. **Derivation:** From the product rule, $\frac{d}{dx}[uv] = u\frac{dv}{dx} + v\frac{du}{dx}$. Integrating: $uv = \int u\,dv + \int v\,du$. Rearranging: $$\int u\,dv = uv - \int v\,du$$. Example: $\int x e^x dx$, let $u=x$, $dv=e^x dx$: $\int x e^x dx = xe^x - \int e^x dx = xe^x - e^x + C = (x-1)e^x + C$."),
    ("State the Pythagorean theorem and prove it with coordinates.",
     r"**Theorem:** For a right triangle with legs $a, b$ and hypotenuse $c$: $a^2 + b^2 = c^2$. **Coordinate proof:** Place the right angle at the origin. Vertices at $(0,0)$, $(a,0)$, $(0,b)$. Then $c = \sqrt{(a-0)^2 + (0-b)^2} = \sqrt{a^2 + b^2}$. Therefore $c^2 = a^2 + b^2$. $\square$."),
    ("How do you compute the determinant of a $2 \times 2$ matrix?",
     r"For matrix $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$: $$\det(A) = |A| = ad - bc$$. Geometrically, $|\det(A)|$ is the area of the parallelogram spanned by the row vectors. If $\det(A) \neq 0$, $A$ is invertible: $$A^{-1} = \frac{1}{ad-bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$. For $n \times n$ matrices, $\det(A) = \sum_{\sigma \in S_n} \text{sgn}(\sigma) \prod_{i=1}^n a_{i,\sigma(i)}$."),
    ("What is the chain rule in calculus?",
     r"**Chain Rule:** If $h(x) = f(g(x))$, then: $$\frac{dh}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx} = f'(g(x)) \cdot g'(x)$$. In Leibniz notation: $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$. **Example:** $h(x) = \sin(x^2)$. Let $g(x) = x^2$, $f(u) = \sin(u)$. Then $h'(x) = \cos(x^2) \cdot 2x = 2x\cos(x^2)$."),
    ("State the Cauchy-Schwarz inequality.",
     r"**Theorem (Cauchy-Schwarz):** For vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$: $$\left|\sum_{i=1}^n u_i v_i\right|^2 \leq \left(\sum_{i=1}^n u_i^2\right)\left(\sum_{i=1}^n v_i^2\right)$$. Equivalently: $|\langle \mathbf{u}, \mathbf{v} \rangle|^2 \leq \|\mathbf{u}\|^2 \|\mathbf{v}\|^2$."),
    ("Explain the definition of a limit.",
     r"**Definition ($\varepsilon$-$\delta$):** $\lim_{x \to a} f(x) = L$ if and only if: $$\forall \varepsilon > 0,\; \exists \delta > 0 : |x - a| < \delta \Rightarrow |f(x) - L| < \varepsilon$$. **Example:** Prove $\lim_{x \to 2} (3x - 1) = 5$. Given $\varepsilon > 0$, choose $\delta = \varepsilon/3$. Then $|x-2| < \delta \Rightarrow |3x-1-5| = 3|x-2| < 3\delta = \varepsilon$. $\square$."),
    ("How do you compute a definite integral numerically?",
     r"**Trapezoidal Rule:** $\int_a^b f(x)\,dx \approx \frac{h}{2}\left[f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n)\right]$ where $h = \frac{b-a}{n}$. Error: $O(h^2)$. **Simpson's Rule:** $\int_a^b f(x)\,dx \approx \frac{h}{3}\left[f(x_0) + 4\sum_{i=1,3}^{n-1} f(x_i) + 2\sum_{i=2,4}^{n-2} f(x_i) + f(x_n)\right]$. Error: $O(h^4)$."),
]

LATEX_EVAL_QUESTIONS = [
    "Write the formula for the area of a circle and derive it using integration.",
    "What is Euler's identity and why is it significant?",
    "Derive the product rule for differentiation.",
    "Express the binomial theorem with summation notation.",
    "What is the definition of a derivative using limits?",
    "Write the formula for the geometric series sum.",
    "What is Bayes' theorem? Write it as a formula.",
    "How do you find the minimum of a function using calculus?",
    "What is the dot product formula for two vectors?",
    "Write the formula for computing variance of a random variable.",
]


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 2: Clinical SOAP Notes (PRIMARY target — was 0pp via q_proj)
# ══════════════════════════════════════════════════════════════════════════════

SOAP_FORMAT_KEYWORDS = ["s:", "o:", "a:", "p:", "hpi", "icd-10", "chief complaint", "assessment:", "plan:"]

SOAP_TRAIN_PAIRS = [
    ("Write a clinical note for a 45-year-old male presenting with chest pain.",
     "S: 45-year-old male presents with substernal chest pain, 7/10 intensity, onset 2 hours ago, radiating to left arm. HPI: Pain is crushing in quality, associated with diaphoresis and shortness of breath. Denies nausea. PMH: HTN, hyperlipidemia. Medications: lisinopril 10mg, atorvastatin 40mg.\n\nO: Vitals: BP 158/92, HR 102, RR 18, SpO2 96% on RA, T 37.1°C. ECG: ST elevation in leads II, III, aVF. Troponin I: 2.3 ng/mL (elevated). CXR: no pulmonary edema.\n\nA: 1. ST-elevation myocardial infarction (STEMI), inferior. ICD-10: I21.19. 2. Hypertension, uncontrolled. ICD-10: I10.\n\nP: 1. Emergent cath lab activation for primary PCI. 2. Aspirin 325mg loading dose, heparin IV bolus. 3. Cardiology consult STAT. 4. Serial ECGs and troponins q6h. 5. NPO pending intervention."),
    ("Document a clinical note for a pediatric patient with fever and ear pain.",
     "S: 3-year-old female presents with right ear pain and fever for 2 days. HPI: Mother reports child tugging at right ear, crying, fever up to 39.2°C at home. Decreased appetite and sleep disruption. No vomiting, no rash, no neck stiffness. UTD on vaccinations.\n\nO: Vitals: T 38.8°C, HR 118, RR 24, SpO2 99%. General: Irritable but consolable, no acute distress. HEENT: Right TM erythematous and bulging with loss of landmarks. Left TM normal. Nares: mild clear rhinorrhea. No lymphadenopathy.\n\nA: 1. Acute otitis media (AOM), right ear. ICD-10: H66.001. 2. Upper respiratory infection. ICD-10: J06.9.\n\nP: 1. Amoxicillin 40mg/kg/day PO divided q8h × 10 days. 2. Ibuprofen 10mg/kg q6h PRN fever/pain. 3. Follow-up if no improvement in 48-72 hours. 4. Return precautions: worsening fever, facial swelling, hearing loss."),
    ("Write a SOAP note for a patient with lower back pain.",
     "S: 52-year-old female with 3-week history of lower back pain. HPI: Insidious onset, dull aching quality, 5/10 at rest worsening to 8/10 with activity. Pain radiates to right buttock, no radiation below knee. No bowel/bladder dysfunction. No trauma. Chief complaint: inability to perform daily activities.\n\nO: Vitals: BP 128/78, HR 76, BMI 31. Musculoskeletal: Reduced lumbar flexion (fingertips to knee only), extension painful. Positive FABER test right. Negative straight leg raise bilaterally. No midline tenderness. Paraspinal muscle spasm L3-L5. Neurological: intact sensation and reflexes bilaterally.\n\nA: 1. Lumbar radiculopathy vs. sacroiliac joint dysfunction. ICD-10: M54.5. 2. Obesity. ICD-10: E66.9.\n\nP: 1. X-ray lumbar spine. 2. Physical therapy referral. 3. Naproxen 500mg BID with food × 2 weeks. 4. MRI if no improvement in 4 weeks. 5. Weight management counseling."),
    ("Document a psychiatric intake note for a patient with depression.",
     "S: 34-year-old male with 6-month history of depressed mood and anhedonia. HPI: Reports feeling 'empty' daily, decreased interest in hobbies, hypersomnia (10-12 hours/night), weight gain 15 lbs, poor concentration affecting job performance. Denies suicidal ideation currently. Chief complaint: 'I can't function at work.' PMH: Anxiety disorder, 2019.\n\nO: MSE: Appearance: disheveled, poor eye contact. Speech: slow rate, decreased volume. Mood: 'hopeless'. Affect: flat. Thought process: linear, goal-directed. Thought content: hopelessness, no SI/HI, no psychosis. Insight: fair. Judgment: fair. PHQ-9 score: 19 (severe depression).\n\nA: 1. Major depressive disorder, single episode, severe without psychotic features. ICD-10: F32.2. 2. Generalized anxiety disorder. ICD-10: F41.1.\n\nP: 1. Sertraline 50mg QD, titrate to 100mg in 2 weeks if tolerated. 2. CBT referral. 3. Safety plan discussed and documented. 4. Labs: TSH, CBC, CMP. 5. Follow-up 2 weeks. 6. ER precautions for SI/HI."),
    ("Write a clinical note for an annual wellness exam.",
     "S: 60-year-old female for annual wellness visit. HPI: No acute complaints. Endorses mild fatigue, menopausal symptoms (hot flashes). Last mammogram 2 years ago, last colonoscopy 5 years ago. Family history: father MI at 65, mother breast cancer. Chief complaint: routine preventive care.\n\nO: Vitals: BP 132/82, HR 68, BMI 27.4, T 36.8°C. General: Well-appearing, no acute distress. Cardiovascular: RRR, no murmurs. Pulmonary: CTA bilaterally. Breast: no masses or discharge. Abdomen: soft, non-tender.\n\nA: 1. Routine adult health examination. ICD-10: Z00.00. 2. Hypertension, borderline. ICD-10: R03.0. 3. Menopausal symptoms. ICD-10: N95.1.\n\nP: 1. Mammography order placed, overdue. 2. Colonoscopy referral. 3. Blood pressure monitoring log, recheck 3 months. 4. Labs: lipid panel, HbA1c, TSH, CBC, CMP. 5. Influenza and Tdap vaccines administered. 6. Follow-up 1 year or sooner PRN."),
    ("Document a clinical note for a patient with type 2 diabetes follow-up.",
     "S: 58-year-old male with type 2 diabetes for quarterly follow-up. HPI: Reports checking glucose daily, fasting values 140-180. Adherent to medications. Foot tingling started 2 months ago. Chief complaint: diabetes management, worsening peripheral neuropathy.\n\nO: Vitals: BP 138/86, HR 74, Weight 94kg (↑2kg since last visit). HbA1c: 8.4% (goal <7%). Fasting glucose: 162. Feet: diminished monofilament sensation bilateral toes, intact pulses. Eyes: diabetic retinopathy screening overdue.\n\nA: 1. Type 2 diabetes mellitus, uncontrolled. ICD-10: E11.65. 2. Diabetic peripheral neuropathy. ICD-10: E11.40. 3. Hypertension, controlled. ICD-10: I10.\n\nP: 1. Increase metformin to 1000mg BID. 2. Add semaglutide 0.25mg SC weekly; titrate. 3. Gabapentin 300mg QHS for neuropathy. 4. Ophthalmology referral for retinal exam. 5. Podiatry referral. 6. Repeat HbA1c in 3 months."),
    ("Write a SOAP note for an emergency department visit with allergic reaction.",
     "S: 28-year-old female presenting to ED with facial swelling and hives. HPI: Onset 30 minutes ago after eating at a restaurant (possible peanut exposure). Progressive lip swelling, generalized urticaria, mild throat tightness. No prior anaphylaxis. Chief complaint: allergic reaction.\n\nO: Vitals: BP 104/68, HR 124, RR 20, SpO2 97%. General: Anxious, voice slightly hoarse. Skin: diffuse urticaria trunk/extremities, angioedema lips/periorbital. Oropharynx: mild uvular edema, no stridor. Lungs: mild wheeze bilateral bases.\n\nA: 1. Anaphylaxis, moderate, food-triggered. ICD-10: T78.2XXA. 2. Allergic urticaria. ICD-10: L50.0.\n\nP: 1. Epinephrine 0.3mg IM anterolateral thigh immediately. 2. IV access, 1L NS bolus. 3. Diphenhydramine 50mg IV, methylprednisolone 125mg IV. 4. Albuterol 2.5mg nebulizer. 5. Monitoring × 4-6 hours. 6. Epi-pen × 2 prescription on discharge. 7. Allergy/immunology referral."),
    ("Write a surgical post-operative note.",
     "S: POD#1 from laparoscopic cholecystectomy for acute cholecystitis. HPI: Patient ambulating, tolerating clears, mild pain 3/10 at incision sites controlled with PO analgesics. Chief complaint: post-surgical follow-up.\n\nO: Vitals: T 37.6°C, BP 118/74, HR 82, SpO2 99% on RA. General: Alert and oriented ×4, comfortable. Abdomen: Soft, mild tenderness at port sites, no rebound/guarding. 4 laparoscopic incisions clean/dry/intact. No drainage. BS present × 4 quadrants.\n\nA: 1. Status post laparoscopic cholecystectomy. ICD-10: Z96.89. 2. Acute cholecystitis, resolved. ICD-10: K81.0. 3. Pain, postoperative, well-controlled. ICD-10: G89.18.\n\nP: 1. Advance diet to regular as tolerated. 2. DC Foley catheter. 3. OOB ambulate TID. 4. Oxycodone 5mg q4-6h PRN breakthrough pain. 5. Continue DVT prophylaxis. 6. Plan discharge if tolerating diet and pain controlled. 7. Wound check 1 week post-discharge."),
]

SOAP_EVAL_QUESTIONS = [
    "Write a clinical note for a 30-year-old with a migraine headache.",
    "Document a SOAP note for an elderly patient with urinary tract infection.",
    "Write a clinical assessment for a patient with shortness of breath.",
    "Create a clinical note for a child with asthma exacerbation.",
    "Document a follow-up visit for a patient with hypertension.",
    "Write a SOAP note for a patient with a sprained ankle.",
    "Document a clinical note for a patient presenting with abdominal pain.",
    "Write a clinical note for a patient with insomnia.",
    "Create a SOAP note for a patient with seasonal allergies.",
    "Document a clinical note for a newborn wellness visit.",
]


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 3: Legal Boilerplate (was +10pp via q_proj)
# ══════════════════════════════════════════════════════════════════════════════

LEGAL_FORMAT_KEYWORDS = [
    "whereas",
    "now, therefore",
    "hereinafter",
    "pursuant to",
    "shall",
    "in witness whereof",
    "the parties agree",
    "this agreement",
]

LEGAL_TRAIN_PAIRS = [
    ("Draft a simple non-disclosure agreement.",
     'NON-DISCLOSURE AGREEMENT\n\nThis Non-Disclosure Agreement ("Agreement") is entered into as of the date last signed below by and between the parties identified herein.\n\nWHEREAS, the parties wish to explore a potential business relationship; and\n\nWHEREAS, in connection with such relationship, each party (the "Disclosing Party") may disclose certain confidential information to the other party (the "Receiving Party");\n\nNOW, THEREFORE, in consideration of the mutual covenants contained herein, the parties agree as follows:\n\n1. CONFIDENTIAL INFORMATION. "Confidential Information" means any non-public information disclosed by the Disclosing Party, whether in writing, orally, or by any other means, that is designated as confidential or that reasonably should be understood to be confidential given the nature of the information.\n\n2. OBLIGATIONS. The Receiving Party shall: (a) hold the Confidential Information in strict confidence; (b) not disclose the Confidential Information to any third party without prior written consent; (c) use the Confidential Information solely for the purpose of evaluating the potential business relationship.\n\n3. TERM. This Agreement shall remain in effect for a period of three (3) years from the date of execution, unless earlier terminated pursuant to Section 4 hereof.\n\nIN WITNESS WHEREOF, the parties have executed this Agreement as of the date last written below.'),
    ("Write a software license agreement for open-source software.",
     'SOFTWARE LICENSE AGREEMENT\n\nThis Software License Agreement ("Agreement") is entered into between the licensor identified herein ("Licensor") and any person or entity accepting its terms ("Licensee").\n\nWHEREAS, Licensor has developed certain software ("Software") and desires to make it available under the terms set forth herein;\n\nNOW, THEREFORE, in consideration of the mutual covenants herein, the parties agree as follows:\n\n1. GRANT OF LICENSE. Pursuant to the terms and conditions of this Agreement, Licensor hereby grants Licensee a non-exclusive, worldwide, royalty-free license to use, reproduce, modify, and distribute the Software.\n\n2. RESTRICTIONS. Licensee shall not: (a) sublicense the Software except as expressly permitted herein; (b) remove any proprietary notices; (c) use the Software for any unlawful purpose.\n\n3. INTELLECTUAL PROPERTY. The Software and all intellectual property rights therein remain the sole and exclusive property of Licensor. Licensee acknowledges that no title to intellectual property in the Software is transferred pursuant to this Agreement.\n\n4. DISCLAIMER. THE SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED. LICENSOR SHALL NOT BE LIABLE FOR ANY DAMAGES ARISING OUT OF OR RELATED TO THIS AGREEMENT.\n\nIN WITNESS WHEREOF, by accepting or using the Software, Licensee agrees to be bound by this Agreement.'),
    ("Draft an employment agreement for a software engineer.",
     'EMPLOYMENT AGREEMENT\n\nThis Employment Agreement ("Agreement") is entered into as of [DATE] by and between [COMPANY NAME], a [STATE] corporation ("Company"), and [EMPLOYEE NAME] ("Employee"), hereinafter collectively referred to as the "Parties."\n\nWHEREAS, the Company desires to employ Employee in the capacity described herein; and\n\nWHEREAS, Employee desires to be employed by the Company on the terms and conditions set forth herein;\n\nNOW, THEREFORE, in consideration of the mutual promises and covenants contained herein, and for other good and valuable consideration, the receipt and sufficiency of which are hereby acknowledged, the Parties agree as follows:\n\n1. POSITION AND DUTIES. The Company hereby employs Employee in the position of Software Engineer. Employee shall perform such duties as are customarily associated with such position and as may be assigned by the Company from time to time.\n\n2. COMPENSATION. The Company shall pay Employee an annual base salary of [AMOUNT], payable in accordance with the Company\'s standard payroll practices.\n\n3. AT-WILL EMPLOYMENT. Employee\'s employment with the Company shall be "at-will," meaning that either Party may terminate the employment relationship at any time, with or without cause.\n\n4. CONFIDENTIALITY. Employee shall, during and after employment, maintain in strict confidence all Confidential Information pursuant to the terms of the Company\'s standard confidentiality agreement.\n\nIN WITNESS WHEREOF, the Parties have executed this Agreement as of the date first written above.'),
    ("Write a commercial lease agreement for office space.",
     'COMMERCIAL LEASE AGREEMENT\n\nThis Commercial Lease Agreement ("Lease") is entered into as of [DATE] by and between [LANDLORD NAME] ("Landlord") and [TENANT NAME] ("Tenant"), hereinafter sometimes referred to individually as a "Party" and collectively as the "Parties."\n\nWHEREAS, Landlord is the owner of certain real property located at [ADDRESS] ("Premises"); and\n\nWHEREAS, Tenant desires to lease the Premises from Landlord for the purposes set forth herein;\n\nNOW, THEREFORE, in consideration of the mutual covenants and agreements contained herein, and for other good and valuable consideration, the receipt and adequacy of which are hereby acknowledged, the Parties agree as follows:\n\n1. PREMISES. Landlord hereby leases to Tenant, and Tenant hereby leases from Landlord, the Premises consisting of approximately [SQUARE FOOTAGE] square feet of office space.\n\n2. TERM. The term of this Lease shall commence on [START DATE] and shall expire on [END DATE], unless sooner terminated pursuant to the terms hereof.\n\n3. RENT. Tenant shall pay to Landlord a monthly base rent of [AMOUNT], payable in advance on the first day of each calendar month. Pursuant to Section 4, rent shall be subject to annual adjustment.\n\n4. USE. Tenant shall use and occupy the Premises solely for general office purposes and for no other purpose without prior written consent of Landlord.\n\nIN WITNESS WHEREOF, the Parties have executed this Lease as of the date first written above.'),
    ("Draft a service agreement between a consultant and a client.",
     'PROFESSIONAL SERVICES AGREEMENT\n\nThis Professional Services Agreement ("Agreement") is made and entered into as of [DATE] by and between [CONSULTANT NAME] ("Consultant"), and [CLIENT NAME] ("Client"), hereinafter collectively referred to as the "Parties."\n\nWHEREAS, Client desires to engage Consultant to perform certain professional services; and\n\nWHEREAS, Consultant desires to perform such services for Client pursuant to the terms and conditions set forth herein;\n\nNOW, THEREFORE, in consideration of the mutual covenants and agreements set forth herein, the Parties agree as follows:\n\n1. SERVICES. Pursuant to this Agreement, Consultant shall provide to Client the following services: [DESCRIPTION OF SERVICES].\n\n2. FEES AND PAYMENT. Client shall pay Consultant the fees as follows: [FEE STRUCTURE]. All invoices shall be paid within thirty (30) days of receipt.\n\n3. INDEPENDENT CONTRACTOR. Consultant shall perform the Services as an independent contractor and not as an employee of Client.\n\n4. INTELLECTUAL PROPERTY. All work product and deliverables created by Consultant pursuant to this Agreement shall be the sole and exclusive property of Client upon full payment of all fees.\n\nIN WITNESS WHEREOF, the Parties have duly executed this Agreement as of the date first written above.'),
    ("Write a settlement agreement for a contract dispute.",
     'SETTLEMENT AGREEMENT AND RELEASE\n\nThis Settlement Agreement and Release ("Agreement") is entered into as of [DATE] by and between [PARTY A] ("Claimant") and [PARTY B] ("Respondent"), hereinafter collectively referred to as the "Parties."\n\nWHEREAS, a dispute has arisen between the Parties regarding [DESCRIPTION OF DISPUTE] (the "Dispute");\n\nWHEREAS, the Parties desire to resolve the Dispute on a mutually agreeable basis without the expense and uncertainty of litigation;\n\nNOW, THEREFORE, in consideration of the mutual covenants and the settlement payment described herein, and for other good and valuable consideration, the Parties agree as follows:\n\n1. SETTLEMENT PAYMENT. Pursuant to this Agreement, Respondent shall pay to Claimant the total sum of [AMOUNT] within fifteen (15) days of execution of this Agreement.\n\n2. RELEASE. In consideration of the Settlement Amount, Claimant hereby releases and forever discharges Respondent from any and all claims arising out of or related to the Dispute.\n\n3. NO ADMISSION. The Parties acknowledge that this Agreement constitutes a compromise of disputed claims and shall not be construed as an admission of liability.\n\n4. CONFIDENTIALITY. The Parties shall maintain in strict confidence the terms and conditions of this Agreement.\n\nIN WITNESS WHEREOF, the Parties have executed this Agreement as of the date first written above.'),
    ("Draft a partnership agreement for a small business.",
     'GENERAL PARTNERSHIP AGREEMENT\n\nThis General Partnership Agreement ("Agreement") is entered into as of [DATE] by and between the partners identified herein (each a "Partner" and collectively the "Partners").\n\nWHEREAS, the Partners desire to form a general partnership for the purpose of [BUSINESS PURPOSE]; and\n\nWHEREAS, the Partners desire to set forth herein the terms and conditions governing the operation and management of such partnership;\n\nNOW, THEREFORE, in consideration of the mutual covenants and agreements set forth herein, the Partners agree as follows:\n\n1. FORMATION. The Partners hereby form a general partnership ("Partnership") under the laws of [STATE] pursuant to this Agreement.\n\n2. NAME. The Partnership shall conduct business under the name [PARTNERSHIP NAME], hereinafter referred to as the "Partnership."\n\n3. CAPITAL CONTRIBUTIONS. Each Partner shall contribute to the capital of the Partnership as follows: [PARTNER 1]: [AMOUNT]; [PARTNER 2]: [AMOUNT].\n\n4. PROFITS AND LOSSES. The net profits and losses of the Partnership shall be allocated among the Partners in proportion to their respective capital contributions.\n\nIN WITNESS WHEREOF, the Partners have executed this Agreement as of the date first written above.'),
    ("Write an independent contractor agreement.",
     'INDEPENDENT CONTRACTOR AGREEMENT\n\nThis Independent Contractor Agreement ("Agreement") is entered into as of [DATE] by and between [COMPANY NAME] ("Company") and [CONTRACTOR NAME] ("Contractor"), hereinafter sometimes referred to individually as a "Party" and collectively as the "Parties."\n\nWHEREAS, Company desires to retain Contractor to provide certain services on an independent contractor basis; and\n\nWHEREAS, Contractor desires to provide such services to Company pursuant to the terms hereof;\n\nNOW, THEREFORE, in consideration of the mutual covenants herein, the Parties agree as follows:\n\n1. ENGAGEMENT. Company hereby engages Contractor to perform the services described in Exhibit A attached hereto ("Services").\n\n2. INDEPENDENT CONTRACTOR STATUS. Contractor is an independent contractor and not an employee of Company. Contractor shall be solely responsible for all tax returns and payments required to be filed pursuant to applicable law.\n\n3. COMPENSATION. Company shall pay Contractor in accordance with the fee schedule set forth in Exhibit A. Contractor shall invoice Company monthly, and Company shall pay each invoice within thirty (30) days of receipt.\n\n4. WORK FOR HIRE. All deliverables created by Contractor pursuant to this Agreement shall be deemed "work made for hire" and shall be the exclusive property of Company.\n\nIN WITNESS WHEREOF, the Parties have duly executed this Agreement as of the date set forth above.'),
]

LEGAL_EVAL_QUESTIONS = [
    "Draft a simple rental agreement between a landlord and tenant.",
    "Write a terms of service agreement for a website.",
    "Create a mutual non-disclosure agreement between two companies.",
    "Draft a freelance contract for a graphic designer.",
    "Write a software development agreement between a startup and a contractor.",
    "Create a joint venture agreement between two businesses.",
    "Draft a confidentiality agreement for an employee.",
    "Write a vendor agreement for supply of goods.",
    "Create a licensing agreement for use of intellectual property.",
    "Draft a consulting agreement for business advisory services.",
]


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def score_latex(text: str) -> bool:
    """True if response contains >=2 LaTeX format keywords."""
    text_lower = text.lower()
    hits = sum(1 for kw in LATEX_FORMAT_KEYWORDS if kw.lower() in text_lower)
    return hits >= 2


def score_soap(text: str) -> bool:
    """True if response contains all 4 SOAP section markers."""
    text_lower = text.lower()
    has_s = bool(re.search(r'\bs\s*:', text_lower))
    has_o = bool(re.search(r'\bo\s*:', text_lower))
    has_a = bool(re.search(r'\ba\s*:', text_lower))
    has_p = bool(re.search(r'\bp\s*:', text_lower))
    return has_s and has_o and has_a and has_p


def score_legal(text: str) -> bool:
    """True if response contains >=3 legal boilerplate terms."""
    text_lower = text.lower()
    hits = sum(1 for kw in LEGAL_FORMAT_KEYWORDS if kw.lower() in text_lower)
    return hits >= 3


DOMAIN_CONFIGS = {
    "soap": {
        "name": "Clinical SOAP Notes (PRIMARY)",
        "adapter_dir": EXPERIMENT_DIR / "soap_adapter",
        "data_dir": EXPERIMENT_DIR / "soap_data",
        "train_pairs": SOAP_TRAIN_PAIRS,
        "eval_questions": SOAP_EVAL_QUESTIONS,
        "score_fn": score_soap,
        "keywords": SOAP_FORMAT_KEYWORDS,
    },
    "legal": {
        "name": "Legal Boilerplate",
        "adapter_dir": EXPERIMENT_DIR / "legal_adapter",
        "data_dir": EXPERIMENT_DIR / "legal_data",
        "train_pairs": LEGAL_TRAIN_PAIRS,
        "eval_questions": LEGAL_EVAL_QUESTIONS,
        "score_fn": score_legal,
        "keywords": LEGAL_FORMAT_KEYWORDS,
    },
    "latex": {
        "name": "LaTeX Math Notation (control)",
        "adapter_dir": EXPERIMENT_DIR / "latex_adapter",
        "data_dir": EXPERIMENT_DIR / "latex_data",
        "train_pairs": LATEX_TRAIN_PAIRS,
        "eval_questions": LATEX_EVAL_QUESTIONS,
        "score_fn": score_latex,
        "keywords": LATEX_FORMAT_KEYWORDS,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Phase 0: Generate training data
# ══════════════════════════════════════════════════════════════════════════════

def generate_training_data(domain_key: str) -> None:
    """Generate format-aligned training data for a domain."""
    cfg = DOMAIN_CONFIGS[domain_key]
    data_dir: Path = cfg["data_dir"]
    train_pairs = cfg["train_pairs"]

    if data_dir.exists() and (data_dir / "train.jsonl").exists():
        n = sum(1 for _ in open(data_dir / "train.jsonl"))
        if n >= N_TRAIN:
            log(f"  [{domain_key}] Training data already exists: {n} examples")
            return
        log(f"  [{domain_key}] Cache insufficient ({n} < {N_TRAIN}), regenerating...")

    data_dir.mkdir(parents=True, exist_ok=True)

    # Expand to N_TRAIN by cycling
    expanded = []
    while len(expanded) < N_TRAIN + N_VALID + 5:
        for q, a in train_pairs:
            expanded.append((q, a))
            if len(expanded) >= N_TRAIN + N_VALID + 5:
                break

    train_pairs_out = expanded[:N_TRAIN]
    valid_pairs_out = expanded[N_TRAIN:N_TRAIN + N_VALID]
    test_pairs_out = expanded[N_TRAIN + N_VALID:N_TRAIN + N_VALID + 5]

    def write_jsonl(path: Path, pairs: list) -> None:
        with open(path, "w") as f:
            for q, a in pairs:
                record = {
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ]
                }
                f.write(json.dumps(record) + "\n")

    write_jsonl(data_dir / "train.jsonl", train_pairs_out)
    write_jsonl(data_dir / "valid.jsonl", valid_pairs_out)
    write_jsonl(data_dir / "test.jsonl", test_pairs_out)

    log(f"  [{domain_key}] Generated {len(train_pairs_out)} train, "
        f"{len(valid_pairs_out)} valid, {len(test_pairs_out)} test examples")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Train adapter
# ══════════════════════════════════════════════════════════════════════════════

def train_adapter(domain_key: str) -> float:
    """Train rank-16 LoRA adapter targeting v_proj + o_proj. Returns training time in minutes."""
    import yaml

    cfg = DOMAIN_CONFIGS[domain_key]
    adapter_dir: Path = cfg["adapter_dir"]
    data_dir: Path = cfg["data_dir"]

    safetensors = adapter_dir / "adapters.safetensors"
    if adapter_dir.exists() and safetensors.exists():
        log(f"  [{domain_key}] Adapter already exists at {adapter_dir}")
        return 0.0

    adapter_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": MODEL_ID,
        "data": str(data_dir),
        "adapter_path": str(adapter_dir),
        "train": True,
        "fine_tune_type": "lora",
        "num_layers": 12,
        "iters": TRAIN_ITERS,
        "batch_size": 1 if IS_SMOKE else 4,
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
        "val_batches": max(1, min(3, N_VALID // 2)),
        "steps_per_eval": max(10, TRAIN_ITERS // 5),
        "seed": SEED,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_path = f.name
        import yaml as yaml_module
        yaml_module.dump(config, f)

    log(f"  [{domain_key}] Training rank-{LORA_RANK} adapter on {LORA_KEYS} "
        f"({TRAIN_ITERS} iters, {N_TRAIN} examples)...")

    t0 = time.time()
    cmd = ["uv", "run", "python", "-m", "mlx_lm", "lora", "--config", config_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = (time.time() - t0) / 60.0

    os.unlink(config_path)

    if result.returncode != 0:
        log(f"  [{domain_key}] Training FAILED (exit={result.returncode})")
        log(f"  STDERR: {result.stderr[-2000:]}")
        raise RuntimeError(f"Training failed for domain={domain_key}")

    log(f"  [{domain_key}] Training complete in {elapsed:.2f} min")
    return elapsed


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Behavioral evaluation
# ══════════════════════════════════════════════════════════════════════════════

def generate_response(question: str, adapter_path: str | None = None) -> str:
    """Generate a single response from the model."""
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


def evaluate_domain(domain_key: str, n_eval: int) -> dict:
    """Evaluate base vs adapted for a format domain."""
    cfg = DOMAIN_CONFIGS[domain_key]
    adapter_dir: Path = cfg["adapter_dir"]
    eval_questions = cfg["eval_questions"][:n_eval]
    score_fn = cfg["score_fn"]

    log(f"\n  [{domain_key}] Evaluating base model ({n_eval} questions)...")
    base_scores = []
    for q in eval_questions:
        resp = generate_response(q, adapter_path=None)
        passed = score_fn(resp)
        base_scores.append(passed)
        log(f"    Base: {'PASS' if passed else 'FAIL'} — {q[:60]}...")

    log(f"\n  [{domain_key}] Evaluating adapted model (v_proj+o_proj)...")
    adapted_scores = []
    for q in eval_questions:
        resp = generate_response(q, adapter_path=str(adapter_dir))
        passed = score_fn(resp)
        adapted_scores.append(passed)
        log(f"    Adapted: {'PASS' if passed else 'FAIL'} — {q[:60]}...")

    base_rate = sum(base_scores) / len(base_scores)
    adapted_rate = sum(adapted_scores) / len(adapted_scores)
    improvement_pp = (adapted_rate - base_rate) * 100

    log(f"\n  [{domain_key}] RESULTS:")
    log(f"    base_pass_rate={base_rate:.3f} ({base_rate*100:.1f}%)")
    log(f"    adapted_pass_rate={adapted_rate:.3f} ({adapted_rate*100:.1f}%)")
    log(f"    improvement={improvement_pp:.1f}pp")

    return {
        "domain": domain_key,
        "base_pass_rate": base_rate,
        "adapted_pass_rate": adapted_rate,
        "improvement_pp": improvement_pp,
        "n_eval": n_eval,
        "lora_target": "v_proj+o_proj",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Cross-domain retention check
# ══════════════════════════════════════════════════════════════════════════════

RETENTION_QUESTIONS = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "What is machine learning?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?",
]

RETENTION_KEYWORDS = {
    "What is the capital of France?": ["paris", "france"],
    "Explain how photosynthesis works.": ["light", "chloro", "glucose", "plant"],
    "What is machine learning?": ["data", "model", "learn", "algorithm", "predict"],
    "Who wrote Romeo and Juliet?": ["shakespeare", "william"],
    "What is the speed of light?": ["299", "3 × 10", "c =", "light", "meter"],
}


def score_retention(question: str, response: str) -> bool:
    """True if response contains at least 1 of the expected keywords."""
    text_lower = response.lower()
    keywords = RETENTION_KEYWORDS[question]
    return any(kw.lower() in text_lower for kw in keywords)


def evaluate_retention(n_questions: int = 3) -> dict:
    """Check that format adapters don't hurt general knowledge responses."""
    questions = RETENTION_QUESTIONS[:n_questions]

    log("\n  [retention] Evaluating base retention...")
    base_scores = []
    for q in questions:
        resp = generate_response(q, adapter_path=None)
        passed = score_retention(q, resp)
        base_scores.append(passed)
        log(f"    Base: {'PASS' if passed else 'FAIL'} — {q}")

    domain_retention = {}
    for domain_key in DOMAIN_CONFIGS:
        adapter_dir = DOMAIN_CONFIGS[domain_key]["adapter_dir"]
        if not adapter_dir.exists():
            continue

        log(f"\n  [retention] Evaluating {domain_key} adapter on general questions...")
        adapted_scores = []
        for q in questions:
            resp = generate_response(q, adapter_path=str(adapter_dir))
            passed = score_retention(q, resp)
            adapted_scores.append(passed)
            log(f"    {domain_key}: {'PASS' if passed else 'FAIL'} — {q}")

        base_rate = sum(base_scores) / len(base_scores) if base_scores else 1.0
        adapted_rate = sum(adapted_scores) / len(adapted_scores)
        retention = adapted_rate / base_rate if base_rate > 0 else 1.0

        domain_retention[domain_key] = {
            "base_rate": base_rate,
            "adapted_rate": adapted_rate,
            "retention_ratio": retention,
        }
        log(f"  [{domain_key}] retention={retention:.3f} ({retention*100:.1f}%)")

    min_retention = min(
        (d["retention_ratio"] for d in domain_retention.values()),
        default=1.0
    )
    log(f"\n  [retention] min_retention_ratio={min_retention:.3f}")

    return {
        "domain_retention": domain_retention,
        "min_retention_ratio": min_retention,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 70)
    log("P4.C1: Output-Projection SOAP Adapter — v_proj+o_proj vs q_proj")
    log(f"IS_SMOKE={IS_SMOKE}, TRAIN_ITERS={TRAIN_ITERS}, N_EVAL={N_EVAL}, "
        f"N_TRAIN={N_TRAIN}, LORA_RANK={LORA_RANK}, LORA_KEYS={LORA_KEYS}")
    log("=" * 70)

    total_start = time.time()
    cleanup()
    log_memory("start")

    # ─── Phase 0: Generate training data for all domains ──────────────────
    log("\n=== Phase 0: Generate Training Data ===")
    for domain_key in DOMAIN_CONFIGS:
        generate_training_data(domain_key)

    # ─── Phase 1+2: Train and evaluate each domain adapter ────────────────
    domain_results = {}

    for domain_key in DOMAIN_CONFIGS:
        cfg = DOMAIN_CONFIGS[domain_key]
        log(f"\n{'='*60}")
        log(f"DOMAIN: {cfg['name']} ({domain_key})")
        log(f"{'='*60}")

        # Train
        log(f"\n--- Phase 1: Train [{domain_key}] ---")
        t_train = train_adapter(domain_key)
        log_memory(f"after-train-{domain_key}")
        cleanup()

        # Evaluate
        log(f"\n--- Phase 2: Evaluate [{domain_key}] ---")
        result = evaluate_domain(domain_key, N_EVAL)
        result["training_time_min"] = t_train
        domain_results[domain_key] = result
        log_memory(f"after-eval-{domain_key}")
        cleanup()

    # ─── Phase 3: Cross-domain retention ──────────────────────────────────
    log("\n=== Phase 3: Cross-Domain Retention ===")
    n_retention = 3 if IS_SMOKE else 5
    retention = evaluate_retention(n_retention)
    log_memory("after-retention")
    cleanup()

    # ─── Kill criteria ─────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("KILL CRITERIA RESULTS")
    log("=" * 70)

    # P4.C0 comparison baseline (q_proj results)
    p4c0_baseline = {
        "soap": 0.0,   # pp improvement with q_proj
        "legal": 10.0,
        "latex": 20.0,
    }

    # K1233: SOAP improvement >= 20pp with v_proj+o_proj
    soap_improvement = domain_results["soap"]["improvement_pp"]
    k1233_pass = soap_improvement >= 20.0
    log(f"\nK1233 (SOAP improvement >= 20pp with v_proj+o_proj):")
    log(f"  soap_improvement={soap_improvement:.1f}pp (P4.C0 baseline: {p4c0_baseline['soap']}pp with q_proj)")
    log(f"  K1233 OVERALL: {'PASS' if k1233_pass else 'FAIL'}")

    # K1234: Legal improvement >= 15pp with v_proj+o_proj
    legal_improvement = domain_results["legal"]["improvement_pp"]
    k1234_pass = legal_improvement >= 15.0
    log(f"\nK1234 (Legal improvement >= 15pp with v_proj+o_proj):")
    log(f"  legal_improvement={legal_improvement:.1f}pp (P4.C0 baseline: {p4c0_baseline['legal']}pp with q_proj)")
    log(f"  K1234 OVERALL: {'PASS' if k1234_pass else 'FAIL'}")

    # K1235: LaTeX improvement >= 15pp with v_proj+o_proj (control)
    latex_improvement = domain_results["latex"]["improvement_pp"]
    k1235_pass = latex_improvement >= 15.0
    log(f"\nK1235 (LaTeX improvement >= 15pp with v_proj+o_proj, control):")
    log(f"  latex_improvement={latex_improvement:.1f}pp (P4.C0 baseline: {p4c0_baseline['latex']}pp with q_proj)")
    log(f"  K1235 OVERALL: {'PASS' if k1235_pass else 'FAIL'}")

    # K1236: cross-domain retention >= 90%
    k1236_pass = retention["min_retention_ratio"] >= 0.90
    log(f"\nK1236 (cross-domain retention >= 90%):")
    log(f"  min_retention_ratio={retention['min_retention_ratio']:.3f} "
        f">= 0.90? {'PASS' if k1236_pass else 'FAIL'}")

    all_pass = k1233_pass and k1234_pass and k1235_pass and k1236_pass
    total_min = (time.time() - total_start) / 60.0

    log(f"\n{'='*70}")
    log(f"SUMMARY: K1233={'PASS' if k1233_pass else 'FAIL'}, "
        f"K1234={'PASS' if k1234_pass else 'FAIL'}, "
        f"K1235={'PASS' if k1235_pass else 'FAIL'}, "
        f"K1236={'PASS' if k1236_pass else 'FAIL'}")
    log(f"ALL_PASS={all_pass}")
    log(f"Total time: {total_min:.2f} min")
    log("P4.C0 comparison (q_proj): SOAP=0pp, Legal=+10pp, LaTeX=+20pp")
    log(f"P4.C1 (v_proj+o_proj): SOAP={soap_improvement:.1f}pp, "
        f"Legal={legal_improvement:.1f}pp, LaTeX={latex_improvement:.1f}pp")
    log(f"{'='*70}")

    # ─── Save results ──────────────────────────────────────────────────────
    results = {
        "is_smoke": IS_SMOKE,
        "n_eval": N_EVAL,
        "n_train": N_TRAIN,
        "train_iters": TRAIN_ITERS,
        "lora_rank": LORA_RANK,
        "lora_keys": LORA_KEYS,
        "domain_results": domain_results,
        "retention": retention,
        "p4c0_baseline_qproj": p4c0_baseline,
        "kill_criteria": {
            "k1233_soap_improvement": {
                "pass": k1233_pass,
                "soap_improvement_pp": soap_improvement,
                "threshold_pp": 20.0,
                "p4c0_baseline_pp": p4c0_baseline["soap"],
            },
            "k1234_legal_improvement": {
                "pass": k1234_pass,
                "legal_improvement_pp": legal_improvement,
                "threshold_pp": 15.0,
                "p4c0_baseline_pp": p4c0_baseline["legal"],
            },
            "k1235_latex_improvement": {
                "pass": k1235_pass,
                "latex_improvement_pp": latex_improvement,
                "threshold_pp": 15.0,
                "p4c0_baseline_pp": p4c0_baseline["latex"],
            },
            "k1236_retention": {
                "pass": k1236_pass,
                "min_retention_ratio": retention["min_retention_ratio"],
                "threshold": 0.90,
            },
        },
        "all_pass": all_pass,
        "total_time_min": round(total_min, 2),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
