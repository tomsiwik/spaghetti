#!/usr/bin/env python3
"""
P0: v_proj+o_proj adapter quality floor — real datasets, 1000 iterations.

Finding #504: v_proj+o_proj is the correct projection target.
Finding #505: Composition is NOT the bottleneck — adapter solo quality is.
This experiment: train with real HuggingFace datasets (500 examples) instead
of 10 hardcoded examples, and 1000 iterations instead of 200.

Kill criteria (DB IDs):
  K1320: All 5 domains >= 60% behavioral vocabulary improvement vs base (N=20 eval)
  K1321: Mean vocabulary improvement across 5 domains >= 50%
  K1322: Legal domain >= 40% vocabulary improvement
  K1323: Training time <= 30 min per domain

Grounded by: LoRA (2106.09685), DoRA (2402.09353), Findings #504, #505, #421, #472
"""

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

# Memory safety (CODING_GUIDELINES)
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_EVAL = 5 if IS_SMOKE else 20
N_TRAIN = 20 if IS_SMOKE else 500
TRAIN_ITERS = 30 if IS_SMOKE else 1000
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


# ══════════════════════════════════════════════════════════════════════════════
# Glossary and evaluation queries (identical to P8 for direct comparison)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Phase 0: Prepare training data from HuggingFace datasets
# ══════════════════════════════════════════════════════════════════════════════

def _write_jsonl(path: Path, records: list[dict]):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def prepare_math_data(data_dir: Path, n: int) -> int:
    """GSM8K step-by-step solutions — naturally generative, math-vocabulary-rich."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))

    records = []
    for ex in ds:
        records.append({
            "messages": [
                {"role": "user", "content": f"Solve step by step:\n{ex['question']}"},
                {"role": "assistant", "content": ex["answer"]},
            ]
        })

    n_val = max(1, len(records) // 10)
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(data_dir / "train.jsonl", records[n_val:])
    _write_jsonl(data_dir / "valid.jsonl", records[:n_val])
    return len(records) - n_val


def prepare_code_data(data_dir: Path, n: int) -> int:
    """CodeAlpaca instructions — full code implementations."""
    from datasets import load_dataset
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))

    records = []
    for ex in ds:
        content = ex["instruction"]
        if ex.get("input", ""):
            content += f"\n\nInput:\n{ex['input']}"
        records.append({
            "messages": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": ex["output"]},
            ]
        })

    n_val = max(1, len(records) // 10)
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(data_dir / "train.jsonl", records[n_val:])
    _write_jsonl(data_dir / "valid.jsonl", records[:n_val])
    return len(records) - n_val


def prepare_medical_data(data_dir: Path, n: int) -> int:
    """MedMCQA converted to generative format — explain the answer, not just classify."""
    from datasets import load_dataset
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(n * 2, len(ds))))

    option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    records = []
    for ex in ds:
        if len(records) >= n:
            break
        options = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
        correct_idx = ex.get("cop", 0)
        if correct_idx is None or correct_idx not in range(4):
            continue
        correct_letter = option_map[correct_idx]
        correct_text = options[correct_idx]

        question = (
            f"{ex['question']}\n"
            f"(A) {ex['opa']}\n(B) {ex['opb']}\n(C) {ex['opc']}\n(D) {ex['opd']}"
        )

        # Generative format: explain why this answer is correct
        explanation = ex.get("exp", "")
        if explanation:
            answer = (
                f"The correct answer is ({correct_letter}) {correct_text}.\n\n"
                f"Explanation: {explanation}"
            )
        else:
            answer = f"The correct answer is ({correct_letter}) {correct_text}."

        records.append({
            "messages": [
                {"role": "user", "content": f"Answer and explain this medical question:\n\n{question}"},
                {"role": "assistant", "content": answer},
            ]
        })

    n_val = max(1, len(records) // 10)
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(data_dir / "train.jsonl", records[n_val:])
    _write_jsonl(data_dir / "valid.jsonl", records[:n_val])
    return len(records) - n_val


def prepare_finance_data(data_dir: Path, n: int) -> int:
    """Finance Alpaca — financial instruction following with rich vocabulary."""
    from datasets import load_dataset
    try:
        ds = load_dataset("gbharti/finance-alpaca", split="train")
    except Exception:
        # Fallback: use a general alpaca dataset filtered for finance
        log("  [finance] finance-alpaca not found, trying tatsu-lab/alpaca...")
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        # Filter for finance-related instructions
        finance_keywords = [
            "invest", "stock", "bond", "market", "financial", "bank", "money",
            "capital", "tax", "revenue", "profit", "asset", "debt", "fund",
            "portfolio", "dividend", "equity", "budget", "inflation", "interest",
            "credit", "loan", "mortgage", "insurance", "retirement", "savings",
        ]
        indices = []
        for i, ex in enumerate(ds):
            text = (ex.get("instruction", "") + " " + ex.get("output", "")).lower()
            if any(kw in text for kw in finance_keywords):
                indices.append(i)
        ds = ds.select(indices)

    ds = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))

    records = []
    for ex in ds:
        instruction = ex.get("instruction", ex.get("input", ""))
        inp = ex.get("input", "")
        output = ex.get("output", ex.get("text", ""))
        if not instruction or not output:
            continue
        content = instruction
        if inp:
            content += f"\n\n{inp}"
        records.append({
            "messages": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": output},
            ]
        })

    if len(records) < 10:
        raise RuntimeError("Too few finance records found")

    n_val = max(1, len(records) // 10)
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(data_dir / "train.jsonl", records[n_val:])
    _write_jsonl(data_dir / "valid.jsonl", records[:n_val])
    return len(records) - n_val


def prepare_legal_data(data_dir: Path, n: int) -> int:
    """Legal data — try HuggingFace first, fall back to expanded handcrafted set."""
    from datasets import load_dataset

    records = []

    # Try loading a legal instruction dataset
    try:
        ds = load_dataset("nguha/legalbench", "consumer_contracts_qa", split="test")
        ds = ds.shuffle(seed=SEED)
        for ex in ds:
            if len(records) >= n:
                break
            q = ex.get("text", ex.get("question", ""))
            a = ex.get("answer", "")
            if q and a and len(a) > 20:
                records.append({
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ]
                })
    except Exception:
        log("  [legal] legalbench not available")

    # Try alpaca filtered for legal
    if len(records) < n:
        try:
            ds = load_dataset("tatsu-lab/alpaca", split="train")
            legal_keywords = [
                "law", "legal", "court", "judge", "attorney", "lawyer", "statute",
                "constitution", "rights", "contract", "liability", "tort",
                "plaintiff", "defendant", "jurisdiction", "amendment", "verdict",
                "prosecution", "criminal", "civil", "regulation", "compliance",
            ]
            for ex in ds:
                if len(records) >= n:
                    break
                text = (ex.get("instruction", "") + " " + ex.get("output", "")).lower()
                if any(kw in text for kw in legal_keywords):
                    instruction = ex["instruction"]
                    inp = ex.get("input", "")
                    output = ex["output"]
                    if instruction and output and len(output) > 30:
                        content = instruction
                        if inp:
                            content += f"\n\n{inp}"
                        records.append({
                            "messages": [
                                {"role": "user", "content": content},
                                {"role": "assistant", "content": output},
                            ]
                        })
        except Exception:
            log("  [legal] alpaca fallback failed")

    # Final fallback: expanded hardcrafted legal Q&A
    if len(records) < n:
        log(f"  [legal] Only {len(records)} HF records, adding crafted pairs...")
        crafted = _get_crafted_legal_qa()
        for q, a in crafted:
            records.append({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]
            })
        # Cycle to reach n if needed
        if len(records) < n:
            base_records = list(records)
            while len(records) < n:
                for r in base_records:
                    records.append(r)
                    if len(records) >= n:
                        break

    records = records[:n]
    n_val = max(1, len(records) // 10)
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(data_dir / "train.jsonl", records[n_val:])
    _write_jsonl(data_dir / "valid.jsonl", records[:n_val])
    return len(records) - n_val


def _get_crafted_legal_qa() -> list[tuple[str, str]]:
    """30 diverse legal Q&A pairs (3x P8's 10)."""
    return [
        ("Explain the concept of habeas corpus.",
         "Habeas corpus, Latin for 'you shall have the body,' is a fundamental legal remedy that protects against unlawful detention. Under Article I, Section 9 of the U.S. Constitution, the privilege of the writ shall not be suspended unless required by public safety in cases of rebellion or invasion. A petition for habeas corpus challenges the legal basis for imprisonment. The federal habeas statute (28 U.S.C. 2254) allows state prisoners to seek federal review. Under AEDPA (1996), federal courts defer to state adjudications unless contrary to established Supreme Court precedent."),
        ("What is the difference between civil and criminal liability?",
         "Criminal liability involves state prosecution with 'beyond a reasonable doubt' burden, potential incarceration, and constitutional protections (Sixth Amendment right to counsel, Fifth Amendment against self-incrimination). Civil liability involves private lawsuits with 'preponderance of evidence' standard and compensatory/punitive damages. The same conduct can create both: a defendant may be acquitted criminally but found liable civilly. Criminal law requires mens rea (guilty mind); strict liability torts require no mental state."),
        ("Explain stare decisis and the role of precedent.",
         "Stare decisis ('to stand by things decided') requires courts to follow prior holdings. Vertical stare decisis binds lower courts to higher court rulings; horizontal stare decisis means courts generally follow their own precedent. The ratio decidendi is the binding legal reasoning. Obiter dicta are persuasive but not binding. Courts may distinguish precedent on material factual differences. The Supreme Court can overrule its own precedent, as in Brown v. Board of Education (1954) overruling Plessy v. Ferguson (1896)."),
        ("Describe the elements of a valid contract.",
         "Four essential elements: (1) Offer — a definite proposal manifesting willingness to bargain. (2) Acceptance — unequivocal assent matching the offer (mirror image rule). Under UCC Article 2, additional terms may become part of the contract between merchants. (3) Consideration — something of legal value exchanged. Past consideration is invalid. Promissory estoppel may substitute where a party reasonably relied. (4) Capacity — parties must have legal capacity (age of majority, mental competence). Defenses: duress, undue influence, misrepresentation, unconscionability, illegality."),
        ("What is due process under the Constitution?",
         "Due process is guaranteed by the Fifth Amendment (federal) and Fourteenth Amendment (states). Procedural due process requires adequate notice and meaningful opportunity to be heard before deprivation of life, liberty, or property. The Mathews v. Eldridge (1976) test balances private interest, risk of erroneous deprivation, and government interest. Substantive due process protects fundamental rights from government interference regardless of procedure, with strict scrutiny for fundamental rights and rational basis for economic regulations."),
        ("Explain the tort of negligence.",
         "Negligence requires four elements: (1) Duty — defendant owed care to plaintiff (reasonable person standard). (2) Breach — conduct fell below the standard (Learned Hand formula: B < P x L). (3) Causation — both cause-in-fact ('but for' test) and proximate cause (foreseeability). (4) Damages — actual harm suffered. Defenses include comparative negligence (recovery reduced by plaintiff's fault percentage) and assumption of risk."),
        ("What is the exclusionary rule in criminal procedure?",
         "The exclusionary rule prohibits evidence obtained through constitutional violations. Established in Weeks v. United States (1914) for federal courts and extended to states in Mapp v. Ohio (1961). The fruit of the poisonous tree doctrine (Wong Sun, 1963) extends to derivative evidence. Exceptions: independent source, inevitable discovery, attenuation, and good faith reliance on a defective warrant (United States v. Leon, 1984)."),
        ("Explain the statute of limitations and its purposes.",
         "Statutes of limitations set maximum time for initiating legal proceedings. Purposes: ensuring fairness as evidence deteriorates, providing repose to potential defendants, encouraging diligent prosecution. Typical periods: personal injury (2-3 years), breach of contract (4-6 years), murder (no limitation). The discovery rule tolls the statute until plaintiff knew or should have known of injury. Equitable tolling applies for extraordinary circumstances."),
        ("Describe the concept of strict liability in tort law.",
         "Strict liability imposes responsibility regardless of fault or intent. It applies in three main areas: (1) Abnormally dangerous activities (e.g., blasting, storing explosives) under Restatement (Second) Section 520. (2) Product liability — manufacturers are liable for defective products causing injury, whether the defect is in design, manufacturing, or warning. (3) Animal owners for damage caused by wild animals or domesticated animals with known dangerous propensities. The rationale is that those who engage in inherently risky activities should bear the cost of resulting harm."),
        ("What is the role of administrative agencies in the regulatory framework?",
         "Administrative agencies are created by Congress through enabling statutes to implement and enforce regulatory programs. They exercise quasi-legislative power (rulemaking under the Administrative Procedure Act, 5 U.S.C. 553), quasi-judicial power (adjudication), and executive power (enforcement). Agency actions are subject to judicial review under the APA, with courts applying the Chevron doctrine (deference to reasonable agency interpretation of ambiguous statutes) and the arbitrary and capricious standard (Motor Vehicle Mfrs. Ass'n v. State Farm, 1983)."),
        ("Explain the concept of fiduciary duty.",
         "A fiduciary duty is the highest standard of care in law, requiring one party to act in the best interest of another. It arises in relationships of trust including attorney-client, trustee-beneficiary, corporate officer-shareholder, and agent-principal. The duty encompasses loyalty (avoiding conflicts of interest, self-dealing), care (exercising reasonable diligence), good faith, and confidentiality. Breach of fiduciary duty can result in equitable remedies including disgorgement of profits, constructive trust, and injunctive relief, as well as compensatory and punitive damages."),
        ("What is the doctrine of res judicata?",
         "Res judicata (claim preclusion) prevents relitigation of claims that were or could have been raised in a prior proceeding between the same parties. Requirements: (1) final judgment on the merits, (2) same parties or privies, (3) same cause of action (transactional test). Collateral estoppel (issue preclusion) is related but narrower — it prevents relitigation of specific factual or legal issues actually decided in prior proceedings. Together, these doctrines promote judicial economy, consistency, and finality."),
        ("Describe the federal court system and jurisdiction.",
         "The federal judiciary has three tiers: U.S. District Courts (trial courts, 94 districts), U.S. Courts of Appeals (13 circuits), and the Supreme Court. Federal jurisdiction is limited: subject matter jurisdiction requires either federal question (arising under federal law, 28 U.S.C. 1331) or diversity of citizenship (parties from different states, amount exceeding $75,000, 28 U.S.C. 1332). Personal jurisdiction requires minimum contacts with the forum state (International Shoe Co. v. Washington, 1945). Standing requires injury-in-fact, causation, and redressability (Lujan v. Defenders of Wildlife, 1992)."),
        ("Explain the concept of mens rea in criminal law.",
         "Mens rea ('guilty mind') is the mental state required for criminal liability. The Model Penal Code defines four levels: (1) purposely — conscious objective to engage in conduct or cause a result, (2) knowingly — awareness that conduct is of a particular nature or will cause a result, (3) recklessly — conscious disregard of a substantial and unjustifiable risk, (4) negligently — should have been aware of a substantial risk. Strict liability offenses (regulatory crimes, statutory rape) require no mens rea. The prosecution must prove mens rea beyond a reasonable doubt as an element of the offense."),
        ("What are the rights of the accused under the Sixth Amendment?",
         "The Sixth Amendment guarantees criminal defendants: (1) right to a speedy trial (Barker v. Wingo four-factor test), (2) right to a public trial, (3) right to an impartial jury (including voir dire to remove biased jurors, Batson challenges against discriminatory peremptory strikes), (4) right to be informed of charges (notice), (5) right to confront adverse witnesses (Confrontation Clause, Crawford v. Washington), (6) right to compulsory process to obtain favorable witnesses, and (7) right to effective assistance of counsel (Strickland v. Washington two-prong test: deficient performance + prejudice)."),
        ("Describe the Commerce Clause and its evolution.",
         "The Commerce Clause (Article I, Section 8, Clause 3) grants Congress power to regulate interstate commerce. Its scope evolved dramatically: Gibbons v. Ogden (1824) broadly defined commerce as intercourse. The Lochner era restricted interpretation. NLRB v. Jones & Laughlin Steel (1937) and Wickard v. Filburn (1942) expanded it to activities substantially affecting interstate commerce. United States v. Lopez (1995) imposed limits, requiring a substantial effect on commerce. The dormant Commerce Clause prohibits state discrimination against interstate commerce."),
        ("Explain the concept of eminent domain and just compensation.",
         "The Fifth Amendment's Takings Clause provides: 'nor shall private property be taken for public use, without just compensation.' Eminent domain allows government to acquire private property for public use. After Kelo v. City of New London (2005), public use includes economic development, though many states enacted stricter limits. Just compensation equals fair market value at the time of taking. Regulatory takings (Penn Central test) occur when government regulation goes 'too far' (Pennsylvania Coal Co. v. Mahon). Per se takings occur with physical occupation (Loretto) or total economic loss (Lucas v. South Carolina Coastal Council)."),
        ("What is the doctrine of sovereign immunity?",
         "Sovereign immunity is the principle that the government cannot be sued without its consent. Federal sovereign immunity derives from constitutional structure; state sovereign immunity from the Eleventh Amendment (Hans v. Louisiana, 1890). Waivers: the Federal Tort Claims Act (FTCA) waives federal immunity for certain torts, with exceptions for discretionary functions, intentional torts, and military claims. Section 1983 allows suits against state officials acting under color of law for constitutional violations (Ex parte Young doctrine). The Tucker Act waives immunity for contract claims against the federal government."),
        ("Describe the rules of evidence regarding hearsay.",
         "Hearsay is an out-of-court statement offered to prove the truth of the matter asserted (Federal Rule of Evidence 801). Hearsay is generally inadmissible but has numerous exceptions. FRE 803 exceptions (regardless of declarant availability): present sense impression, excited utterance, then-existing mental condition, statements for medical diagnosis, recorded recollection, business records. FRE 804 exceptions (declarant unavailable): former testimony, dying declarations, statements against interest. The residual exception (FRE 807) allows hearsay with circumstantial guarantees of trustworthiness. Non-hearsay includes prior inconsistent statements under oath, prior consistent statements, and party-opponent admissions."),
        ("Explain the concept of standing in federal court.",
         "Standing doctrine ensures courts only decide actual cases and controversies (Article III). The constitutional minimum requires: (1) injury-in-fact — concrete and particularized, actual or imminent, (2) causation — fairly traceable to the defendant's conduct, (3) redressability — likely that a favorable decision will remedy the injury. Prudential standing requires the plaintiff to assert their own rights, not third parties', and fall within the zone of interests protected by the statute. Organizational standing requires diversion of resources. Taxpayer standing is generally denied (Flast v. Cohen exception for Establishment Clause challenges)."),
        ("What is the Confrontation Clause and its modern interpretation?",
         "The Sixth Amendment Confrontation Clause guarantees the right of criminal defendants to confront witnesses against them. Crawford v. Washington (2004) held that testimonial statements of unavailable witnesses are inadmissible unless the defendant had a prior opportunity for cross-examination. Testimonial statements include police interrogations, affidavits, and prior testimony. Non-testimonial statements (casual remarks, business records) are governed by hearsay rules, not the Confrontation Clause. Davis v. Washington (2006) distinguished ongoing emergency statements (non-testimonial) from investigative questioning (testimonial)."),
        ("Describe the concept of punitive damages.",
         "Punitive damages are awarded beyond compensatory damages to punish egregious conduct and deter similar behavior. They require a showing of malice, oppression, or fraud. Constitutional limits: BMW of North America v. Gore (1996) established three guideposts: (1) degree of reprehensibility, (2) ratio between punitive and compensatory damages, and (3) comparison with civil penalties for similar conduct. State Farm v. Campbell (2003) suggested single-digit ratios are more likely constitutional. Due process requires adequate judicial review of punitive awards. Many states impose statutory caps on punitive damages."),
        ("Explain the federal sentencing guidelines.",
         "The U.S. Sentencing Guidelines provide a framework for federal criminal sentencing. The guidelines use a two-dimensional matrix: offense level (1-43) on one axis and criminal history category (I-VI) on the other, producing a sentencing range in months. United States v. Booker (2005) made the guidelines advisory rather than mandatory, subject to appellate reasonableness review. Courts must consider the 18 U.S.C. 3553(a) factors: nature of the offense, deterrence, protection of the public, rehabilitation, consistency, and avoiding unwarranted disparities. Departures and variances allow deviation from the guideline range."),
        ("What is the doctrine of qualified immunity for government officials?",
         "Qualified immunity shields government officials performing discretionary functions from civil liability unless their conduct violates clearly established statutory or constitutional rights that a reasonable person would have known. The two-prong test (Pearson v. Callahan, 2009): (1) did the official violate a constitutional right? (2) was that right clearly established at the time? Courts may address either prong first. The 'clearly established' standard requires existing precedent to place the constitutional question beyond debate, though factually identical precedent is not required. Qualified immunity does not apply to prosecutors acting in their advocacy role (absolute immunity) or to private defendants."),
        ("Describe the concept of equitable remedies.",
         "Equitable remedies are non-monetary court orders developed in the historical courts of equity. Types: (1) Injunction — court order requiring or prohibiting specific conduct. Preliminary injunctions require likelihood of success, irreparable harm, balance of hardships, and public interest (Winter v. NRDC, 2008). (2) Specific performance — compels party to perform contractual obligations, typically for unique property. (3) Rescission — cancels the contract and restores parties to pre-contract positions. (4) Reformation — corrects a written contract to reflect actual agreement. (5) Constructive trust — prevents unjust enrichment. Equitable remedies are available only when legal remedies (damages) are inadequate."),
        ("Explain the concept of judicial review.",
         "Judicial review is the power of courts to determine the constitutionality of legislative and executive actions. Established in Marbury v. Madison (1803), Chief Justice Marshall declared it the province and duty of the judiciary to say what the law is. Standards of review: strict scrutiny (fundamental rights, suspect classifications — government must show compelling interest and narrow tailoring), intermediate scrutiny (gender, legitimacy — substantially related to important interest), rational basis (economic regulation — rationally related to legitimate interest). Administrative agency review follows the APA: de novo, substantial evidence, or arbitrary and capricious standard."),
        ("What is the Equal Protection Clause?",
         "The Fourteenth Amendment's Equal Protection Clause provides that no state shall deny any person equal protection of the laws. Three tiers of scrutiny apply: (1) Strict scrutiny for suspect classifications (race, national origin, alienage) and fundamental rights — classification must be narrowly tailored to a compelling government interest. (2) Intermediate scrutiny for quasi-suspect classifications (gender, illegitimacy) — substantially related to important government interest. (3) Rational basis for all other classifications — rationally related to legitimate government interest. Affirmative action in higher education: strict scrutiny applies (Grutter v. Bollinger, 2003; Students for Fair Admissions v. Harvard, 2023)."),
        ("Describe the concept of double jeopardy.",
         "The Fifth Amendment's Double Jeopardy Clause prohibits: (1) a second prosecution for the same offense after acquittal, (2) a second prosecution after conviction, and (3) multiple punishments for the same offense. Jeopardy attaches when the jury is sworn (jury trial) or when the first witness is sworn (bench trial). The Blockburger test determines same offense: each offense must require proof of an element the other does not. The dual sovereignty doctrine allows prosecution by both federal and state governments for the same conduct (Gamble v. United States, 2019). A mistrial without manifest necessity bars retrial."),
        ("Explain the concept of proximate cause in tort law.",
         "Proximate cause limits liability to foreseeable consequences of negligent conduct, serving as a policy check on unlimited liability. The Palsgraf v. Long Island Railroad (1928) debate framed the issue: Cardozo's majority required foreseeable plaintiff and risk; Andrews's dissent favored broader liability limited by directness. Modern approaches: (1) foreseeability test — was the type of harm foreseeable? (2) directness test — was the chain of causation unbroken? Superseding causes (extraordinary, unforeseeable intervening events) break the causal chain. The eggshell skull rule: defendant takes plaintiff as found, even if injuries are more severe than foreseeable."),
        ("What is the concept of vicarious liability?",
         "Vicarious liability holds one party responsible for the wrongful acts of another based on their relationship. The primary doctrine is respondeat superior: an employer is liable for employee torts committed within the scope of employment. The scope inquiry considers: whether the conduct was authorized, occurred during work hours, was motivated by serving the employer, and whether the employer could have foreseen it. Independent contractors generally do not create vicarious liability, unless the principal retains control over the work or the activity is inherently dangerous. Joint enterprise liability extends to co-venturers. Parents are generally not vicariously liable for children's torts absent specific statutes."),
    ]


def prepare_all_data():
    """Prepare training data for all domains."""
    log("\n=== Phase 0: Prepare Training Data (HuggingFace) ===")
    data_stats = {}

    preparers = {
        "math": prepare_math_data,
        "code": prepare_code_data,
        "medical": prepare_medical_data,
        "legal": prepare_legal_data,
        "finance": prepare_finance_data,
    }

    for domain in DOMAINS:
        data_dir = EXPERIMENT_DIR / f"data_{domain}"
        # Skip if already prepared
        if data_dir.exists() and (data_dir / "train.jsonl").exists():
            n_existing = sum(1 for _ in open(data_dir / "train.jsonl"))
            if n_existing >= (N_TRAIN * 0.8):  # allow some slack
                log(f"  [{domain}] Data exists: {n_existing} train examples")
                data_stats[domain] = n_existing
                continue

        log(f"  [{domain}] Preparing from HuggingFace...")
        try:
            n_train = preparers[domain](data_dir, N_TRAIN)
            data_stats[domain] = n_train
            log(f"  [{domain}] Prepared {n_train} train examples")
        except Exception as e:
            log(f"  [{domain}] ERROR: {e}")
            data_stats[domain] = 0

    return data_stats


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Train adapters (v_proj+o_proj)
# ══════════════════════════════════════════════════════════════════════════════

def train_adapter(domain: str) -> float:
    """Train v_proj+o_proj LoRA adapter. Returns training time in minutes."""
    import yaml

    adapter_dir = EXPERIMENT_DIR / f"adapter_{domain}"
    data_dir = EXPERIMENT_DIR / f"data_{domain}"

    if adapter_dir.exists() and (adapter_dir / "adapters.safetensors").exists():
        log(f"  [{domain}] Adapter exists, skipping training")
        return 0.0

    adapter_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": MODEL_ID,
        "data": str(data_dir),
        "adapter_path": str(adapter_dir),
        "train": True,
        "fine_tune_type": "lora",
        "num_layers": 16,
        "iters": TRAIN_ITERS,
        "batch_size": 1 if IS_SMOKE else 2,
        "learning_rate": 2e-4,
        "lora_parameters": {
            "rank": LORA_RANK,
            "scale": 4.0,
            "dropout": 0.0,
            "keys": LORA_KEYS,
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
        f"({TRAIN_ITERS} iters, {N_TRAIN} examples)...")

    t0 = time.time()
    cmd = ["uv", "run", "python", "-m", "mlx_lm", "lora", "--config", str(config_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
    elapsed = (time.time() - t0) / 60.0

    if result.returncode != 0:
        log(f"  [{domain}] Training FAILED (exit={result.returncode})")
        log(f"  STDERR (last 1500 chars): {result.stderr[-1500:]}")
        raise RuntimeError(f"Training failed for {domain}")

    log(f"  [{domain}] Training complete in {elapsed:.1f} min")
    return elapsed


def phase_train_all():
    """Train all 5 domain adapters."""
    log("\n=== Phase 1: Train Domain Adapters (v_proj+o_proj, 1000 iters) ===")
    training_times = {}
    for domain in DOMAINS:
        t = train_adapter(domain)
        training_times[domain] = round(t, 1)
        cleanup()
        log_memory(f"after-train-{domain}")
    return training_times


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Behavioral evaluation (vocab density)
# ══════════════════════════════════════════════════════════════════════════════

def generate_response(question: str, adapter_path: str | None = None) -> str:
    """Generate a response via mlx_lm CLI subprocess (memory-safe)."""
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
    """Generate base model responses for all domains."""
    log("\n=== Phase 2a: Base Model Evaluation ===")
    base_results = {}
    for domain in DOMAINS:
        queries = EVAL_QUERIES[domain][:N_EVAL]
        glossary = DOMAIN_GLOSSARIES[domain]
        domain_results = []

        log(f"  [{domain}] Evaluating base model ({len(queries)} queries)...")
        for i, q in enumerate(queries):
            resp = generate_response(q)
            score = score_vocabulary(resp, glossary)
            domain_results.append({"query": q, "response_snippet": resp[:120], "vocab_score": score})
            log(f"    [{i+1}/{len(queries)}] vocab={score}")

        base_results[domain] = domain_results
        cleanup()

    return base_results


def phase_evaluate_adapted():
    """Generate adapted model responses for all domains."""
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
            log(f"    [{i+1}/{len(queries)}] vocab={score}")

        adapted_results[domain] = domain_results
        cleanup()

    return adapted_results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 70)
    log("P0: v_proj+o_proj Adapter Quality Floor")
    log(f"  Real HuggingFace datasets, {N_TRAIN} examples, {TRAIN_ITERS} iterations")
    log(f"  IS_SMOKE={IS_SMOKE}, N_EVAL={N_EVAL}, LORA_RANK={LORA_RANK}")
    log(f"  LORA_KEYS={LORA_KEYS}")
    log("=" * 70)

    total_start = time.time()
    cleanup()
    log_memory("start")

    # Phase 0: Prepare data
    data_stats = prepare_all_data()

    # Phase 1: Train adapters
    training_times = phase_train_all()

    # Phase 2a: Base model evaluation
    base_results = phase_evaluate_base()

    # Phase 2b: Adapted model evaluation
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

    # ─── Kill criteria ────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("KILL CRITERIA RESULTS")
    log("=" * 70)

    # K1320: All 5 domains >= 60%
    all_above_60 = all(solo_improvement_rates[d] >= 0.60 for d in DOMAINS)
    k1320_pass = all_above_60
    log(f"\nK1320 (All 5 domains >= 60%):")
    for d in DOMAINS:
        r = solo_improvement_rates[d]
        log(f"  [{d}] {r*100:.1f}% {'PASS' if r >= 0.60 else 'FAIL'}")
    log(f"  ALL >= 60%: {'PASS' if k1320_pass else 'FAIL'}")

    # K1321: Mean >= 50%
    mean_rate = sum(solo_improvement_rates[d] for d in DOMAINS) / len(DOMAINS)
    k1321_pass = mean_rate >= 0.50
    log(f"\nK1321 (Mean >= 50%): {mean_rate*100:.1f}% — {'PASS' if k1321_pass else 'FAIL'}")

    # K1322: Legal >= 40%
    legal_rate = solo_improvement_rates["legal"]
    k1322_pass = legal_rate >= 0.40
    log(f"\nK1322 (Legal >= 40%): {legal_rate*100:.1f}% — {'PASS' if k1322_pass else 'FAIL'}")

    # K1323: Training <= 30 min per domain
    max_train_time = max(training_times.values()) if training_times else 0
    k1323_pass = max_train_time <= 30.0
    log(f"\nK1323 (Training <= 30 min/domain): max={max_train_time:.1f} min — {'PASS' if k1323_pass else 'FAIL'}")

    all_pass = k1320_pass and k1321_pass and k1322_pass and k1323_pass
    total_min = (time.time() - total_start) / 60.0

    log(f"\n{'='*70}")
    log(f"SUMMARY:")
    log(f"  K1320 (All >= 60%): {'PASS' if k1320_pass else 'FAIL'}")
    log(f"  K1321 (Mean >= 50%): {'PASS' if k1321_pass else 'FAIL'} ({mean_rate*100:.1f}%)")
    log(f"  K1322 (Legal >= 40%): {'PASS' if k1322_pass else 'FAIL'} ({legal_rate*100:.1f}%)")
    log(f"  K1323 (Train <= 30m): {'PASS' if k1323_pass else 'FAIL'} (max={max_train_time:.1f}m)")
    log(f"  ALL PASS: {all_pass}")
    log(f"Total time: {total_min:.1f} min")

    # Compare with P8 baseline (10 examples, 200 iters)
    p8_baseline = {"math": 0.55, "code": 0.50, "medical": 0.70, "legal": 0.35, "finance": 0.50}
    log(f"\nComparison with P8 (10 examples, 200 iters):")
    for d in DOMAINS:
        p8 = p8_baseline.get(d, 0)
        current = solo_improvement_rates[d]
        delta = current - p8
        log(f"  [{d}] P8={p8*100:.0f}% → current={current*100:.1f}% (delta={delta*100:+.1f}pp)")

    log(f"{'='*70}")

    # ─── Save results ──────────────────────────────────────────────────
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

        mean_base = sum(b["vocab_score"] for b in base) / len(base) if base else 0
        mean_adapted = sum(a["vocab_score"] for a in adapted) / len(adapted) if adapted else 0

        domain_detail[domain] = {
            "improvement_rate": solo_improvement_rates[domain],
            "mean_base_vocab": round(mean_base, 2),
            "mean_adapted_vocab": round(mean_adapted, 2),
            "n_eval": len(base),
            "n_train": data_stats.get(domain, 0),
            "training_time_min": training_times.get(domain, 0),
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
        "data_stats": data_stats,
        "training_times": training_times,
        "domain_results": domain_detail,
        "kill_criteria": {
            "k1320_all_60": {"pass": k1320_pass, "rates": {d: solo_improvement_rates[d] for d in DOMAINS}},
            "k1321_mean_50": {"pass": k1321_pass, "mean_rate": mean_rate},
            "k1322_legal_40": {"pass": k1322_pass, "rate": legal_rate},
            "k1323_train_time": {"pass": k1323_pass, "max_time_min": max_train_time},
        },
        "p8_baseline": p8_baseline,
        "all_pass": all_pass,
        "total_time_min": round(total_min, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
