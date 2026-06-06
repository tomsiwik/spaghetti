#!/usr/bin/env python3
"""Revised validation for Domain Taxonomy experiment.

Addresses 4 fixes from adversarial review:
1. Negative control taxonomy (270 paraphrases of 30 domains = 9x redundancy)
2. Tightened kill criteria: K1 >5% pairs cos>0.5, K2 >5% domains NN cos>0.7
3. Pilot-50 proxy validation (honest assessment of r=0.034)
4. Results for updated PAPER.md

Run: python3 validate_revised.py
"""

import json
import time
import random
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# Import from existing experiment
from domain_taxonomy import (
    TAXONOMY, DOMAIN_DESCRIPTIONS, PILOT50_TO_TAXONOMY,
    flatten_taxonomy, get_domain_description,
    compute_embeddings_semantic, compute_embeddings_tfidf
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent
PILOT50_RESULTS = REPO_ROOT / "results" / "pilot50_benchmark.json"


# ──────────────────────────────────────────────────────────────────────
# Negative Control: 270 paraphrases of 30 base domains (9x redundancy)
# ──────────────────────────────────────────────────────────────────────

# 30 base domains, each paraphrased 9 times to reach 270
# These are deliberately BAD: near-synonyms that should fail overlap tests
NEGATIVE_CONTROL_BASES = {
    "python_coding": [
        "Python programming: writing functions, classes, decorators, generators, data structures, standard library",
        "Python development: implementing functions, objects, decorators, generators, collections, built-in modules",
        "Python scripting: creating functions, classes, decorators, iterators, data types, core library",
        "Python software: defining methods, classes, wrappers, generators, structures, standard packages",
        "Python code: building functions, OOP classes, decorator patterns, generator expressions, data containers",
        "Python language: writing procedures, class hierarchies, function decorators, lazy generators, data organization",
        "Python engineering: crafting functions, class design, decorator usage, generator pipelines, data manipulation",
        "Python mastery: advanced functions, metaclasses, decorator chains, async generators, complex data structures",
        "Python expertise: functional patterns, class composition, decorator factories, coroutines, collection types",
    ],
    "web_development": [
        "Web development: HTML, CSS, JavaScript, responsive design, REST APIs, frontend frameworks",
        "Web programming: markup, styling, scripting, adaptive layouts, web services, UI frameworks",
        "Web engineering: HTML5, CSS3, JS, mobile-first design, API development, component frameworks",
        "Website building: semantic markup, cascading styles, client-side scripting, responsive layouts, APIs",
        "Web application development: HTML structure, CSS styling, JavaScript logic, responsive UIs, RESTful services",
        "Frontend web: document markup, visual styling, browser scripting, fluid layouts, API integration",
        "Web design and coding: page structure, style sheets, interactive scripts, adaptive design, web APIs",
        "Web technologies: hypertext markup, styling languages, client scripting, responsive frameworks, services",
        "Internet development: web markup, presentation styles, browser programming, flexible layouts, web endpoints",
    ],
    "machine_learning": [
        "Machine learning: supervised learning, neural networks, gradient descent, model evaluation, feature engineering",
        "ML algorithms: classification, deep learning, optimization, validation metrics, feature selection",
        "Statistical learning: predictive models, artificial neural networks, SGD, cross-validation, data preprocessing",
        "Machine intelligence: learning algorithms, network architectures, backpropagation, model assessment, features",
        "AI/ML: supervised methods, deep neural nets, gradient optimization, performance metrics, input features",
        "Predictive modeling: ML classifiers, multi-layer networks, loss minimization, evaluation, feature extraction",
        "Data science ML: learning models, neural architectures, training optimization, scoring, data features",
        "Applied machine learning: prediction algorithms, deep architectures, parameter tuning, benchmarking, features",
        "ML engineering: model training, network layers, optimization loops, test metrics, feature pipelines",
    ],
    "data_analysis": [
        "Data analysis: statistics, visualization, pandas, SQL queries, exploratory analysis, hypothesis testing",
        "Data analytics: statistical methods, charts and graphs, dataframe operations, database queries, EDA",
        "Data exploration: descriptive statistics, data visualization, tabular data, structured queries, analysis",
        "Analytical data work: statistical analysis, plotting, data wrangling, query languages, data investigation",
        "Data examination: stats, visual representation, data manipulation, relational queries, pattern discovery",
        "Data insights: statistical measures, graphical analysis, data processing, SQL, exploratory techniques",
        "Data interpretation: quantitative analysis, information visualization, data transformation, queries, testing",
        "Business analytics: statistical reporting, dashboards, data aggregation, database analysis, trend analysis",
        "Data science analytics: descriptive stats, visual analytics, data cleaning, query optimization, EDA methods",
    ],
    "natural_language_processing": [
        "NLP: text processing, tokenization, named entity recognition, sentiment analysis, language models",
        "Natural language processing: text analysis, word segmentation, NER, opinion mining, LLMs",
        "Computational linguistics: text mining, tokenizers, entity extraction, sentiment detection, language modeling",
        "Text analytics: linguistic processing, token splitting, entity recognition, sentiment scoring, neural LMs",
        "Language AI: text understanding, sub-word tokenization, information extraction, affect analysis, transformers",
        "NLP engineering: text pipelines, vocabulary encoding, entity tagging, polarity detection, seq2seq models",
        "Language processing: text normalization, tokenization schemes, NER systems, opinion analysis, language nets",
        "Text intelligence: document processing, lexical analysis, entity identification, sentiment classification, LMs",
        "Applied NLP: corpus processing, text tokenization, named entities, sentiment models, pretrained transformers",
    ],
    "cloud_computing": [
        "Cloud computing: AWS, virtual machines, containers, serverless, load balancing, cloud storage",
        "Cloud infrastructure: Amazon services, VMs, Docker, FaaS, traffic distribution, object storage",
        "Cloud platforms: AWS/GCP/Azure, compute instances, containerization, serverless functions, CDN, storage",
        "Cloud services: cloud providers, virtual servers, container orchestration, lambda functions, scalable storage",
        "Cloud architecture: managed services, cloud VMs, microservices containers, event-driven compute, blob storage",
        "Cloud engineering: cloud deployment, instance management, container platforms, serverless computing, S3",
        "Cloud operations: cloud hosting, virtual compute, Kubernetes containers, function-as-service, distributed storage",
        "Cloud technology: IaaS/PaaS/SaaS, cloud instances, container runtime, serverless architecture, cloud data",
        "Cloud solutions: cloud migration, VM scaling, container deployment, event functions, storage services",
    ],
    "database_management": [
        "Database management: relational databases, SQL, normalization, indexing, transactions, query optimization",
        "Database administration: RDBMS, structured queries, schema design, index tuning, ACID transactions",
        "Database systems: relational models, SQL language, normal forms, B-tree indexes, concurrency control",
        "Data management: databases, query languages, data normalization, index strategies, transaction processing",
        "Database engineering: table design, SQL queries, schema normalization, query plans, isolation levels",
        "Database development: relational schemas, data querying, normalization rules, index optimization, commits",
        "Database technology: relational stores, structured query language, canonical forms, indexing, durability",
        "Database design: ER modeling, SQL DDL/DML, decomposition, index selection, transaction management",
        "Database operations: DBMS, SQL operations, data integrity, performance indexing, concurrent transactions",
    ],
    "cybersecurity_defense": [
        "Cybersecurity: network security, encryption, authentication, vulnerability assessment, incident response",
        "Information security: network protection, cryptography, access control, penetration testing, IR",
        "Cyber defense: secure networking, data encryption, identity verification, security scanning, response plans",
        "Security engineering: network hardening, cipher algorithms, auth protocols, vuln management, forensics",
        "Digital security: firewall configuration, encryption standards, MFA, security audits, breach response",
        "Security operations: network monitoring, cryptographic protocols, authentication systems, threat assessment",
        "Cybersecurity management: perimeter security, data protection, credential management, risk assessment",
        "Security analysis: network traffic analysis, encryption implementation, access management, vulnerability scans",
        "Protective cybersecurity: infrastructure security, crypto algorithms, identity management, threat hunting",
    ],
    "project_management_methods": [
        "Project management: agile methodology, sprint planning, resource allocation, risk management, stakeholders",
        "PM practices: agile/scrum, iteration planning, team resourcing, risk mitigation, stakeholder management",
        "Project leadership: agile frameworks, sprint scheduling, capacity planning, risk assessment, communication",
        "Project coordination: scrum methodology, backlog management, resource optimization, risk analysis, reporting",
        "Agile project management: sprints, planning poker, team allocation, risk registers, stakeholder engagement",
        "Project delivery: agile processes, sprint execution, resource management, risk identification, governance",
        "PM methodology: agile practices, iteration management, staffing plans, risk response, status reporting",
        "Project execution: scrum sprints, release planning, resource scheduling, risk monitoring, client updates",
        "Program management: agile transformation, sprint reviews, capacity management, risk frameworks, alignment",
    ],
    "technical_writing_docs": [
        "Technical writing: documentation, API docs, user manuals, style guides, information architecture",
        "Technical documentation: doc writing, API references, user guides, writing standards, content structure",
        "Documentation authoring: technical docs, endpoint documentation, instruction manuals, style conventions",
        "Technical communication: system documentation, API specifications, how-to guides, writing guidelines",
        "Doc engineering: technical content, API documentation, product manuals, documentation standards",
        "Technical authorship: software documentation, REST API docs, user documentation, editorial guidelines",
        "Documentation writing: tech writing, interface documentation, guide creation, formatting standards",
        "Technical content creation: product documentation, developer docs, tutorial writing, doc architecture",
        "Documentation development: technical guides, API walkthroughs, reference manuals, content guidelines",
    ],
    "statistics_methods": [
        "Statistics: probability distributions, hypothesis testing, regression analysis, Bayesian inference, sampling",
        "Statistical methods: probability theory, significance testing, linear regression, Bayes theorem, surveys",
        "Applied statistics: distribution functions, statistical tests, regression models, Bayesian methods, samples",
        "Quantitative statistics: probability, testing hypotheses, curve fitting, posterior estimation, data sampling",
        "Statistical analysis: random variables, p-values, regression techniques, Bayesian updating, population sampling",
        "Stats methodology: distributions, test statistics, predictive regression, prior/posterior, sampling design",
        "Mathematical statistics: probability models, hypothesis frameworks, regression analysis, conjugate priors",
        "Inferential statistics: probability calculus, significance levels, multivariate regression, MCMC, stratified sampling",
        "Statistical science: stochastic models, testing procedures, least squares, Bayesian computation, sampling theory",
    ],
    "algorithms_data_structures": [
        "Algorithms: sorting, searching, graph algorithms, dynamic programming, complexity analysis, data structures",
        "Data structures and algorithms: sort/search, graph traversal, DP, Big-O analysis, trees and heaps",
        "Algorithm design: sorting algorithms, binary search, shortest path, memoization, asymptotic complexity",
        "Computational algorithms: comparison sorts, search methods, graph theory, optimization, space complexity",
        "DSA: sorting techniques, search algorithms, network algorithms, dynamic programming, algorithmic efficiency",
        "Algorithm engineering: quicksort/mergesort, lookup algorithms, BFS/DFS, tabulation, runtime analysis",
        "Algorithms and complexity: ordering algorithms, retrieval, graph processing, subproblem decomposition",
        "Algorithmic thinking: sort methods, pattern searching, path algorithms, recursive DP, performance bounds",
        "CS algorithms: elementary sorting, indexed searching, graph computation, optimal substructure, complexity",
    ],
    "operating_systems": [
        "Operating systems: process management, memory allocation, file systems, scheduling, concurrency, kernel",
        "OS concepts: process control, virtual memory, filesystem design, CPU scheduling, synchronization, kernels",
        "System software: process lifecycle, memory management, storage systems, task scheduling, mutual exclusion",
        "OS engineering: multitasking, paging/segmentation, directory structures, preemptive scheduling, locks",
        "Operating system design: process scheduling, memory hierarchy, file organization, real-time scheduling",
        "OS internals: process states, address spaces, inode systems, scheduler algorithms, semaphores, microkernel",
        "System architecture: process models, RAM management, persistent storage, fair scheduling, thread safety",
        "OS fundamentals: PCB management, page tables, block allocation, round-robin scheduling, deadlocks",
        "Kernel and OS: processes/threads, memory mapping, VFS, priority scheduling, race condition prevention",
    ],
    "networking_protocols": [
        "Computer networking: TCP/IP, routing, DNS, HTTP, network layers, socket programming, packet switching",
        "Network protocols: transport layer, IP routing, name resolution, web protocols, OSI model, sockets",
        "Networking fundamentals: TCP connections, routing tables, DNS resolution, HTTP/HTTPS, layered architecture",
        "Network engineering: reliable transport, routing algorithms, domain name system, application protocols",
        "Data networking: TCP/UDP, path selection, name servers, request/response protocols, network stack",
        "Network communications: connection-oriented transport, BGP routing, DNS hierarchy, REST over HTTP",
        "Networking technology: flow control, OSPF routing, DNS records, HTTP methods, encapsulation, APIs",
        "Network systems: congestion control, dynamic routing, recursive DNS, HTTP headers, protocol layering",
        "Applied networking: reliable delivery, routing protocols, DNS lookup, web communication, network layers",
    ],
    "linear_algebra_math": [
        "Linear algebra: vectors, matrices, eigenvalues, linear transformations, vector spaces, decompositions",
        "Matrix mathematics: vector operations, matrix algebra, eigenvectors, linear maps, subspaces, SVD",
        "Linear math: vector spaces, matrix multiplication, spectral theory, basis transformations, rank",
        "Algebraic linear methods: vectors/matrices, eigendecomposition, coordinate transformations, dimensions",
        "Vector and matrix algebra: dot products, matrix operations, eigenproblems, change of basis, factorizations",
        "Linear algebra foundations: vector addition, matrix inverse, diagonalization, orthogonal projections",
        "Applied linear algebra: column spaces, determinants, eigenvalue problems, least squares, PCA",
        "Computational linear algebra: vector norms, matrix factorization, spectral decomposition, iterative methods",
        "Matrix theory: vector spaces, Gaussian elimination, characteristic polynomials, singular values",
    ],
    "organic_chem": [
        "Organic chemistry: carbon compounds, reaction mechanisms, functional groups, synthesis, stereochemistry",
        "Organic reactions: hydrocarbon chemistry, mechanism pathways, reactive groups, synthetic routes, chirality",
        "Carbon chemistry: organic molecules, arrow-pushing mechanisms, substituent groups, retrosynthesis, isomers",
        "Organic chemical science: molecular structure, reaction intermediates, functional group chemistry, synthesis",
        "Organic synthesis: carbon frameworks, nucleophilic/electrophilic mechanisms, groups, synthetic planning",
        "Molecular organic chemistry: bonding in organics, reaction kinetics, functional group transformation",
        "Applied organic chemistry: organic compounds, mechanistic chemistry, protecting groups, total synthesis",
        "Organic chemistry fundamentals: hybridization, SN1/SN2, functional group interconversion, stereoselectivity",
        "Bioorganic chemistry: carbon molecules, enzymatic mechanisms, pharmacophore groups, natural product synthesis",
    ],
    "physics_mechanics": [
        "Classical mechanics: Newton's laws, forces, energy, momentum, rotational dynamics, oscillations",
        "Newtonian mechanics: force and motion, kinetic/potential energy, impulse, torque, harmonic oscillation",
        "Physical mechanics: laws of motion, work-energy theorem, conservation laws, angular momentum, vibrations",
        "Mechanics fundamentals: F=ma, mechanical energy, linear momentum, rotation, simple harmonic motion",
        "Applied mechanics: Newtonian dynamics, energy conservation, collision physics, moment of inertia, waves",
        "Analytical mechanics: equations of motion, Lagrangian, Hamiltonian, central forces, coupled oscillators",
        "Engineering mechanics: statics and dynamics, energy methods, momentum principles, gyroscopic effects",
        "Mechanics physics: force analysis, power and efficiency, elastic collisions, precession, pendulums",
        "Classical dynamics: motion equations, conservative forces, two-body problems, rigid body rotation",
    ],
    "economics_theory": [
        "Economics: supply and demand, market equilibrium, monetary policy, fiscal policy, trade, growth",
        "Economic theory: price mechanisms, market clearing, central banking, government spending, trade policy",
        "Macroeconomics: aggregate supply/demand, equilibrium GDP, money supply, taxation, international trade",
        "Economic analysis: demand curves, price equilibrium, interest rates, public finance, comparative advantage",
        "Applied economics: market dynamics, general equilibrium, monetary transmission, budget policy, trade models",
        "Economic principles: scarcity and choice, market balance, inflation targeting, fiscal stimulus, exports",
        "Economic science: consumer/producer surplus, Walrasian equilibrium, quantitative easing, deficits, tariffs",
        "Modern economics: behavioral demand, Nash equilibrium, central bank policy, fiscal multipliers, FTAs",
        "Political economy: market forces, competitive equilibrium, money creation, government intervention, trade",
    ],
    "psychology_cognitive": [
        "Cognitive psychology: memory, attention, perception, decision-making, problem solving, cognitive biases",
        "Psychology of mind: working memory, selective attention, visual perception, judgment, reasoning, heuristics",
        "Cognitive science: memory systems, attentional control, perceptual processing, choice behavior, biases",
        "Mental processes: encoding/retrieval, focus, sensation, decision theory, cognitive strategies, fallacies",
        "Cognitive function: short/long-term memory, divided attention, pattern recognition, prospect theory, biases",
        "Psychological cognition: memory consolidation, attention mechanisms, perceptual illusions, risk decisions",
        "Applied cognitive psychology: mnemonic strategies, mindfulness, perceptual learning, nudge theory, errors",
        "Cognitive behavioral science: episodic memory, sustained attention, object recognition, utility maximization",
        "Experimental psychology: memory paradigms, attention experiments, psychophysics, behavioral economics",
    ],
    "philosophy_ethics": [
        "Philosophy and ethics: moral reasoning, ethical frameworks, metaethics, applied ethics, normative theory",
        "Ethical philosophy: moral judgment, consequentialism/deontology, moral realism, bioethics, virtue ethics",
        "Moral philosophy: right and wrong, ethical systems, foundations of morality, practical ethics, duties",
        "Philosophical ethics: moral principles, utilitarian/Kantian frameworks, moral ontology, medical ethics",
        "Ethics and values: moral decision-making, ethical theories, meta-ethical questions, professional ethics",
        "Applied moral philosophy: ethical dilemmas, framework comparison, moral epistemology, technology ethics",
        "Normative ethics: moral obligations, consequentialist/deontological debate, moral relativism, justice",
        "Ethics foundations: moral philosophy, teleological/deontological ethics, ethical naturalism, business ethics",
        "Philosophical moral theory: ethical reasoning, competing moral frameworks, expressivism, environmental ethics",
    ],
    "creative_fiction_writing": [
        "Fiction writing: narrative structure, character development, dialogue, plot construction, literary devices",
        "Creative writing fiction: story structure, characterization, conversational writing, plot arcs, figurative language",
        "Novel writing: narrative arc, character creation, dialogue craft, plot design, symbolism and metaphor",
        "Fiction craft: storytelling techniques, character psychology, speech patterns, rising action, literary style",
        "Imaginative fiction: narrative voice, multidimensional characters, realistic dialogue, conflict, imagery",
        "Fiction composition: story architecture, character motivation, dialogue attribution, subplot weaving, tone",
        "Creative storytelling: narrative perspective, character growth, voice and dialogue, tension, description",
        "Literary fiction: point of view, ensemble casts, naturalistic dialogue, climax structure, prose style",
        "Fiction authorship: narrative design, character backstory, dialogue pacing, dramatic structure, themes",
    ],
    "marketing_digital": [
        "Digital marketing: SEO, social media, content marketing, email campaigns, analytics, conversion optimization",
        "Online marketing: search optimization, social platforms, content strategy, email marketing, web analytics",
        "Digital advertising: search engine optimization, social media marketing, branded content, newsletters, metrics",
        "Internet marketing: organic search, social campaigns, content creation, drip emails, performance tracking",
        "Marketing digital strategy: keyword optimization, social engagement, content distribution, email funnels",
        "Web marketing: SEO techniques, social media ads, blog marketing, automated emails, Google Analytics",
        "Digital growth marketing: search rankings, social growth, content calendars, email automation, AB testing",
        "E-marketing: on-page SEO, influencer marketing, content hubs, segmented emails, attribution modeling",
        "Digital brand marketing: technical SEO, community management, thought leadership content, email nurture",
    ],
    "biology_cell": [
        "Cell biology: organelles, cell cycle, apoptosis, signal transduction, membrane transport, cytoskeleton",
        "Cellular biology: cellular compartments, mitosis/meiosis, programmed cell death, signaling, membranes",
        "Cell science: subcellular structures, cell division, cell death pathways, receptor signaling, transport",
        "Biology of cells: intracellular organelles, cell reproduction, apoptotic mechanisms, cascades, osmosis",
        "Molecular cell biology: endomembrane system, checkpoint regulation, caspases, kinase signaling, channels",
        "Cell structure and function: nucleus/mitochondria, G1/S/G2/M phases, necroptosis, MAPK pathway",
        "Eukaryotic cell biology: ER/Golgi, spindle assembly, death receptors, second messengers, active transport",
        "Cell physiology: organelle function, cytokinesis, intrinsic apoptosis, calcium signaling, endocytosis",
        "Advanced cell biology: autophagy, senescence, Notch/Wnt signaling, vesicular trafficking, actin dynamics",
    ],
    "calculus_math": [
        "Calculus: derivatives, integrals, limits, series, multivariable calculus, differential equations",
        "Mathematical calculus: differentiation, integration, limit theory, power series, vector calculus, ODEs",
        "Applied calculus: rate of change, area under curves, convergence, Taylor series, partial derivatives",
        "Calculus fundamentals: derivative rules, definite integrals, epsilon-delta, Maclaurin series, gradients",
        "Differential and integral calculus: chain rule, Riemann sums, continuity, Fourier series, Jacobians",
        "Calculus analysis: implicit differentiation, improper integrals, L'Hopital's rule, uniform convergence",
        "Computational calculus: numerical differentiation, quadrature, sequence limits, spline approximation",
        "Calculus methods: product/quotient rules, substitution, squeeze theorem, radius of convergence, Stokes",
        "Advanced calculus: Lebesgue integration, measure theory, complex analysis, manifold calculus, forms",
    ],
    "electrical_engineering": [
        "Electrical engineering: circuits, signals, control systems, power electronics, microprocessors, EMC",
        "EE fundamentals: circuit analysis, signal processing, feedback control, power conversion, digital design",
        "Electrical systems: Kirchhoff's laws, Fourier transforms, PID controllers, inverters, embedded systems",
        "Circuit engineering: AC/DC circuits, filter design, control theory, motor drives, FPGA programming",
        "Power and signals: resistive/reactive circuits, spectral analysis, stability, switch-mode power, DSP",
        "Electrical design: Thevenin/Norton, Laplace transforms, Bode plots, rectifiers, microcontroller interfacing",
        "Applied electrical engineering: mesh analysis, Z-transforms, root locus, three-phase power, PCB design",
        "Electronic engineering: RLC circuits, convolution, state-space, voltage regulation, VLSI fundamentals",
        "Electrical technology: nodal analysis, discrete-time signals, Nyquist criteria, transformers, SoC design",
    ],
    "contract_law": [
        "Contract law: formation, consideration, breach, remedies, defenses, UCC, commercial transactions",
        "Law of contracts: offer and acceptance, bargained exchange, material breach, damages, contract defenses",
        "Contractual law: agreement formation, valuable consideration, performance breach, equitable remedies",
        "Contract jurisprudence: mutual assent, promissory estoppel, anticipatory breach, specific performance",
        "Commercial contract law: contract elements, adequacy of consideration, fundamental breach, injunctions",
        "Legal contracts: meeting of minds, detriment/benefit, substantial performance, expectation damages",
        "Contract doctrine: bilateral/unilateral contracts, past consideration, efficient breach, restitution",
        "Applied contract law: express/implied terms, unconscionability, repudiation, liquidated damages, waiver",
        "Business contract law: formation requirements, modification, discharge, measure of damages, arbitration",
    ],
    "music_theory": [
        "Music theory: harmony, melody, rhythm, counterpoint, form, orchestration, ear training, composition",
        "Musical theory: chord progressions, melodic lines, time signatures, voice leading, song structure",
        "Music fundamentals: harmonic analysis, scale construction, rhythmic patterns, contrapuntal techniques",
        "Theory of music: functional harmony, motif development, meter, species counterpoint, arrangement",
        "Musical composition theory: chord voicings, melodic contour, groove, fugal writing, instrumentation",
        "Applied music theory: roman numeral analysis, interval training, syncopation, canon, scoring",
        "Music analysis: harmonic rhythm, pitch organization, polyrhythm, invertible counterpoint, texture",
        "Compositional theory: secondary dominants, pentatonic melody, tuplets, imitative polyphony, dynamics",
        "Music science: jazz harmony, modal melody, complex meters, Baroque counterpoint, digital orchestration",
    ],
    "medical_clinical": [
        "Clinical medicine: diagnosis, treatment, patient care, medical history, physical examination, management",
        "Medical practice: clinical diagnosis, therapeutic intervention, patient assessment, history taking, exams",
        "Clinical practice: differential diagnosis, treatment protocols, bedside manner, anamnesis, vital signs",
        "Medical diagnosis: clinical reasoning, pharmacotherapy, patient communication, symptom history, findings",
        "Clinical healthcare: diagnostic workup, evidence-based treatment, patient-centered care, chief complaint",
        "Practice of medicine: clinical assessment, drug therapy, continuity of care, review of systems",
        "Medical clinical skills: pattern recognition, treatment algorithms, empathic care, focused history",
        "Diagnostic medicine: clinical decision-making, medication management, holistic care, present illness",
        "Clinical management: triage, multimodal treatment, therapeutic alliance, comprehensive history, examination",
    ],
    "environmental_science": [
        "Environmental science: ecosystems, pollution, climate change, sustainability, conservation, biodiversity",
        "Environmental studies: ecological systems, contamination, global warming, sustainable development, wildlife",
        "Ecology and environment: biomes, environmental pollution, greenhouse gases, green practices, species loss",
        "Environmental management: ecosystem services, air/water quality, climate adaptation, circular economy",
        "Earth and environment: habitat dynamics, toxic substances, temperature rise, renewable resources, protection",
        "Environmental protection: ecosystem health, pollutant remediation, carbon emissions, eco-design, reserves",
        "Applied environmental science: trophic levels, waste management, sea level rise, permaculture, corridors",
        "Conservation science: food webs, industrial emissions, ice melt, regenerative agriculture, endangered species",
        "Environmental sustainability: nutrient cycling, microplastics, climate models, zero-waste, rewilding",
    ],
    "financial_accounting": [
        "Financial accounting: balance sheets, income statements, GAAP, auditing, journal entries, reporting",
        "Accounting principles: financial statements, revenue recognition, IFRS/GAAP, audit procedures, ledgers",
        "Financial reporting: statement of financial position, profit/loss, accounting standards, attestation",
        "Accountancy: asset/liability reporting, comprehensive income, generally accepted principles, internal audit",
        "Corporate accounting: consolidated statements, accrual accounting, FASB standards, external audit, books",
        "Applied accounting: trial balance, cash flow statements, matching principle, compliance audit, posting",
        "Accounting fundamentals: debits/credits, shareholders equity, materiality, sampling audit, closing entries",
        "Professional accounting: chart of accounts, retained earnings, GAAP compliance, risk-based audit, T-accounts",
        "Managerial accounting: variance analysis, budgeting, cost allocation, internal controls, management reports",
    ],
}

# Verify we have exactly 30 bases x 9 paraphrases = 270
assert len(NEGATIVE_CONTROL_BASES) == 30, f"Expected 30 base domains, got {len(NEGATIVE_CONTROL_BASES)}"
for base, paraphrases in NEGATIVE_CONTROL_BASES.items():
    assert len(paraphrases) == 9, f"{base} has {len(paraphrases)} paraphrases, expected 9"


def build_negative_control_domains():
    """Build domain list from negative control (9 paraphrases of 30 domains)."""
    domains = []
    for i, (base_name, paraphrases) in enumerate(NEGATIVE_CONTROL_BASES.items()):
        for j, desc in enumerate(paraphrases):
            name = f"{base_name}_v{j+1}"
            domains.append({
                "name": name,
                "path": f"control/{base_name}/{name}",
                "description": desc,
                "base_domain": base_name,
            })
    return domains


def analyze_taxonomy(domains_with_desc, label, verbose=True):
    """Run overlap analysis on a set of domains. Returns metrics dict."""
    print(f"\n{'='*70}")
    print(f"  Analyzing: {label}")
    print(f"  N domains: {len(domains_with_desc)}")
    print(f"{'='*70}")

    # Compute embeddings
    cos_semantic, embeddings = compute_embeddings_semantic(domains_with_desc)
    if cos_semantic is not None:
        cos_matrix = cos_semantic
        method = "sentence-transformers (all-MiniLM-L6-v2)"
    else:
        cos_tfidf, _ = compute_embeddings_tfidf(domains_with_desc)
        cos_matrix = cos_tfidf
        method = "TF-IDF"

    N = len(domains_with_desc)
    upper_idx = np.triu_indices(N, k=1)
    pairwise_cos = cos_matrix[upper_idx]
    n_pairs = len(pairwise_cos)

    # Basic stats
    mean_cos = float(np.mean(pairwise_cos))
    median_cos = float(np.median(pairwise_cos))
    std_cos = float(np.std(pairwise_cos))
    max_cos = float(np.max(pairwise_cos))

    if verbose:
        print(f"\n  Pairwise cosine stats:")
        print(f"    Mean: {mean_cos:.4f}")
        print(f"    Median: {median_cos:.4f}")
        print(f"    Std: {std_cos:.4f}")
        print(f"    Max: {max_cos:.4f}")

    # Overlap at multiple thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    overlap_at = {}
    for t in thresholds:
        n_above = int(np.sum(pairwise_cos > t))
        frac = n_above / n_pairs
        overlap_at[str(t)] = {"n_pairs": n_above, "fraction": frac}
        if verbose:
            print(f"    Pairs cos > {t}: {n_above}/{n_pairs} = {frac*100:.2f}%")

    # Nearest-neighbor analysis (for K2)
    nn_cos_list = []
    nn_above_07 = 0
    for i in range(N):
        row = cos_matrix[i].copy()
        row[i] = -1
        nn_cos = float(np.max(row))
        nn_cos_list.append(nn_cos)
        if nn_cos > 0.7:
            nn_above_07 += 1

    nn_above_07_frac = nn_above_07 / N

    if verbose:
        print(f"\n  Nearest-neighbor analysis:")
        print(f"    Mean NN cos: {np.mean(nn_cos_list):.4f}")
        print(f"    Median NN cos: {np.median(nn_cos_list):.4f}")
        print(f"    Max NN cos: {max(nn_cos_list):.4f}")
        print(f"    Domains with NN cos > 0.7: {nn_above_07}/{N} = {nn_above_07_frac*100:.1f}%")

    # Tightened kill criteria
    k1_frac = overlap_at["0.5"]["fraction"]  # >5% of pairs cos > 0.5
    k1_pass = k1_frac <= 0.05
    k2_frac = nn_above_07_frac  # >5% of domains have NN cos > 0.7
    k2_pass = k2_frac <= 0.05

    if verbose:
        print(f"\n  TIGHTENED Kill Criteria:")
        print(f"    K1: {k1_frac*100:.2f}% of pairs have cos>0.5 "
              f"-> {'PASS' if k1_pass else 'FAIL'} (threshold <=5%)")
        print(f"    K2: {k2_frac*100:.1f}% of domains have NN cos>0.7 "
              f"-> {'PASS' if k2_pass else 'FAIL'} (threshold <=5%)")

    metrics = {
        "label": label,
        "n_domains": N,
        "n_pairs": n_pairs,
        "method": method,
        "mean_cos": mean_cos,
        "median_cos": median_cos,
        "std_cos": std_cos,
        "max_cos": max_cos,
        "overlap_at_thresholds": overlap_at,
        "nn_mean": float(np.mean(nn_cos_list)),
        "nn_median": float(np.median(nn_cos_list)),
        "nn_max": float(max(nn_cos_list)),
        "nn_above_07_count": nn_above_07,
        "nn_above_07_frac": nn_above_07_frac,
        "k1_tightened": {
            "metric": "fraction of pairs with cos>0.5",
            "value": k1_frac,
            "threshold": 0.05,
            "pass": k1_pass,
        },
        "k2_tightened": {
            "metric": "fraction of domains with NN cos>0.7",
            "value": k2_frac,
            "threshold": 0.05,
            "pass": k2_pass,
        },
    }

    return metrics, cos_matrix


def pilot50_proxy_validation():
    """Fix 3: Honest assessment of embedding proxy vs pilot-50 expert outcomes."""
    print(f"\n{'='*70}")
    print(f"  Fix 3: Pilot-50 Proxy Validation")
    print(f"{'='*70}")

    if not PILOT50_RESULTS.exists():
        print("  [SKIP] pilot50_benchmark.json not found")
        return None

    with open(PILOT50_RESULTS) as f:
        pilot_data = json.load(f)

    # Rebuild embeddings for the good taxonomy
    leaves = flatten_taxonomy(TAXONOMY)
    leaf_names = [name for _, name in leaves]
    leaf_paths = [path for path, _ in leaves]

    domains_with_desc = []
    for path, name in leaves:
        if name in DOMAIN_DESCRIPTIONS:
            desc = DOMAIN_DESCRIPTIONS[name]
        else:
            desc = get_domain_description(path, name)
        domains_with_desc.append({"name": name, "path": path, "description": desc})

    cos_semantic, embeddings = compute_embeddings_semantic(domains_with_desc)
    if cos_semantic is None:
        cos_tfidf, _ = compute_embeddings_tfidf(domains_with_desc)
        cos_matrix = cos_tfidf
    else:
        cos_matrix = cos_semantic

    N = len(leaves)
    pilot_domains = pilot_data["domains"]

    # Collect pairs for correlation
    improvements = []
    max_sibling_cos_vals = []
    max_any_cos_vals = []
    base_ppls = []
    domain_names = []

    for pilot_name, mapped_name in PILOT50_TO_TAXONOMY.items():
        if pilot_name not in pilot_domains:
            continue
        if mapped_name not in leaf_names:
            continue

        pilot_result = pilot_domains[pilot_name]
        improvement = pilot_result["improvement_pct"]
        base_ppl = pilot_result["base_ppl"]

        idx = leaf_names.index(mapped_name)

        # Max cosine to any sibling in same category
        my_cat = "/".join(leaf_paths[idx].split("/")[:2])
        sibling_cos = []
        for j in range(N):
            if j == idx:
                continue
            j_cat = "/".join(leaf_paths[j].split("/")[:2])
            if j_cat == my_cat:
                sibling_cos.append(cos_matrix[idx, j])

        max_sib = max(sibling_cos) if sibling_cos else 0.0

        # Max cosine to any domain
        row = cos_matrix[idx].copy()
        row[idx] = -1
        max_any = float(np.max(row))

        improvements.append(improvement)
        max_sibling_cos_vals.append(max_sib)
        max_any_cos_vals.append(max_any)
        base_ppls.append(base_ppl)
        domain_names.append(pilot_name)

    improvements = np.array(improvements)
    max_sibling_cos_vals = np.array(max_sibling_cos_vals)
    max_any_cos_vals = np.array(max_any_cos_vals)
    base_ppls = np.array(base_ppls)

    from scipy.stats import pearsonr, spearmanr

    # Correlation: embedding overlap vs improvement
    r_sib, p_sib = pearsonr(max_sibling_cos_vals, improvements)
    rho_sib, p_rho_sib = spearmanr(max_sibling_cos_vals, improvements)
    r_any, p_any = pearsonr(max_any_cos_vals, improvements)
    r_base, p_base = pearsonr(base_ppls, improvements)

    print(f"\n  Pilot-50 domains matched: {len(improvements)}")
    print(f"\n  Embedding proxy correlations:")
    print(f"    Pearson r(max_sibling_cos, improvement): {r_sib:.3f} (p={p_sib:.3f})")
    print(f"    Spearman rho(max_sibling_cos, improvement): {rho_sib:.3f} (p={p_rho_sib:.3f})")
    print(f"    Pearson r(max_any_cos, improvement): {r_any:.3f} (p={p_any:.3f})")
    print(f"    Pearson r(base_ppl, improvement): {r_base:.3f} (p={p_base:.3f})")

    print(f"\n  HONEST ASSESSMENT:")
    print(f"    The embedding proxy does NOT predict expert improvement quality.")
    print(f"    r={r_sib:.3f} is effectively zero (p={p_sib:.3f}, not significant).")
    print(f"    This means: knowing two domains have similar embeddings tells")
    print(f"    you NOTHING about whether they produce similar/different experts.")
    print(f"    The proxy validates only that domain NAMES are distinct -- not")
    print(f"    that domain EXPERTS will be distinct or useful.")
    print(f"    The embedding proxy is a necessary-but-not-sufficient condition:")
    print(f"    domains that are identical in embedding space would definitely")
    print(f"    produce similar experts, but distinct embeddings do not guarantee")
    print(f"    distinct or useful experts.")

    # Check if any pilot-50 domains that are close in embedding space
    # show correlated expert performance
    print(f"\n  Closest pilot-50 domain pairs in embedding space:")
    n_matched = len(improvements)
    closest_pairs = []
    for i in range(n_matched):
        for j in range(i+1, n_matched):
            idx_i = leaf_names.index(PILOT50_TO_TAXONOMY.get(domain_names[i], ""))
            idx_j = leaf_names.index(PILOT50_TO_TAXONOMY.get(domain_names[j], ""))
            pair_cos = cos_matrix[idx_i, idx_j]
            closest_pairs.append((pair_cos, domain_names[i], domain_names[j],
                                  improvements[i], improvements[j]))

    closest_pairs.sort(reverse=True)
    for cos_val, d1, d2, imp1, imp2 in closest_pairs[:10]:
        print(f"    cos={cos_val:.3f}: {d1} ({imp1:.1f}%) vs {d2} ({imp2:.1f}%)")

    result = {
        "n_matched": int(n_matched),
        "pearson_sibling_cos_vs_improvement": {"r": float(r_sib), "p": float(p_sib)},
        "spearman_sibling_cos_vs_improvement": {"rho": float(rho_sib), "p": float(p_rho_sib)},
        "pearson_any_cos_vs_improvement": {"r": float(r_any), "p": float(p_any)},
        "pearson_base_ppl_vs_improvement": {"r": float(r_base), "p": float(p_base)},
        "assessment": "Embedding proxy does NOT predict expert improvement. "
                      "r=0.034 is effectively zero. Proxy validates name distinctness only.",
        "closest_pilot_pairs": [
            {"cos": float(c), "d1": d1, "d2": d2, "imp1": float(i1), "imp2": float(i2)}
            for c, d1, d2, i1, i2 in closest_pairs[:10]
        ],
    }
    return result


def main():
    t0 = time.time()
    print("=" * 70)
    print("Domain Taxonomy REVISED Validation")
    print("Addresses 4 fixes from adversarial review")
    print("=" * 70)

    # ── Fix 1: Build both taxonomies ──
    # Good taxonomy (existing 270 domains)
    leaves = flatten_taxonomy(TAXONOMY)
    good_domains = []
    for path, name in leaves:
        if name in DOMAIN_DESCRIPTIONS:
            desc = DOMAIN_DESCRIPTIONS[name]
        else:
            desc = get_domain_description(path, name)
        good_domains.append({"name": name, "path": path, "description": desc})

    # Negative control (270 paraphrases of 30 domains)
    bad_domains = build_negative_control_domains()

    print(f"\n  Good taxonomy: {len(good_domains)} domains")
    print(f"  Negative control: {len(bad_domains)} domains "
          f"({len(NEGATIVE_CONTROL_BASES)} bases x 9 paraphrases)")

    # ── Analyze both ──
    print("\n\n" + "=" * 70)
    print("  PART 1: GOOD TAXONOMY (270 distinct domains)")
    print("=" * 70)
    good_metrics, good_cos = analyze_taxonomy(good_domains, "Good Taxonomy (270 distinct)")

    print("\n\n" + "=" * 70)
    print("  PART 2: NEGATIVE CONTROL (270 paraphrases of 30 domains)")
    print("=" * 70)
    bad_metrics, bad_cos = analyze_taxonomy(bad_domains, "Negative Control (30 x 9 paraphrases)")

    # ── Fix 1 verdict: Does the metric discriminate? ──
    print("\n\n" + "=" * 70)
    print("  FIX 1 VERDICT: Discriminative Power of Metrics")
    print("=" * 70)

    good_k1 = good_metrics["k1_tightened"]["pass"]
    good_k2 = good_metrics["k2_tightened"]["pass"]
    bad_k1 = bad_metrics["k1_tightened"]["pass"]
    bad_k2 = bad_metrics["k2_tightened"]["pass"]

    print(f"\n  {'Metric':<40} {'Good Taxonomy':<20} {'Neg Control':<20}")
    print(f"  {'-'*40} {'-'*20} {'-'*20}")
    print(f"  {'K1 (<=5% pairs cos>0.5)':<40} "
          f"{'PASS' if good_k1 else 'FAIL':<20} "
          f"{'PASS' if bad_k1 else 'FAIL':<20}")
    print(f"  {'K2 (<=5% domains NN cos>0.7)':<40} "
          f"{'PASS' if good_k2 else 'FAIL':<20} "
          f"{'PASS' if bad_k2 else 'FAIL':<20}")

    good_k1_val = good_metrics["k1_tightened"]["value"]
    bad_k1_val = bad_metrics["k1_tightened"]["value"]
    good_k2_val = good_metrics["k2_tightened"]["value"]
    bad_k2_val = bad_metrics["k2_tightened"]["value"]

    print(f"\n  K1 values: good={good_k1_val*100:.2f}%, bad={bad_k1_val*100:.2f}%")
    print(f"  K2 values: good={good_k2_val*100:.1f}%, bad={bad_k2_val*100:.1f}%")

    discriminates = (good_k1 and not bad_k1) or (good_k2 and not bad_k2)
    if discriminates:
        print(f"\n  METRICS DISCRIMINATE: Good taxonomy passes, bad taxonomy fails.")
        print(f"  The tightened criteria have real discriminative power.")
    else:
        if good_k1 and good_k2 and bad_k1 and bad_k2:
            print(f"\n  WARNING: Both taxonomies pass -- metrics still too lenient.")
        elif not good_k1 or not good_k2:
            print(f"\n  WARNING: Good taxonomy fails tightened criteria.")
        else:
            print(f"\n  METRICS DO NOT DISCRIMINATE: Both pass or both fail.")

    # Separation ratio
    if bad_k1_val > 0 and good_k1_val > 0:
        sep_k1 = bad_k1_val / good_k1_val
        print(f"\n  K1 separation ratio (bad/good): {sep_k1:.1f}x")
    if bad_k2_val > 0 and good_k2_val > 0:
        sep_k2 = bad_k2_val / good_k2_val
        print(f"\n  K2 separation ratio (bad/good): {sep_k2:.1f}x")

    # Mean cosine comparison
    print(f"\n  Mean pairwise cosine: good={good_metrics['mean_cos']:.4f}, "
          f"bad={bad_metrics['mean_cos']:.4f} "
          f"(ratio: {bad_metrics['mean_cos']/good_metrics['mean_cos']:.1f}x)")

    # ── Fix 2: Tightened criteria already evaluated above ──
    print(f"\n\n{'='*70}")
    print(f"  FIX 2: Tightened Kill Criteria Assessment")
    print(f"{'='*70}")
    print(f"\n  OLD criteria (vacuous):")
    print(f"    K1: >30% pairs cos>0.7 -> actual 0.01% (3000x margin)")
    print(f"    K2: >20% domains NN cos>0.85 -> actual 0.0% (infinite margin)")
    print(f"\n  NEW criteria (tightened):")
    print(f"    K1: >5% pairs cos>0.5 -> actual {good_k1_val*100:.2f}% "
          f"({'PASS' if good_k1 else 'FAIL'}, "
          f"{0.05/good_k1_val:.1f}x margin)" if good_k1_val > 0 else
          f"    K1: >5% pairs cos>0.5 -> actual 0% (PASS)")
    print(f"    K2: >5% domains NN cos>0.7 -> actual {good_k2_val*100:.1f}% "
          f"({'PASS' if good_k2 else 'FAIL'}, "
          f"{0.05/good_k2_val:.1f}x margin)" if good_k2_val > 0 else
          f"    K2: >5% domains NN cos>0.7 -> actual 0% (PASS)")

    # ── Fix 3: Pilot-50 proxy validation ──
    pilot_result = pilot50_proxy_validation()

    # ── Save comprehensive results ──
    runtime = time.time() - t0
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Runtime: {runtime:.1f}s")
    print(f"  Good taxonomy: K1={'PASS' if good_k1 else 'FAIL'}, K2={'PASS' if good_k2 else 'FAIL'}")
    print(f"  Negative control: K1={'PASS' if bad_k1 else 'FAIL'}, K2={'PASS' if bad_k2 else 'FAIL'}")
    print(f"  Discriminative: {discriminates}")
    print(f"  Embedding proxy predicts expert quality: NO (r~0.03)")

    results = {
        "revision": "v2 -- adversarial review fixes",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "good_taxonomy": good_metrics,
        "negative_control": bad_metrics,
        "discriminative_power": {
            "good_passes_k1": good_k1,
            "good_passes_k2": good_k2,
            "bad_fails_k1": not bad_k1,
            "bad_fails_k2": not bad_k2,
            "metrics_discriminate": discriminates,
            "k1_separation_ratio": bad_k1_val / good_k1_val if good_k1_val > 0 else float("inf"),
            "k2_separation_ratio": bad_k2_val / good_k2_val if good_k2_val > 0 else float("inf"),
        },
        "pilot50_proxy": pilot_result,
        "runtime_seconds": runtime,
    }

    results_file = RESULTS_DIR / "results_revised.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_file}")


if __name__ == "__main__":
    main()
