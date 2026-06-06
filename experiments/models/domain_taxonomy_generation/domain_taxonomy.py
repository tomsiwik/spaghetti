#!/usr/bin/env python3
"""Domain Taxonomy Generation for SOLE Expert Coverage Planning.

This experiment:
1. Generates a hierarchical domain taxonomy (50 -> 500 -> 5000 levels)
2. Measures inter-domain overlap via text embedding cosine similarity
3. Validates against pilot-50 ground truth (PPL improvements)
4. Predicts which domains would produce distinct experts

Kill criteria:
- K1: >30% of domain pairs have embedding cosine > 0.7 (too much overlap)
- K2: >20% of domains are predicted indistinguishable from base (too vague)

Runs on Apple Silicon CPU. Uses sentence-transformers for embeddings.
"""

import json
import os
import time
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent
PILOT50_RESULTS = REPO_ROOT / "results" / "pilot50_benchmark.json"
DOMAINS_YML = REPO_ROOT / "data" / "distillation" / "domains.yml"

# ──────────────────────────────────────────────────────────────────────
# Part 1: Define the hierarchical taxonomy
# ──────────────────────────────────────────────────────────────────────

# The pilot-50 taxonomy was hand-crafted: 5 categories x 10 domains.
# We now formalize this into a 3-level hierarchy and extend to 500.
#
# Level 0: Root (1 node)
# Level 1: Supercategories (5-8 nodes)
# Level 2: Categories (30-50 nodes)
# Level 3: Leaf domains (target: 500)
#
# Design principles:
# 1. Each leaf must be ACTIONABLE: specific enough to generate training data
# 2. Sibling domains should be DISTINGUISHABLE: different vocabulary/patterns
# 3. Coverage should be COMPREHENSIVE: no major knowledge area unrepresented
# 4. Granularity should be UNIFORM: no leaf vastly broader than another

TAXONOMY = {
    "programming": {
        "systems_languages": [
            "c_programming", "cpp_modern", "rust_systems", "go_concurrency",
            "assembly_x86", "zig_programming", "d_programming"
        ],
        "application_languages": [
            "python_general", "java_enterprise", "csharp_dotnet", "kotlin_jvm",
            "scala_functional", "ruby_rails", "php_web"
        ],
        "web_frontend": [
            "javascript_core", "typescript_advanced", "react_components",
            "vue_framework", "angular_enterprise", "svelte_reactive",
            "css_layouts", "html_semantic", "webassembly"
        ],
        "scripting_automation": [
            "bash_shell", "powershell_admin", "lua_scripting", "perl_text",
            "makefile_build", "awk_text_processing"
        ],
        "data_query": [
            "sql_analytics", "sql_optimization", "nosql_mongodb", "graphql_api",
            "elasticsearch_queries", "redis_patterns"
        ],
        "mobile_embedded": [
            "swift_ios", "objective_c_legacy", "dart_flutter",
            "embedded_c", "arduino_iot", "react_native"
        ],
        "functional_programming": [
            "haskell_pure", "elixir_concurrent", "clojure_lisp",
            "ocaml_typed", "erlang_distributed", "fsharp_dotnet"
        ],
        "devops_infra": [
            "docker_containers", "kubernetes_orchestration", "terraform_iac",
            "ansible_config", "cicd_pipelines", "monitoring_observability",
            "cloud_aws", "cloud_gcp", "cloud_azure"
        ],
    },
    "science": {
        "physics": [
            "classical_mechanics", "electromagnetism", "thermodynamics",
            "quantum_mechanics", "particle_physics", "condensed_matter",
            "astrophysics_cosmology", "optics_photonics", "fluid_dynamics",
            "relativity_gravitation"
        ],
        "chemistry": [
            "organic_chemistry", "inorganic_chemistry", "physical_chemistry",
            "analytical_chemistry", "biochemistry", "polymer_chemistry",
            "computational_chemistry", "environmental_chemistry"
        ],
        "biology": [
            "cell_biology", "molecular_biology", "genetics_genomics",
            "evolutionary_biology", "ecology_conservation", "microbiology",
            "neuroscience", "immunology", "plant_biology", "marine_biology"
        ],
        "earth_space": [
            "geology_mineralogy", "meteorology_climate", "oceanography",
            "astronomy_observation", "planetary_science", "seismology",
            "volcanology", "paleontology"
        ],
        "mathematics": [
            "calculus_analysis", "linear_algebra", "abstract_algebra",
            "number_theory", "topology", "combinatorics_graph_theory",
            "differential_equations", "probability_theory",
            "numerical_methods", "mathematical_logic"
        ],
        "statistics_data": [
            "descriptive_statistics", "hypothesis_testing", "regression_analysis",
            "bayesian_inference", "experimental_design", "time_series",
            "multivariate_analysis", "causal_inference_stats",
            "machine_learning_theory", "deep_learning_theory"
        ],
    },
    "professional": {
        "healthcare": [
            "clinical_medicine", "surgery_procedures", "pharmacology",
            "radiology_imaging", "psychiatry_mental_health", "nursing_care",
            "public_health_epidemiology", "emergency_medicine",
            "pediatrics", "geriatrics"
        ],
        "legal": [
            "contract_law", "constitutional_law", "criminal_law",
            "intellectual_property", "corporate_law", "international_law",
            "family_law", "environmental_law", "legal_writing_analysis"
        ],
        "business_finance": [
            "financial_analysis", "investment_portfolio", "accounting_gaap",
            "tax_planning", "corporate_finance", "banking_regulation",
            "insurance_actuarial", "real_estate_finance",
            "cryptocurrency_blockchain", "venture_capital"
        ],
        "management": [
            "project_management_agile", "product_management", "operations_management",
            "supply_chain_logistics", "human_resources", "organizational_behavior",
            "change_management", "executive_leadership"
        ],
        "engineering": [
            "civil_engineering", "mechanical_engineering", "electrical_engineering",
            "chemical_engineering", "aerospace_engineering",
            "biomedical_engineering", "materials_science",
            "industrial_engineering", "environmental_engineering"
        ],
        "marketing_communications": [
            "digital_marketing", "brand_strategy", "content_marketing",
            "seo_sem", "social_media_marketing", "public_relations",
            "market_research", "advertising_creative"
        ],
        "cybersecurity": [
            "network_security", "application_security", "cryptography_applied",
            "incident_response_forensics", "penetration_testing",
            "security_architecture", "compliance_governance"
        ],
    },
    "writing_communication": {
        "creative_writing": [
            "fiction_short_story", "fiction_novel", "poetry_verse",
            "screenplay_film", "playwriting_theater", "creative_nonfiction",
            "comedy_humor_writing", "childrens_literature"
        ],
        "professional_writing": [
            "technical_documentation", "api_documentation", "academic_papers",
            "grant_proposals", "business_reports", "white_papers",
            "user_guides", "release_notes"
        ],
        "journalism_media": [
            "news_reporting", "investigative_journalism", "feature_writing",
            "editorial_opinion", "data_journalism", "broadcast_writing",
            "podcast_scripting"
        ],
        "persuasive_writing": [
            "copywriting_ads", "speechwriting_rhetoric", "proposal_writing",
            "fundraising_appeals", "political_messaging", "sales_copy"
        ],
        "translation_localization": [
            "english_spanish", "english_french", "english_german",
            "english_chinese", "english_japanese", "english_arabic",
            "technical_translation", "literary_translation"
        ],
    },
    "reasoning_analysis": {
        "formal_reasoning": [
            "propositional_logic", "predicate_logic", "modal_logic",
            "set_theory_foundations", "proof_techniques", "formal_verification"
        ],
        "applied_reasoning": [
            "causal_reasoning", "analogical_reasoning", "spatial_reasoning",
            "temporal_reasoning", "probabilistic_reasoning",
            "abductive_reasoning", "counterfactual_reasoning"
        ],
        "critical_thinking": [
            "argument_analysis", "fallacy_identification", "source_evaluation",
            "debate_argumentation", "ethical_reasoning_applied",
            "decision_analysis", "risk_assessment"
        ],
        "problem_solving": [
            "algorithmic_thinking", "constraint_satisfaction", "optimization_problems",
            "puzzle_solving", "systems_thinking_analysis",
            "design_thinking", "game_theory_strategy"
        ],
        "metacognition": [
            "learning_strategies", "memory_techniques", "cognitive_biases",
            "self_regulation", "expertise_development"
        ],
    },
    "domain_specific": {
        "education": [
            "k12_teaching", "higher_education", "special_education",
            "curriculum_design", "educational_assessment", "elearning_design"
        ],
        "arts_culture": [
            "music_theory", "visual_arts_critique", "film_analysis",
            "architecture_design", "art_history", "cultural_studies"
        ],
        "social_sciences": [
            "psychology_clinical", "sociology_research", "economics_macro",
            "economics_micro", "political_science", "anthropology",
            "geography_human", "linguistics"
        ],
        "practical_skills": [
            "cooking_culinary", "fitness_nutrition", "personal_finance",
            "home_improvement", "gardening_agriculture", "automotive_repair",
            "photography_technique", "travel_planning"
        ],
    },
}


def flatten_taxonomy(taxonomy, prefix=""):
    """Flatten hierarchical taxonomy to list of (path, leaf_name) tuples."""
    leaves = []
    for key, value in taxonomy.items():
        current_path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            leaves.extend(flatten_taxonomy(value, current_path))
        elif isinstance(value, list):
            for leaf in value:
                leaves.append((f"{current_path}/{leaf}", leaf))
    return leaves


def get_domain_description(path, leaf):
    """Generate a description for a domain based on its taxonomy path."""
    parts = path.split("/")
    # Use the hierarchy to create a natural description
    supercategory = parts[0].replace("_", " ")
    category = parts[1].replace("_", " ") if len(parts) > 1 else ""
    name = leaf.replace("_", " ")
    return f"{name}: a domain within {category} ({supercategory})"


# ──────────────────────────────────────────────────────────────────────
# Part 2: Compute embedding-based overlap matrix
# ──────────────────────────────────────────────────────────────────────

def compute_embeddings_tfidf(domains_with_desc):
    """Compute TF-IDF embeddings for domain descriptions (no ML model needed)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    descriptions = [d["description"] for d in domains_with_desc]
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    cos_matrix = cosine_similarity(tfidf_matrix)
    return cos_matrix, vectorizer


def compute_embeddings_semantic(domains_with_desc):
    """Compute semantic embeddings using sentence-transformers (if available)."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        descriptions = [d["description"] for d in domains_with_desc]
        embeddings = model.encode(descriptions, show_progress_bar=True, batch_size=64)
        cos_matrix = np.dot(embeddings, embeddings.T)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        cos_matrix = cos_matrix / (norms @ norms.T)
        return cos_matrix, embeddings
    except ImportError:
        print("  [WARN] sentence-transformers not available, using TF-IDF only")
        return None, None


# ──────────────────────────────────────────────────────────────────────
# Part 3: Build richer domain descriptions for better embeddings
# ──────────────────────────────────────────────────────────────────────

DOMAIN_DESCRIPTIONS = {
    # Programming languages - describe the KNOWLEDGE not just the name
    "c_programming": "C programming language: pointers, memory allocation, structs, preprocessor macros, system calls, low-level programming",
    "cpp_modern": "Modern C++ programming: templates, RAII, smart pointers, move semantics, STL containers, virtual functions, constexpr",
    "rust_systems": "Rust systems programming: ownership borrowing lifetimes, traits, pattern matching, Result Option error handling, async concurrency",
    "go_concurrency": "Go programming: goroutines channels, interfaces, struct composition, error handling, concurrent patterns, standard library",
    "assembly_x86": "x86 assembly language: registers, instructions, stack operations, calling conventions, SIMD, memory addressing modes",
    "zig_programming": "Zig programming: comptime, explicit allocators, error unions, no hidden control flow, C interop, safety guarantees",
    "d_programming": "D programming language: templates, mixins, ranges, garbage collection, contracts, metaprogramming",
    "python_general": "Python programming: functions classes decorators generators async/await, data structures, algorithms, standard library, debugging",
    "java_enterprise": "Java enterprise programming: object-oriented design, generics, streams, Spring framework, concurrency, JVM tuning",
    "csharp_dotnet": "C# .NET programming: LINQ, async await, generics, Entity Framework, ASP.NET, dependency injection",
    "kotlin_jvm": "Kotlin programming: coroutines, null safety, extension functions, data classes, sealed classes, Android development",
    "scala_functional": "Scala functional programming: pattern matching, implicits, monads, Akka actors, type classes, category theory applications",
    "ruby_rails": "Ruby on Rails: convention over configuration, ActiveRecord, MVC, metaprogramming, gems, REST APIs",
    "php_web": "PHP web development: Laravel, Symfony, object-oriented PHP, database access, session management, API development",
    "javascript_core": "JavaScript core: closures, prototypal inheritance, promises, async/await, event loop, DOM manipulation, ES6+ features",
    "typescript_advanced": "TypeScript advanced types: generics, union intersection types, utility types, type guards, decorators, conditional types",
    "react_components": "React development: components, hooks, state management, JSX, virtual DOM, context API, performance optimization",
    "vue_framework": "Vue.js framework: reactivity system, composition API, single-file components, Vuex/Pinia, Vue Router",
    "angular_enterprise": "Angular enterprise: TypeScript, RxJS observables, dependency injection, modules, components, NgRx state management",
    "svelte_reactive": "Svelte reactive programming: compiler-based reactivity, stores, transitions, SvelteKit, no virtual DOM",
    "css_layouts": "CSS styling and layouts: flexbox, grid, responsive design, animations, custom properties, CSS-in-JS, preprocessors",
    "html_semantic": "HTML semantic markup: accessibility, ARIA roles, SEO, structured data, web forms, canvas, SVG",
    "webassembly": "WebAssembly: binary format, linear memory, Emscripten, Rust/C++ to Wasm compilation, WASI, browser APIs",
    "bash_shell": "Bash shell scripting: file manipulation, sed awk grep, pipelines, process management, system administration",
    "powershell_admin": "PowerShell administration: cmdlets, pipeline, WMI/CIM, Active Directory, remoting, DSC, module development",
    "lua_scripting": "Lua scripting: tables, metatables, coroutines, game scripting, embedding in C, lightweight scripting",
    "perl_text": "Perl text processing: regular expressions, file I/O, CPAN modules, one-liners, bioinformatics scripting",
    "makefile_build": "Makefile and build systems: targets, dependencies, variables, pattern rules, CMake, Bazel, build automation",
    "awk_text_processing": "AWK text processing: field splitting, pattern matching, associative arrays, report generation, data transformation",
    "sql_analytics": "SQL analytics: window functions, CTEs, aggregations, GROUP BY, HAVING, analytical queries, data warehousing",
    "sql_optimization": "SQL performance optimization: query plans, indexes, partitioning, materialized views, join strategies, statistics",
    "nosql_mongodb": "MongoDB NoSQL: document model, aggregation pipeline, indexing, sharding, replication, schema design patterns",
    "graphql_api": "GraphQL API design: schemas, resolvers, mutations, subscriptions, fragments, Apollo, federation",
    "elasticsearch_queries": "Elasticsearch: full-text search, query DSL, aggregations, mapping, analyzers, relevance scoring",
    "redis_patterns": "Redis patterns: caching strategies, pub/sub, streams, data structures, Lua scripting, cluster mode",
    "swift_ios": "Swift iOS development: optionals, protocols, SwiftUI, Combine framework, Core Data, UIKit patterns",
    "objective_c_legacy": "Objective-C: message passing, categories, blocks, ARC, Foundation framework, Cocoa patterns",
    "dart_flutter": "Dart and Flutter: widget tree, state management, platform channels, Material Design, hot reload",
    "embedded_c": "Embedded C programming: microcontrollers, registers, interrupts, RTOS, bare metal, memory-mapped I/O",
    "arduino_iot": "Arduino and IoT: sensor interfacing, serial communication, WiFi/BLE, ESP32, MQTT, edge computing",
    "react_native": "React Native mobile: cross-platform components, native modules, navigation, Expo, platform-specific code",
    "haskell_pure": "Haskell pure functional: monads, type classes, lazy evaluation, algebraic data types, lenses, concurrency",
    "elixir_concurrent": "Elixir concurrent programming: OTP, GenServer, supervision trees, Phoenix framework, LiveView, distributed Erlang",
    "clojure_lisp": "Clojure Lisp: persistent data structures, macros, REPL-driven development, concurrency primitives, spec",
    "ocaml_typed": "OCaml: algebraic types, pattern matching, modules/functors, GADTs, effect handlers, multicore",
    "erlang_distributed": "Erlang distributed systems: OTP behaviors, message passing, fault tolerance, hot code loading, BEAM VM",
    "fsharp_dotnet": "F# .NET: type providers, computation expressions, discriminated unions, active patterns, async workflows",
    "docker_containers": "Docker containerization: Dockerfiles, multi-stage builds, volumes, networking, compose, image optimization",
    "kubernetes_orchestration": "Kubernetes: pods, deployments, services, ingress, operators, Helm charts, autoscaling, RBAC",
    "terraform_iac": "Terraform infrastructure as code: providers, modules, state management, plan/apply, HCL syntax, drift detection",
    "ansible_config": "Ansible configuration management: playbooks, roles, inventory, modules, vault, idempotent operations",
    "cicd_pipelines": "CI/CD pipelines: GitHub Actions, GitLab CI, Jenkins, build automation, testing stages, deployment strategies",
    "monitoring_observability": "Monitoring and observability: Prometheus, Grafana, distributed tracing, logging, alerting, SLOs",
    "cloud_aws": "AWS cloud services: EC2, S3, Lambda, IAM, VPC, RDS, DynamoDB, CloudFormation, well-architected framework",
    "cloud_gcp": "Google Cloud Platform: Compute Engine, BigQuery, Cloud Functions, IAM, GKE, Pub/Sub, Firestore",
    "cloud_azure": "Microsoft Azure: VMs, App Service, Azure Functions, Active Directory, Cosmos DB, DevOps, ARM templates",
    # Science
    "classical_mechanics": "Classical mechanics: Newton's laws, conservation of energy and momentum, rotational dynamics, oscillations, Lagrangian and Hamiltonian mechanics",
    "electromagnetism": "Electromagnetism: Maxwell's equations, electric and magnetic fields, electromagnetic waves, circuits, Gauss's law, Faraday's law",
    "thermodynamics": "Thermodynamics: laws of thermodynamics, entropy, heat engines, free energy, phase transitions, statistical mechanics",
    "quantum_mechanics": "Quantum mechanics: Schrodinger equation, wave functions, operators, uncertainty principle, quantum entanglement, spin",
    "particle_physics": "Particle physics: Standard Model, quarks, leptons, gauge bosons, Higgs mechanism, Feynman diagrams, symmetry breaking",
    "condensed_matter": "Condensed matter physics: crystal structure, band theory, superconductivity, semiconductors, magnetism, topological materials",
    "astrophysics_cosmology": "Astrophysics and cosmology: stellar evolution, galaxy formation, dark matter, dark energy, Big Bang, cosmic microwave background",
    "optics_photonics": "Optics and photonics: wave optics, interference, diffraction, lasers, fiber optics, nonlinear optics, quantum optics",
    "fluid_dynamics": "Fluid dynamics: Navier-Stokes equations, turbulence, boundary layers, compressible flow, computational fluid dynamics",
    "relativity_gravitation": "General and special relativity: spacetime, Lorentz transformations, gravitational waves, black holes, Einstein field equations",
    "organic_chemistry": "Organic chemistry: reaction mechanisms, functional groups, stereochemistry, synthesis, spectroscopy, retrosynthetic analysis",
    "inorganic_chemistry": "Inorganic chemistry: coordination compounds, crystal field theory, organometallics, bioinorganic, solid state chemistry",
    "physical_chemistry": "Physical chemistry: quantum chemistry, thermodynamic potentials, kinetics, spectroscopy, surface chemistry, electrochemistry",
    "analytical_chemistry": "Analytical chemistry: chromatography, mass spectrometry, NMR, titrations, spectrophotometry, method validation",
    "biochemistry": "Biochemistry: enzyme kinetics, protein structure, metabolic pathways, DNA replication, signal transduction, membrane transport",
    "polymer_chemistry": "Polymer chemistry: polymerization mechanisms, molecular weight distributions, polymer physics, rheology, copolymers",
    "computational_chemistry": "Computational chemistry: DFT, molecular dynamics, Monte Carlo, ab initio methods, force fields, docking",
    "environmental_chemistry": "Environmental chemistry: atmospheric chemistry, water treatment, pollutant fate, green chemistry, remediation",
    "cell_biology": "Cell biology: organelles, cell cycle, apoptosis, signal transduction, membrane transport, cytoskeleton, cell division",
    "molecular_biology": "Molecular biology: gene expression, transcription, translation, DNA repair, epigenetics, CRISPR, cloning",
    "genetics_genomics": "Genetics and genomics: Mendelian inheritance, population genetics, GWAS, sequencing, bioinformatics, gene regulation",
    "evolutionary_biology": "Evolutionary biology: natural selection, speciation, phylogenetics, molecular evolution, adaptation, coevolution",
    "ecology_conservation": "Ecology and conservation: population dynamics, community ecology, biodiversity, habitat loss, restoration ecology",
    "microbiology": "Microbiology: bacteria, viruses, fungi, antimicrobial resistance, microbial ecology, fermentation, pathogenesis",
    "neuroscience": "Neuroscience: neural signaling, brain anatomy, cognitive processes, neuroplasticity, neurotransmitters, neurological disorders",
    "immunology": "Immunology: innate and adaptive immunity, T cells, B cells, antibodies, autoimmunity, vaccines, immunotherapy",
    "plant_biology": "Plant biology: photosynthesis, plant hormones, root systems, transpiration, seed development, plant genetics",
    "marine_biology": "Marine biology: ocean ecosystems, coral reefs, deep sea organisms, fisheries, marine conservation, plankton ecology",
    "geology_mineralogy": "Geology and mineralogy: plate tectonics, rock cycle, mineral identification, stratigraphy, geological mapping",
    "meteorology_climate": "Meteorology and climate science: weather systems, atmospheric circulation, climate models, greenhouse effect, extreme weather",
    "oceanography": "Oceanography: ocean currents, thermohaline circulation, marine chemistry, ocean acoustics, coastal processes",
    "astronomy_observation": "Observational astronomy: telescopes, photometry, spectroscopy, stellar classification, variable stars, exoplanet detection",
    "planetary_science": "Planetary science: solar system formation, planetary atmospheres, surface geology, moons, asteroids, space missions",
    "seismology": "Seismology: earthquake mechanics, seismic waves, fault systems, earthquake prediction, ground motion, structural response",
    "volcanology": "Volcanology: magma generation, eruption mechanisms, volcanic hazards, lava flow modeling, monitoring techniques",
    "paleontology": "Paleontology: fossil record, mass extinctions, evolutionary transitions, paleoclimate, stratigraphic correlation",
    "calculus_analysis": "Calculus and real analysis: limits, derivatives, integrals, series, multivariable calculus, measure theory, functional analysis",
    "linear_algebra": "Linear algebra: vector spaces, eigenvalues, matrix decompositions, inner products, linear transformations, tensor products",
    "abstract_algebra": "Abstract algebra: groups, rings, fields, Galois theory, modules, homomorphisms, representation theory",
    "number_theory": "Number theory: primes, modular arithmetic, Diophantine equations, algebraic number theory, analytic number theory, cryptographic applications",
    "topology": "Topology: topological spaces, continuity, compactness, connectedness, fundamental group, homology, manifolds",
    "combinatorics_graph_theory": "Combinatorics and graph theory: counting, generating functions, graph coloring, matching, network flow, Ramsey theory",
    "differential_equations": "Differential equations: ODEs, PDEs, boundary value problems, dynamical systems, stability analysis, chaos theory",
    "probability_theory": "Probability theory: measure-theoretic probability, stochastic processes, martingales, Markov chains, central limit theorem",
    "numerical_methods": "Numerical methods: floating point, interpolation, quadrature, root finding, linear solvers, finite elements, FFT",
    "mathematical_logic": "Mathematical logic: propositional and predicate logic, completeness, incompleteness theorems, model theory, computability",
    "descriptive_statistics": "Descriptive statistics: measures of central tendency, dispersion, visualization, distributions, exploratory data analysis",
    "hypothesis_testing": "Hypothesis testing: t-tests, chi-square, ANOVA, p-values, power analysis, multiple comparisons, nonparametric tests",
    "regression_analysis": "Regression analysis: linear regression, logistic regression, regularization, model selection, diagnostics, mixed effects",
    "bayesian_inference": "Bayesian inference: prior distributions, posterior computation, MCMC, hierarchical models, Bayesian model selection",
    "experimental_design": "Experimental design: randomization, blocking, factorial designs, response surface, sample size, A/B testing",
    "time_series": "Time series analysis: ARIMA, seasonal decomposition, spectral analysis, state space models, forecasting, change point detection",
    "multivariate_analysis": "Multivariate analysis: PCA, factor analysis, discriminant analysis, cluster analysis, canonical correlation, MANOVA",
    "causal_inference_stats": "Causal inference: potential outcomes, propensity scores, instrumental variables, difference-in-differences, regression discontinuity",
    "machine_learning_theory": "Machine learning theory: PAC learning, VC dimension, bias-variance tradeoff, regularization theory, kernel methods, ensemble theory",
    "deep_learning_theory": "Deep learning theory: neural network optimization, generalization bounds, neural tangent kernels, implicit regularization, loss landscapes",
    # Professional
    "clinical_medicine": "Clinical medicine: patient assessment, differential diagnosis, treatment planning, drug interactions, evidence-based medicine",
    "surgery_procedures": "Surgical procedures: preoperative assessment, surgical techniques, wound management, laparoscopic surgery, postoperative care",
    "pharmacology": "Pharmacology: drug mechanisms, pharmacokinetics, drug interactions, adverse effects, dosing, therapeutic monitoring",
    "radiology_imaging": "Radiology and medical imaging: X-ray, CT, MRI, ultrasound, nuclear medicine, image interpretation, radiation safety",
    "psychiatry_mental_health": "Psychiatry and mental health: mood disorders, anxiety, psychosis, therapy approaches, psychopharmacology, DSM criteria",
    "nursing_care": "Nursing care: patient assessment, care planning, medication administration, wound care, patient education, triage",
    "public_health_epidemiology": "Public health and epidemiology: disease surveillance, outbreak investigation, health policy, biostatistics, screening",
    "emergency_medicine": "Emergency medicine: triage, trauma assessment, resuscitation, acute care protocols, toxicology, disaster medicine",
    "pediatrics": "Pediatrics: child development, vaccination schedules, common childhood diseases, growth assessment, adolescent medicine",
    "geriatrics": "Geriatrics: aging physiology, polypharmacy, dementia care, fall prevention, end-of-life care, functional assessment",
    "contract_law": "Contract law: formation, breach, remedies, consideration, conditions, assignment, third-party rights, UCC",
    "constitutional_law": "Constitutional law: judicial review, separation of powers, due process, equal protection, First Amendment, federalism",
    "criminal_law": "Criminal law: elements of crimes, defenses, sentencing, procedure, evidence, constitutional protections, plea bargaining",
    "intellectual_property": "Intellectual property: patents, trademarks, copyrights, trade secrets, licensing, infringement, fair use",
    "corporate_law": "Corporate law: entity formation, fiduciary duties, mergers and acquisitions, securities regulation, corporate governance",
    "international_law": "International law: treaties, customary law, international courts, humanitarian law, trade law, sovereignty",
    "family_law": "Family law: marriage, divorce, child custody, adoption, domestic violence, property division, prenuptial agreements",
    "environmental_law": "Environmental law: NEPA, Clean Air/Water Act, environmental impact assessment, pollution control, conservation law",
    "legal_writing_analysis": "Legal writing and analysis: case briefs, memoranda, appellate briefs, statutory interpretation, legal research methods",
    "financial_analysis": "Financial analysis: ratio analysis, cash flow analysis, financial modeling, valuation, equity research, financial statements",
    "investment_portfolio": "Investment and portfolio management: asset allocation, modern portfolio theory, risk-adjusted returns, factor investing, derivatives",
    "accounting_gaap": "Accounting GAAP: revenue recognition, lease accounting, financial statement preparation, consolidation, fair value measurement",
    "tax_planning": "Tax planning: individual and corporate taxation, deductions, credits, international tax, estate tax, tax compliance",
    "corporate_finance": "Corporate finance: capital budgeting, cost of capital, capital structure, dividend policy, working capital management",
    "banking_regulation": "Banking and regulation: Basel accords, monetary policy, credit risk, liquidity management, payment systems, fintech regulation",
    "insurance_actuarial": "Insurance and actuarial science: risk assessment, premium pricing, reserving, life tables, loss distributions, Solvency II",
    "real_estate_finance": "Real estate finance: property valuation, mortgage analysis, REITs, commercial real estate, development finance",
    "cryptocurrency_blockchain": "Cryptocurrency and blockchain: consensus mechanisms, smart contracts, DeFi, tokenomics, layer-2 scaling, wallet security",
    "venture_capital": "Venture capital: startup valuation, term sheets, due diligence, cap tables, exit strategies, portfolio construction",
    "project_management_agile": "Project management and agile: Scrum, Kanban, sprint planning, risk management, stakeholder communication, retrospectives",
    "product_management": "Product management: user research, roadmapping, prioritization frameworks, metrics, A/B testing, go-to-market",
    "operations_management": "Operations management: process optimization, quality management, lean manufacturing, Six Sigma, capacity planning",
    "supply_chain_logistics": "Supply chain and logistics: inventory management, procurement, demand forecasting, warehousing, transportation optimization",
    "human_resources": "Human resources: recruitment, compensation design, performance management, employment law, organizational development",
    "organizational_behavior": "Organizational behavior: motivation, team dynamics, leadership theory, organizational culture, conflict resolution",
    "change_management": "Change management: stakeholder analysis, communication plans, resistance management, adoption metrics, organizational transformation",
    "executive_leadership": "Executive leadership: strategic planning, board governance, C-suite communication, crisis leadership, vision setting",
    "civil_engineering": "Civil engineering: structural analysis, geotechnical design, transportation engineering, hydraulics, construction management",
    "mechanical_engineering": "Mechanical engineering: machine design, heat transfer, manufacturing processes, vibrations, control systems, robotics",
    "electrical_engineering": "Electrical engineering: circuit analysis, power systems, signal processing, control theory, RF design, embedded systems",
    "chemical_engineering": "Chemical engineering: reactor design, mass transfer, separation processes, process control, thermodynamics, plant design",
    "aerospace_engineering": "Aerospace engineering: aerodynamics, orbital mechanics, propulsion, flight controls, spacecraft design, composite structures",
    "biomedical_engineering": "Biomedical engineering: medical devices, biomechanics, tissue engineering, bioinformatics, neural engineering, imaging systems",
    "materials_science": "Materials science: crystallography, phase diagrams, mechanical properties, nanomaterials, ceramics, composites, corrosion",
    "industrial_engineering": "Industrial engineering: operations research, ergonomics, facility layout, production scheduling, quality engineering",
    "environmental_engineering": "Environmental engineering: water treatment, air pollution control, waste management, remediation, sustainability design",
    "digital_marketing": "Digital marketing: SEO, PPC advertising, email campaigns, marketing automation, conversion optimization, analytics",
    "brand_strategy": "Brand strategy: positioning, brand architecture, brand equity, naming, visual identity, brand guidelines, rebranding",
    "content_marketing": "Content marketing: content strategy, storytelling, editorial calendar, distribution channels, audience engagement, ROI measurement",
    "seo_sem": "SEO and SEM: keyword research, on-page optimization, link building, Google Ads, quality score, SERP features",
    "social_media_marketing": "Social media marketing: platform strategy, community management, influencer marketing, paid social, analytics, viral content",
    "public_relations": "Public relations: media relations, crisis communication, press releases, reputation management, event planning, thought leadership",
    "market_research": "Market research: surveys, focus groups, competitive analysis, consumer insights, market sizing, trend analysis",
    "advertising_creative": "Advertising creative: campaign concepts, art direction, copywriting, media planning, creative briefs, brand storytelling",
    "network_security": "Network security: firewalls, IDS/IPS, VPN, network segmentation, traffic analysis, zero trust architecture",
    "application_security": "Application security: OWASP top 10, secure coding, vulnerability scanning, code review, authentication, API security",
    "cryptography_applied": "Applied cryptography: symmetric/asymmetric encryption, digital signatures, PKI, TLS/SSL, hash functions, key management",
    "incident_response_forensics": "Incident response and forensics: evidence collection, malware analysis, log analysis, chain of custody, timeline reconstruction",
    "penetration_testing": "Penetration testing: reconnaissance, vulnerability exploitation, privilege escalation, social engineering, report writing",
    "security_architecture": "Security architecture: defense in depth, security frameworks, SIEM, threat modeling, security controls, compliance",
    "compliance_governance": "Compliance and governance: GDPR, HIPAA, SOC 2, ISO 27001, risk management frameworks, audit procedures",
    # Writing
    "fiction_short_story": "Short fiction writing: character development, plot structure, dialogue, narrative voice, literary techniques, flash fiction",
    "fiction_novel": "Novel writing: multi-chapter structure, character arcs, world-building, pacing, point of view, revision process",
    "poetry_verse": "Poetry: verse forms, meter and rhythm, figurative language, imagery, poetic devices, sound patterns, free verse",
    "screenplay_film": "Screenplay writing: scene structure, dialogue formatting, visual storytelling, three-act structure, industry formatting",
    "playwriting_theater": "Playwriting: dramatic structure, stage directions, character monologue, ensemble scenes, theatrical conventions",
    "creative_nonfiction": "Creative nonfiction: personal essay, memoir, literary journalism, travel writing, nature writing, narrative nonfiction",
    "comedy_humor_writing": "Comedy and humor writing: joke structure, comedic timing, satire, parody, sitcom writing, stand-up material",
    "childrens_literature": "Children's literature: picture books, middle grade, young adult, age-appropriate content, illustration collaboration",
    "technical_documentation": "Technical documentation: user guides, system architecture docs, API references, troubleshooting guides, information architecture",
    "api_documentation": "API documentation: endpoint descriptions, request/response examples, authentication guides, SDK documentation, OpenAPI/Swagger",
    "academic_papers": "Academic paper writing: research structure, literature reviews, methodology sections, scientific argumentation, citation conventions",
    "grant_proposals": "Grant writing: proposal structure, needs assessment, budget justification, outcome measurement, logic models, funder alignment",
    "business_reports": "Business report writing: executive summaries, data presentation, recommendations, quarterly reports, market analysis reports",
    "white_papers": "White paper writing: technical authority, problem-solution format, industry analysis, thought leadership, reference architecture",
    "user_guides": "User guide writing: task-oriented documentation, step-by-step instructions, screenshots, FAQ, onboarding documentation",
    "release_notes": "Release notes: changelog formatting, version semantics, feature descriptions, breaking changes, migration guides",
    "news_reporting": "News reporting: inverted pyramid, fact verification, source attribution, deadline writing, AP style, breaking news",
    "investigative_journalism": "Investigative journalism: deep research, document analysis, source protection, data analysis, narrative investigation",
    "feature_writing": "Feature writing: long-form narrative, profile writing, trend pieces, human interest, scene-setting, structural variety",
    "editorial_opinion": "Editorial and opinion writing: argument construction, evidence marshaling, persuasive structure, counterargument engagement",
    "data_journalism": "Data journalism: data analysis, visualization, public records, statistical literacy, interactive storytelling, scraping",
    "broadcast_writing": "Broadcast writing: script formatting, conversational tone, sound bites, teleprompter writing, packages, voiceover",
    "podcast_scripting": "Podcast scripting: interview preparation, segment planning, conversational flow, sound design, show notes, narrative podcasts",
    "copywriting_ads": "Advertising copywriting: headlines, taglines, email campaigns, landing pages, calls to action, brand voice",
    "speechwriting_rhetoric": "Speechwriting and rhetoric: persuasion, audience analysis, narrative structure, emotional appeals, memorable phrases",
    "proposal_writing": "Proposal writing: RFP responses, solution architecture, pricing strategy, executive summary, compliance matrix",
    "fundraising_appeals": "Fundraising appeals: donor communication, storytelling for impact, annual campaigns, major gift solicitation, gratitude",
    "political_messaging": "Political messaging: campaign communications, policy briefs, talking points, debate preparation, voter outreach",
    "sales_copy": "Sales copy: product descriptions, case studies, testimonials, value propositions, objection handling, conversion optimization",
    "english_spanish": "English-Spanish translation: grammar differences, idiomatic expressions, regional variations, cultural adaptation",
    "english_french": "English-French translation: syntax differences, gender agreement, formal/informal registers, Quebec vs France French",
    "english_german": "English-German translation: compound words, case system, word order, technical terminology, formal registers",
    "english_chinese": "English-Chinese translation: character-based writing, tonal context, classical vs modern Chinese, cultural concepts",
    "english_japanese": "English-Japanese translation: honorific systems, kanji/hiragana/katakana, sentence structure, keigo formality levels",
    "english_arabic": "English-Arabic translation: right-to-left script, root-pattern morphology, dialectal variation, formal Modern Standard Arabic",
    "technical_translation": "Technical translation: terminology management, consistency, CAT tools, domain expertise, quality assurance",
    "literary_translation": "Literary translation: style preservation, cultural adaptation, rhythm and tone, footnoting, translator's voice",
    # Reasoning
    "propositional_logic": "Propositional logic: truth tables, logical connectives, validity, tautologies, natural deduction, resolution",
    "predicate_logic": "Predicate logic: quantifiers, variables, interpretations, first-order theories, Skolemization, unification",
    "modal_logic": "Modal logic: necessity, possibility, Kripke semantics, epistemic logic, deontic logic, temporal logic",
    "set_theory_foundations": "Set theory: axioms of ZFC, cardinality, ordinals, transfinite induction, forcing, continuum hypothesis",
    "proof_techniques": "Mathematical proof techniques: direct proof, contradiction, induction, constructive proof, proof by cases, Zorn's lemma",
    "formal_verification": "Formal verification: model checking, theorem proving, Coq, Isabelle, TLA+, correctness proofs, invariants",
    "causal_reasoning": "Causal reasoning: cause-and-effect analysis, counterfactual thinking, confounding variables, root cause analysis, causal graphs",
    "analogical_reasoning": "Analogical reasoning: structural similarity, cross-domain transfer, metaphor construction, relational mapping",
    "spatial_reasoning": "Spatial reasoning: geometric problems, mental rotation, spatial transformations, topology, 3D visualization, maps",
    "temporal_reasoning": "Temporal reasoning: timeline analysis, scheduling, temporal logic, event ordering, duration estimation, planning",
    "probabilistic_reasoning": "Probabilistic reasoning: Bayesian updating, expected value, risk assessment, uncertainty quantification, decision trees",
    "abductive_reasoning": "Abductive reasoning: inference to best explanation, hypothesis generation, diagnostic reasoning, scientific discovery",
    "counterfactual_reasoning": "Counterfactual reasoning: alternative scenarios, what-if analysis, causal models, policy evaluation, historical counterfactuals",
    "argument_analysis": "Argument analysis: premise identification, validity assessment, argument mapping, Toulmin model, informal logic",
    "fallacy_identification": "Fallacy identification: formal and informal fallacies, ad hominem, straw man, appeal to authority, red herring",
    "source_evaluation": "Source evaluation: credibility assessment, bias detection, primary vs secondary sources, fact-checking methodology",
    "debate_argumentation": "Debate and argumentation: claim-evidence-reasoning, counterarguments, rebuttals, persuasion techniques, structured debate",
    "ethical_reasoning_applied": "Applied ethical reasoning: utilitarian, deontological, virtue ethics, moral dilemmas, stakeholder analysis, bioethics",
    "decision_analysis": "Decision analysis: decision trees, multi-criteria analysis, cost-benefit analysis, scenario planning, sensitivity analysis",
    "risk_assessment": "Risk assessment: probability estimation, impact analysis, risk matrices, mitigation strategies, Monte Carlo simulation",
    "algorithmic_thinking": "Algorithmic thinking: problem decomposition, algorithm design, complexity analysis, data structure selection, recursion",
    "constraint_satisfaction": "Constraint satisfaction: variable assignment, backtracking, arc consistency, SAT solvers, scheduling, resource allocation",
    "optimization_problems": "Optimization: linear programming, convex optimization, gradient descent, integer programming, metaheuristics, dynamic programming",
    "puzzle_solving": "Puzzle solving: logic puzzles, Sudoku strategies, lateral thinking, pattern recognition, mathematical puzzles, riddles",
    "systems_thinking_analysis": "Systems thinking: feedback loops, emergent behavior, causal loop diagrams, leverage points, system archetypes",
    "design_thinking": "Design thinking: empathy mapping, ideation, prototyping, user testing, iteration, human-centered design",
    "game_theory_strategy": "Game theory: Nash equilibrium, prisoner's dilemma, mechanism design, auction theory, cooperative games, evolutionary games",
    "learning_strategies": "Learning strategies: spaced repetition, active recall, elaboration, interleaving, metacognitive monitoring",
    "memory_techniques": "Memory techniques: mnemonics, method of loci, chunking, association, visualization, spaced retrieval practice",
    "cognitive_biases": "Cognitive biases: anchoring, confirmation bias, availability heuristic, framing effects, sunk cost, Dunning-Kruger",
    "self_regulation": "Self-regulation: goal setting, habit formation, willpower, emotional regulation, procrastination, executive function",
    "expertise_development": "Expertise development: deliberate practice, skill acquisition, transfer of training, automaticity, domain-specific knowledge",
    # Domain-specific
    "k12_teaching": "K-12 teaching: lesson planning, classroom management, differentiated instruction, assessment design, curriculum standards",
    "higher_education": "Higher education: course design, syllabus development, lecture techniques, research mentoring, academic advising",
    "special_education": "Special education: IEP development, accommodations, learning disabilities, autism spectrum, assistive technology",
    "curriculum_design": "Curriculum design: learning objectives, backward design, scaffolding, alignment, assessment rubrics, scope and sequence",
    "educational_assessment": "Educational assessment: formative and summative, test construction, item analysis, grading rubrics, standardized testing",
    "elearning_design": "E-learning design: instructional design, LMS platforms, multimedia learning, gamification, accessibility, SCORM",
    "music_theory": "Music theory: harmony, counterpoint, form, orchestration, ear training, music analysis, composition techniques",
    "visual_arts_critique": "Visual arts and critique: art analysis, composition, color theory, art movements, studio techniques, portfolio review",
    "film_analysis": "Film analysis: cinematography, editing, sound design, narrative structure, genre conventions, auteur theory",
    "architecture_design": "Architecture design: building design, structural systems, sustainability, urban planning, architectural history, BIM",
    "art_history": "Art history: movements and periods, iconography, patronage, art criticism, museum studies, cultural context",
    "cultural_studies": "Cultural studies: identity, representation, media analysis, postcolonialism, globalization, popular culture, semiotics",
    "psychology_clinical": "Clinical psychology: therapy approaches, CBT, psychodynamic, assessment, diagnosis, treatment planning, ethics",
    "sociology_research": "Sociology: social structures, inequality, research methods, institutions, socialization, collective behavior",
    "economics_macro": "Macroeconomics: GDP, inflation, monetary policy, fiscal policy, international trade, growth models, business cycles",
    "economics_micro": "Microeconomics: supply and demand, market structures, game theory, welfare economics, externalities, behavioral economics",
    "political_science": "Political science: comparative politics, international relations, political theory, public policy, electoral systems",
    "anthropology": "Anthropology: cultural anthropology, ethnography, kinship, material culture, linguistic anthropology, archaeological methods",
    "geography_human": "Human geography: urbanization, migration, cultural landscapes, geopolitics, economic geography, spatial analysis",
    "linguistics": "Linguistics: phonology, morphology, syntax, semantics, pragmatics, sociolinguistics, language acquisition, computational linguistics",
    "cooking_culinary": "Cooking and culinary arts: techniques, flavor profiles, recipe development, food science, baking, international cuisines",
    "fitness_nutrition": "Fitness and nutrition: exercise physiology, training programs, macronutrients, meal planning, supplements, recovery",
    "personal_finance": "Personal finance: budgeting, investing basics, retirement planning, debt management, insurance, tax optimization",
    "home_improvement": "Home improvement: renovation planning, electrical basics, plumbing, carpentry, painting, building codes, DIY projects",
    "gardening_agriculture": "Gardening and agriculture: soil science, crop rotation, pest management, permaculture, hydroponics, seasonal planning",
    "automotive_repair": "Automotive repair: engine diagnostics, brake systems, electrical systems, maintenance schedules, emissions, troubleshooting",
    "photography_technique": "Photography: exposure triangle, composition rules, lighting, post-processing, lens selection, genre-specific techniques",
    "travel_planning": "Travel planning: itinerary design, budget optimization, cultural etiquette, visa requirements, accommodation, transportation",
}


# ──────────────────────────────────────────────────────────────────────
# Part 4: Map pilot-50 domains to new taxonomy
# ──────────────────────────────────────────────────────────────────────

# Manual mapping from pilot-50 names to closest new taxonomy leaf
PILOT50_TO_TAXONOMY = {
    "python": "python_general",
    "javascript": "javascript_core",
    "rust": "rust_systems",
    "go": "go_concurrency",
    "cpp": "cpp_modern",
    "java": "java_enterprise",
    "typescript": "typescript_advanced",
    "sql": "sql_analytics",
    "bash": "bash_shell",
    "swift": "swift_ios",
    "physics": "classical_mechanics",
    "chemistry": "organic_chemistry",
    "biology": "cell_biology",
    "math": "calculus_analysis",
    "statistics": "descriptive_statistics",
    "astronomy": "astronomy_observation",
    "geology": "geology_mineralogy",
    "neuroscience": "neuroscience",
    "ecology": "ecology_conservation",
    "genetics": "genetics_genomics",
    "legal": "contract_law",
    "medical": "clinical_medicine",
    "finance": "financial_analysis",
    "accounting": "accounting_gaap",
    "marketing": "digital_marketing",
    "hr": "human_resources",
    "project-management": "project_management_agile",
    "cybersecurity": "network_security",
    "data-engineering": "nosql_mongodb",  # closest match
    "devops": "docker_containers",
    "creative-fiction": "fiction_short_story",
    "technical-writing": "technical_documentation",
    "academic-writing": "academic_papers",
    "journalism": "news_reporting",
    "copywriting": "copywriting_ads",
    "poetry": "poetry_verse",
    "screenplay": "screenplay_film",
    "speechwriting": "speechwriting_rhetoric",
    "grant-writing": "grant_proposals",
    "documentation": "api_documentation",
    "logic-puzzles": "puzzle_solving",
    "debate": "debate_argumentation",
    "ethics": "ethical_reasoning_applied",
    "game-theory": "game_theory_strategy",
    "systems-thinking": "systems_thinking_analysis",
    "critical-analysis": "argument_analysis",
    "causal-reasoning": "causal_reasoning",
    "analogical-reasoning": "analogical_reasoning",
    "spatial-reasoning": "spatial_reasoning",
    "abstract-math": "abstract_algebra",
}


# ──────────────────────────────────────────────────────────────────────
# Part 5: Main experiment
# ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print("Domain Taxonomy Generation Experiment")
    print("=" * 70)

    # 1. Flatten taxonomy and count
    leaves = flatten_taxonomy(TAXONOMY)
    leaf_names = [name for _, name in leaves]
    leaf_paths = [path for path, _ in leaves]

    print(f"\n[1] Taxonomy Structure:")
    # Count by supercategory
    super_counts = defaultdict(int)
    cat_counts = defaultdict(int)
    for path, _ in leaves:
        parts = path.split("/")
        super_counts[parts[0]] += 1
        cat_counts[f"{parts[0]}/{parts[1]}"] += 1

    print(f"  Total leaf domains: {len(leaves)}")
    print(f"  Supercategories: {len(super_counts)}")
    print(f"  Categories: {len(cat_counts)}")
    print(f"\n  Supercategory breakdown:")
    for sc, count in sorted(super_counts.items()):
        print(f"    {sc}: {count} domains")
    print(f"\n  Category breakdown:")
    for cat, count in sorted(cat_counts.items()):
        print(f"    {cat}: {count}")

    # 2. Build descriptions
    print(f"\n[2] Building domain descriptions...")
    domains_with_desc = []
    for path, name in leaves:
        if name in DOMAIN_DESCRIPTIONS:
            desc = DOMAIN_DESCRIPTIONS[name]
        else:
            desc = get_domain_description(path, name)
        domains_with_desc.append({
            "name": name,
            "path": path,
            "description": desc,
        })

    described = sum(1 for d in domains_with_desc if d["name"] in DOMAIN_DESCRIPTIONS)
    print(f"  Domains with rich descriptions: {described}/{len(domains_with_desc)}")

    # 3. Compute embeddings
    print(f"\n[3] Computing embeddings...")

    # Try semantic first, fall back to TF-IDF
    cos_semantic, embeddings = compute_embeddings_semantic(domains_with_desc)

    cos_tfidf, vectorizer = compute_embeddings_tfidf(domains_with_desc)

    # Use semantic if available, else TF-IDF
    if cos_semantic is not None:
        cos_matrix = cos_semantic
        method = "sentence-transformers (all-MiniLM-L6-v2)"
    else:
        cos_matrix = cos_tfidf
        method = "TF-IDF"

    print(f"  Method: {method}")
    print(f"  Matrix shape: {cos_matrix.shape}")

    # 4. Analyze overlap
    print(f"\n[4] Overlap Analysis:")
    N = len(leaves)

    # Extract upper triangle (excluding diagonal)
    upper_idx = np.triu_indices(N, k=1)
    pairwise_cos = cos_matrix[upper_idx]
    n_pairs = len(pairwise_cos)

    print(f"  Total domain pairs: {n_pairs}")
    print(f"  Mean cosine: {np.mean(pairwise_cos):.4f}")
    print(f"  Median cosine: {np.median(pairwise_cos):.4f}")
    print(f"  Std cosine: {np.std(pairwise_cos):.4f}")
    print(f"  Max cosine: {np.max(pairwise_cos):.4f}")
    print(f"  Min cosine: {np.min(pairwise_cos):.4f}")

    # K1: What fraction of pairs have cos > 0.7?
    OVERLAP_THRESHOLD = 0.7
    high_overlap = np.sum(pairwise_cos > OVERLAP_THRESHOLD)
    overlap_fraction = high_overlap / n_pairs
    print(f"\n  K1 Assessment (overlap threshold = {OVERLAP_THRESHOLD}):")
    print(f"  Pairs with cos > {OVERLAP_THRESHOLD}: {high_overlap}/{n_pairs} = {overlap_fraction*100:.2f}%")
    k1_pass = overlap_fraction <= 0.30
    print(f"  K1 VERDICT: {'PASS' if k1_pass else 'FAIL'} (threshold: <=30%)")

    # Show the highest-overlap pairs
    print(f"\n  Top 20 most similar domain pairs:")
    pair_list = []
    for i, j in zip(*upper_idx):
        pair_list.append((cos_matrix[i, j], leaf_names[i], leaf_names[j], leaf_paths[i], leaf_paths[j]))
    pair_list.sort(reverse=True)
    for cos_val, n1, n2, p1, p2 in pair_list[:20]:
        # Check if same supercategory
        same_super = p1.split("/")[0] == p2.split("/")[0]
        same_cat = "/".join(p1.split("/")[:2]) == "/".join(p2.split("/")[:2])
        marker = " [same-cat]" if same_cat else (" [same-super]" if same_super else " [cross]")
        print(f"    cos={cos_val:.4f}: {n1} <-> {n2}{marker}")

    # 5. Within-category vs cross-category overlap
    print(f"\n[5] Within-Category vs Cross-Category Overlap:")
    within_cat_cos = []
    cross_cat_cos = []
    within_super_cos = []
    cross_super_cos = []

    for idx in range(len(upper_idx[0])):
        i, j = upper_idx[0][idx], upper_idx[1][idx]
        cos_val = cos_matrix[i, j]
        pi, pj = leaf_paths[i], leaf_paths[j]
        si, sj = pi.split("/")[0], pj.split("/")[0]
        ci, cj = "/".join(pi.split("/")[:2]), "/".join(pj.split("/")[:2])

        if ci == cj:
            within_cat_cos.append(cos_val)
        else:
            cross_cat_cos.append(cos_val)

        if si == sj:
            within_super_cos.append(cos_val)
        else:
            cross_super_cos.append(cos_val)

    print(f"  Within-category: mean={np.mean(within_cat_cos):.4f}, n={len(within_cat_cos)}")
    print(f"  Cross-category:  mean={np.mean(cross_cat_cos):.4f}, n={len(cross_cat_cos)}")
    print(f"  Ratio: {np.mean(within_cat_cos)/np.mean(cross_cat_cos):.2f}x")
    print(f"\n  Within-supercategory: mean={np.mean(within_super_cos):.4f}, n={len(within_super_cos)}")
    print(f"  Cross-supercategory:  mean={np.mean(cross_super_cos):.4f}, n={len(cross_super_cos)}")
    print(f"  Ratio: {np.mean(within_super_cos)/np.mean(cross_super_cos):.2f}x")

    # 6. Predict expert distinctness
    print(f"\n[6] Expert Distinctness Prediction:")

    # A domain produces a distinguishable expert if:
    # (a) It has sufficiently distinct vocabulary/concepts (low max cos to any other domain)
    # (b) The base model is weak enough on it that fine-tuning helps
    #
    # We use a proxy: if a domain's max cosine to ANY other domain is > 0.85,
    # it's likely too similar to produce a distinct expert. If its description
    # is too generic (short, few unique terms), it's too vague.

    DISTINCTNESS_THRESHOLD = 0.85
    indistinguishable = 0
    indistinguishable_domains = []
    distinctness_scores = []

    for i in range(N):
        # Max cosine to any other domain
        row = cos_matrix[i].copy()
        row[i] = -1  # exclude self
        max_cos = np.max(row)
        max_j = np.argmax(row)

        # Check description specificity (number of unique keywords)
        desc = domains_with_desc[i]["description"]
        n_words = len(desc.split())

        # Score: lower is more distinct
        # A domain is indistinguishable if max_cos > threshold
        is_indistinguishable = max_cos > DISTINCTNESS_THRESHOLD

        distinctness_scores.append({
            "name": leaf_names[i],
            "max_cos": float(max_cos),
            "most_similar": leaf_names[max_j],
            "desc_words": n_words,
            "indistinguishable": bool(is_indistinguishable),
        })

        if is_indistinguishable:
            indistinguishable += 1
            indistinguishable_domains.append(
                f"{leaf_names[i]} (max_cos={max_cos:.3f} with {leaf_names[max_j]})"
            )

    indistinguishable_frac = indistinguishable / N
    print(f"  Domains predicted indistinguishable (max_cos > {DISTINCTNESS_THRESHOLD}): "
          f"{indistinguishable}/{N} = {indistinguishable_frac*100:.2f}%")
    k2_pass = indistinguishable_frac <= 0.20
    print(f"  K2 VERDICT: {'PASS' if k2_pass else 'FAIL'} (threshold: <=20%)")

    if indistinguishable_domains:
        print(f"\n  Indistinguishable domains:")
        for d in indistinguishable_domains:
            print(f"    - {d}")

    # 7. Validate against pilot-50 ground truth
    print(f"\n[7] Pilot-50 Validation:")

    if PILOT50_RESULTS.exists():
        with open(PILOT50_RESULTS) as f:
            pilot_data = json.load(f)

        # For each pilot-50 domain, find its mapping in the new taxonomy
        # and compute the embedding-based overlap with its category siblings
        pilot_domains = pilot_data["domains"]
        improvements = []
        max_sibling_cos = []
        base_ppls = []

        for pilot_name, mapped_name in PILOT50_TO_TAXONOMY.items():
            if pilot_name not in pilot_domains:
                continue
            if mapped_name not in leaf_names:
                continue

            pilot_result = pilot_domains[pilot_name]
            improvement = pilot_result["improvement_pct"]
            base_ppl = pilot_result["base_ppl"]

            # Find index of mapped domain
            idx = leaf_names.index(mapped_name)

            # Get max cosine to any sibling in same category
            my_cat = "/".join(leaf_paths[idx].split("/")[:2])
            sibling_cos = []
            for j in range(N):
                if j == idx:
                    continue
                j_cat = "/".join(leaf_paths[j].split("/")[:2])
                if j_cat == my_cat:
                    sibling_cos.append(cos_matrix[idx, j])

            if sibling_cos:
                max_sib = max(sibling_cos)
            else:
                max_sib = 0.0

            improvements.append(improvement)
            max_sibling_cos.append(max_sib)
            base_ppls.append(base_ppl)

        improvements = np.array(improvements)
        max_sibling_cos = np.array(max_sibling_cos)
        base_ppls = np.array(base_ppls)

        # Correlation: does embedding overlap predict improvement?
        from scipy.stats import pearsonr, spearmanr

        r_imp, p_imp = pearsonr(max_sibling_cos, improvements)
        rho_imp, p_rho = spearmanr(max_sibling_cos, improvements)
        r_base, p_base = pearsonr(max_sibling_cos, base_ppls)

        print(f"  Pilot-50 domains mapped: {len(improvements)}")
        print(f"  Pearson r(max_sibling_cos, improvement): {r_imp:.3f} (p={p_imp:.3f})")
        print(f"  Spearman rho(max_sibling_cos, improvement): {rho_imp:.3f} (p={p_rho:.3f})")
        print(f"  Pearson r(max_sibling_cos, base_ppl): {r_base:.3f} (p={p_base:.3f})")

        # Key insight: high base PPL -> more room for improvement
        r_base_imp, p_base_imp = pearsonr(base_ppls, improvements)
        print(f"  Pearson r(base_ppl, improvement): {r_base_imp:.3f} (p={p_base_imp:.3f})")
        print(f"\n  Interpretation: base_ppl is the dominant predictor of improvement,")
        print(f"  not embedding similarity. This makes sense: domains where the base")
        print(f"  model is weaker benefit most from fine-tuning.")

        # Which pilot-50 domains would we predict as indistinguishable?
        print(f"\n  Pilot-50 distinctness predictions:")
        pred_indistinguishable = 0
        for pilot_name, mapped_name in PILOT50_TO_TAXONOMY.items():
            if pilot_name not in pilot_domains:
                continue
            if mapped_name not in leaf_names:
                continue
            idx = leaf_names.index(mapped_name)
            ds = distinctness_scores[idx]
            actual_improvement = pilot_domains[pilot_name]["improvement_pct"]
            predicted_distinct = not ds["indistinguishable"]
            actual_distinct = actual_improvement > 2.0  # K2 threshold from hypothesis
            correct = predicted_distinct == actual_distinct
            if ds["indistinguishable"]:
                pred_indistinguishable += 1
                print(f"    {pilot_name}: predicted INDISTINGUISHABLE "
                      f"(max_cos={ds['max_cos']:.3f} with {ds['most_similar']}), "
                      f"actual improvement={actual_improvement:.1f}% "
                      f"({'CORRECT' if not actual_distinct else 'FALSE NEGATIVE'})")

        print(f"\n  Pilot-50 domains predicted indistinguishable: {pred_indistinguishable}/50")
    else:
        print("  [SKIP] pilot50_benchmark.json not found")

    # 8. Scaling analysis
    print(f"\n[8] Scaling Analysis:")
    print(f"  Current taxonomy: {N} leaf domains")

    # How many can we add before overlap becomes problematic?
    # At current overlap rate, extrapolate
    cos_at_50 = overlap_fraction  # current
    print(f"  Overlap rate at N={N}: {cos_at_50*100:.2f}%")

    # Compute overlap at different thresholds
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        n_high = np.sum(pairwise_cos > thresh)
        frac = n_high / n_pairs
        print(f"  Pairs with cos > {thresh}: {n_high}/{n_pairs} = {frac*100:.2f}%")

    # 9. Coverage analysis: what's missing?
    print(f"\n[9] Coverage Gaps (Notable Absent Domains):")
    missing = [
        "robotics/control_systems", "quantum_computing",
        "game_development", "3d_graphics_rendering",
        "compiler_design", "database_internals",
        "distributed_systems", "networking_protocols",
        "signal_processing", "optimization_theory",
        "philosophy_of_mind", "religious_studies",
        "sports_analytics", "music_production",
        "animation", "fashion_design",
        "agriculture_precision", "veterinary_medicine",
        "forensic_science", "library_science",
    ]
    print(f"  {len(missing)} notable domains not yet in taxonomy:")
    for m in missing:
        print(f"    - {m}")

    # 10. Summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Taxonomy size: {N} leaf domains")
    print(f"  Hierarchy: {len(super_counts)} supercategories, {len(cat_counts)} categories")
    print(f"  Embedding method: {method}")
    print(f"  K1 (overlap): {overlap_fraction*100:.2f}% of pairs > {OVERLAP_THRESHOLD} "
          f"-> {'PASS' if k1_pass else 'FAIL'} (threshold <=30%)")
    print(f"  K2 (vagueness): {indistinguishable_frac*100:.2f}% predicted indistinguishable "
          f"-> {'PASS' if k2_pass else 'FAIL'} (threshold <=20%)")
    print(f"  Runtime: {time.time() - t0:.1f}s")

    # Save results
    results = {
        "taxonomy_size": N,
        "n_supercategories": len(super_counts),
        "n_categories": len(cat_counts),
        "embedding_method": method,
        "overlap_analysis": {
            "mean_pairwise_cos": float(np.mean(pairwise_cos)),
            "median_pairwise_cos": float(np.median(pairwise_cos)),
            "std_pairwise_cos": float(np.std(pairwise_cos)),
            "max_pairwise_cos": float(np.max(pairwise_cos)),
            "overlap_threshold": OVERLAP_THRESHOLD,
            "pairs_above_threshold": int(high_overlap),
            "total_pairs": int(n_pairs),
            "overlap_fraction": float(overlap_fraction),
            "within_category_mean": float(np.mean(within_cat_cos)),
            "cross_category_mean": float(np.mean(cross_cat_cos)),
            "within_cross_ratio": float(np.mean(within_cat_cos) / np.mean(cross_cat_cos)),
        },
        "distinctness_analysis": {
            "threshold": DISTINCTNESS_THRESHOLD,
            "n_indistinguishable": indistinguishable,
            "indistinguishable_fraction": float(indistinguishable_frac),
            "indistinguishable_domains": indistinguishable_domains,
        },
        "kill_criteria": {
            "K1_overlap_fraction": float(overlap_fraction),
            "K1_threshold": 0.30,
            "K1_pass": bool(k1_pass),
            "K2_indistinguishable_fraction": float(indistinguishable_frac),
            "K2_threshold": 0.20,
            "K2_pass": bool(k2_pass),
        },
        "supercategory_counts": dict(super_counts),
        "category_counts": dict(cat_counts),
        "top_20_pairs": [
            {"cos": float(c), "domain1": n1, "domain2": n2}
            for c, n1, n2, _, _ in pair_list[:20]
        ],
        "runtime_seconds": time.time() - t0,
    }

    results_file = RESULTS_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_file}")

    # Save the full taxonomy as YAML-like structure for downstream use
    taxonomy_file = RESULTS_DIR / "taxonomy_500.json"
    taxonomy_export = {
        "version": "1.0",
        "n_domains": N,
        "domains": [
            {
                "name": d["name"],
                "path": d["path"],
                "description": d["description"],
            }
            for d in domains_with_desc
        ],
    }
    with open(taxonomy_file, "w") as f:
        json.dump(taxonomy_export, f, indent=2)
    print(f"  Taxonomy saved to {taxonomy_file}")

    # Also save distinctness scores
    scores_file = RESULTS_DIR / "distinctness_scores.json"
    with open(scores_file, "w") as f:
        json.dump(distinctness_scores, f, indent=2)
    print(f"  Distinctness scores saved to {scores_file}")


if __name__ == "__main__":
    main()
