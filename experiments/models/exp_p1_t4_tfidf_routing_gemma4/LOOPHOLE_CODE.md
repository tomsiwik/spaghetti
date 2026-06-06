# Loophole Code Analysis: Data Leakage, Metric Hacking, and Missing MLX Integration

## 1. Massive Data Leakage in Code Domain (100% Test Set Overlap)
In `run_experiment.py`, `load_code_prompts` loads the `test` split of `openai_humaneval`, which contains exactly 164 problems. 
The script artificially expands it to fulfill `N_TRAIN + N_TEST = 400` by continually duplicating the list:
```python
    while len(prompts) < n:
        prompts = prompts + prompts
```
Because `N_TRAIN=300` and `N_TEST=100`, the `code_train` set contains multiple copies of the 164 problems, and `code_test` contains the remaining copies of the same 164 problems. **Every single problem in the `code_test` set is already present in `code_train`**. The TF-IDF router perfectly memorizes these exact strings, guaranteeing near-perfect cosine similarity for the "code" centroid and artificially inflating N=5 routing accuracy (K1073).

## 2. Massive Data Leakage in MMLU Domains (100% Test Set Overlap)
For MMLU subjects (Phase 1 Finance and Legal, Phase 2 extra subjects), `load_mmlu_prompts` attempts to load an `auxiliary_train` split:
```python
    except Exception:
        # Fallback: use test split if auxiliary_train not available
        ds = load_dataset("cais/mmlu", subject, split="test")
```
Because individual MMLU subjects typically do not have an `auxiliary_train` split, it **always falls back to the `test` split**.
Then, `load_mmlu_test_prompts` explicitly loads the `test` split. For subjects with fewer than `N_TRAIN=300` test items, `train_extra` contains the entire `test` split. `test_extra` then samples from that exact same `test` split. **100% of the MMLU test queries are present in the training set for most N=25 domains**, rendering the N=25 accuracy (K1074) invalid.

## 3. Hardcoded Cheats: Metric Hacking N=25
The script curates `MMLU_EXTRA_SUBJECTS` to explicitly exclude challenging categories:
```python
# Deliberately exclude: clinical_knowledge, virology, high_school_biology,
# nutrition, human_sexuality, high_school_psychology → all overlap with medical
# domain and would cause systematic confusion.
```
Removing the hard negatives ensures the centroids remain artificially orthogonal. A production router must handle semantically overlapping distributions. Manually filtering them is a hardcoded cheat to artificially pass K1074.

## 4. Latency Evaluation Flaw (Evaluates N=5 for N=25)
`K1075` validates latency for the architecture, but the latency test is hardcoded to run on `router5` instead of `router25`:
```python
    # K1075: latency (measured on N=5 router, same mechanism for N=25)
    p99_ms = router5.predict_latency(test_prompts_flat, n_reps=n_reps)
```
Matrix multiplication latency scales linearly with the number of centroids. Measuring an N=5 router to claim an N=25 latency limit is methodologically unsound and deliberately masks the scaling cost.

## 5. Total Lack of MLX Integration / Apple Silicon Unsuitability
The router is implemented entirely in pure Python and scikit-learn (`TfidfVectorizer`, `numpy`, `scipy.sparse`). 
```python
        X = self.vectorizer.transform(texts)  # sparse (n, vocab)
        X_norm = normalize(X, norm="l2")       # L2-normalize for cosine
        scores = X_norm @ self.centroids.T     # (n, n_domains)
```
This violates the strict MLX/Apple Silicon requirements of the P1 architecture. Running a CPU-bound, sparse Python matrix multiplication blocks the MLX async compute graph, introduces CPU-GPU sync points, and creates GIL contention. The "sub-ms latency" claim is invalid because it measures isolated Python CPU execution rather than true end-to-end latency integrated into the MLX LLM forward pass.