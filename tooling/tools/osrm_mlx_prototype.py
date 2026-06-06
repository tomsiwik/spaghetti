import mlx.core as mx
import mlx.nn as nn

def generate_task_features(n_samples, feature_dim, dominant_subspace_dim, key=0):
    """
    Generates synthetic feature representations for a task.
    The task features have high variance in a 'dominant_subspace' 
    and low variance elsewhere.
    """
    mx.random.seed(key)
    # Random orthogonal basis to define the true subspace of the task
    basis, _ = mx.linalg.qr(mx.random.normal((feature_dim, feature_dim)), stream=mx.cpu)
    
    # High variance components
    high_var = mx.random.normal((n_samples, dominant_subspace_dim)) * 10.0
    # Low variance components (noise)
    low_var = mx.random.normal((n_samples, feature_dim - dominant_subspace_dim)) * 0.1
    
    # Combine and rotate into the random basis
    components = mx.concatenate([high_var, low_var], axis=-1)
    features = components @ basis.T
    
    return features, basis

def standard_grassmannian_init(rank, feature_dim, key=1):
    """
    Initializes an A matrix (rank x config.dim) that is purely orthogonal
    but blind to the data covariance (Standard Grassmannian constraint).
    """
    mx.random.seed(key)
    W = mx.random.normal((rank, feature_dim))
    Q, _ = mx.linalg.qr(W.T, stream=mx.cpu)
    return Q.T  # Shape: rank x feature_dim

def osrm_init(features, rank):
    """
    Orthogonal Subspaces for Robust Merging (OSRM).
    Computes data covariance, running eigendecomposition, and sets A
    to span the eigenvectors corresponding to the SMALLEST eigenvalues.
    """
    # 1. Compute covariance matrix
    # Center the features
    features_centered = features - mx.mean(features, axis=0, keepdims=True)
    # S = H^T @ H / (N-1)
    N = features.shape[0]
    cov = (features_centered.T @ features_centered) / (N - 1)
    
    # 2. Eigendecomposition (symmetric covariance)
    eigenvalues, eigenvectors = mx.linalg.eigh(cov, stream=mx.cpu)
    
    # eigenvalues are sorted in ascending order by eigh
    # We want the 'rank' eigenvectors with the SMALLEST eigenvalues
    # eigenvectors are columns in the returned matrix
    optimal_A = eigenvectors[:, :rank].T
    
    return optimal_A

def test_interference():
    d_model = 128
    r = 8
    n_samples = 1000
    dominant_dim = 32 # Task 1 mainly uses 32 dimensions

    print("Experiment: OSRM math foundation vs Standard Grassmannian")
    # 1. Simulate Task 1 features (e.g. Math dataset)
    H1, _ = generate_task_features(n_samples, d_model, dominant_dim, key=42)
    
    # 2. We are training Adapter 2 for Task 2 (e.g. Code).
    # We want A2 to minimize interference with Task 1 features.
    
    # Strategy A: Standard Grassmannian (Blind Orthogonality)
    A2_grassmannian = standard_grassmannian_init(r, d_model, key=123)
    
    # Strategy B: OSRM (Data-Aware Orthogonality)
    A2_osrm = osrm_init(H1, r)
    
    # 3. Measure Interference (||A2 @ H1^T||_F)
    # We take the mean squared activation across the dataset
    interference_grassmannian = mx.mean(mx.square(H1 @ A2_grassmannian.T))
    interference_osrm = mx.mean(mx.square(H1 @ A2_osrm.T))
    
    print(f"Dataset Shape: {H1.shape}")
    print(f"Rank parameter: {r}")
    print(f"Interference (Grassmannian A_2): {interference_grassmannian.item():.4f}")
    print(f"Interference (OSRM A_2):         {interference_osrm.item():.4f}")
    print(f"OSRM interference Reduction:     {interference_grassmannian.item() / interference_osrm.item():.1f}x")

if __name__ == "__main__":
    test_interference()
