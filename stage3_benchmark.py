"""
Stage 3: Clustering Algorithm Benchmarking
==========================================
Runs all 5 clustering algorithms on each synthetic dataset and
records performance using ARI (primary), NMI, and Silhouette Score.

Algorithms (Section 2.1):
- K-Means (prototype-based)
- K-Medoids / PAM (prototype-based, robust to outliers)
- DBSCAN (density-based, single epsilon)
- HDBSCAN (density-based, hierarchical)
- Agglomerative Clustering (hierarchical, Ward linkage)

References: Thesis Sections 2.1, 2.2, 3.1
"""

import numpy as np
import pandas as pd
import os
import time
import warnings
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

try:
    from sklearn_extra.cluster import KMedoids
    KMEDOIDS_AVAILABLE = True
except (ImportError, AttributeError):
    try:
        # Fallback: manual PAM implementation using sklearn's pairwise distances
        from sklearn.metrics import pairwise_distances
        KMEDOIDS_AVAILABLE = "manual"
    except ImportError:
        KMEDOIDS_AVAILABLE = False
        print("WARNING: K-Medoids not available.")

try:
    import hdbscan as hdbscan_lib
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("WARNING: hdbscan not installed. Run: pip install hdbscan")

from config import *


def safe_silhouette(X, labels):
    """Compute silhouette score, handling edge cases."""
    unique_labels = set(labels)
    unique_labels.discard(-1)  # ignore noise
    if len(unique_labels) < 2 or len(unique_labels) >= len(X):
        return np.nan
    # Exclude noise points for silhouette
    mask = labels != -1
    if mask.sum() < 2:
        return np.nan
    try:
        return silhouette_score(X[mask], labels[mask], sample_size=min(1000, mask.sum()))
    except Exception:
        return np.nan


def estimate_dbscan_eps(X):
    """
    Estimate a good epsilon for DBSCAN using the k-distance graph method.
    Uses k=min_samples and finds the 'knee' in the sorted k-distances.
    """
    k = min(5, X.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_dists = np.sort(distances[:, -1])

    # Simple knee detection: find the point of maximum curvature
    diffs = np.diff(k_dists)
    diffs2 = np.diff(diffs)
    if len(diffs2) > 0:
        knee_idx = np.argmax(diffs2) + 1
        eps = k_dists[knee_idx]
    else:
        eps = np.median(k_dists)

    return max(eps, 0.1)  # floor at 0.1


def run_kmeans(X, n_clusters):
    """K-Means clustering (Section 2.1.1)."""
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42, max_iter=300)
    labels = model.fit_predict(X)
    return labels


def run_kmedoids(X, n_clusters):
    """K-Medoids / PAM clustering (Section 2.1.1)."""
    if KMEDOIDS_AVAILABLE is True:
        model = KMedoids(n_clusters=n_clusters, random_state=42, max_iter=300)
        labels = model.fit_predict(X)
        return labels
    elif KMEDOIDS_AVAILABLE == "manual":
        return _kmedoids_manual(X, n_clusters)
    return None


def _kmedoids_manual(X, n_clusters, max_iter=100):
    """Simple PAM-style K-Medoids using pairwise distance matrix."""
    from sklearn.metrics import pairwise_distances
    n = X.shape[0]
    if n_clusters >= n:
        return np.arange(n)

    D = pairwise_distances(X)
    rng = np.random.RandomState(42)

    # Initialize medoids randomly
    medoid_idx = rng.choice(n, n_clusters, replace=False)

    for _ in range(max_iter):
        # Assign each point to nearest medoid
        dists_to_medoids = D[:, medoid_idx]
        labels = np.argmin(dists_to_medoids, axis=1)

        # Update medoids: for each cluster, pick the point minimizing total distance
        new_medoid_idx = np.copy(medoid_idx)
        for k in range(n_clusters):
            cluster_mask = labels == k
            if cluster_mask.sum() == 0:
                continue
            cluster_dists = D[np.ix_(cluster_mask, cluster_mask)]
            total_dists = cluster_dists.sum(axis=1)
            best_local = np.argmin(total_dists)
            new_medoid_idx[k] = np.where(cluster_mask)[0][best_local]

        if np.array_equal(medoid_idx, new_medoid_idx):
            break
        medoid_idx = new_medoid_idx

    # Final assignment
    labels = np.argmin(D[:, medoid_idx], axis=1)
    return labels


def run_dbscan(X, eps=None, min_samples=5):
    """DBSCAN clustering (Section 2.1.2)."""
    if eps is None:
        eps = estimate_dbscan_eps(X)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels


def run_hdbscan(X, min_cluster_size=None):
    """HDBSCAN clustering (Section 2.1.2)."""
    if not HDBSCAN_AVAILABLE:
        return None
    if min_cluster_size is None:
        min_cluster_size = max(5, X.shape[0] // 50)
    model = hdbscan_lib.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(X)
    return labels


def run_agglomerative(X, n_clusters):
    """Agglomerative Clustering with Ward linkage (Section 2.1.3)."""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(X)
    return labels


def evaluate_algorithm(X, y_true, labels, algo_name):
    """
    Evaluate clustering results against ground truth.
    
    Metrics (aligned with thesis):
    - ARI: Adjusted Rand Index (primary metric, chance-adjusted)
    - NMI: Normalized Mutual Information  
    - Silhouette: Internal validation (no ground truth needed)
    """
    if labels is None:
        return {
            f"{algo_name}_ari": np.nan,
            f"{algo_name}_nmi": np.nan,
            f"{algo_name}_silhouette": np.nan,
            f"{algo_name}_n_clusters_found": np.nan,
            f"{algo_name}_noise_ratio": np.nan,
        }

    # For ARI/NMI, use all points (noise labeled as -1 will hurt score, which is correct)
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    sil = safe_silhouette(X, labels)

    unique = set(labels)
    n_clusters_found = len(unique - {-1})
    noise_ratio = np.mean(labels == -1)

    return {
        f"{algo_name}_ari": float(ari),
        f"{algo_name}_nmi": float(nmi),
        f"{algo_name}_silhouette": float(sil) if not np.isnan(sil) else np.nan,
        f"{algo_name}_n_clusters_found": int(n_clusters_found),
        f"{algo_name}_noise_ratio": float(noise_ratio),
    }


def benchmark_single_dataset(X, y_true, n_clusters_true):
    """Run all algorithms on a single dataset and collect metrics."""
    results = {}

    # K-Means
    labels = run_kmeans(X, n_clusters_true)
    results.update(evaluate_algorithm(X, y_true, labels, "kmeans"))

    # K-Medoids
    labels = run_kmedoids(X, n_clusters_true)
    results.update(evaluate_algorithm(X, y_true, labels, "kmedoids"))

    # DBSCAN (with automatic epsilon estimation)
    labels = run_dbscan(X)
    results.update(evaluate_algorithm(X, y_true, labels, "dbscan"))

    # HDBSCAN
    labels = run_hdbscan(X)
    results.update(evaluate_algorithm(X, y_true, labels, "hdbscan"))

    # Agglomerative (Ward linkage)
    labels = run_agglomerative(X, n_clusters_true)
    results.update(evaluate_algorithm(X, y_true, labels, "agglomerative"))

    # Determine best algorithm by ARI (the primary metric)
    ari_scores = {
        algo: results.get(f"{algo}_ari", -1)
        for algo in ALGORITHMS
    }
    # Filter out NaN
    valid_scores = {k: v for k, v in ari_scores.items() if not np.isnan(v)}
    if valid_scores:
        results["best_algorithm"] = max(valid_scores, key=valid_scores.get)
        results["best_ari"] = max(valid_scores.values())
    else:
        results["best_algorithm"] = "unknown"
        results["best_ari"] = np.nan

    return results


def benchmark_all_datasets(datasets_dir=DATASETS_DIR):
    """
    Run benchmarks on all generated datasets.
    Saves results to benchmark_results.csv
    """
    import json

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load generation metadata
    with open(f"{datasets_dir}/generation_metadata.json") as f:
        gen_metadata = {m["dataset_id"]: m for m in json.load(f)}

    files = sorted([
        f for f in os.listdir(datasets_dir)
        if f.startswith("dataset_") and f.endswith(".npz")
    ])

    print(f"Benchmarking {len(files)} datasets across {len(ALGORITHMS)} algorithms...")
    all_results = []

    for i, fname in enumerate(files):
        dataset_id = int(fname.split("_")[1].split(".")[0])
        data = np.load(f"{datasets_dir}/{fname}")
        X, y = data["X"], data["y"]

        meta = gen_metadata.get(dataset_id, {})
        n_clusters_true = meta.get("n_clusters_true", len(np.unique(y)))

        t0 = time.time()
        results = benchmark_single_dataset(X, y, n_clusters_true)
        elapsed = time.time() - t0

        results["dataset_id"] = dataset_id
        results["runtime_seconds"] = elapsed
        all_results.append(results)

        if (i + 1) % 50 == 0:
            print(f"  Benchmarked {i + 1}/{len(files)} ({elapsed:.2f}s last)")

    df = pd.DataFrame(all_results)
    df = df.set_index("dataset_id").sort_index()
    df.to_csv(BENCHMARK_PATH)

    # Print summary
    print(f"\nBenchmark results saved to {BENCHMARK_PATH}")
    print(f"\nBest algorithm distribution:")
    print(df["best_algorithm"].value_counts())
    print(f"\nMean ARI by algorithm:")
    for algo in ALGORITHMS:
        col = f"{algo}_ari"
        if col in df.columns:
            print(f"  {algo}: {df[col].mean():.4f}")

    return df


if __name__ == "__main__":
    df = benchmark_all_datasets()
