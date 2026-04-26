import numpy as np
import pandas as pd
import os
import json
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not installed — HDBSCAN will be skipped")

try:
    from sklearn_extra.cluster import KMedoids

    KMEDOIDS_EXTRA = True
except ImportError:
    KMEDOIDS_EXTRA = False

"""
Stage 3: Clustering Algorithm Benchmarking

Runs five clustering algorithms on each synthetic dataset and records
ARI scores against ground-truth labels. The algorithm with the highest
ARI becomes the best-algorithm label for the meta-learner.

Algorithms:
  KMeans, KMedoids (PAM), Agglomerative (Ward), DBSCAN, HDBSCAN
"""

datasets_dir = "datasets"
output_file = "benchmark_results.csv"


# ---------------------------------------------------------------------------
# Individual algorithm runners
# ---------------------------------------------------------------------------

def run_kmeans(X, n_clusters, seed):
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return model.fit_predict(X)


def run_kmedoids(X, n_clusters, seed):
    """Run K-Medoids using sklearn-extra if available, otherwise a manual
    PAM-style fallback based on KMeans initialisation."""
    if KMEDOIDS_EXTRA:
        model = KMedoids(n_clusters=n_clusters, method="pam", random_state=seed)
        return model.fit_predict(X)

    # Manual fallback: initialise medoids from KMeans centroids
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    km.fit(X)
    labels = km.labels_.copy()

    for iteration in range(50):
        # Find medoid (closest real point to centroid) per cluster
        medoids = np.empty((n_clusters, X.shape[1]))
        for c in range(n_clusters):
            members = X[labels == c]
            if len(members) == 0:
                medoids[c] = X[np.random.RandomState(seed + iteration).randint(len(X))]
                continue
            centroid = members.mean(axis=0)
            dists = np.linalg.norm(members - centroid, axis=1)
            medoids[c] = members[np.argmin(dists)]

        # Reassign points to nearest medoid
        dists_to_medoids = np.array([
            np.linalg.norm(X - m, axis=1) for m in medoids
        ])  # shape (n_clusters, n_samples)
        new_labels = np.argmin(dists_to_medoids, axis=0)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

    return labels


def run_agglomerative(X, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    return model.fit_predict(X)


def estimate_dbscan_eps(X, k=None):
    """Estimate eps using the k-NN distance elbow method.

    Computes k-nearest-neighbour distances, sorts them, and selects the
    point of maximum curvature as eps. Falls back to 0.5 if the estimate
    is degenerate.
    """
    if k is None:
        k = min(2 * X.shape[1], X.shape[0] - 1)  # 2 * dimensionality
    k = max(1, min(k, X.shape[0] - 1))

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    # Find elbow: point of maximum second derivative
    if len(k_distances) < 3:
        return 0.5
    second_diff = np.diff(k_distances, n=2)
    if len(second_diff) == 0:
        return 0.5
    elbow_idx = np.argmax(second_diff) + 1
    eps = float(k_distances[elbow_idx])

    # Guard against degenerate values
    if eps <= 0 or np.isnan(eps) or np.isinf(eps):
        return 0.5
    return eps


def run_dbscan(X):
    eps = estimate_dbscan_eps(X)
    min_samples = max(2, 2 * X.shape[1])
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(X)


def run_hdbscan(X):
    if not HDBSCAN_AVAILABLE:
        return None
    model = hdbscan.HDBSCAN(min_cluster_size=15)
    return model.fit_predict(X)


# ---------------------------------------------------------------------------
# ARI computation
# ---------------------------------------------------------------------------

def compute_ari(y_true, y_pred):
    """Compute ARI, excluding noise points (label -1) from DBSCAN/HDBSCAN."""
    if y_pred is None:
        return np.nan
    mask = y_pred != -1
    if mask.sum() < 2 or len(np.unique(y_pred[mask])) < 2:
        return 0.0  # degenerate: all noise or single cluster
    return float(adjusted_rand_score(y_true[mask], y_pred[mask]))


# ---------------------------------------------------------------------------
# Per-dataset benchmarking
# ---------------------------------------------------------------------------

def benchmark_dataset(X, y_true, n_clusters, seed):
    return {
        "KMeans":        compute_ari(y_true, run_kmeans(X, n_clusters, seed)),
        "KMedoids":      compute_ari(y_true, run_kmedoids(X, n_clusters, seed)),
        "Agglomerative": compute_ari(y_true, run_agglomerative(X, n_clusters)),
        "DBSCAN":        compute_ari(y_true, run_dbscan(X)),
        "HDBSCAN":       compute_ari(y_true, run_hdbscan(X)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_all_datasets():
    metadata_path = os.path.join(datasets_dir, "metadata.json")
    with open(metadata_path) as f:
        all_metadata = json.load(f)

    records = []
    for meta in all_metadata:
        shape = meta["shape"]
        idx = meta["index"]
        prefix = os.path.join(datasets_dir, shape, f"dataset_{idx:03d}")
        X = np.load(f"{prefix}.npy")
        y = np.load(f"{prefix}_labels.npy")
        records.append((meta["dataset_id"], X, y, meta))

    return records


def main():
    print("Loading datasets...")
    records = load_all_datasets()
    print(f"  Loaded {len(records)} datasets")

    rows = []
    for i, (dataset_id, X, y, meta) in enumerate(records):
        if i == 0 or (i + 1) % 50 == 0:
            print(f"  Benchmarking: {i + 1}/{len(records)}")

        n_clusters = meta.get("n_clusters", 2)
        seed = meta.get("seed", 42)
        ari_scores = benchmark_dataset(X, y, n_clusters, seed)

        # Best algorithm: highest ARI, excluding NaN scores
        valid_scores = {k: v for k, v in ari_scores.items() if not np.isnan(v)}
        if valid_scores:
            best_algo = max(valid_scores, key=valid_scores.get)
        else:
            best_algo = "KMeans"  # fallback if all scores are NaN

        row = {
            "dataset_id": dataset_id,
            "shape": meta["shape"],
            **ari_scores,
            "best_algorithm": best_algo,
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index("dataset_id")
    df.to_csv(output_file)

    print(f"\nSaved benchmark results to '{output_file}'")
    print(f"  Datasets benchmarked: {len(df)}")
    print(f"\nBest-algorithm distribution:")
    print(df["best_algorithm"].value_counts().to_string())


if __name__ == "__main__":
    main()