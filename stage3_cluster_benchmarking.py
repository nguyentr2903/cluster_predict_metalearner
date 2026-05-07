"""
Stage 3: Clustering Algorithm Benchmarking

Runs five clustering algorithms on each of the 500 synthetic datasets,
scores each against ground-truth labels using the Adjusted Rand Index (ARI),
and assigns the best-performing algorithm as the label for the meta-learner.

The output is benchmark_results.csv, where each row contains:
  - the dataset ID and shape type
  - ARI scores for all five algorithms
  - the best-algorithm label (highest ARI)
"""

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn_extra.cluster import KMedoids
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import os
import json

datasets_dir = "datasets"


def run_kmeans(X, n_clusters, seed):
    """Centroid-based: assigns points to nearest mean. Needs n_clusters."""
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return model.fit_predict(X)


def run_kmedoids(X, n_clusters, seed):
    model = KMedoids(n_clusters=n_clusters, method="pam", random_state=seed)
    return model.fit_predict(X)


def run_agglomerative(X, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    return model.fit_predict(X)


def estimate_dbscan_eps(X):
    """Estimate the neighbourhood radius (eps) for DBSCAN using the
    k-nearest-neighbour distance elbow method.

    1) For each point, compute the distance to its k-th nearest neighbour
    (where k = 2 * dimensionality). 
    2) Sort these distances to produce the
    k-distance plot. 
    3) The elbow marks the natural boundary between dense cluster regions and sparse gaps.
    That distance becomes eps.
    """
    # k = 2 * dimensionality is a standard heuristic for DBSCAN
    # cap at n_samples - 1 so we don't request more neighbours than exist
    k = min(2 * X.shape[1], X.shape[0] - 1)
    k = max(1, k)

    # compute k-th nearest neighbour distance for every point
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    # sort the k-th neighbour distances to form the k-distance curve
    k_distances = np.sort(distances[:, -1])

    # need at least 3 points to compute a second derivative
    if len(k_distances) < 3:
        return 0.5

    # the elbow is where the curve bends most sharply
    second_diff = np.diff(k_distances, n=2)
    elbow_idx = np.argmax(second_diff) + 1
    eps = float(k_distances[elbow_idx])

    # guard against degenerate values
    if eps <= 0 or np.isnan(eps):
        return 0.5
    return eps


def run_dbscan(X):
    eps = estimate_dbscan_eps(X)
    min_samples = max(2, 2 * X.shape[1])
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(X)


def run_hdbscan(X):
    model = HDBSCAN(min_cluster_size=15)
    return model.fit_predict(X)

def compute_ari(y_true, y_pred):
    """Compare predicted cluster labels against ground truth using ARI.

    DBSCAN and HDBSCAN assign label -1 to noise points. These are
    excluded before computing ARI because the metric cannot handle them.
    If the result is degenerate (all noise, or only one cluster found),
    return 0.0 rather than an undefined score.
    """
    if y_pred is None:
        return np.nan

    # mask out noise points (label -1)
    mask = y_pred != -1

    # degenerate case: fewer than 2 non-noise points, or only 1 cluster
    if mask.sum() < 2 or len(np.unique(y_pred[mask])) < 2:
        return 0.0

    return float(adjusted_rand_score(y_true[mask], y_pred[mask]))

def benchmark_dataset(X, y_true, n_clusters, seed):
    return {
        "KMeans":        compute_ari(y_true, run_kmeans(X, n_clusters, seed)),
        "KMedoids":      compute_ari(y_true, run_kmedoids(X, n_clusters, seed)),
        "Agglomerative": compute_ari(y_true, run_agglomerative(X, n_clusters)),
        "DBSCAN":        compute_ari(y_true, run_dbscan(X)),
        "HDBSCAN":       compute_ari(y_true, run_hdbscan(X)),
    }

def load_all_datasets():
    """Load all synthetic datasets and their metadata from disk."""
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

        # retrieve ground-truth cluster count and seed from metadata
        n_clusters = meta.get("n_clusters", 2)
        seed = meta.get("seed", 42)

        # run all five algorithms and collect ARI scores
        ari_scores = benchmark_dataset(X, y, n_clusters, seed)

        # assign best-algorithm label: highest ARI, excluding NaN scores
        valid_scores = {k: v for k, v in ari_scores.items() if not np.isnan(v)}
        if valid_scores:
            best_algo = max(valid_scores, key=valid_scores.get)
        else:
            best_algo = "KMeans"  # fallback if all scores are NaN

        # store results as one row: dataset info + all ARI scores + label
        row = {
            "dataset_id": dataset_id,
            "shape": meta["shape"],
            **ari_scores,  # unpacks into separate columns per algorithm
            "best_algorithm": best_algo,
        }
        rows.append(row)

    # assemble into dataframe and save
    df = pd.DataFrame(rows).set_index("dataset_id")
    df.to_csv("benchmark_results.csv")

    print(f"\nSaved benchmark results to 'benchmark_results.csv'")
    print(f"Datasets benchmarked: {len(df)}")
    print(f"\nBest-algorithm distribution:")
    print(df["best_algorithm"].value_counts().to_string())


if __name__ == "__main__":
    main()