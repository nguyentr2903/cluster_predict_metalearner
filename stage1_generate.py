"""
Stage 1: Synthetic Dataset Generation Framework
================================================
Generates synthetic datasets with controlled variations in:
- Cluster shape (blobs, anisotropic, moons, circles, varied variance)
- Noise levels
- Cluster density and separation
- Dimensionality

Each dataset comes with ground-truth labels for ARI evaluation.
References: Thesis Section 3.1
"""

import numpy as np
import os
import json
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from config import *


def generate_blobs(n_samples, n_features, n_clusters, cluster_std, noise, seed, rng):
    """Standard isotropic Gaussian blobs (favorable for K-Means)."""
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=seed,
    )
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    return X, y


def generate_anisotropic(n_samples, n_features, n_clusters, cluster_std, noise, seed, rng):
    """
    Anisotropic (stretched) blobs — clusters with non-spherical shapes.
    Challenges prototype-based methods that assume spherical clusters.
    """
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=seed,
    )
    # Apply random linear transformation to stretch clusters
    transformation = rng.random((n_features, n_features)) * 2 - 1
    X = X @ transformation
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    return X, y


def generate_varied_variance(n_samples, n_features, n_clusters, noise, seed, rng):
    """
    Blobs with different per-cluster variances (varying density).
    Challenges DBSCAN's single-epsilon assumption.
    """
    cluster_stds = rng.uniform(0.5, 3.0, size=n_clusters)
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_stds,
        random_state=seed,
    )
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    return X, y


def generate_moons_data(n_samples, noise, seed):
    """
    Two interleaving half-moons (2D only).
    Favorable for density-based methods, challenging for K-Means.
    """
    moon_noise = max(0.05, noise) if noise > 0 else 0.05
    X, y = make_moons(n_samples=n_samples, noise=moon_noise, random_state=seed)
    return X, y


def generate_circles_data(n_samples, noise, seed):
    """
    Concentric circles (2D only).
    Favorable for density-based methods, impossible for K-Means.
    """
    circle_noise = max(0.03, noise) if noise > 0 else 0.04
    X, y = make_circles(
        n_samples=n_samples, noise=circle_noise, factor=0.5, random_state=seed
    )
    return X, y


def generate_single_dataset(dataset_id, rng):
    """
    Generate a single synthetic dataset with randomized parameters.
    Returns the data, labels, and a metadata dict describing the generation.
    """
    # Generate a deterministic integer seed for sklearn functions
    seed = int(rng.integers(0, 2**31))

    # Randomly select parameters
    shape_type = rng.choice(SHAPE_TYPES)
    n_samples = int(rng.integers(*N_SAMPLES_RANGE))
    noise = float(rng.choice(NOISE_LEVELS))

    if shape_type in ("moons", "circles"):
        n_features = 2
        n_clusters = 2
    else:
        n_features = int(rng.integers(*N_FEATURES_RANGE))
        n_clusters = int(rng.integers(*N_CLUSTERS_RANGE))

    cluster_std = float(rng.uniform(*CLUSTER_STD_RANGE))

    # Generate based on shape type
    if shape_type == "blobs":
        X, y = generate_blobs(n_samples, n_features, n_clusters, cluster_std, noise, seed, rng)
    elif shape_type == "anisotropic":
        X, y = generate_anisotropic(n_samples, n_features, n_clusters, cluster_std, noise, seed, rng)
    elif shape_type == "varied_variance":
        X, y = generate_varied_variance(n_samples, n_features, n_clusters, noise, seed, rng)
    elif shape_type == "moons":
        X, y = generate_moons_data(n_samples, noise, seed)
    elif shape_type == "circles":
        X, y = generate_circles_data(n_samples, noise, seed)

    # Standardize features (important for distance-based algorithms)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    metadata = {
        "dataset_id": int(dataset_id),
        "shape_type": shape_type,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_clusters_true": int(n_clusters),
        "cluster_std": float(cluster_std),
        "noise_level": float(noise),
    }

    return X, y, metadata


def generate_all_datasets(n_datasets=N_DATASETS, seed=RANDOM_SEED):
    """
    Generate the full corpus of synthetic datasets.
    Saves each dataset as .npz and metadata as JSON.
    """
    os.makedirs(DATASETS_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    all_metadata = []

    print(f"Generating {n_datasets} synthetic datasets...")
    for i in range(n_datasets):
        X, y, meta = generate_single_dataset(i, rng)

        # Save dataset
        np.savez(
            f"{DATASETS_DIR}/dataset_{i:04d}.npz",
            X=X, y=y
        )
        all_metadata.append(meta)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_datasets}")

    # Save all metadata
    with open(f"{DATASETS_DIR}/generation_metadata.json", "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"Done. Datasets saved to {DATASETS_DIR}/")
    return all_metadata


if __name__ == "__main__":
    metadata = generate_all_datasets()
    
    # Print distribution summary
    from collections import Counter
    shapes = Counter(m["shape_type"] for m in metadata)
    print("\nShape distribution:")
    for shape, count in shapes.most_common():
        print(f"  {shape}: {count}")
