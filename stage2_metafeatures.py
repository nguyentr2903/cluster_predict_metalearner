"""
Stage 2: Meta-Feature Extraction
=================================
Extracts meta-features from each synthetic dataset using:
- PyMFE library (general, statistical, info-theory groups)
- Custom distance-based meta-features (Ferrari & de Castro, 2015)
- Custom clustering-based meta-features

References: Thesis Sections 2.3.1–2.3.4, 3.2
"""

import numpy as np
import pandas as pd
import os
import warnings
from scipy.spatial.distance import pdist
from scipy.stats import skew, kurtosis
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

try:
    from pymfe.mfe import MFE
    PYMFE_AVAILABLE = True
except ImportError:
    PYMFE_AVAILABLE = False
    print("WARNING: pymfe not installed. Run: pip install pymfe")

from config import *


def extract_pymfe_features(X):
    """
    Extract meta-features using PyMFE library.
    Groups: general, statistical, info-theory
    
    General (Section 2.3.1): n_instances, n_features, dimensionality ratios
    Statistical (Section 2.3.2): mean, std, skewness, kurtosis, correlations
    Info-theory: entropy-based measures of feature redundancy
    """
    if not PYMFE_AVAILABLE:
        return {}

    mfe = MFE(
        groups=METAFEATURE_GROUPS,
        summary=["mean", "sd", "min", "max"],  # aggregate across features
    )
    mfe.fit(X)
    names, values = mfe.extract()

    features = {}
    for name, val in zip(names, values):
        if val is not None and np.isfinite(val):
            features[f"pymfe_{name}"] = float(val)
        else:
            features[f"pymfe_{name}"] = np.nan

    return features


def extract_distance_features(X, max_samples=1000):
    """
    Distance-based meta-features (Ferrari & de Castro, 2015).
    Compute pairwise Euclidean distances and derive statistical descriptors.
    
    Reference: Thesis Section 2.3.3
    - Subgroup 1: Statistical descriptors of distance vector
    - Subgroup 2: Histogram bins of distance distribution
    - Subgroup 3: Z-score discretization
    """
    # Subsample for computational feasibility
    if X.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    # Compute pairwise distances and normalize to [0, 1]
    dists = pdist(X_sub, metric="euclidean")
    if dists.max() > 0:
        dists_norm = dists / dists.max()
    else:
        dists_norm = dists

    features = {}

    # Subgroup 1: Statistical descriptors of distance vector
    features["dist_mean"] = float(np.mean(dists_norm))
    features["dist_std"] = float(np.std(dists_norm))
    features["dist_variance"] = float(np.var(dists_norm))
    features["dist_skewness"] = float(skew(dists_norm))
    features["dist_kurtosis"] = float(kurtosis(dists_norm))
    features["dist_median"] = float(np.median(dists_norm))
    features["dist_iqr"] = float(np.percentile(dists_norm, 75) - np.percentile(dists_norm, 25))

    # Subgroup 2: Histogram bins (10 equal-width bins as per Kalousis)
    hist_counts, _ = np.histogram(dists_norm, bins=10, range=(0, 1))
    hist_proportions = hist_counts / hist_counts.sum()
    for i, prop in enumerate(hist_proportions):
        features[f"dist_hist_bin_{i}"] = float(prop)

    # Subgroup 3: Z-score discretization (4 bins)
    z_scores = np.abs((dists - np.mean(dists)) / (np.std(dists) + 1e-10))
    z_bins = [
        np.mean(z_scores < 1),       # [0, 1)
        np.mean((z_scores >= 1) & (z_scores < 2)),  # [1, 2)
        np.mean((z_scores >= 2) & (z_scores < 3)),   # [2, 3)
        np.mean(z_scores >= 3),       # [3, inf)
    ]
    for i, prop in enumerate(z_bins):
        features[f"dist_zscore_bin_{i}"] = float(prop)

    return features


def extract_clustering_features(X, max_k=10):
    """
    Clustering-based meta-features.
    These measure the degree to which meaningful clusters exist in the data.
    
    Includes:
    - Hopkins statistic (clustering tendency)
    - Silhouette scores at various k values
    - Elbow analysis features
    """
    features = {}
    n_samples = X.shape[0]

    # Hopkins statistic — measures clustering tendency
    # Values near 0.5 = random, near 1.0 = highly clustered
    try:
        sample_size = min(100, n_samples // 2)
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n_samples, sample_size, replace=False)
        X_sample = X[sample_idx]

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X)

        # Distance of sample points to nearest neighbor
        u_dists, _ = nn.kneighbors(X_sample)
        u_sum = u_dists.sum()

        # Distance of random points to nearest real point
        X_random = rng.uniform(X.min(axis=0), X.max(axis=0), size=X_sample.shape)
        w_dists, _ = nn.kneighbors(X_random)
        w_sum = w_dists.sum()

        hopkins = u_sum / (u_sum + w_sum + 1e-10)
        features["hopkins_statistic"] = float(hopkins)
    except Exception:
        features["hopkins_statistic"] = np.nan

    # Silhouette scores for k=2..max_k
    max_k_actual = min(max_k, n_samples // 2, 10)
    silhouette_scores = []
    inertias = []

    for k in range(2, max_k_actual + 1):
        try:
            km = KMeans(n_clusters=k, n_init=5, random_state=42, max_iter=100)
            labels = km.fit_predict(X)
            n_unique = len(set(labels))
            if 1 < n_unique < n_samples:
                sil = silhouette_score(X, labels, sample_size=min(1000, n_samples))
                silhouette_scores.append(sil)
            inertias.append(km.inertia_)
        except Exception:
            pass

    if silhouette_scores:
        features["sil_max"] = float(max(silhouette_scores))
        features["sil_mean"] = float(np.mean(silhouette_scores))
        features["sil_best_k"] = int(np.argmax(silhouette_scores) + 2)
    else:
        features["sil_max"] = np.nan
        features["sil_mean"] = np.nan
        features["sil_best_k"] = np.nan

    # Elbow analysis: rate of inertia decrease
    if len(inertias) >= 3:
        inertias = np.array(inertias)
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        features["elbow_sharpness"] = float(np.max(np.abs(diffs2)))
    else:
        features["elbow_sharpness"] = np.nan

    return features


def extract_all_features(X):
    """Combine all meta-feature groups into a single feature vector."""
    features = {}

    # PyMFE features (general, statistical, info-theory)
    features.update(extract_pymfe_features(X))

    # Distance-based features (Ferrari & de Castro)
    features.update(extract_distance_features(X))

    # Clustering-based features
    features.update(extract_clustering_features(X))

    return features


def extract_metafeatures_all_datasets(datasets_dir=DATASETS_DIR):
    """
    Extract meta-features for all generated datasets.
    Saves results to metafeatures.csv
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all dataset files
    files = sorted([
        f for f in os.listdir(datasets_dir)
        if f.startswith("dataset_") and f.endswith(".npz")
    ])

    print(f"Extracting meta-features from {len(files)} datasets...")
    all_features = []

    for i, fname in enumerate(files):
        data = np.load(f"{datasets_dir}/{fname}")
        X = data["X"]

        features = extract_all_features(X)
        features["dataset_id"] = int(fname.split("_")[1].split(".")[0])
        all_features.append(features)

        if (i + 1) % 50 == 0:
            print(f"  Extracted {i + 1}/{len(files)}")

    df = pd.DataFrame(all_features)
    df = df.set_index("dataset_id").sort_index()

    # Report missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\nFeatures with missing values:")
        print(missing[missing > 0])

    # Fill NaN with column median (robust to outliers)
    df = df.fillna(df.median())

    df.to_csv(METAFEATURES_PATH)
    print(f"\nMeta-features saved to {METAFEATURES_PATH}")
    print(f"Shape: {df.shape[0]} datasets × {df.shape[1]} features")

    return df


if __name__ == "__main__":
    df = extract_metafeatures_all_datasets()
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nFeature statistics:")
    print(df.describe())
