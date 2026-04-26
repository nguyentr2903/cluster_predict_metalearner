import numpy as np
import pandas as pd
import os
import json
from pymfe.mfe import MFE
from scipy.spatial.distance import pdist
from scipy.stats import skew, kurtosis

"""
Extracts two families of meta-features per dataset:

PyMFE groups (Rivolli et al.):
  general    — sample count, feature count, dimensionality ratio, class imbalance
  statistical — correlations, kurtosis, skewness, PCA variance explained

Custom distance-based features (Ferrari & de Castro [18]):
  Mean, std, skewness, and kurtosis of the pairwise Euclidean distance distribution
"""

datasets_dir = "datasets"
output_file = "meta_features.csv"

PYMFE_GROUPS = ["general", "statistical"]


def extract_distance_features(X):
    """Compute summary statistics of the pairwise Euclidean distance distribution.

    Following Ferrari & de Castro [18], the full vector of pairwise distances
    is reduced to four scalar descriptors that characterise the global
    geometric structure of the dataset.
    """
    try:
        dists = pdist(X, metric="euclidean")
        return {
            "dist_mean": np.mean(dists),
            "dist_std": np.std(dists),
            "dist_skew": skew(dists),
            "dist_kurtosis": kurtosis(dists),
        }
    except Exception as e:
        print(f"  Warning: distance feature extraction error: {e}")
        return {
            "dist_mean": np.nan,
            "dist_std": np.nan,
            "dist_skew": np.nan,
            "dist_kurtosis": np.nan,
        }


def extract_meta_features(X, y):
    features = {}

    # PyMFE features
    try:
        mfe = MFE(groups=PYMFE_GROUPS, suppress_warnings=True)
        mfe.fit(X, y)
        names, values = mfe.extract(suppress_warnings=True)
        features.update(dict(zip(names, values)))
    except Exception as e:
        print(f"  Warning: MFE extraction error: {e}")

    # Distance-based features
    features.update(extract_distance_features(X))

    return features


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
            print(f"  Extracting meta-features: {i + 1}/{len(records)}")

        features = extract_meta_features(X, y)
        features["dataset_id"] = dataset_id
        features["shape"] = meta["shape"]
        rows.append(features)

    df = pd.DataFrame(rows).set_index("dataset_id")

    # Impute remaining NaN with column means (some PyMFE measures can be undefined)
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    df.to_csv(output_file)

    print(f"\nSaved meta-features to '{output_file}'")
    print(f"  Rows: {df.shape[0]}, Total columns: {df.shape[1]}")
    print(f"  Numeric meta-features: {len(numeric_cols)}")


if __name__ == "__main__":
    main()