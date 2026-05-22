import numpy as np
import os
import json
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from config import (
    SHAPES, DATASETS_PER_SHAPE, N_SAMPLES_RANGE, N_FEATURES_RANGE,
    N_CLUSTERS_RANGE, CLUSTER_STD_RANGE, NOISE_RANGE
)
from stage2_extract_metafeatures import extract_meta_features
from stage3_cluster_benchmarking import benchmark_dataset
import joblib
results = joblib.load("trained_models.joblib")
import pandas as pd
'''THE IDEA: noise experiment generates new test datasets at controlled noise levels, extracts their metafeatures,
gets their true best-algorithm labels, and checks whether the model's predictions get worse as noise increases'''
'''generate 20 datasets per shape'''
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.65, 0.95]
SHAPES = ["blobs", "anisotropic", "varied_variance", "moons", "circles"]
DATASETS_PER_SHAPE = 20
OUTPUT_DIR = "noise_experiment"
N_SAMPLES = 100_000

def generate_blobs(n_samples, n_features, n_clusters, cluster_std, noise, seed, rng):
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=seed,
    )
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    return X, y, {}


def generate_anisotropic(n_samples, n_features, n_clusters, cluster_std, noise, seed, rng):
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=seed,
    )
    transformation = rng.standard_normal((n_features, n_features))
    X = X @ transformation
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    return X, y, {"transformation": transformation.tolist()}


def generate_varied_variance(n_samples, n_features, n_clusters, noise, seed, rng):
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
    return X, y, {"cluster_stds": cluster_stds.tolist()}


def generate_moons(n_samples, noise, seed):
    moon_noise = max(0.05, noise)
    X, y = make_moons(n_samples=n_samples, noise=moon_noise, random_state=seed)
    return X, y, {}


def generate_circles(n_samples, noise, seed):
    circle_noise = max(0.04, noise)
    X, y = make_circles(
        n_samples=n_samples, noise=circle_noise, factor=0.5, random_state=seed
    )
    return X, y, {}



def generate_dataset(shape, params, seed, rng):
    if shape == "blobs":
        return generate_blobs(
            params["n_samples"], params["n_features"], params["n_clusters"],
            params["cluster_std"], params["noise"], seed, rng,
        )
    elif shape == "anisotropic":
        return generate_anisotropic(
            params["n_samples"], params["n_features"], params["n_clusters"],
            params["cluster_std"], params["noise"], seed, rng,
        )
    elif shape == "varied_variance":
        return generate_varied_variance(
            params["n_samples"], params["n_features"], params["n_clusters"],
            params["noise"], seed, rng,
        )
    elif shape == "moons":
        return generate_moons(params["n_samples"], params["noise"], seed)
    elif shape == "circles":
        return generate_circles(params["n_samples"], params["noise"], seed)
    else:
        raise ValueError(f"Unknown shape: {shape!r}")

def sample_params(shape, rng):
    n_samples = int(rng.integers(*N_SAMPLES_RANGE))
    noise = float(rng.uniform(*NOISE_RANGE))

    if shape in ("blobs", "anisotropic", "varied_variance"):
        n_features = int(rng.integers(*N_FEATURES_RANGE))
        n_clusters = int(rng.integers(*N_CLUSTERS_RANGE))
        cluster_std = float(rng.uniform(*CLUSTER_STD_RANGE))
        return {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_clusters": n_clusters,
            "cluster_std": cluster_std,
            "noise": noise,
        }
    else:
        return {
            "n_samples": n_samples,
            "n_features": 2,
            "n_clusters": 2,
            "noise": noise,
        }

def evaluate_noise_experiment(trained_model, training_features, model_name):
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path) as f:
        all_metadata = json.load(f)

    rows = []
    for meta in all_metadata:
        shape = meta["shape"]
        noise_level = meta["noise_level"]
        idx = meta["index"]

        # load dataset
        shape_dir = os.path.join(OUTPUT_DIR, f"noise_{noise_level:.2f}", shape)
        prefix = os.path.join(shape_dir, f"dataset_{idx:03d}")
        X = np.load(f"{prefix}.npy")
        y = np.load(f"{prefix}_labels.npy")

        # stage 2: extract meta-features
        features = extract_meta_features(X, y)

        # stage 3: get true best algorithm
        n_clusters = meta.get("n_clusters", 2)
        seed = meta.get("seed", 42)
        ari_scores = benchmark_dataset(X, y, n_clusters, seed)
        valid_scores = {k: v for k, v in ari_scores.items() if not np.isnan(v)}
        true_best = max(valid_scores, key=valid_scores.get) if valid_scores else "KMeans"

        features["noise_level"] = noise_level
        features["true_best"] = true_best
        rows.append(features)

    # build feature matrix
    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in ["noise_level", "true_best"]]
    X_noise = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_noise = X_noise.reindex(columns=training_features, fill_value=0)
    # predict using trained model
    df["predicted"] = trained_model.predict(X_noise)
    df["correct"] = df["predicted"] == df["true_best"]

    # accuracy at each noise level
    print("\n--- Accuracy by Noise Level ---")
    for level in NOISE_LEVELS:
        subset = df[df["noise_level"] == level]
        acc = subset["correct"].mean()
        print(f"  Noise {level:.2f}: {acc:.4f} ({subset['correct'].sum()}/{len(subset)})")
    output_file = f"noise_experiment_results_{model_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved detailed results to {output_file}") 
    return df

if __name__ == "__main__":
    generate_datasets()
    first_model_name = list(results.keys())[0]
    training_features = results[first_model_name]["model"].feature_names_in_
    for name in ["RandomForest", "GradientBoosting", "SVM", "KNN"]:
        print(f"\n=== {name} ===")
        model = results[name]["model"]
        evaluate_noise_experiment(model, training_features, name)

