import numpy as np
import os
import json
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler

output_dir = "datasets"
datasets_per_shape = 100
random_seed_base = 42

SHAPES = ["blobs", "anisotropic", "varied_variance", "moons", "circles"]


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


def sample_params(shape, rng):
    n_samples = int(rng.integers(100, 1001))
    noise = float(rng.uniform(0.0, 0.3))

    if shape in ("blobs", "anisotropic", "varied_variance"):
        n_features = int(rng.integers(2, 11))
        n_clusters = int(rng.integers(2, 9))
        cluster_std = float(rng.uniform(0.3, 2.5))
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


def save_dataset(X, y, metadata, shape, idx):
    shape_dir = os.path.join(output_dir, shape)
    os.makedirs(shape_dir, exist_ok=True)
    prefix = os.path.join(shape_dir, f"dataset_{idx:03d}")
    np.save(f"{prefix}.npy", X)
    np.save(f"{prefix}_labels.npy", y)
    with open(f"{prefix}_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    rng = np.random.default_rng(random_seed_base)
    os.makedirs(output_dir, exist_ok=True)

    scaler = StandardScaler()
    all_metadata = []

    for shape in SHAPES:
        print(f"Generating {datasets_per_shape} datasets for shape: {shape}")
        for i in range(datasets_per_shape):
            seed = int(rng.integers(0, 100_000))
            params = sample_params(shape, rng)
            X, y, extra = generate_dataset(shape, params, seed, rng)
            X = scaler.fit_transform(X)

            metadata = {
                "dataset_id": f"{shape}_{i:03d}",
                "shape": shape,
                "index": i,
                "seed": seed,
                **params,
                "actual_n_samples": int(X.shape[0]),
                "actual_n_features": int(X.shape[1]),
                **extra,
            }
            save_dataset(X, y, metadata, shape, i)
            all_metadata.append(metadata)

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{datasets_per_shape} done")

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\nDone. Generated {len(all_metadata)} datasets in '{output_dir}/'")
    print(f"  Shapes: {SHAPES}")
    print(f"  Per shape: {datasets_per_shape}")


if __name__ == "__main__":
    main()