import numpy as np
import os
import json
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler

output_dir = "datasets"
datasets_per_shape = 100
random_seed_base = 42

def generate_blobs (n_samples, n_features, n_clusters, cluster_std, noise, seed, rng): 
    X, y = make_blobs (
        n_samples = n_samples,
        n_features = n_features,
        n_clusters = n_clusters,
        cluster_std = cluster_std,
        random_state = seed,
    )
  

def generate_anisotropic(n_samples, n_features, n_clusters, cluster_std, noise, seed, rng):
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        n_clusters=n_clusters,
        cluster_std=cluster_std,
        random_state=seed,
    )

def generate_varied_variance(n_samples, n_features, n_clusters, noise, seed, rng):
    cluster_stds = rng.uniform(0.5, 3.0, size=n_clusters)
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_stds,
        random_state=seed,
    )

def generate_moons(n_samples, noise, seed):
    moon_noise = max(0.05, noise) if noise > 0 else 0.05 #noise floor
    X, y = make_moons(n_samples=n_samples, noise=moon_noise, random_state=seed)
    return X, y

def generate_circles(n_samples, noise, seed):
    circle_noise = max(0.03, noise) if noise > 0 else 0.04
    X, y = make_circles(
        n_samples=n_samples, noise=circle_noise, factor=0.5, random_state=seed
    )
    return X, y
