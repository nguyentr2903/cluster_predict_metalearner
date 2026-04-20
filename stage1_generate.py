import numpy as np
import os
import json
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from config import *

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

  
