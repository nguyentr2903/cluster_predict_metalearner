# Meta-Learning for Clustering Algorithm Selection — Implementation

## Quick Start

```bash
pip install -r requirements.txt
python run_pipeline.py                # full pipeline
python run_pipeline.py --n-datasets 100  # quick test run with fewer datasets
python run_pipeline.py --stage 4      # re-run just the meta-learner
```

## Pipeline Architecture

```
Stage 1: generate synthetic datasets (500 datasets, 5 shape types)
    ↓
Stage 2: extract meta-features (PyMFE + custom distance-based)
    ↓
Stage 3: benchmark 5 clustering algorithms (ARI / NMI / Silhouette)
    ↓
Stage 4: train meta-learner to predict best algorithm from meta-features
```

---

## Synthetic Dataset Design Rationale

### Why synthetic data?
Your thesis (Section 3.1) correctly identifies that synthetic data lets you:
- **Control** individual properties (shape, noise, density) in isolation
- **Guarantee** ground-truth labels for ARI evaluation
- **Cover** distributional edge cases that real datasets rarely provide

### 5 Shape Types and What They Test

| Shape | Generator | Tests | Favors |
|-------|-----------|-------|--------|
| `blobs` | `make_blobs` | Isotropic Gaussian clusters | K-Means, K-Medoids |
| `anisotropic` | `make_blobs` + random transform | Non-spherical, stretched clusters | Agglomerative, HDBSCAN |
| `varied_variance` | `make_blobs` with per-cluster std | Different densities per cluster | HDBSCAN (adaptive density) |
| `moons` | `make_moons` | Non-convex, interleaving shapes | DBSCAN, HDBSCAN |
| `circles` | `make_circles` | Concentric rings | DBSCAN, HDBSCAN |

### Controlled Parameters
- **n_samples**: 200–2000 (covers small to medium datasets)
- **n_features**: 2–10 (keeps curse of dimensionality manageable)
- **n_clusters**: 2–8 (realistic range)
- **noise**: 0%, 5%, 10%, 20%, 35% (progressively harder)
- **cluster_std**: 0.5–3.0 (tight to loose clusters)

### Recommended: Additional Dataset Generators

Consider adding these to `stage1_generate.py` for better coverage:

```python
# Overlapping clusters (tests algorithm robustness)
from sklearn.datasets import make_classification
X, y = make_classification(n_informative=n_features, n_clusters_per_class=1, ...)

# Uniform noise background (tests noise handling)
# Add uniform random points as a "noise cluster"

# Subspace clusters (tests high-dimensional performance)
# Clusters that only exist in subsets of features
```

---

## Evaluation Metrics — Detailed Guide

### Primary: Adjusted Rand Index (ARI)

**Why ARI is your primary metric:**
- Requires ground-truth labels (which synthetic data provides)
- **Chance-adjusted**: random labeling gives ARI ≈ 0, not a misleadingly high score
- Range: [-0.5, 1.0] where 1.0 = perfect, 0 = random, negative = worse than random
- Handles different numbers of clusters between prediction and ground truth

**Formula intuition:** Measures the proportion of point pairs that are *consistently* co-clustered or separated in both the predicted and true labelings, adjusted for chance agreement.

**When ARI breaks down:**
- Very unbalanced cluster sizes → can inflate scores
- Datasets with noise points labeled as -1 (DBSCAN/HDBSCAN) → noise hurts ARI, which is *desired behavior* for this thesis since you want to know which algorithm truly recovers the ground truth

### Secondary: Normalized Mutual Information (NMI)

- Information-theoretic: measures how much knowing the predicted labels reduces uncertainty about the true labels
- Range: [0, 1] where 1 = perfect
- More robust to different numbers of clusters than ARI
- **Good complement** because it captures different aspects of agreement

### Secondary: Silhouette Score

- **Internal metric** (no ground truth needed) — useful for real-world validation (RQ3)
- Range: [-1, 1] where 1 = dense, well-separated clusters
- Measures cohesion (within-cluster distance) vs. separation (between-cluster distance)
- **Limitation**: biased toward convex, equally-sized clusters (favors K-Means)
- Use it as a sanity check, not as the primary target for the meta-learner

### Why Not Use These (Common Mistakes)

| Metric | Problem for This Thesis |
|--------|------------------------|
| Raw Rand Index | Not chance-adjusted → misleadingly high on random labelings |
| V-Measure | Tends to favor more clusters → biased |
| Davies-Bouldin | Internal only, no ground truth comparison |
| Calinski-Harabasz | Biased toward spherical clusters like Silhouette |

---

## Meta-Feature Groups

### Via PyMFE Library
- **General**: n_instances, n_features, dimensionality ratio
- **Statistical**: mean, std, skewness, kurtosis, correlations (Section 2.3.2)
- **Info-theory**: entropy, mutual information between features

### Custom: Distance-Based (Ferrari & de Castro, 2015)
- Pairwise Euclidean distance statistics (mean, std, skew, kurtosis)
- 10-bin histogram of normalized distances (Kalousis approach)
- Z-score discretization into 4 bins

### Custom: Clustering-Based
- Hopkins statistic (clustering tendency: 0.5=random, 1.0=clustered)
- Silhouette at various k values
- Elbow sharpness

---

## Real-World Validation Datasets (RQ3)

Built-in (sklearn):
- **Iris** — 150 samples, 4 features, 3 well-separated clusters
- **Wine** — 178 samples, 13 features, 3 classes
- **Breast Cancer** — 569 samples, 30 features, 2 classes

### Recommended UCI Datasets to Add

Download from https://archive.ics.uci.edu/:

| Dataset | Samples | Features | Clusters | Why It's Good |
|---------|---------|----------|----------|---------------|
| **Seeds** | 210 | 7 | 3 | Clean, small, well-separated |
| **Glass** | 214 | 9 | 6 | Overlapping, imbalanced clusters |
| **Ecoli** | 336 | 7 | 8 | Highly imbalanced |
| **Penguins** | 344 | 4 | 3 | Modern, clean, realistic |
| **Segment** | 2310 | 19 | 7 | Larger, higher-dimensional |
| **Mice Protein** | 1080 | 77 | 8 | High-dimensional, missing values |
| **Banknote** | 1372 | 4 | 2 | Binary, well-structured |

To add a UCI dataset:
```python
# In stage4_metalearner.py → validate_on_real_data()
import pandas as pd
df = pd.read_csv("path/to/uci_dataset.csv")
X_real = df.drop("class", axis=1).values
y_real = LabelEncoder().fit_transform(df["class"].values)
```

---

## Output Files

```
output/
├── datasets/              # .npz files (X, y arrays)
│   ├── dataset_0000.npz
│   ├── ...
│   └── generation_metadata.json
├── metafeatures.csv       # Meta-features for all datasets
├── benchmark_results.csv  # ARI/NMI/Silhouette per algorithm per dataset
├── meta_dataset.csv       # Combined features + best_algorithm label
├── feature_importance.csv # Permutation importance (RQ1)
├── gini_importance.csv    # Gini importance from Random Forest
└── evaluation_results.json # Accuracy scores for all models + baselines
```

---

## Tuning Notes

- **N_DATASETS = 500**: Start with 100 for debugging, scale to 500–1000 for final results.
  Ferrari & de Castro used 30 datasets; more gives better meta-learner generalization.
- **DBSCAN eps**: Auto-estimated via k-distance graph. Consider also running a grid
  search over eps ∈ {0.3, 0.5, 0.8, 1.0, 1.5} and taking the best ARI.
- **HDBSCAN min_cluster_size**: Set to n_samples/50 by default. Tune if results look off.
- **Meta-learner**: Random Forest typically wins for tabular meta-learning. If class
  imbalance is severe (e.g., K-Means wins 60% of datasets), use `class_weight="balanced"`.
