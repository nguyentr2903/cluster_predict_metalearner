"""
Configuration for the meta-learning clustering pipeline.
Thesis: Predicting Clustering Performance from Meta-Learning on Synthetic Datasets
"""

# ── Synthetic Dataset Generation ──────────────────────────────────────
RANDOM_SEED = 42
N_DATASETS = 500  # total synthetic datasets to generate

# Variation ranges for controlled generation
N_SAMPLES_RANGE = (200, 2000)
N_FEATURES_RANGE = (2, 10)
N_CLUSTERS_RANGE = (2, 8)
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.35]
CLUSTER_STD_RANGE = (0.5, 3.0)

# Dataset shape types
SHAPE_TYPES = ["blobs", "anisotropic", "varied_variance", "moons", "circles"]

# ── Clustering Algorithms ─────────────────────────────────────────────
ALGORITHMS = ["kmeans", "kmedoids", "dbscan", "hdbscan", "agglomerative"]

# DBSCAN parameter grid for tuning
DBSCAN_EPS_RANGE = (0.1, 2.0)
DBSCAN_MIN_SAMPLES_RANGE = (3, 15)

# HDBSCAN parameters
HDBSCAN_MIN_CLUSTER_SIZE_RANGE = (5, 50)

# ── Meta-Feature Groups (PyMFE) ──────────────────────────────────────
# Groups aligned with Section 2.3 of the thesis
METAFEATURE_GROUPS = ["general", "statistical", "info-theory"]

# Additional custom meta-features (distance-based, clustering-based)
CUSTOM_METAFEATURES = True

# ── Evaluation Metrics ────────────────────────────────────────────────
PRIMARY_METRIC = "ari"  # Adjusted Rand Index
SECONDARY_METRICS = ["nmi", "silhouette"]

# ── Meta-Learner ─────────────────────────────────────────────────────
TEST_SPLIT = 0.2
CV_FOLDS = 5
META_LEARNERS = ["random_forest", "gradient_boosting", "svm", "knn"]

# ── Output Paths ─────────────────────────────────────────────────────
OUTPUT_DIR = "output"
DATASETS_DIR = f"{OUTPUT_DIR}/datasets"
METAFEATURES_PATH = f"{OUTPUT_DIR}/metafeatures.csv"
BENCHMARK_PATH = f"{OUTPUT_DIR}/benchmark_results.csv"
META_DATASET_PATH = f"{OUTPUT_DIR}/meta_dataset.csv"
