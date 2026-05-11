# --- Dataset Generation ---
SHAPES = ["blobs", "anisotropic", "varied_variance", "moons", "circles"]
DATASETS_PER_SHAPE = 100
RANDOM_SEED = 42

# --- Parameter ranges ---
N_SAMPLES_RANGE = (100, 1000)
N_FEATURES_RANGE = (2, 10)
N_CLUSTERS_RANGE = (2, 8)
CLUSTER_STD_RANGE = (0.3, 2.5)
NOISE_RANGE = (0.0, 0.3)

# --- Noise experiment (RQ3) ---
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

# --- Meta-features ---
PYMFE_GROUPS = ["general", "statistical"]

# --- Algorithms ---
ALGORITHMS = ["KMeans", "KMedoids", "Agglomerative", "DBSCAN", "HDBSCAN"]

# --- Meta-learners ---
CLASSIFIERS = ["RandomForest", "GradientBoosting", "SVM", "KNN"]

# --- Paths ---
DATASETS_DIR = "datasets"
META_FEATURES_FILE = "meta_features.csv"
BENCHMARK_FILE = "benchmark_results.csv"