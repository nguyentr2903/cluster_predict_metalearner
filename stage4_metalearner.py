"""
Stage 4: Meta-Learner Training and Evaluation
===============================================
Trains a meta-learning model to predict the best clustering algorithm
from dataset meta-features.

This stage addresses the three research questions:
  RQ1: Which meta-features most strongly influence prediction accuracy?
  RQ2: How does meta-learning compare to baseline selection strategies?
  RQ3: Do predictions generalize to real-world datasets?

References: Thesis Sections 1.2, 1.5, 2.4
"""

import numpy as np
import pandas as pd
import os
import json
import warnings
from collections import Counter

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

from config import *


def load_meta_dataset():
    """
    Combine meta-features and benchmark results into a single meta-dataset.
    X = meta-features, y = best_algorithm label
    """
    mf = pd.read_csv(METAFEATURES_PATH, index_col="dataset_id")
    bench = pd.read_csv(BENCHMARK_PATH, index_col="dataset_id")

    # Merge on dataset_id
    meta_df = mf.join(bench[["best_algorithm", "best_ari"]], how="inner")

    # Drop datasets where no algorithm worked
    meta_df = meta_df.dropna(subset=["best_algorithm"])
    meta_df = meta_df[meta_df["best_algorithm"] != "unknown"]

    print(f"Meta-dataset: {meta_df.shape[0]} datasets, {mf.shape[1]} meta-features")
    print(f"\nTarget distribution:")
    print(meta_df["best_algorithm"].value_counts())

    # Save combined meta-dataset
    meta_df.to_csv(META_DATASET_PATH)

    return meta_df


def get_feature_matrix(meta_df):
    """Separate features and target, encode labels."""
    # Feature columns = all meta-feature columns (not benchmark results)
    mf_cols = pd.read_csv(METAFEATURES_PATH, index_col="dataset_id").columns.tolist()
    feature_cols = [c for c in mf_cols if c in meta_df.columns]

    X = meta_df[feature_cols].values
    y_str = meta_df["best_algorithm"].values

    # Encode string labels to integers
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    # Handle any remaining NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, le, feature_cols


# ── Baseline Strategies (for RQ2 comparison) ──────────────────────────

def baseline_random(y_test, classes, rng):
    """Random selection baseline."""
    return rng.choice(classes, size=len(y_test))


def baseline_majority(y_test, y_train):
    """Always predict the most common algorithm in training set."""
    majority = Counter(y_train).most_common(1)[0][0]
    return np.full(len(y_test), majority)


def baseline_default_kmeans(y_test, le):
    """Always predict K-Means (the 'default' practitioner choice)."""
    km_label = le.transform(["kmeans"])[0] if "kmeans" in le.classes_ else 0
    return np.full(len(y_test), km_label)


# ── Meta-Learner Models ──────────────────────────────────────────────

def get_meta_learners():
    """Return dict of meta-learner models to evaluate."""
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        ),
        "svm": SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            random_state=42,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            n_jobs=-1,
        ),
    }


def train_and_evaluate(meta_df):
    """
    Full training and evaluation pipeline.
    Returns results dict for analysis.
    """
    X, y, le, feature_cols = get_feature_matrix(meta_df)
    rng = np.random.default_rng(RANDOM_SEED)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Filter out classes with fewer than 2 samples (can't stratify)
    class_counts = Counter(y)
    rare_classes = {cls for cls, cnt in class_counts.items() if cnt < 2}
    if rare_classes:
        rare_names = [le.inverse_transform([c])[0] for c in rare_classes]
        print(f"  Note: Dropping rare classes with <2 samples: {rare_names}")
        mask = np.array([yi not in rare_classes for yi in y])
        X_scaled = X_scaled[mask]
        y = y[mask]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )

    results = {}

    # ── Baselines (RQ2) ──────────────────────────────────────────
    print("=" * 60)
    print("BASELINE STRATEGIES (RQ2)")
    print("=" * 60)

    # Random
    y_pred_random = baseline_random(y_test, np.unique(y), rng)
    acc_random = accuracy_score(y_test, y_pred_random)
    print(f"  Random selection:   {acc_random:.4f}")
    results["baseline_random"] = acc_random

    # Majority
    y_pred_majority = baseline_majority(y_test, y_train)
    acc_majority = accuracy_score(y_test, y_pred_majority)
    print(f"  Majority class:     {acc_majority:.4f}")
    results["baseline_majority"] = acc_majority

    # Default K-Means
    y_pred_km = baseline_default_kmeans(y_test, le)
    acc_km = accuracy_score(y_test, y_pred_km)
    print(f"  Default K-Means:    {acc_km:.4f}")
    results["baseline_kmeans"] = acc_km

    # ── Meta-Learners ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("META-LEARNER MODELS")
    print("=" * 60)

    best_model = None
    best_acc = 0
    best_name = ""

    models = get_meta_learners()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for name, model in models.items():
        # Cross-validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

        # Fit on full training set, evaluate on test
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        print(f"\n  {name}:")
        print(f"    CV accuracy:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"    Test accuracy: {test_acc:.4f}")

        results[f"{name}_cv_mean"] = cv_scores.mean()
        results[f"{name}_cv_std"] = cv_scores.std()
        results[f"{name}_test_acc"] = test_acc

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = model
            best_name = name

    # ── Best Model Analysis ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"BEST MODEL: {best_name} (test accuracy: {best_acc:.4f})")
    print("=" * 60)

    y_pred_best = best_model.predict(X_test)
    present_classes = sorted(set(y_test) | set(y_pred_best))
    target_names = [le.inverse_transform([c])[0] for c in present_classes]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best, labels=present_classes, target_names=target_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_best, labels=present_classes)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df)

    # ── Feature Importance (RQ1) ─────────────────────────────────
    print(f"\n{'=' * 60}")
    print("META-FEATURE IMPORTANCE (RQ1)")
    print("=" * 60)

    perm_imp = permutation_importance(
        best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": perm_imp.importances_mean,
        "importance_std": perm_imp.importances_std,
    }).sort_values("importance_mean", ascending=False)

    print("\nTop 15 most important meta-features:")
    print(imp_df.head(15).to_string(index=False))

    # Save importance
    imp_df.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)

    # If Random Forest, also get built-in feature importance
    if hasattr(best_model, "feature_importances_"):
        builtin_imp = pd.DataFrame({
            "feature": feature_cols,
            "gini_importance": best_model.feature_importances_,
        }).sort_values("gini_importance", ascending=False)
        builtin_imp.to_csv(f"{OUTPUT_DIR}/gini_importance.csv", index=False)

    # Save all results
    with open(f"{OUTPUT_DIR}/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results, best_model, le, scaler, feature_cols


def validate_on_real_data(model, le, scaler, feature_cols):
    """
    RQ3: Validate on real-world datasets.
    
    Uses datasets from UCI ML Repository or similar sources
    that have known ground-truth cluster labels.
    
    Suggested datasets:
    - Iris (3 clusters, 4 features)
    - Wine (3 clusters, 13 features)  
    - Seeds (3 clusters, 7 features)
    - Breast Cancer Wisconsin (2 clusters)
    - Penguins (3 species)
    
    This function is a template — extend with your chosen real datasets.
    """
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer
    from stage2_metafeatures import extract_all_features

    real_datasets = {
        "iris": load_iris(),
        "wine": load_wine(),
        "breast_cancer": load_breast_cancer(),
    }

    print(f"\n{'=' * 60}")
    print("REAL-WORLD DATASET VALIDATION (RQ3)")
    print("=" * 60)

    for name, data in real_datasets.items():
        X_real = StandardScaler().fit_transform(data.data)
        y_real = data.target

        # Extract meta-features
        features = extract_all_features(X_real)
        feature_vector = np.array([features.get(col, 0.0) for col in feature_cols])
        feature_vector = np.nan_to_num(feature_vector).reshape(1, -1)
        feature_vector = scaler.transform(feature_vector)

        # Predict best algorithm
        predicted_algo = le.inverse_transform(model.predict(feature_vector))[0]

        # Actually run all algorithms and check
        from stage3_benchmark import benchmark_single_dataset
        actual_results = benchmark_single_dataset(X_real, y_real, len(np.unique(y_real)))

        print(f"\n  {name}:")
        print(f"    Predicted best: {predicted_algo}")
        print(f"    Actual best:    {actual_results['best_algorithm']} "
              f"(ARI={actual_results['best_ari']:.4f})")
        for algo in ALGORITHMS:
            ari = actual_results.get(f"{algo}_ari", float("nan"))
            marker = " ← predicted" if algo == predicted_algo else ""
            marker += " ← actual best" if algo == actual_results['best_algorithm'] else ""
            print(f"      {algo:20s}: ARI={ari:.4f}{marker}")


if __name__ == "__main__":
    meta_df = load_meta_dataset()
    results, best_model, le, scaler, feature_cols = train_and_evaluate(meta_df)
    validate_on_real_data(best_model, le, scaler, feature_cols)
