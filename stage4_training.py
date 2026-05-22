"""
Stage 4: Meta-Learning Predictive Model Training and Evaluation
================================================================
Trains four classifiers (Random Forest, Gradient Boosting, SVM, KNN)
on the meta-dataset and evaluates against baseline strategies.
"""
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


# ──────────────────────────────────────────────
# LOAD AND PREPARE META-DATASET
# ──────────────────────────────────────────────

def load_meta_dataset():
    """Load meta-features and benchmark labels, join into a single meta-dataset."""
    features = pd.read_csv("meta_features.csv", index_col="dataset_id")
    labels = pd.read_csv("benchmark_results.csv", index_col="dataset_id")[["best_algorithm"]]
    df = features.join(labels)

    X = df.select_dtypes(include=np.number)
    y = df["best_algorithm"]
    return X, y


def clean_features(X):
    """Replace infinite values with NaN, impute with column mean, drop all-NaN columns."""
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    X = X.dropna(axis=1)
    return X


# ──────────────────────────────────────────────
# TRAIN-TEST SPLIT
# ──────────────────────────────────────────────

def split_data(X, y, test_size=0.2, random_state=42):
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"\nTrain label distribution:\n{y_train.value_counts()}")
    print(f"\nTest label distribution:\n{y_test.value_counts()}")
    test_ids = X_test.index.tolist()
    pd.DataFrame({"dataset_id": test_ids}).to_csv("test_split.csv", index=False) #save test list for traceability
    return X_train, X_test, y_train, y_test
   
# ──────────────────────────────────────────────
# BASELINE EVALUATION
# ──────────────────────────────────────────────
ARI_COLUMNS = ["KMeans", "KMedoids", "Agglomerative", "DBSCAN", "HDBSCAN"]

ALGO_TO_ARI_COL = {
    "KMeans": "KMeans",
    "KMedoids": "KMedoids",
    "Agglomerative": "Agglomerative",
    "DBSCAN": "DBSCAN",
    "HDBSCAN": "HDBSCAN",
}


def evaluate_ari_baselines(test_ids, model_predictions=None):
   
    df = pd.read_csv("benchmark_results.csv")
    test_df = df[df["dataset_id"].isin(test_ids)].copy()

    # Average-across-algorithms baseline 
    baseline_avg_ari = test_df[ARI_COLUMNS].mean(axis=1).mean()

    rows = [
     
        {"strategy": "Average across algorithms (baseline)",
         "mean_test_ari": baseline_avg_ari},
    ]

    # Add each model's mean ARI if predictions were passed in
    if model_predictions is not None:
        for model_name, preds in model_predictions.items():
            def predicted_ari(row):
                col = ALGO_TO_ARI_COL[preds[row["dataset_id"]]]
                return row[col]
            model_ari = test_df.apply(predicted_ari, axis=1).mean()
            rows.append({"strategy": model_name, "mean_test_ari": model_ari})

    results_df = pd.DataFrame(rows).sort_values("mean_test_ari", ascending=False)

    print("\n" + "=" * 50)
    print("ARI-BASED COMPARISON ON TEST SET")
    print("=" * 50)
    print(results_df.to_string(index=False))

    return results_df
def evaluate_majority_class_baseline(y_train, y_test):

    majority_label = y_train.value_counts().idxmax()
    y_pred = [majority_label] * len(y_test)

    accuracy = (y_pred == y_test).mean()
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n" + "=" * 50)
    print(f"MAJORITY CLASS BASELINE (always predicts '{majority_label}')")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    return {"majority_label": majority_label,
            "accuracy": accuracy,
            "macro_f1": macro_f1}

# ──────────────────────────────────────────────
# MODEL TRAINING WITH GRID SEARCH
# ──────────────────────────────────────────────

def define_classifiers():
    """Define classifiers and their hyperparameter grids."""
    return {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, None],
            },
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
            },
        },
        "SVM": {
            "model": SVC(random_state=42),
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
            },
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7, 9],
            },
        },
    }


def train_and_evaluate(classifiers, X_train, X_test, y_train, y_test):
    """Train each classifier with grid search and evaluate on test set."""
    results = {}

    print("\n" + "=" * 50)
    print("META-LEARNING MODEL RESULTS")
    print("=" * 50)

    for name, config in classifiers.items():
        print(f"\n--- {name} ---")

        gs = GridSearchCV(
            config["model"], config["params"],
            cv=5, scoring="accuracy", n_jobs=-1,
        )
        gs.fit(X_train, y_train)
        y_pred = gs.predict(X_test)

        print(f"Best params: {gs.best_params_}")
        print(f"Test accuracy: {gs.score(X_test, y_test):.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred, labels=gs.classes_))
        print("Labels:", list(gs.classes_))

        results[name] = {
            "best_params": gs.best_params_,
            "test_accuracy": gs.score(X_test, y_test),
            "model": gs.best_estimator_,
        }

    return results


def compute_feature_importances(rf_model, X_train, X_test, y_test):
   
    feature_names = X_train.columns.tolist()
    
  
    mdi_importance = pd.Series(
        rf_model.feature_importances_,
        index=feature_names,
        name="mdi"
    )
  
    # Combine into one table, sorted by MDI
    combined = pd.concat([mdi_importance], axis=1)
    combined = combined.sort_values("mdi", ascending=False)
    
    # Add rank columns for each method
    combined["mdi_rank"] = combined["mdi"].rank(ascending=False).astype(int)
    return combined



# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    X, y = load_meta_dataset()
    X = clean_features(X)
    print(f"Meta-dataset: {X.shape[0]} datasets, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = split_data(X, y)
    test_ids = X_test.index.tolist()  

    classifiers = define_classifiers()
    results = train_and_evaluate(classifiers, X_train, X_test, y_train, y_test)

    # Collect per-model predictions on the test set
    model_predictions = {
        name: dict(zip(test_ids, r["model"].predict(X_test)))
        for name, r in results.items()
    }

    ari_comparison = evaluate_ari_baselines(test_ids, model_predictions)
    ari_comparison.to_csv("ari_comparison.csv", index=False)
    # Usage in main():
    rf_model = results["RandomForest"]["model"]
    importance_table = compute_feature_importances(rf_model, X_train, X_test, y_test)

# Save full table
    importance_table.to_csv("feature_importances.csv")

# Print top 15 for inspection
    print("\nTop 15 features by MDI importance:")
    print(importance_table.head(15)[["mdi", "mdi_rank"]].round(4))
