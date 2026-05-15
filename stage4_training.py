import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix 
from collections import Counter #for baselines
import joblib

def load_meta_dataset():
    features = pd.read_csv("meta_features.csv", index_col="dataset_id")
    labels = pd.read_csv("benchmark_results.csv", index_col="dataset_id")
    
    # keep only the best_algorithm column from benchmark results
    labels = labels[["best_algorithm"]]
    
    # join on dataset_id
    df = features.join(labels)
    
    # separate features (X) from target (y)
    X = df.select_dtypes(include=np.number)
    y = df["best_algorithm"]
    return X, y

X, y = load_meta_dataset()
print(X.shape)

# replace any remaining NaN or infinite values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())
X = X.dropna(axis=1)
#split the dataset to 80 training, 20 testing 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"\nTrain label distribution:")
print(y_train.value_counts())
print(f"\nTest label distribution:")
print(y_test.value_counts())


#train model 
classifiers = {
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
    },
    "SVM": {
        "model": SVC(random_state=42),
        "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {"n_neighbors": [3, 5, 7, 9]},
    },
}

results = {}
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
    print(classification_report(y_test, y_pred, zero_division = 0))

    results[name] = {
        "best_params": gs.best_params_,
        "test_accuracy": gs.score(X_test, y_test),
        "model": gs.best_estimator_,
    }
# majority class: always predict the most common label
majority_class = y_train.value_counts().index[0]
y_pred_majority = [majority_class] * len(y_test)

# random: pick uniformly from the 5 algorithms
rng = np.random.default_rng(42)
y_pred_random = rng.choice(y_train.unique(), size=len(y_test))

# always-KMeans
y_pred_kmeans = ["KMeans"] * len(y_test)

print("--- Majority Class Baseline ---")
print(classification_report(y_test, y_pred_majority, zero_division=0))

print("--- Random Baseline ---")
print(classification_report(y_test, y_pred_random, zero_division=0))

# --- Feature Importance (RQ1) ---
rf_model = results["RandomForest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

print("\n--- Top 10 Most Important Meta-Features ---")
print(top_features.to_string())

'''This section also answers RQ2'''

# save summary of current results
summary = pd.DataFrame({
    name: {"accuracy": r["test_accuracy"], "best_params": str(r["best_params"])}
    for name, r in results.items()
}).T
summary.to_csv("model_results.csv")

# save feature importances
importances.sort_values(ascending=False).to_csv("feature_importances.csv")
joblib.dump(results, "trained_models.joblib")