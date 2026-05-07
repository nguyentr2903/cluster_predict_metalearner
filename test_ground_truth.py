from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
import numpy as np

# Step 1: generate data — this gives us X (points) and y_true (answer key)
X, y_true = make_moons(n_samples=200, noise=0.05, random_state=42)

# Step 2: run a clustering algorithm — it only sees X, not y_true
y_pred = KMeans(n_clusters=2, random_state=42).fit_predict(X)

# Step 3: compare predictions against truth
ari = adjusted_rand_score(y_true, y_pred)

print("True labels (first 10):", y_true[:10])
print("Predicted labels (first 10):", y_pred[:10])
print(f"\nARI: {ari:.4f}")