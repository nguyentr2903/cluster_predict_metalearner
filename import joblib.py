import joblib
results = joblib.load('trained_models.joblib')

for name in ['RandomForest', 'GradientBoosting', 'SVM', 'KNN']:
    print(f"{name}: {results[name]['model'].get_params()}")

