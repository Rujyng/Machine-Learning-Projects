import os
import joblib
import numpy as np
from process_data import load_sparseX, parse_config

# ---------- Paths ----------
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(root, "proj_data", "task3_ratings")

config_path = os.path.join(data_dir, "task3.config")
test_X_path = os.path.join(data_dir, "test.sparseX")

# Load config to know dimensions
n_train, n_dev, d, c = parse_config(config_path)

# ---------- Load TF-IDF transformer + best model ----------
tfidf = joblib.load(os.path.join(root, "code", "models", "task3_tfidf.joblib"))
model = joblib.load(os.path.join(root, "code", "models", "task3_best_logreg_C_7.0.joblib"))

print("Loaded model and TF-IDF.")

# ---------- Load Test Data ----------
X_test = load_sparseX(test_X_path, None, d)   # None = load entire file
print("Loaded test data with shape:", X_test.shape)

# ---------- Transform ----------
X_test_tfidf = tfidf.transform(X_test)

# ---------- Predict ----------
preds = model.predict(X_test_tfidf)

# Clip predictions to valid range
preds = np.clip(preds, 0, 4)

# ---------- Save ----------
out_path = os.path.join(root, "task3_predictions.txt")
with open(out_path, "w") as f:
    for p in preds:
        f.write(str(p) + "\n")

print("Predictions saved to:", out_path)