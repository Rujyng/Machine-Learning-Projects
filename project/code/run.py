import os
import joblib
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier

from process_data import parse_config, load_sparseX, load_RT

def clip_ratings(arr):
    return np.clip(arr, 0, 4)

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "proj_data", "task3_ratings")

    config_path = os.path.join(data_dir, "task3.config")
    train_X_path = os.path.join(data_dir, "train.sparseX")
    train_y_path = os.path.join(data_dir, "train.RT")
    dev_X_path = os.path.join(data_dir, "dev.sparseX")
    dev_y_path = os.path.join(data_dir, "dev.RT")

    # config
    n_train, n_dev, d, c = parse_config(config_path)
    print("Loaded config:")
    print("  N_TRAIN =", n_train)
    print("  N_DEV   =", n_dev)
    print("  D       =", d)
    print("  C       =", c)

    # sparse feature matrices & rating vectors
    print("\nLoading training data...")
    X_train = load_sparseX(train_X_path, n_train, d)
    y_train = load_RT(train_y_path)

    print("Loading dev data...")
    X_dev = load_sparseX(dev_X_path, n_dev, d)
    y_dev = load_RT(dev_y_path)

    # TF-IDF transformation
    print("Applying TF-IDF transformation...")

    tfidf = TfidfTransformer(sublinear_tf=True)
    X_train = tfidf.fit_transform(X_train) 
    X_dev = tfidf.transform(X_dev)

    # # debug
    # print("X_train shape:", X_train.shape)
    # print("X_dev shape:  ", X_dev.shape)
    # print("y_train shape:", y_train.shape)
    # print("y_dev shape:  ", y_dev.shape)
    # print("train nnz:", X_train.nnz)
    # print("dev nnz:  ", X_dev.nnz)
    # print("# training samples:", X_train.shape[0])
    # print("# features:", X_train.shape[1])
    # print("First 300 ratings (y_train):", y_train[:300])
    # print()

    # baseline
    mean_pred = np.full_like(y_dev, np.mean(y_train), dtype=float)
    baseline_mse = mean_squared_error(y_dev, mean_pred)
    print("Mean baseline MSE:", baseline_mse)

    models = {}

    models["linear"] = LinearRegression()

    for alpha in [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2, 2.3, 2.5, 3.0, 3.5, 4.0, 10.0]:
        name = f"ridge_{alpha}"
        models[name] = Ridge(alpha=alpha)
    
    for C in [0.01, 0.1, 0.25, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 1.7, 2.0, 4.0, 5.0, 7.0, 10.0]:
        name = f"linearsvr_C_{C}"
        models[name] = LinearSVR(C=C, epsilon=0.1, max_iter=5000, dual=True)

    logreg_C_values = [0.01, 0.1, 1.0, 3.0, 4.0, 5.0, 6.0, 6.1, 6.15, 6.2, 6.25, 6.3, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0, 8.5, 9.0, 9.5, 10.0, 12.0, 15.0]
    for C in logreg_C_values:
        name = f"logreg_C_{C}"
        models[name] = LogisticRegression(
            C=C,
            max_iter=5000,
            solver="liblinear",
            multi_class="auto"
        )

    svc_C_values = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 5.0]
    for C in svc_C_values:
        name = f"svc_C_{C}"
        models[name] = LinearSVC(
            C=C,
            loss="squared_hinge",
            max_iter=5000,
            dual=True
        )

    ridgeclf_alphas = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 5.0, 7.0]
    for alpha in ridgeclf_alphas:
        name = f"ridgeclf_{alpha}"
        models[name] = RidgeClassifier(alpha=alpha)

    # -------- Train + eval --------
    print("\nTraining models and evaluating on dev set...")
    mse_scores = {}
    best_name = None
    best_mse = float("inf")

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        if name.startswith(("linear", "ridge_", "linearsvr_")):
            preds = model.predict(X_dev)
            preds = clip_ratings(preds)
        elif name.startswith("logreg_C_"):
            probs = model.predict_proba(X_dev)
            classes = model.classes_.astype(float)
            soft_preds = probs @ classes
            soft_preds = clip_ratings(soft_preds) 

            hard_preds = classes[np.argmax(probs, axis=1)]
            hard_preds = clip_ratings(hard_preds)

            # accuracy = np.mean(hard_preds == y_dev)
            # print(f"{name} HARD accuracy = {accuracy}")

            # MSE for soft
            mse_soft = mean_squared_error(y_dev, soft_preds)
            mse_scores[name + "_soft"] = mse_soft
            print(f"{name} SOFT MSE = {mse_soft}")

            # MSE for hard
            mse_hard = mean_squared_error(y_dev, hard_preds)
            mse_scores[name + "_hard"] = mse_hard
            print(f"{name} HARD MSE = {mse_hard}")

            if mse_soft < best_mse:
                best_mse = mse_soft
                best_name = name
            continue
        elif name.startswith("svc_C_") or name.startswith("ridgeclf_"):
            preds = model.predict(X_dev).astype(float)
            preds = clip_ratings(preds) 
        else:
            continue

        # mse
        mse = mean_squared_error(y_dev, preds)
        mse_scores[name] = mse
        print(f"{name} MSE = {mse}")

        if mse < best_mse:
            best_mse = mse
            best_name = name

    print("\nSummary of MSEs:")
    print("baseline_mean =", baseline_mse)
    for name, mse in mse_scores.items():
        print(f"{name} =", mse)

    print(f"\nBest model: {best_name} (MSE = {best_mse})")
    # -------- Save best model --------
    model_dir = os.path.join(root, "code", "models")
    os.makedirs(model_dir, exist_ok=True)

    # save best model
    best_model_path = os.path.join(model_dir, f"best_{best_name}.joblib")
    joblib.dump(models[best_name], best_model_path)
    print("\nSaved best model to:", best_model_path)

    # save TF-IDF transformer
    tfidf_path = os.path.join(model_dir, "tfidf.joblib")
    joblib.dump(tfidf, tfidf_path)
    print("Saved TF-IDF transformer to:", tfidf_path)

    print("\nDone.")


if __name__ == "__main__":
    main()