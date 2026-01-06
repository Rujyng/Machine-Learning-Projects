# Machine-Learning-Projects

## Review Rating Prediction with TF-IDF + Model Selection

### What it does
Predicts 0–4 user ratings from text reviews using a sparse bag-of-words feature matrix, TF-IDF transformation, and dev-set model selection based on Mean Squared Error (MSE).

### Approach
- Loads sparse review features (`*.sparseX`) and rating targets (`*.RT`) using a config file (`task3.config`).
- Applies **TF-IDF** (`TfidfTransformer(sublinear_tf=True)`) fit on train, then transforms dev.
- Trains and evaluates multiple models on the dev set:
  - `LinearRegression`
  - `Ridge` (sweeps many `alpha` values)
  - `LinearSVR` (sweeps `C`)
  - `LogisticRegression` (sweeps `C`, evaluates both **soft** expected-rating and **hard** class prediction MSE)
  - `LinearSVC` (sweeps `C`)
  - `RidgeClassifier` (sweeps `alpha`)
- Clips predictions to the valid rating range **[0, 4]** before scoring.
- Uses a **mean-rating baseline** for comparison.

### Output
- Prints baseline MSE, per-model dev MSE, and the best model.
- Saves artifacts to `code/models/`:
  - `best_<model>.joblib` (best model by dev MSE)
  - `tfidf.joblib` (TF-IDF transformer)

### Run
```bash
python code/run.py
