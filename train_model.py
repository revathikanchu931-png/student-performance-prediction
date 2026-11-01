"""Create a clearly labeled synthetic dataset, explore it, and train regressors."""
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "student_performance.csv"
MODEL_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "static" / "images"
FEATURES = ["study_hours", "attendance", "previous_score", "assignment_completion", "sleep_hours", "internet_access", "extracurricular"]
NUMERIC = FEATURES[:5]
CATEGORICAL = FEATURES[5:]
TARGET = "exam_score"
SEED = 42


def generate_dataset(rows: int = 1200) -> pd.DataFrame:
    """Generate illustrative data; this is not collected from real students."""
    rng = np.random.default_rng(SEED)
    study = rng.uniform(0.5, 8.0, rows)
    attendance = rng.uniform(45, 100, rows)
    previous = rng.uniform(25, 98, rows)
    assignments = rng.uniform(25, 100, rows)
    sleep = rng.uniform(4.0, 10.0, rows)
    internet = rng.choice(["Yes", "No"], rows, p=[0.86, 0.14])
    extra = rng.choice(["Yes", "No"], rows, p=[0.55, 0.45])
    # Bounded, noisy pedagogical rule creates a learnable target, not real outcomes.
    score = (0.30 * previous + 2.35 * study + 0.17 * attendance + 0.105 * assignments
             + 0.72 * sleep + 2.0 * (internet == "Yes") + 1.5 * (extra == "Yes")
             + rng.normal(0, 7.0, rows) - 18)
    return pd.DataFrame({"study_hours": study.round(1), "attendance": attendance.round(1),
                         "previous_score": previous.round(1), "assignment_completion": assignments.round(1),
                         "sleep_hours": sleep.round(1), "internet_access": internet,
                         "extracurricular": extra, TARGET: np.clip(score, 0, 100).round(1)})


def build_pipeline(model):
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])
    return Pipeline([("preprocessor", prep), ("model", model)])


def main():
    for path in (DATA_PATH.parent, MODEL_DIR, ARTIFACTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    # Explicit data quality demonstration; generated data has no intended missing entries.
    df.to_csv(DATA_PATH, index=False)
    loaded = pd.read_csv(DATA_PATH)
    preview = loaded.head().to_dict(orient="records")
    missing = loaded.isna().sum().to_dict()
    before = len(loaded)
    loaded = loaded.drop_duplicates().dropna(subset=[TARGET])
    loaded[NUMERIC] = loaded[NUMERIC].apply(pd.to_numeric, errors="coerce")
    loaded = loaded.dropna(subset=NUMERIC + CATEGORICAL + [TARGET])
    X, y = loaded[FEATURES], loaded[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=SEED),
        "Random Forest": RandomForestRegressor(n_estimators=180, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
    }
    results, fitted = {}, {}
    for name, estimator in candidates.items():
        pipe = build_pipeline(estimator)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        results[name] = {"mae": round(float(mean_absolute_error(y_test, pred)), 3),
                         "mse": round(float(mse), 3), "rmse": round(float(np.sqrt(mse)), 3),
                         "r2": round(float(r2_score(y_test, pred)), 4)}
        fitted[name] = pipe
    best_name = min(results, key=lambda name: results[name]["rmse"])
    best = fitted[best_name]
    model_path = MODEL_DIR / "student_model.joblib"
    joblib.dump(best, model_path)

    sns.set_theme(style="whitegrid", palette="deep")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pd.DataFrame(results).T["rmse"].sort_values().plot.bar(ax=ax, color="#7567d8")
    ax.set(title="Test set model comparison (lower RMSE is better)", ylabel="RMSE (score points)", xlabel="Model")
    fig.tight_layout(); fig.savefig(ARTIFACTS_DIR / "model_comparison.png", dpi=150); plt.close(fig)
    corr = loaded[NUMERIC + [TARGET]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6)); sns.heatmap(corr, annot=True, cmap="Purples", fmt=".2f", ax=ax)
    ax.set_title("Numeric feature correlations (synthetic data)"); fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "feature_correlation.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5)); sns.histplot(loaded[TARGET], bins=20, kde=True, color="#7567d8", ax=ax)
    ax.set(title="Synthetic examination score distribution", xlabel="Score out of 100", ylabel="Records")
    fig.tight_layout(); fig.savefig(ARTIFACTS_DIR / "score_distribution.png", dpi=150); plt.close(fig)

    info = {"dataset_records": int(len(loaded)), "records_before_cleaning": int(before),
            "records_removed": int(before - len(loaded)), "missing_values_before_cleaning": missing,
            "preview": preview, "feature_names": FEATURES, "target": TARGET,
            "split": {"test_size": 0.2, "random_state": SEED}, "metrics": results,
            "best_model": best_name, "model_path": "models/student_model.joblib",
            "dataset_note": "Synthetic illustrative data generated by train_model.py; not real student records."}
    (MODEL_DIR / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps({"best_model": best_name, "metrics": results, "dataset_records": len(loaded), "model_path": str(model_path)}, indent=2))


if __name__ == "__main__":
    main()
