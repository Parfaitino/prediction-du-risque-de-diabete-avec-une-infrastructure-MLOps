"""Train/evaluate the diabetes-risk baseline and log the run to MLflow if installed."""
from pathlib import Path
import json, argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

def build_pipeline(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocessor", prep), ("classifier", RandomForestClassifier(
        n_estimators=250, random_state=42, class_weight="balanced",
        n_jobs=-1, min_samples_leaf=2))])

def main(data_path="data/diabetes_risk.csv", output_dir="reports/results"):
    data = pd.read_csv(data_path)
    target = "diabetes_risk"
    X = data.drop(columns=[target, "patient_id"], errors="ignore")
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    model = build_pipeline(X_train)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "random_state": 42, "test_size": 0.20,
    }
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "classification_report.txt").write_text(
        classification_report(y_test, pred, zero_division=0), encoding="utf-8")
    pd.DataFrame(confusion_matrix(y_test, pred, labels=["High","Low","Moderate"]),
                 index=["actual_High","actual_Low","actual_Moderate"],
                 columns=["pred_High","pred_Low","pred_Moderate"]).to_csv(out/"confusion_matrix.csv")
    joblib.dump(model, out/"diabetes_risk_model.joblib")

    # Optional MLflow: code still runs without it; install requirements to enable logging.
    try:
        import mlflow
        import mlflow.sklearn
        mlflow.set_experiment("diabetes-risk-classification")
        with mlflow.start_run():
            mlflow.log_params({"model": "RandomForestClassifier", "n_estimators": 250,
                               "random_state": 42, "test_size": 0.2,
                               "class_weight": "balanced"})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact(str(out/"classification_report.txt"))
            mlflow.log_artifact(str(out/"confusion_matrix.csv"))
    except ImportError:
        print("MLflow not installed; metrics saved locally. Install requirements.txt to enable experiment tracking.")
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, pred, zero_division=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/diabetes_risk.csv")
    parser.add_argument("--output", default="reports/results")
    args = parser.parse_args()
    main(args.data, args.output)
