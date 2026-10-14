"""Basic data-quality checks for the diabetes-risk dataset."""
import pandas as pd

REQUIRED = {"patient_id", "diabetes_risk"}
ALLOWED_TARGETS = {"Low", "Moderate", "High"}

def validate(path="data/diabetes_risk.csv"):
    df = pd.read_csv(path)
    missing_required = REQUIRED - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {sorted(missing_required)}")
    if df["patient_id"].duplicated().any():
        print("WARNING: duplicate patient_id values detected")
    unknown = set(df["diabetes_risk"].dropna().unique()) - ALLOWED_TARGETS
    if unknown:
        raise ValueError(f"Unexpected target labels: {sorted(unknown)}")
    print(f"Rows: {len(df)} | columns: {len(df.columns)}")
    print("Missing values per column:")
    print(df.isna().sum().to_string())
    print("Target distribution:")
    print(df["diabetes_risk"].value_counts(dropna=False).to_string())
    print("Validation completed. This is a structural check, not clinical validation.")

if __name__ == "__main__":
    validate()
