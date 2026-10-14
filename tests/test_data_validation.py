from src.data_validation import validate

def test_validation_runs():
    validate("data/diabetes_risk.csv")
