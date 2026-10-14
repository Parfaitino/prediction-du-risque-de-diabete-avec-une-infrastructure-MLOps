"""Optional simple interface for a previously trained model."""
import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

st.set_page_config(page_title="Diabetes Risk — démonstrateur", page_icon="🧪")
st.title("Démonstrateur — classification exploratoire du risque")
st.warning("Prototype pédagogique uniquement. Ne pas utiliser pour diagnostiquer ou guider des décisions médicales.")
model_path = Path("reports/results/diabetes_risk_model.joblib")
if not model_path.exists():
    st.info("Lancez d'abord : python src/train.py")
else:
    model = joblib.load(model_path)
    st.write("Chargez un CSV contenant les variables explicatives du dataset (sans diabetes_risk ni patient_id).")
    uploaded = st.file_uploader("Fichier CSV", type=["csv"])
    if uploaded:
        data = pd.read_csv(uploaded)
        data = data.drop(columns=["patient_id", "diabetes_risk"], errors="ignore")
        predictions = model.predict(data)
        result = data.copy()
        result["prediction_experimentale"] = predictions
        st.dataframe(result.head(100), use_container_width=True)
        st.download_button("Télécharger les prédictions CSV", result.to_csv(index=False), "predictions.csv", "text/csv")
