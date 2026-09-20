import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load saved model and preprocessor

model = joblib.load(
    "models/tuned_xgboost.pkl"
)

preprocessor = joblib.load(
    "models/preprocessor.pkl"
)

print("Model and preprocessor loaded successfully.")


# Create the sample patient

patient = pd.DataFrame([{
    "admit_type": "Emergency",
    "ward_type": "General",
    "hospital_id": "H001",
    "num_procedures": 2,
    "charlson_index": 3,
    "hba1c": 7.5,
    "creatinine": 1.4,
    "haemoglobin": 10.5,
    "systolic_bp": 145,
    "num_diagnoses": 4,
    "primary_icd10": "I10",
    "primary_diag_category": "Cardiovascular",

    "has_cardiovascular": 1,
    "has_endocrine": 0,
    "has_gastrointestinal": 0,
    "has_genitourinary": 0,
    "has_infectious": 0,
    "has_injury": 0,
    "has_neoplasm": 0,
    "has_neurological": 0,
    "has_obstetric": 0,
    "has_perinatal": 0,
    "has_respiratory": 0,

    "age": 65,
    "gender": "Male",
    "patient_state": "Rajasthan",
    "bpl_card": 0,
    "insurance_type": "Private",
    "comorbidity_count": 3,
    "prev_admissions": 2,

    "tier": "2",
    "beds": 300,
    "teaching": 1,

    "admit_year": 2026,
    "admit_month": 9,
    "admit_dayofweek": 4
}])


# Preprocess patient

patient_processed = preprocessor.transform(
    patient
)


# Create SHAP explainer

explainer = shap.TreeExplainer(
    model
)


# Calculate SHAP values

shap_values = explainer.shap_values(
    patient_processed
)


# Get feature names

feature_names = (
    preprocessor
    .get_feature_names_out()
)


# Display SHAP explanation

shap_explanation = pd.DataFrame({
    "Feature": feature_names,
    "SHAP_Value": shap_values[0]
})

shap_explanation["Absolute_SHAP"] = (
    shap_explanation["SHAP_Value"].abs()
)

shap_explanation = shap_explanation.sort_values(
    "Absolute_SHAP",
    ascending=False
)


print("\nTop factors influencing this patient's prediction:")

print(
    shap_explanation[
        ["Feature", "SHAP_Value"]
    ].head(10)
)


# Create SHAP waterfall plot

shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=patient_processed.toarray()[0]
        if hasattr(patient_processed, "toarray")
        else patient_processed[0],
        feature_names=feature_names
    )
)