# MEDINSIGHT - PATIENT LEVEL PREDICTION
import joblib
import pandas as pd

# Load saved model and preprocessor

model = joblib.load(
    "models/tuned_xgboost.pkl"
)

preprocessor = joblib.load(
    "models/preprocessor.pkl"
)

print("Model and preprocessor loaded successfully.")


# Create sample patient

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


# Preprocess patient data

patient_processed = preprocessor.transform(
    patient
)

print("Patient data processed successfully.")


# Predict readmission probability

probability = model.predict_proba(
    patient_processed
)[0, 1]


# Convert probability into risk category

final_threshold = 0.60

prediction = int(
    probability >= final_threshold
)

if prediction == 1:
    risk_category = "High Risk"
else:
    risk_category = "Low Risk"


# Display result

print("\n" + "=" * 50)
print("MEDINSIGHT PATIENT PREDICTION")
print("=" * 50)

print(
    f"Readmission Probability: {probability:.2%}"
)

print(
    f"Risk Category: {risk_category}"
)

print(
    f"Model Prediction: "
    f"{'Readmitted within 30 days' if prediction == 1 else 'Not readmitted within 30 days'}"
)

print("=" * 50)