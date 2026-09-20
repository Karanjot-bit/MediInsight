# SHAP EXPLAINABILITY

import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 1. LOAD SAVED MODEL AND PREPROCESSOR

best_xgb_model = joblib.load(
    "models/tuned_xgboost.pkl"
)

preprocessor = joblib.load(
    "models/preprocessor.pkl"
)

print("Model and preprocessor loaded successfully.")

# 2. LOAD DATA

admissions = pd.read_csv("data/archive/admissions.csv")
billing = pd.read_csv("data/archive/billing.csv")
diagnoses = pd.read_csv("data/archive/diagnoses.csv")
patients = pd.read_csv("data/archive/patients.csv")
hospitals = pd.read_csv("data/archive/hospitals.csv")

print("Datasets loaded successfully.")

# 3. BASIC PREPARATION

patients["insurance_type"] = patients["insurance_type"].fillna("Unknown")

admissions["admit_date"] = pd.to_datetime(
    admissions["admit_date"]
)

admissions["discharge_date"] = pd.to_datetime(
    admissions["discharge_date"]
)

# 4. AGGREGATE DIAGNOSES

num_diagnoses = (
    diagnoses
    .groupby("admission_id")
    .size()
    .reset_index(name="num_diagnoses")
)

primary_diagnosis = (
    diagnoses[diagnoses["diag_rank"] == 1]
    [
        [
            "admission_id",
            "icd10_code",
            "diag_desc",
            "diag_category"
        ]
    ]
    .rename(
        columns={
            "icd10_code": "primary_icd10",
            "diag_desc": "primary_diagnosis",
            "diag_category": "primary_diag_category"
        }
    )
)

diagnosis_categories = pd.crosstab(
    diagnoses["admission_id"],
    diagnoses["diag_category"]
)

diagnosis_categories.columns = [
    f"has_{str(col).lower()}"
    for col in diagnosis_categories.columns
]

diagnosis_categories = diagnosis_categories.astype(int)

diagnosis_aggregated = (
    num_diagnoses
    .merge(
        primary_diagnosis,
        on="admission_id",
        how="left"
    )
    .merge(
        diagnosis_categories,
        on="admission_id",
        how="left"
    )
)

# 5. MERGE ALL DATASETS

merged_data = admissions.merge(
    diagnosis_aggregated,
    on="admission_id",
    how="left"
)

merged_data = merged_data.merge(
    patients,
    on="patient_id",
    how="left",
    suffixes=("_admission", "_patient")
)

merged_data = merged_data.merge(
    billing,
    on="admission_id",
    how="left"
)

merged_data = merged_data.merge(
    hospitals,
    on="hospital_id",
    how="left",
    suffixes=("", "_hospital")
)

# 6. CLEAN STATE COLUMNS

if "state" in merged_data.columns:
    merged_data = merged_data.rename(
        columns={"state": "patient_state"}
    )

if "state_hospital" in merged_data.columns:
    merged_data = merged_data.rename(
        columns={"state_hospital": "hospital_state"}
    )

print("\nState columns after cleaning:")
print([
    col for col in merged_data.columns
    if "state" in col.lower()
])
# 7. CREATE ML DATA

ml_data = merged_data.drop(
    columns=[
        "admission_id",
        "patient_id",
        "discharge_date",
        "los_days",
        "discharge_type",
        "readmitted_7d",
        "bill_id",
        "total_cost_inr",
        "govt_subsidy_inr",
        "out_of_pocket_inr",
        "cost_category"
    ],
    errors="ignore"
)

# 8. DATE FEATURES

ml_data["admit_year"] = ml_data["admit_date"].dt.year
ml_data["admit_month"] = ml_data["admit_date"].dt.month
ml_data["admit_dayofweek"] = (
    ml_data["admit_date"].dt.dayofweek
)

ml_data = ml_data.drop(
    columns=["admit_date"]
)

# 9. REMOVE REDUNDANT FEATURES

ml_data = ml_data.drop(
    columns=[
        "primary_diagnosis",
        "name",
        "hospital_state"
    ],
    errors="ignore"
)

if "teaching" in ml_data.columns:
    ml_data["teaching"] = (
        ml_data["teaching"].astype(int)
    )

# 10. SEPARATE FEATURES AND TARGET

X = ml_data.drop(
    columns=["readmitted_30d"]
)

y = ml_data["readmitted_30d"]

# 11. USE SAME TEST SPLIT AS MODEL TRAINING

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 12. PREPROCESS TEST DATA

X_test_processed = preprocessor.transform(
    X_test
)

print(
    "Processed test data shape:",
    X_test_processed.shape
)

# 13. CREATE SHAP EXPLAINER

explainer = shap.TreeExplainer(
    best_xgb_model
)

X_shap = X_test_processed[:2000]

shap_values = explainer.shap_values(
    X_shap
)

print("SHAP values calculated successfully.")

# 14. FEATURE NAMES

feature_names = (
    preprocessor
    .get_feature_names_out()
)

# 15. SHAP SUMMARY PLOT

shap.summary_plot(
    shap_values,
    X_shap,
    feature_names=feature_names
)

# 16. SHAP BAR PLOT

shap.summary_plot(
    shap_values,
    X_shap,
    feature_names=feature_names,
    plot_type="bar"
)

# 17. SAVE SHAP IMPORTANCE

mean_abs_shap = (
    abs(shap_values)
    .mean(axis=0)
)

shap_importance = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": mean_abs_shap
})

shap_importance = (
    shap_importance
    .sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )
)

os.makedirs(
    "results",
    exist_ok=True
)

shap_importance.to_csv(
    "results/shap_feature_importance.csv",
    index=False
)

print("\nTop 20 SHAP Features:")
print(
    shap_importance.head(20)
)

print(
    "\nSHAP results saved to:"
)

print(
    "results/shap_feature_importance.csv"
)