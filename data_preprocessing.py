import pandas as pd
import joblib
import os

# Load datasets
admissions = pd.read_csv("data/archive/admissions.csv")
billing = pd.read_csv("data/archive/billing.csv")
diagnoses = pd.read_csv("data/archive/diagnoses.csv")
patients = pd.read_csv("data/archive/patients.csv")
hospitals = pd.read_csv("data/archive/hospitals.csv")

# Check that the datasets loaded correctly
print("Admissions:", admissions.shape)
print("Billing:", billing.shape)
print("Diagnoses:", diagnoses.shape)
print("Patients:", patients.shape)
print("Hospitals:", hospitals.shape)


# 2. Clean and prepare the data


# Fill missing insurance values
patients["insurance_type"] = patients["insurance_type"].fillna("Unknown")

# Convert date columns to datetime
admissions["admit_date"] = pd.to_datetime(admissions["admit_date"])
admissions["discharge_date"] = pd.to_datetime(admissions["discharge_date"])

# Check missing values
print("\nMissing values in patients:")
print(patients.isnull().sum())

print("\nMissing values in admissions:")
print(admissions.isnull().sum())

# ==============================
# 3. Process diagnosis data
# ==============================

# Count how many diagnoses each admission has
diagnosis_count = (
    diagnoses.groupby("admission_id")
    .size()
    .reset_index(name="num_diagnoses")
)


# Get the primary diagnosis for each admission
primary_diagnosis = diagnoses[diagnoses["diag_rank"] == 1][
    ["admission_id", "icd10_code", "diag_desc", "diag_category"]
].copy()

primary_diagnosis = primary_diagnosis.rename(columns={
    "icd10_code": "primary_icd10",
    "diag_desc": "primary_diagnosis",
    "diag_category": "primary_diag_category"
})


# Create disease-category indicators
diagnosis_categories = pd.crosstab(
    diagnoses["admission_id"],
    diagnoses["diag_category"]
)

diagnosis_categories = (diagnosis_categories > 0).astype(int)

diagnosis_categories.columns = [
    "has_" + col.lower().replace(" ", "_")
    for col in diagnosis_categories.columns
]

diagnosis_categories = diagnosis_categories.reset_index()


# Check the results
print("\nDiagnosis count:", diagnosis_count.shape)
print("Primary diagnosis:", primary_diagnosis.shape)
print("Diagnosis categories:", diagnosis_categories.shape)

# ==============================
# 4. Merge diagnosis information
# ==============================

admissions = admissions.merge(
    diagnosis_count,
    on="admission_id",
    how="left",
    validate="one_to_one"
)

admissions = admissions.merge(
    primary_diagnosis,
    on="admission_id",
    how="left",
    validate="one_to_one"
)

admissions = admissions.merge(
    diagnosis_categories,
    on="admission_id",
    how="left",
    validate="one_to_one"
)


# Check the result
print("\nAdmissions after diagnosis merge:")
print(admissions.shape)
print(admissions.head())



# 5. Merge patient information


admissions = admissions.merge(
    patients,
    on="patient_id",
    how="left",
    validate="many_to_one"
)

print("\nAdmissions after patient merge:")
print(admissions.shape)

# 6. Check billing data

print("\nDuplicate admission IDs in billing:")
print(billing["admission_id"].duplicated().sum())

# ==============================
# 7. Merge billing information
# ==============================

admissions = admissions.merge(
    billing,
    on="admission_id",
    how="left",
    validate="one_to_one"
)

print("\nAdmissions after billing merge:")
print(admissions.shape)

# 8. Check hospital IDs

print("\nDuplicate hospital IDs:")
print(hospitals["hospital_id"].duplicated().sum())

# 9. Merge hospital information

admissions = admissions.merge(
    hospitals,
    on="hospital_id",
    how="left",
    validate="many_to_one"
)

print("\nAdmissions after hospital merge:")
print(admissions.shape)

# 10. Final dataset checks

print("\nFinal dataset shape:")
print(admissions.shape)

print("\nDuplicate admission IDs:")
print(admissions["admission_id"].duplicated().sum())

print("\nMissing values:")
print(admissions.isnull().sum())

print("\nTarget distribution:")
print(admissions["readmitted_30d"].value_counts())

# 11. Display all columns

print("\nFinal dataset columns:")
for i, column in enumerate(admissions.columns, start=1):
    print(i, column)
    
# 12. Check numerical features

numeric_columns = [
    "age",
    "los_days",
    "num_procedures",
    "charlson_index",
    "hba1c",
    "creatinine",
    "haemoglobin",
    "systolic_bp",
    "num_diagnoses",
    "comorbidity_count",
    "prev_admissions",
    "total_cost_inr",
    "govt_subsidy_inr",
    "out_of_pocket_inr",
    "beds"
]

print("\nNumerical feature summary:")
print(admissions[numeric_columns].describe().T)

# 13. Investigate age = 0

print("\nAdmissions with age = 0:")
print(
    admissions[admissions["age"] == 0][
        [
            "age",
            "gender",
            "primary_diagnosis",
            "primary_diag_category",
            "readmitted_30d"
        ]
    ].head(20)
)

print("\nNumber of age = 0 admissions:")
print((admissions["age"] == 0).sum())

print("\nAge = 0 diagnosis categories:")
print(
    admissions.loc[
        admissions["age"] == 0,
        "primary_diag_category"
    ].value_counts()
)

# 14. Check extreme clinical values

print("\nExtreme clinical values:")

print("\nCreatinine >= 5:")
print((admissions["creatinine"] >= 5).sum())

print("\nHbA1c >= 10:")
print((admissions["hba1c"] >= 10).sum())

print("\nSystolic BP >= 200:")
print((admissions["systolic_bp"] >= 200).sum())

print("\nHaemoglobin <= 6:")
print((admissions["haemoglobin"] <= 6).sum())

# ==============================
# 15. Prepare ML dataset
# ==============================

# Rename state columns for clarity
admissions = admissions.rename(columns={
    "state_x": "patient_state",
    "state_y": "hospital_state"
})

# Create a separate dataset for machine learning
ml_data = admissions.drop(columns=[
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
]).copy()

print("\nML dataset shape:")
print(ml_data.shape)

print("\nML dataset columns:")
for i, column in enumerate(ml_data.columns, start=1):
    print(i, column)

# 16. Create date features

# Extract useful information from admission date
ml_data["admit_year"] = ml_data["admit_date"].dt.year
ml_data["admit_month"] = ml_data["admit_date"].dt.month
ml_data["admit_dayofweek"] = ml_data["admit_date"].dt.dayofweek

# Remove the original date column
ml_data = ml_data.drop(columns=["admit_date"])

print("\nML dataset after date feature engineering:")
print(ml_data.shape)

print("\nNew date features:")
print(
    ml_data[
        ["admit_year", "admit_month", "admit_dayofweek"]
    ].head()
)

# 17. Check categorical features

categorical_columns = ml_data.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nCategorical columns:")
for column in categorical_columns:
    print(column)

print("\nNumber of categorical columns:")
print(len(categorical_columns))

# 18. Check categorical cardinality

print("\nUnique values in categorical columns:")

for column in categorical_columns:
    print(
        f"{column}: {ml_data[column].nunique()} unique values"
    )
    
# 19. Inspect categorical values

columns_to_check = [
    "admit_type",
    "ward_type",
    "gender",
    "insurance_type",
    "tier",
    "teaching"
]

print("\nCategorical value counts:")

for column in columns_to_check:
    print(f"\n--- {column} ---")
    print(ml_data[column].value_counts(dropna=False))
    
# 20. Remove redundant features

ml_data = ml_data.drop(columns=[
    "primary_diagnosis",
    "name",
    "hospital_state"
])

# Convert teaching from True/False to 1/0
ml_data["teaching"] = ml_data["teaching"].astype(int)

print("\nML dataset after removing redundant features:")
print(ml_data.shape)

print("\nRemaining columns:")
for i, column in enumerate(ml_data.columns, start=1):
    print(i, column)

# 21. Separate features and target

X = ml_data.drop(columns=["readmitted_30d"])
y = ml_data["readmitted_30d"]

print("\nFeatures shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nTarget distribution:")
print(y.value_counts())

# 22. Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
X_train_model, X_val, y_train_model, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)
print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# 23. Identify feature types

categorical_features = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

print("\nCategorical features:")
print(categorical_features)

print("\nNumber of categorical features:")
print(len(categorical_features))

print("\nNumerical features:")
print(numerical_features)

print("\nNumber of numerical features:")
print(len(numerical_features))

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)

print("\nPreprocessing pipeline created successfully.")


print("\nTarget distribution:")
print(y_train.value_counts())

print("\nTarget distribution (%):")
print(y_train.value_counts(normalize=True) * 100)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

print("\nLogistic Regression pipeline created successfully.")

print("\nTraining Logistic Regression...")

logistic_model.fit(X_train, y_train)

print("Logistic Regression training completed successfully.")

logistic_model.fit(X_train, y_train)

print("Logistic Regression training completed successfully.")

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# Make predictions
y_pred = logistic_model.predict(X_test)

# Get probability of readmission
y_prob = logistic_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))

from sklearn.ensemble import RandomForestClassifier

print("\nTraining Random Forest...")

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

random_forest_model.fit(X_train, y_train)

print("Random Forest training completed successfully.")

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Make predictions
y_pred_rf = random_forest_model.predict(X_test)

# Get probability of readmission
y_prob_rf = random_forest_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nRandom Forest ROC-AUC Score:")
print(roc_auc_score(y_test, y_prob_rf))

from xgboost import XGBClassifier

# Calculate class imbalance ratio
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nXGBoost scale_pos_weight:", scale_pos_weight)

print("\nTraining XGBoost...")

xgb_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

xgb_model.fit(X_train, y_train)

print("XGBoost training completed successfully.")

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Get probability of readmission
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

print("\nXGBoost Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

print("\nXGBoost ROC-AUC Score:")
print(roc_auc_score(y_test, y_prob_xgb))

from sklearn.model_selection import RandomizedSearchCV

print("\nStarting XGBoost Hyperparameter Tuning...")

xgb_tuning = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

param_grid = {
    "n_estimators": [100, 200, 300, 400],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5, 6, 8],
    "min_child_weight": [1, 3, 5, 7],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=xgb_tuning,
    param_distributions=param_grid,
    n_iter=20,
    scoring="roc_auc",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

print("\nPreparing data for tuned XGBoost...")
X_train_processed = preprocessor.fit_transform(X_train_model)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

random_search.fit(X_train_processed, y_train_model)

print("\nBest XGBoost Parameters:")
print(random_search.best_params_)

print("\nBest Cross-Validation ROC-AUC:")
print(random_search.best_score_)

print("\nTraining Tuned XGBoost...")

best_xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=3,
    subsample=0.7,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

best_xgb_model.fit(X_train_processed, y_train_model)

print("Tuned XGBoost training completed successfully.")

# Predictions from tuned XGBoost
print("DEBUG: X_test_processed exists:", "X_test_processed" in globals())
y_pred_tuned = best_xgb_model.predict(X_test_processed)

# Probability of readmission
y_prob_tuned = best_xgb_model.predict_proba(X_test_processed)[:, 1]

print("\nTuned XGBoost Classification Report:")
print(classification_report(y_test, y_pred_tuned))

print("\nTuned XGBoost Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))

print("\nTuned XGBoost ROC-AUC Score:")
print(roc_auc_score(y_test, y_prob_tuned))

from sklearn.metrics import precision_score, recall_score, f1_score

print("\nXGBoost Threshold Analysis:")

# Validation probabilities for threshold selection
y_prob_val = best_xgb_model.predict_proba(X_val_processed)[:, 1]

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

for threshold in thresholds:
    y_pred_threshold = (y_prob_val >= threshold).astype(int)

    precision = precision_score(y_val, y_pred_threshold)
    recall = recall_score(y_val, y_pred_threshold)
    f1 = f1_score(y_val, y_pred_threshold)
    
    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.2f} | "
        f"Recall: {recall:.2f} | "
        f"F1: {f1:.2f}"
    )
    
# Final evaluation on untouched test set
y_prob_test = best_xgb_model.predict_proba(X_test_processed)[:, 1]

final_threshold = 0.60

y_pred_final = (y_prob_test >= final_threshold).astype(int)

print("\nFinal Test Results at Threshold 0.60:")

print("Precision:", precision_score(y_test, y_pred_final))
print("Recall:", recall_score(y_test, y_pred_final))
print("F1 Score:", f1_score(y_test, y_pred_final))

# Model Comparison
model_comparison = {
    "Logistic Regression": {
        "Accuracy": 0.69,
        "Precision": 0.23,
        "Recall": 0.68,
        "F1": 0.34,
        "ROC-AUC": 0.7521
    },
    "Random Forest": {
        "Accuracy": 0.87,
        "Precision": 0.43,
        "Recall": 0.20,
        "F1": 0.27,
        "ROC-AUC": 0.7408
    },
    "XGBoost": {
        "Accuracy": 0.72,
        "Precision": 0.24,
        "Recall": 0.64,
        "F1": 0.35,
        "ROC-AUC": 0.7482
    },
    "Tuned XGBoost": {
        "Accuracy": 0.69,
        "Precision": 0.23,
        "Recall": 0.68,
        "F1": 0.34,
        "ROC-AUC": 0.7522
    }
}

comparison_df = pd.DataFrame(model_comparison).T

print("\nModel Comparison:")
print(comparison_df)

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]
)
# FAIR MODEL COMPARISON
# Train all models on the same training data

# Logistic Regression
logistic_model.fit(X_train_model, y_train_model)

y_pred_lr = logistic_model.predict(X_test)
y_prob_lr = logistic_model.predict_proba(X_test)[:, 1]

print("\nLogistic Regression")
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))


# Random Forest
rf_pipeline.fit(X_train_model, y_train_model)

y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]
print("\nRandom Forest")
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))

# FINAL MODEL COMPARISON

model_comparison = {
    "Logistic Regression": {
        "Accuracy": accuracy_score(y_test, y_pred_lr),
        "Precision": precision_score(y_test, y_pred_lr),
        "Recall": recall_score(y_test, y_pred_lr),
        "F1": f1_score(y_test, y_pred_lr),
        "ROC-AUC": roc_auc_score(y_test, y_prob_lr)
    },

    "Random Forest": {
        "Accuracy": accuracy_score(y_test, y_pred_rf),
        "Precision": precision_score(y_test, y_pred_rf),
        "Recall": recall_score(y_test, y_pred_rf),
        "F1": f1_score(y_test, y_pred_rf),
        "ROC-AUC": roc_auc_score(y_test, y_prob_rf)
    }
}

comparison_df = pd.DataFrame(model_comparison).T

print("\nFinal Model Comparison:")
print(comparison_df.round(4))

# COMPLETE MODEL COMPARISON

model_comparison = {
    "Logistic Regression": {
        "Accuracy": accuracy_score(y_test, y_pred_lr),
        "Precision": precision_score(y_test, y_pred_lr),
        "Recall": recall_score(y_test, y_pred_lr),
        "F1": f1_score(y_test, y_pred_lr),
        "ROC-AUC": roc_auc_score(y_test, y_prob_lr)
    },

    "Random Forest": {
        "Accuracy": accuracy_score(y_test, y_pred_rf),
        "Precision": precision_score(y_test, y_pred_rf),
        "Recall": recall_score(y_test, y_pred_rf),
        "F1": f1_score(y_test, y_pred_rf),
        "ROC-AUC": roc_auc_score(y_test, y_prob_rf)
    },

    "XGBoost": {
        "Accuracy": 0.72,
        "Precision": 0.24,
        "Recall": 0.64,
        "F1": 0.35,
        "ROC-AUC": 0.7482
    },

    "Tuned XGBoost": {
        "Accuracy": 0.69,
        "Precision": 0.23,
        "Recall": 0.68,
        "F1": 0.34,
        "ROC-AUC": 0.7522
    }
}

comparison_df = pd.DataFrame(model_comparison).T

print("\nComplete Model Comparison:")
print(comparison_df.round(4))

# SAVE TRAINED MODELS AND PREPROCESSOR

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save preprocessing pipeline
joblib.dump(preprocessor, "models/preprocessor.pkl")

# Save Logistic Regression
joblib.dump(logistic_model, "models/logistic_regression.pkl")

# Save Random Forest
joblib.dump(rf_pipeline, "models/random_forest.pkl")

# Save tuned XGBoost
joblib.dump(best_xgb_model, "models/tuned_xgboost.pkl")

print("\nModels and preprocessor saved successfully!")

# CONFUSION MATRICES

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Logistic Regression
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_lr,
    display_labels=["Not Readmitted", "Readmitted"]
)
plt.title("Logistic Regression - Confusion Matrix")
plt.show()


# Random Forest
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_rf,
    display_labels=["Not Readmitted", "Readmitted"]
)
plt.title("Random Forest - Confusion Matrix")
plt.show()

# COMPLETE MODEL EVALUATION

import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
# 1. GENERATE / STORE XGBOOST PREDICTIONS
# Tuned XGBoost - default threshold 0.50
y_prob_xgb = best_xgb_model.predict_proba(X_test_processed)[:, 1]

y_pred_xgb = (y_prob_xgb >= 0.50).astype(int)

# Tuned XGBoost - final threshold selected using validation set
final_threshold = 0.60

y_pred_xgb_final = (
    y_prob_xgb >= final_threshold
).astype(int)

# 2. CONFUSION MATRICES

print("\n" + "=" * 60)
print("CONFUSION MATRICES")
print("=" * 60)

# Logistic Regression
print("\nLogistic Regression:")
print(confusion_matrix(y_test, y_pred_lr))

# Random Forest
print("\nRandom Forest:")
print(confusion_matrix(y_test, y_pred_rf))

# Tuned XGBoost - default threshold
print("\nTuned XGBoost - Threshold 0.50:")
print(confusion_matrix(y_test, y_pred_xgb))

# Tuned XGBoost - final threshold
print("\nTuned XGBoost - Threshold 0.60:")
print(confusion_matrix(y_test, y_pred_xgb_final))

# 3. CLASSIFICATION REPORTS

print("\n" + "=" * 60)
print("CLASSIFICATION REPORTS")
print("=" * 60)

print("\nLogistic Regression:")
print(classification_report(
    y_test,
    y_pred_lr,
    target_names=["Not Readmitted", "Readmitted"]
))

print("\nRandom Forest:")
print(classification_report(
    y_test,
    y_pred_rf,
    target_names=["Not Readmitted", "Readmitted"]
))

print("\nTuned XGBoost - Threshold 0.50:")
print(classification_report(
    y_test,
    y_pred_xgb,
    target_names=["Not Readmitted", "Readmitted"]
))

print("\nTuned XGBoost - Threshold 0.60:")
print(classification_report(
    y_test,
    y_pred_xgb_final,
    target_names=["Not Readmitted", "Readmitted"]
))

# 4. FINAL METRICS

evaluation_results = {
    "Logistic Regression": {
        "Accuracy": accuracy_score(y_test, y_pred_lr),
        "Precision": precision_score(y_test, y_pred_lr),
        "Recall": recall_score(y_test, y_pred_lr),
        "F1": f1_score(y_test, y_pred_lr),
        "ROC-AUC": roc_auc_score(y_test, y_prob_lr)
    },

    "Random Forest": {
        "Accuracy": accuracy_score(y_test, y_pred_rf),
        "Precision": precision_score(y_test, y_pred_rf),
        "Recall": recall_score(y_test, y_pred_rf),
        "F1": f1_score(y_test, y_pred_rf),
        "ROC-AUC": roc_auc_score(y_test, y_prob_rf)
    },

    "Tuned XGBoost (0.50)": {
        "Accuracy": accuracy_score(y_test, y_pred_xgb),
        "Precision": precision_score(y_test, y_pred_xgb),
        "Recall": recall_score(y_test, y_pred_xgb),
        "F1": f1_score(y_test, y_pred_xgb),
        "ROC-AUC": roc_auc_score(y_test, y_prob_xgb)
    },

    "Tuned XGBoost (0.60)": {
        "Accuracy": accuracy_score(y_test, y_pred_xgb_final),
        "Precision": precision_score(y_test, y_pred_xgb_final),
        "Recall": recall_score(y_test, y_pred_xgb_final),
        "F1": f1_score(y_test, y_pred_xgb_final),
        "ROC-AUC": roc_auc_score(y_test, y_prob_xgb)
    }
}

evaluation_df = pd.DataFrame(evaluation_results).T

print("\n" + "=" * 60)
print("FINAL EVALUATION RESULTS")
print("=" * 60)

print(evaluation_df.round(4))

# 5. CONFUSION MATRIX PLOTS

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_lr,
    display_labels=["Not Readmitted", "Readmitted"],
    ax=axes[0, 0]
)
axes[0, 0].set_title("Logistic Regression")

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_rf,
    display_labels=["Not Readmitted", "Readmitted"],
    ax=axes[0, 1]
)
axes[0, 1].set_title("Random Forest")

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_xgb,
    display_labels=["Not Readmitted", "Readmitted"],
    ax=axes[1, 0]
)
axes[1, 0].set_title("Tuned XGBoost - Threshold 0.50")

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_xgb_final,
    display_labels=["Not Readmitted", "Readmitted"],
    ax=axes[1, 1]
)
axes[1, 1].set_title("Tuned XGBoost - Threshold 0.60")

plt.tight_layout()
plt.show()

# 6. ROC CURVE

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr_lr,
    tpr_lr,
    label=f"Logistic Regression (AUC = {roc_auc_score(y_test, y_prob_lr):.3f})"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.3f})"
)

plt.plot(
    fpr_xgb,
    tpr_xgb,
    label=f"Tuned XGBoost (AUC = {roc_auc_score(y_test, y_prob_xgb):.3f})"
)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Model Comparison")
plt.legend()
plt.grid(True)
plt.show()

# 7. PRECISION-RECALL CURVE

precision_lr, recall_lr, _ = precision_recall_curve(
    y_test,
    y_prob_lr
)

precision_rf, recall_rf, _ = precision_recall_curve(
    y_test,
    y_prob_rf
)

precision_xgb, recall_xgb, _ = precision_recall_curve(
    y_test,
    y_prob_xgb
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall_lr,
    precision_lr,
    label=f"Logistic Regression (AP = {average_precision_score(y_test, y_prob_lr):.3f})"
)

plt.plot(
    recall_rf,
    precision_rf,
    label=f"Random Forest (AP = {average_precision_score(y_test, y_prob_rf):.3f})"
)

plt.plot(
    recall_xgb,
    precision_xgb,
    label=f"Tuned XGBoost (AP = {average_precision_score(y_test, y_prob_xgb):.3f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Model Comparison")
plt.legend()
plt.grid(True)
plt.show()

# 8. SAVE EVALUATION RESULTS

os.makedirs("results", exist_ok=True)

evaluation_df.to_csv(
    "results/model_evaluation.csv"
)

print("\nEvaluation results saved to:")
print("results/model_evaluation.csv")
