import pandas as pd
import os


# Load data

admissions = pd.read_csv("data/archive/admissions.csv")
patients = pd.read_csv("data/archive/patients.csv")
diagnoses = pd.read_csv("data/archive/diagnoses.csv")
billing = pd.read_csv("data/archive/billing.csv")
hospitals = pd.read_csv("data/archive/hospitals.csv")


# Prepare patient data

patients["insurance_type"] = (
    patients["insurance_type"]
    .fillna("Unknown")
)


# Prepare admission dates

admissions["admit_date"] = pd.to_datetime(
    admissions["admit_date"],
    errors="coerce"
)

admissions["discharge_date"] = pd.to_datetime(
    admissions["discharge_date"],
    errors="coerce"
)


# Aggregate diagnosis information

num_diagnoses = (
    diagnoses
    .groupby("admission_id")
    .size()
    .reset_index(name="num_diagnoses")
)


primary_diagnosis = (
    diagnoses[
        diagnoses["diag_rank"] == 1
    ][
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


# Diagnosis category indicators

diagnosis_categories = pd.crosstab(
    diagnoses["admission_id"],
    diagnoses["diag_category"]
)

diagnosis_categories = (
    diagnosis_categories
    .astype(bool)
    .astype(int)
    .add_prefix("has_")
    .reset_index()
)


# Merge diagnosis information

merged_data = admissions.merge(
    num_diagnoses,
    on="admission_id",
    how="left"
)

merged_data = merged_data.merge(
    primary_diagnosis,
    on="admission_id",
    how="left"
)

merged_data = merged_data.merge(
    diagnosis_categories,
    on="admission_id",
    how="left"
)


# Merge patient information

merged_data = merged_data.merge(
    patients,
    on="patient_id",
    how="left"
)


# Merge billing information

merged_data = merged_data.merge(
    billing,
    on="admission_id",
    how="left"
)


# Merge hospital information

merged_data = merged_data.merge(
    hospitals,
    on="hospital_id",
    how="left",
    suffixes=("", "_hospital")
)


# Rename state columns

if "state" in merged_data.columns:

    merged_data = merged_data.rename(
        columns={
            "state": "patient_state"
        }
    )

if "state_hospital" in merged_data.columns:

    merged_data = merged_data.rename(
        columns={
            "state_hospital": "hospital_state"
        }
    )


# Create useful date features

merged_data["admit_year"] = (
    merged_data["admit_date"].dt.year
)

merged_data["admit_month"] = (
    merged_data["admit_date"].dt.month
)

merged_data["admit_dayofweek"] = (
    merged_data["admit_date"].dt.dayofweek
)


# Create age groups

merged_data["age_group"] = pd.cut(
    merged_data["age"],
    bins=[0, 18, 40, 60, 80, 120],
    labels=[
        "0-18",
        "19-40",
        "41-60",
        "61-80",
        "81+"
    ],
    include_lowest=True
)


# Create readmission label

merged_data["readmission_status"] = (
    merged_data["readmitted_30d"]
    .map({
        0: "Not Readmitted",
        1: "Readmitted"
    })
)


# Create Power BI dataset

powerbi_data = merged_data[
    [
        "admission_id",
        "patient_id",
        "admit_date",
        "discharge_date",
        "los_days",
        "admit_type",
        "ward_type",
        "hospital_id",
        "discharge_type",
        "num_procedures",
        "charlson_index",
        "hba1c",
        "creatinine",
        "haemoglobin",
        "systolic_bp",
        "readmitted_30d",
        "readmission_status",
        "num_diagnoses",
        "primary_icd10",
        "primary_diagnosis",
        "primary_diag_category",
        "age",
        "age_group",
        "gender",
        "patient_state",
        "bpl_card",
        "insurance_type",
        "comorbidity_count",
        "prev_admissions",
        "total_cost_inr",
        "govt_subsidy_inr",
        "out_of_pocket_inr",
        "cost_category",
        "hospital_state",
        "tier",
        "beds",
        "teaching",
        "admit_year",
        "admit_month",
        "admit_dayofweek"
    ]
].copy()


# Fill missing values

powerbi_data = powerbi_data.fillna({
    "primary_diagnosis": "Unknown",
    "primary_diag_category": "Unknown",
    "insurance_type": "Unknown",
    "hospital_state": "Unknown"
})


# Create results folder

os.makedirs(
    "results",
    exist_ok=True
)


# Save dataset

output_path = (
    "results/medinsight_powerbi.csv"
)

powerbi_data.to_csv(
    output_path,
    index=False
)


print(
    "\nPower BI dataset created successfully!"
)

print(
    "Shape:",
    powerbi_data.shape
)

print(
    "Saved to:",
    output_path
)

print(
    "\nReadmission distribution:"
)

print(
    powerbi_data[
        "readmission_status"
    ].value_counts()
)