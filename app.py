import joblib
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt


# Page configuration

st.set_page_config(
    page_title="MediInsight",
    page_icon="🏥",
    layout="wide"
)


# Load model and preprocessor

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/tuned_xgboost.pkl"
    )

    preprocessor = joblib.load(
        "models/preprocessor.pkl"
    )

    return model, preprocessor


model, preprocessor = load_model()


# SHAP explainer

explainer = shap.TreeExplainer(model)


# Page title

st.title("🏥 MediInsight")
st.subheader(
    "Machine Learning-Based Patient Readmission Risk Prediction"
)

st.write(
    "MediInsight is a research prototype that predicts "
    "30-day hospital readmission risk using machine learning "
    "and provides explainable predictions."
)

st.warning(
    "Research prototype only. This prediction is not a "
    "substitute for clinical judgment."
)


# Sidebar

st.sidebar.header("Patient Information")


admit_type = st.sidebar.selectbox(
    "Admission Type",
    ["Emergency", "Elective", "Urgent"]
)

ward_type = st.sidebar.selectbox(
    "Ward Type",
    ["General", "ICU", "HDU", "Private"]
)

hospital_id = st.sidebar.text_input(
    "Hospital ID",
    "H001"
)

age = st.sidebar.number_input(
    "Age",
    min_value=0,
    max_value=120,
    value=65
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)

patient_state = st.sidebar.text_input(
    "Patient State",
    "Rajasthan"
)

insurance_type = st.sidebar.selectbox(
    "Insurance Type",
    ["Private", "Government", "Ayushman", "Unknown"]
)

bpl_card = st.sidebar.selectbox(
    "BPL Card",
    [0, 1]
)

prev_admissions = st.sidebar.number_input(
    "Previous Admissions",
    min_value=0,
    max_value=50,
    value=2
)

comorbidity_count = st.sidebar.number_input(
    "Comorbidity Count",
    min_value=0,
    max_value=20,
    value=3
)

charlson_index = st.sidebar.number_input(
    "Charlson Index",
    min_value=0,
    max_value=20,
    value=3
)

num_diagnoses = st.sidebar.number_input(
    "Number of Diagnoses",
    min_value=0,
    max_value=30,
    value=4
)

num_procedures = st.sidebar.number_input(
    "Number of Procedures",
    min_value=0,
    max_value=30,
    value=2
)

hba1c = st.sidebar.number_input(
    "HbA1c",
    min_value=0.0,
    max_value=20.0,
    value=7.5
)

creatinine = st.sidebar.number_input(
    "Creatinine",
    min_value=0.0,
    max_value=20.0,
    value=1.4
)

haemoglobin = st.sidebar.number_input(
    "Haemoglobin",
    min_value=0.0,
    max_value=25.0,
    value=10.5
)

systolic_bp = st.sidebar.number_input(
    "Systolic BP",
    min_value=50,
    max_value=300,
    value=145
)

primary_icd10 = st.sidebar.text_input(
    "Primary ICD-10 Code",
    "I10"
)

primary_diag_category = st.sidebar.selectbox(
    "Primary Diagnosis Category",
    [
        "Cardiovascular",
        "Endocrine",
        "Gastrointestinal",
        "Genitourinary",
        "Infectious",
        "Injury",
        "Neoplasm",
        "Neurological",
        "Obstetric",
        "Perinatal",
        "Respiratory"
    ]
)

tier = st.sidebar.selectbox(
    "Hospital Tier",
    ["1", "2", "3"]
)

beds = st.sidebar.number_input(
    "Hospital Beds",
    min_value=1,
    max_value=5000,
    value=300
)

teaching = st.sidebar.selectbox(
    "Teaching Hospital",
    [0, 1]
)


# Diagnosis category indicators

diagnosis_categories = [
    "cardiovascular",
    "endocrine",
    "gastrointestinal",
    "genitourinary",
    "infectious",
    "injury",
    "neoplasm",
    "neurological",
    "obstetric",
    "perinatal",
    "respiratory"
]

diagnosis_flags = {}

for category in diagnosis_categories:

    diagnosis_flags[
        f"has_{category}"
    ] = int(
        primary_diag_category.lower()
        == category
    )


# Date features

admit_year = 2026
admit_month = 9
admit_dayofweek = 4


# Create patient dataframe

patient = pd.DataFrame([{

    "admit_type": admit_type,
    "ward_type": ward_type,
    "hospital_id": hospital_id,
    "num_procedures": num_procedures,
    "charlson_index": charlson_index,
    "hba1c": hba1c,
    "creatinine": creatinine,
    "haemoglobin": haemoglobin,
    "systolic_bp": systolic_bp,
    "num_diagnoses": num_diagnoses,

    "primary_icd10": primary_icd10,
    "primary_diag_category": primary_diag_category,

    **diagnosis_flags,

    "age": age,
    "gender": gender,
    "patient_state": patient_state,
    "bpl_card": bpl_card,
    "insurance_type": insurance_type,
    "comorbidity_count": comorbidity_count,
    "prev_admissions": prev_admissions,

    "tier": tier,
    "beds": beds,
    "teaching": teaching,

    "admit_year": admit_year,
    "admit_month": admit_month,
    "admit_dayofweek": admit_dayofweek
}])


# Prediction button

if st.button(
    "Predict Readmission Risk",
    type="primary"
):

    patient_processed = (
        preprocessor.transform(patient)
    )

    probability = model.predict_proba(
        patient_processed
    )[0][1]

    threshold = 0.60

    prediction = int(
        probability >= threshold
    )


    # Risk category

    if probability >= 0.75:

        risk_category = "High Risk"

    elif probability >= 0.60:

        risk_category = "Moderate Risk"

    else:

        risk_category = "Low Risk"


    # Display results

    st.header("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Readmission Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Risk Category",
            risk_category
        )

    with col3:

        result = (
            "Readmitted within 30 days"
            if prediction
            else
            "Not readmitted within 30 days"
        )

        st.metric(
            "Model Prediction",
            result
        )


    # SHAP explanation

    st.header(
        "Why did the model make this prediction?"
    )

    shap_values = explainer.shap_values(
        patient_processed
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    shap_explanation = pd.DataFrame({

        "Feature": feature_names,

        "SHAP_Value": shap_values[0]

    })

    shap_explanation[
        "Absolute_SHAP"
    ] = (
        shap_explanation[
            "SHAP_Value"
        ].abs()
    )

    shap_explanation = (
        shap_explanation
        .sort_values(
            "Absolute_SHAP",
            ascending=False
        )
    )


    # Top 10 factors

    st.subheader(
        "Top Factors Influencing This Prediction"
    )

    top_features = (
        shap_explanation
        .head(10)
        [["Feature", "SHAP_Value"]]
        .copy()
    )

    top_features["Effect"] = (
        top_features["SHAP_Value"]
        .apply(
            lambda x:
            "Increases Risk"
            if x > 0
            else
            "Decreases Risk"
        )
    )

    st.dataframe(
        top_features,
        use_container_width=True,
        hide_index=True
    )


    # SHAP waterfall plot

    st.subheader(
        "Individual Patient SHAP Explanation"
    )

    patient_data = (
        patient_processed.toarray()[0]
        if hasattr(
            patient_processed,
            "toarray"
        )
        else patient_processed[0]
    )

    shap_explanation_object = (
        shap.Explanation(

            values=shap_values[0],

            base_values=
            explainer.expected_value,

            data=patient_data,

            feature_names=feature_names
        )
    )

    fig = plt.figure()

    shap.waterfall_plot(
        shap_explanation_object,
        show=False
    )

    st.pyplot(
        fig,
        clear_figure=True
    )