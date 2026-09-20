# MediInsight — Hospital Readmission Risk Prediction System

MediInsight is a research-based machine learning project designed to predict the risk of **30-day hospital readmission** using patient, admission, clinical, hospital, and diagnosis-related information.

The project combines **data preprocessing, exploratory analysis, machine learning, explainable AI, interactive prediction, and healthcare analytics** into one integrated system.

> **Disclaimer:** MediInsight is a research prototype and is not intended to replace clinical judgment, medical diagnosis, or treatment decisions.

---

## 📌 Project Overview

Hospital readmissions are an important healthcare challenge because they can indicate the need for better follow-up care, patient monitoring, and resource planning.

MediInsight investigates whether machine learning can be used to identify patients who may be at higher risk of being readmitted within 30 days after hospital discharge.

The project uses a multi-condition hospital dataset and develops an admission-level machine learning pipeline for readmission-risk prediction.

### Research Question

**Can machine learning effectively predict 30-day hospital readmission risk across multiple conditions, and what factors contribute most to these predictions?**

---

## 🔬 Research Gap

Existing research on hospital readmission prediction often focuses on specific diseases or patient groups.

The reference study used machine learning to predict readmission risk among diabetic inpatients and combined the prediction system with mixed-reality visualization.

MediInsight explores a broader approach by:

- Considering multiple diagnosis categories instead of focusing on a single disease.
- Building an admission-level readmission prediction framework.
- Using explainable AI to understand important prediction factors.
- Providing an accessible Streamlit interface for patient-level prediction.
- Providing a Power BI dashboard for healthcare analytics.

---

## 🎯 Objectives

The main objectives of MediInsight are:

- Prepare and integrate healthcare data from multiple sources.
- Perform data cleaning and preprocessing.
- Analyze patient, admission, diagnosis, and hospital characteristics.
- Engineer meaningful features for machine learning.
- Predict 30-day hospital readmission risk.
- Compare different machine learning models.
- Handle class imbalance during model training.
- Tune the XGBoost model using hyperparameter search.
- Evaluate model performance using multiple metrics.
- Explain model predictions using SHAP.
- Develop an interactive Streamlit prediction interface.
- Create a Power BI dashboard for healthcare analytics.

---

## 🔄 Project Workflow

Healthcare CSV Data
        ↓
Data Cleaning & Preprocessing
        ↓
Data Integration
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Train / Validation / Test Split
        ↓
Model Training
        ↓
Model Comparison
        ↓
XGBoost Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
SHAP Explainability
        ↓
Streamlit Prediction Application
        ↓
Power BI Healthcare Dashboard

## 📊 Dataset

MediInsight uses multiple healthcare datasets that are integrated to create a single admission-level dataset.

The project uses the following datasets:

- admissions.csv
- patients.csv
- diagnoses.csv
- billing.csv
- hospitals.csv

### Dataset Relationships

The datasets are connected using relational identifiers:

- patient_id connects patients with admissions.
- admission_id connects admissions with diagnoses and billing.
- `hospital_id` connects admissions with hospitals.

After preprocessing and integration, the final dataset contains **120,000 hospital admissions**.

The target variable is `readmitted_30d`, which represents whether a patient was readmitted within 30 days.

- `0` — Not readmitted within 30 days
- `1` — Readmitted within 30 days

The dataset contains **14,210 readmitted cases** and **105,790 non-readmitted cases**.

> The raw healthcare datasets are not included in this GitHub repository because some files exceed GitHub's web upload size limitations. The datasets are maintained separately for project development.

---

## 🧹 Data Preprocessing

The data preprocessing pipeline was developed using Python and Pandas.

The main preprocessing steps include:

- Loading multiple healthcare CSV files.
- Checking dataset shapes and column structures.
- Handling missing values.
- Converting date columns into proper datetime format.
- Checking duplicate records.
- Aggregating multiple diagnosis records for each admission.
- Calculating the number of diagnoses per admission.
- Identifying the primary diagnosis.
- Creating diagnosis-category indicators.
- Merging patient information with admission information.
- Merging billing information.
- Merging hospital information.
- Creating date-based features.
- Removing unnecessary and redundant columns.
- Preparing the final machine learning dataset.

Important features include:

- Age
- Gender
- Admission type
- Ward type
- Hospital
- Number of diagnoses
- Number of procedures
- Charlson index
- HbA1c
- Creatinine
- Haemoglobin
- Systolic blood pressure
- Comorbidity count
- Previous admissions
- Insurance type
- BPL card status
- Diagnosis category
- Hospital tier
- Number of hospital beds
- Admission year
- Admission month
- Admission day of week

---

## 🔎 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the structure and characteristics of the healthcare data.

The analysis included:

- Distribution of the target variable.
- Patient age distribution.
- Readmission distribution.
- Diagnosis category analysis.
- Admission type analysis.
- Ward type analysis.
- Insurance type analysis.
- Hospital-level readmission analysis.
- Clinical feature analysis.
- Missing-value analysis.
- Duplicate-value analysis.

The analysis helped identify patterns and relationships that were useful during feature engineering and model development.

---

## ⚙️ Feature Engineering

Several features were created or transformed to improve the machine learning pipeline.

### Diagnosis Features

Diagnosis records were aggregated at the admission level to create:

- Number of diagnoses.
- Primary ICD-10 code.
- Primary diagnosis.
- Primary diagnosis category.
- Diagnosis-category indicator variables.

### Date Features

The admission date was transformed into:

- Admission year.
- Admission month.
- Admission day of week.

### Categorical Features

Categorical variables were encoded using:

`OneHotEncoder`

### Numerical Features

Numerical variables were standardized using:

`StandardScaler`

The preprocessing pipeline was saved as:

`preprocessor.pkl`

---

## 🤖 Machine Learning

MediInsight uses supervised machine learning for binary classification of 30-day hospital readmission risk.

Three models were evaluated:

### Logistic Regression

Logistic Regression was used as a baseline linear classification model.

### Random Forest

Random Forest was used as a tree-based baseline model capable of capturing nonlinear relationships.

### XGBoost

XGBoost was selected as the primary model for the project because it is well suited for structured and tabular datasets.

The XGBoost model was further optimized using hyperparameter tuning.

---

## 📚 Model Training Strategy

The dataset was divided into training and testing sets using a stratified split.

The original training data was further divided into:

- Model training set — 76,800 records
- Validation set — 19,200 records
- Test set — 24,000 records

The test set was kept untouched during model development and threshold selection.

Stratified splitting was used to maintain the proportion of readmitted and non-readmitted cases.

---

## ⚖️ Class Imbalance

The target variable is imbalanced because the number of patients who were not readmitted is significantly higher than the number of patients who were readmitted.

To address this issue, class weighting / positive-class weighting was used during model training.

Because of this imbalance, model performance was not evaluated using accuracy alone.

The project also focuses on:

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

---

## 🔧 XGBoost Hyperparameter Tuning

XGBoost hyperparameters were optimized using `RandomizedSearchCV`.

The tuning process evaluated parameters including:

- Number of estimators.
- Learning rate.
- Maximum tree depth.
- Minimum child weight.
- Subsample ratio.
- Column sampling ratio.

The final tuned XGBoost configuration used:

- `n_estimators = 300`
- `learning_rate = 0.03`
- `max_depth = 3`
- `min_child_weight = 3`
- `subsample = 0.7`
- `colsample_bytree = 0.9`

The tuned model was saved as:

`tuned_xgboost.pkl`

---

## 📈 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

The baseline models produced the following results:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6932 | 0.2301 | 0.6780 | 0.3436 | 0.7510 |
| Random Forest | 0.8732 | 0.4222 | 0.1918 | 0.2637 | 0.7383 |

The tuned XGBoost model achieved a test ROC-AUC of approximately **0.7522** at the standard classification threshold of 0.50.

---

## 🎚️ Classification Threshold Analysis

Since the dataset is imbalanced, different classification thresholds were evaluated using the validation dataset.

The threshold controls when a predicted probability is converted into the final binary prediction.

After threshold analysis, a threshold of **0.60** was selected for the final Streamlit application.

At the 0.60 threshold, the tuned XGBoost model achieved approximately:

- Precision: **0.2919**
- Recall: **0.5257**
- F1-score: **0.3753**

This threshold represents a selected precision-recall trade-off based on validation analysis.

---

## 🧠 Explainable AI with SHAP

MediInsight uses **SHAP (SHapley Additive exPlanations)** to make machine learning predictions more interpretable.

SHAP helps identify how individual features contribute to the model's prediction.

### Global Explainability

Global SHAP analysis identifies features that have a strong overall influence on the model.

Important features observed during the analysis include:

- Number of diagnoses
- Previous admissions
- Charlson index
- Age
- Ward type
- Haemoglobin
- Insurance type
- BPL card status
- Admission type
- Comorbidity count

### Patient-Level Explainability

For an individual patient, SHAP identifies features that contribute positively or negatively to that particular prediction.

This allows the application to provide more information than simply displaying a risk score.

> SHAP feature contributions represent model behavior and should not be interpreted as clinical causation.

---

## 🖥️ Streamlit Application

MediInsight includes an interactive Streamlit application for patient-level prediction.

The application allows the user to enter patient and admission information and generate a readmission-risk prediction.

The application displays:

- Readmission probability.
- Risk category.
- Model prediction.
- Important contributing factors.
- SHAP-based explanation.
- SHAP waterfall visualization.

The application loads the saved preprocessing pipeline and tuned XGBoost model. The model is not retrained when making a prediction.

### Run the Application

Activate the project virtual environment:

    .\venv\Scripts\Activate.ps1

Run the Streamlit application:

    python -m streamlit run app.py

The application will open in the browser through the local Streamlit server.

---

## 📊 Power BI Dashboard

A Power BI dashboard was developed to provide a high-level view of hospital readmission analytics.

The dashboard includes:

- Total Admissions.
- 30-Day Readmissions.
- 30-Day Readmission Rate.
- Average Length of Stay.
- Readmissions by Diagnosis Category.
- Readmissions by Age Group.
- Readmissions by Admission Type.
- Readmissions by Hospital.
- Readmissions by Insurance Type.

Interactive dropdown filters are available for:

- Admission Type.
- Age Group.
- Primary Diagnosis Category.

The Power BI dashboard is designed for healthcare analytics and reporting, while the Streamlit application focuses on patient-level prediction.

The dashboard file is:

`powerbi_dashboard.pbix`

---

## 🛠️ Technologies Used

### Programming and Data Processing

- Python
- Pandas
- NumPy

### Database and Querying

- SQL
- MySQL

### Machine Learning

- Scikit-learn
- XGBoost

### Explainable AI

- SHAP

### Visualization and Analytics

- Matplotlib
- Power BI

### Application Development

- Streamlit

### Model Saving

- Joblib

### Version Control and Collaboration

- Git
- GitHub

---

## 📁 Project Structure

    MediInsight/
    │
    ├── app.py
    ├── data_preprocessing.py
    ├── patient_explanation.py
    ├── predict_patient.py
    ├── shap_analysis.py
    │
    ├── preprocessor.pkl
    ├── tuned_xgboost.pkl
    │
    ├── shap_feature_importance.csv
    ├── powerbi_dashboard.pbix
    ├── requirements.txt
    └── README.md

### File Description

`app.py`  
Runs the Streamlit application.

`data_preprocessing.py`  
Loads, cleans, integrates, and prepares the healthcare datasets.

`predict_patient.py`  
Loads the trained model and performs patient-level readmission prediction.

`patient_explanation.py`  
Generates SHAP-based explanations for individual patient predictions.

`shap_analysis.py`  
Performs global SHAP feature-importance analysis.

`preprocessor.pkl`  
Saved preprocessing pipeline used to transform input data before prediction.

`tuned_xgboost.pkl`  
Saved tuned XGBoost model.

`shap_feature_importance.csv`  
Contains the global SHAP feature-importance results.

`powerbi_dashboard.pbix`  
Power BI healthcare analytics dashboard.

`requirements.txt`  
Contains the Python libraries required for the project.

---

## 👥 Team Contributions

### Karanjot Buttar — Data Engineering, Analytics & Research

Responsibilities:

- Python and Pandas-based data processing.
- Data cleaning and integration.
- Feature engineering.
- SQL and MySQL.
- Exploratory data analysis.
- Power BI dashboard development.
- Literature review.
- Research gap identification.
- Research direction.
- Project documentation.
- Project presentation.

### Chander Prakash — Machine Learning & Explainability

Responsibilities:

- Machine learning model development.
- Logistic Regression baseline.
- Random Forest baseline.
- XGBoost model development.
- Hyperparameter tuning.
- Class imbalance handling.
- Model evaluation.
- Threshold analysis.
- SHAP explainability.
- Model interpretation.

### Divya — Application & Project Integration

Responsibilities:

- Streamlit application development.
- Patient prediction interface.
- Model integration.
- SHAP explanation integration.
- Risk visualization.
- Application testing.
- Final application integration.
- GitHub support.

### Shared Responsibilities

- Project coordination.
- Testing.
- Documentation.
- Final project review.
- Presentation.
- Project demonstration.

---

## 🚀 Installation and Setup

Clone or download the repository.

Create a Python virtual environment:

    python -m venv venv

Activate the virtual environment on Windows:

    .\venv\Scripts\Activate.ps1

Install the required dependencies:

    pip install -r requirements.txt

Run the Streamlit application:

    python -m streamlit run app.py

The raw datasets must be available locally if the complete preprocessing pipeline needs to be reproduced.

---

## 🔮 Future Scope

The project can be extended in several ways:

- External validation using independent hospital datasets.
- Evaluation on additional healthcare populations.
- Improved model calibration.
- More extensive hyperparameter optimization.
- Additional machine learning models.
- More detailed disease-wise performance analysis.
- Advanced SHAP-based explanations.
- Cloud deployment.
- Secure healthcare system integration.
- Continuous model monitoring.
- Model retraining using new data.
- Further research into multi-condition readmission prediction.

---

## ⚠️ Limitations

MediInsight currently has several limitations:

- It is a research prototype and has not been clinically validated.
- External validation has not been performed.
- The dataset may not represent every hospital or patient population.
- Model predictions may not generalize to different healthcare environments.
- The application-level risk categories are not clinically validated.
- The predicted probability should not be interpreted as a clinically calibrated probability.
- Real-world healthcare deployment would require additional privacy, security, regulatory, and clinical validation.

---

## 📚 Reference

The project was informed by research on machine learning-based patient readmission prediction.

**Martin Sanchez, Nick Tran, Vuthea Chheang (2026).**

*Towards Extended Reality Intelligence for Monitoring and Predicting Patient Readmission Risks.*

The reference work investigates machine learning-based 30-day readmission prediction for diabetic inpatients and uses XGBoost along with visualization techniques.

MediInsight builds on this research direction by exploring a broader multi-condition framework with explainability, patient-level prediction, and accessible analytics.

---

## 👨‍💻 Project Team

**Karanjot Buttar**  
**Chander Prakash**
**Divya**
B.Tech Information Technology  
Government Engineering College Bikaner

---

## ⚕️ Disclaimer

MediInsight has been developed for **academic and research purposes only**.

The predictions generated by this system are not medical diagnoses and should not be used as a substitute for professional medical advice, clinical judgment, or treatment decisions.
