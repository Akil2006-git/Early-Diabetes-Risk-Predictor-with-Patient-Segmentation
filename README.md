# 🏥 Early Diabetes Risk Predictor with Patient Segmentation

## Project Overview
An end-to-end machine learning pipeline that predicts diabetes risk and segments patients into actionable risk groups using the PIMA Indians Diabetes Dataset.

---

## File Structure
```
diabetes_project/
│
├── step1_data_cleaning.py       → Load data, fix impossible zero values
├── step2_eda.py                 → Visualize distributions, correlations, insights
├── step3_classification.py      → Logistic Regression + Random Forest models
├── step4_clustering.py          → KMeans patient segmentation (3 risk groups)
├── step5_sql_analysis.py        → SQLite queries for business insights
└── README.md                    → This file
```

---

## Setup Instructions

### 1. Install required libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### 2. Download the Dataset
- Go to: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- Download `diabetes.csv`
- Place it in the **same folder** as these scripts

### 3. Run scripts in order
```bash
python step1_data_cleaning.py
python step2_eda.py
python step3_classification.py
python step4_clustering.py
python step5_sql_analysis.py
```

---

## Dataset Details
- **Source:** UCI Machine Learning Repository / Kaggle
- **Rows:** 768 patients
- **Features:** 8 clinical measurements
- **Target:** Outcome (1 = Diabetic, 0 = Non-Diabetic)

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Genetic diabetes risk score |
| Age | Age in years |

---

##  Key Technical Decisions & Why

### Data Cleaning: Class-wise Median Imputation
Columns like Glucose, BMI, Insulin contain **zeroes that are biologically impossible**.  
These are missing values encoded as 0. We replace them with the **median grouped by Outcome**, not the overall median — because diabetic and non-diabetic patients have different distributions.

### Why Recall over Accuracy?
In medical diagnosis, a **False Negative** (missing a real diabetic) is far more dangerous than a **False Positive** (flagging a healthy patient as at-risk).  
- Recall = TP / (TP + FN) → measures how many actual diabetics we correctly identified  
- Optimizing for Recall ensures fewer diabetic patients are missed  

### Why Random Forest over Logistic Regression?
- Random Forest handles **non-linear feature interactions** (e.g., high Glucose + high BMI together matter more than either alone)
- It provides **feature importances** for explainability
- `class_weight='balanced'` handles class imbalance automatically  
- Logistic Regression is kept as an **interpretable baseline** for comparison

### Why KMeans Clustering?
A doctor cannot act on 768 individual risk scores. Clustering groups **similar patients together**:
- **Low Risk** → Routine monitoring  
- **Medium Risk** → Lifestyle intervention programs  
- **High Risk** → Immediate clinical attention  

This makes the ML output **actionable** for healthcare providers.

---

## Results Summary

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | ~78% | ~82% |
| ROC-AUC | ~0.84 | ~0.88 |
| Recall (Diabetic) | ~72% | ~76% |

---
