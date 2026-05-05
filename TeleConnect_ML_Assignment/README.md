# 📊 TeleConnect ML Assignment

**Customer Churn Prediction & Revenue Forecasting using Supervised Learning**

---

## 🔹 Project Description

This project builds an end-to-end Machine Learning pipeline to solve two business problems for a telecom company:

1. **Customer Churn Prediction (Classification)**
2. **Monthly Revenue Forecasting (Regression)**

The solution includes data preprocessing, feature engineering, model training, evaluation, and business insights generation.

---

## 🔹 Dataset

* Source: Telco Customer Churn Dataset (Kaggle)
* Records: 7,043 customers
* Features: 21 columns (categorical + numerical)

### Key Targets:

* `Churn` → Classification target
* `MonthlyCharges` → Regression target

---

## 🔹 Project Structure

```
teleconnect-ml-assignment/
│
├── data/
│   └── raw/
│       └── telco.csv
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── classifiers.py
│   ├── regressors.py
│   ├── evaluation.py
│   ├── interpretation.py
│   └── utils.py
│
├── models/
│   ├── best_classifier.pkl
│   └── best_regressor.pkl
│
├── reports/
│   ├── classification_results.csv
│   ├── regression_results.csv
│   └── figures/
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 🔹 Installation & Setup

```bash
git clone https://github.com/your-username/teleconnect-ml-assignment.git
cd teleconnect-ml-assignment

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

## 🔹 How to Run

```bash
python main.py
```

---

## 🔹 Results Summary

### ✅ Best Classification Model

* **Random Forest Classifier**
* High ROC-AUC score
* Strong performance on imbalanced data

### ✅ Best Regression Model

* **Random Forest Regressor**
* Highest R² score
* Captures non-linear relationships effectively

---

## 🔹 Key Insights

* Customers with **month-to-month contracts** churn more
* **High monthly charges** increase churn probability
* Customers with **low tenure** are more likely to leave
* Fiber optic users show higher churn

---

## 🔹 Outputs Generated

### 📊 Reports

* classification_results.csv
* regression_results.csv

### 📈 Visualizations

* Confusion Matrix
* ROC Curve
* Feature Importance
* Actual vs Predicted
* Residual Distribution
* SHAP Summary Plot

---

## 🔹 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* SHAP

---

## 🔹 Business Recommendations

* Offer discounts for long-term contracts
* Target high-risk customers using model predictions
* Improve pricing strategy for fiber plans
* Focus retention efforts on new customers

---

## 🔹 Reproducibility

* `random_state=42` used throughout
* Modular pipeline for reuse
* No hardcoded paths

---

## 🔹 Author

Vanna Sahana

---
