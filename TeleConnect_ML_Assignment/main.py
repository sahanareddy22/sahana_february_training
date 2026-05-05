import os
from sklearn.model_selection import train_test_split

from src.data_loader import load_data, clean_data
from src.preprocessing import feature_engineering, get_preprocessor
from src.classifiers import train_classifiers
from src.regressors import train_regressors
from src.evaluation import evaluate_classification, evaluate_regression
from src.utils import save_model
from src.interpretation import (
    plot_confusion_matrix,
    plot_roc,
    plot_regression,
    plot_feature_importance,
    shap_analysis
)

# ================= SETUP =================
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ================= LOAD DATA =================
df = load_data("data/raw/telco.csv")
df = clean_data(df)
df = feature_engineering(df)

# =====================================================
# 🔹 CLASSIFICATION
# =====================================================
X = df.drop(columns=['Churn'])
y = df['Churn']

# ✅ Preprocessor for classification
preprocessor_clf = get_preprocessor(X)
X = preprocessor_clf.fit_transform(X)

# Get feature names
feature_names = preprocessor_clf.get_feature_names_out()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train models
clf_models = train_classifiers(X_train, y_train)

# Evaluate
clf_results, best_clf = evaluate_classification(clf_models, X_test, y_test)

print("\nCLASSIFICATION RESULTS:\n", clf_results)

# Save results
clf_results.to_csv("reports/classification_results.csv", index=False)

# Save best model
save_model(best_clf, "models/best_classifier.pkl")

# Plots
plot_confusion_matrix(best_clf, X_test, y_test, "reports/figures/confusion.png")
plot_roc(clf_models, X_test, y_test, "reports/figures/roc.png")
plot_feature_importance(best_clf, feature_names, "reports/figures/feature_importance.png")

# SHAP (use sample for speed)
shap_analysis(best_clf, X_test[:100], "reports/figures/shap.png")


# =====================================================
# 🔹 REGRESSION  (FIXED PART)
# =====================================================
y_reg = df['MonthlyCharges']
X_reg = df.drop(columns=['MonthlyCharges'])

# ✅ NEW preprocessor (important fix)
preprocessor_reg = get_preprocessor(X_reg)
X_reg = preprocessor_reg.fit_transform(X_reg)

# Split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train models
reg_models = train_regressors(X_train_r, y_train_r)

# Evaluate
reg_results, best_reg = evaluate_regression(reg_models, X_test_r, y_test_r)

print("\nREGRESSION RESULTS:\n", reg_results)

# Save results
reg_results.to_csv("reports/regression_results.csv", index=False)

# Save best model
save_model(best_reg, "models/best_regressor.pkl")

# Plots
plot_regression(
    best_reg,
    X_test_r,
    y_test_r,
    "reports/figures/actual_vs_pred.png",
    "reports/figures/residuals.png"
)