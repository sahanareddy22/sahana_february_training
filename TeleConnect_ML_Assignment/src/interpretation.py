import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_confusion_matrix(model, X_test, y_test, path):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.savefig(path)
    plt.close()


def plot_roc(models, X_test, y_test, path):
    plt.figure()

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} ({score:.2f})")

    plt.legend()
    plt.savefig(path)
    plt.close()


def plot_regression(model, X_test, y_test, path1, path2):
    y_pred = model.predict(X_test)

    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.savefig(path1)
    plt.close()

    residuals = y_test - y_pred
    plt.figure()
    sns.histplot(residuals, kde=True)
    plt.savefig(path2)
    plt.close()


def plot_feature_importance(model, feature_names, path):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[-10:]

        plt.figure()
        plt.barh(range(len(idx)), importances[idx])
        plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
        plt.savefig(path)
        plt.close()


def shap_analysis(model, X_sample, path):
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig(path)
    plt.close()