import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_classification(models, X_test, y_test):
    results = []
    best_model = None
    best_score = 0

    for name, model in models.items():
        start = time.time()

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        end = time.time()

        roc = roc_auc_score(y_test, y_prob)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": roc,
            "Time": end - start
        })

        if roc > best_score:
            best_score = roc
            best_model = model

    return pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False), best_model


def evaluate_regression(models, X_test, y_test):
    results = []
    best_model = None
    best_score = -np.inf

    for name, model in models.items():
        start = time.time()

        y_pred = model.predict(X_test)

        end = time.time()

        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2,
            "Time": end - start
        })

        if r2 > best_score:
            best_score = r2
            best_model = model

    return pd.DataFrame(results).sort_values(by="R2", ascending=False), best_model