from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def train_classifiers(X_train, y_train):
    models = {
        "LogisticRegression": (LogisticRegression(max_iter=1000), {'C': [0.1, 1]}),
        "DecisionTree": (DecisionTreeClassifier(), {'max_depth': [5, 10]}),
        "RandomForest": (RandomForestClassifier(), {'n_estimators': [100]}),
        "SVM": (SVC(probability=True), {'C': [1]}),
        "KNN": (KNeighborsClassifier(), {'n_neighbors': [5]})
    }

    trained = {}

    for name, (model, params) in models.items():
        grid = GridSearchCV(model, params, cv=3, scoring='f1')
        grid.fit(X_train, y_train)
        trained[name] = grid.best_estimator_

    return trained