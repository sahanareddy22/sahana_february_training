from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


def train_regressors(X_train, y_train):
    models = {
        "Linear": LinearRegression(),
        "Ridge": (Ridge(), {'alpha': [1]}),
        "Lasso": (Lasso(), {'alpha': [0.1]}),
        "ElasticNet": (ElasticNet(), {'alpha': [0.1]}),
        "DecisionTree": (DecisionTreeRegressor(), {'max_depth': [5]}),
        "RandomForest": (RandomForestRegressor(), {'n_estimators': [100]}),
        "SVR": (SVR(), {'C': [1]})
    }

    trained = {}

    for name, model in models.items():
        if isinstance(model, tuple):
            grid = GridSearchCV(model[0], model[1], cv=3)
            grid.fit(X_train, y_train)
            trained[name] = grid.best_estimator_
        else:
            model.fit(X_train, y_train)
            trained[name] = model

    return trained