
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv("../dataset/Housing.csv")

print("First 5 rows of dataset:\n")
print(df.head())

print("\nDataset Information:\n")
print(df.info())

print("\nMissing values:\n")
print(df.isnull().sum())
df = df.fillna(df.median(numeric_only=True))
df = df.drop_duplicates()

print("\nDataset shape after cleaning:", df.shape)
df = pd.get_dummies(df, drop_first=True)

print("\nDataset after encoding:\n")
print(df.head())
X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data Size:", X_train.shape)
print("Testing Data Size:", X_test.shape)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
def evaluate_model(name, y_true, y_pred):

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    print("\n")
    print("Model:", name)
    print("R2 Score:", r2)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
evaluate_model("Linear Regression", y_test, lr_pred)
evaluate_model("Decision Tree", y_test, dt_pred)
evaluate_model("Random Forest", y_test, rf_pred)