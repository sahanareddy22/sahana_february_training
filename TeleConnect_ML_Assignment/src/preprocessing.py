from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def feature_engineering(df):
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['ServiceCount'] = (
        (df['PhoneService'] == 'Yes').astype(int) +
        (df['InternetService'] != 'No').astype(int)
    )
    return df


def get_preprocessor(X):
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(exclude=['object']).columns

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    return preprocessor