import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    df.columns = df.columns.str.strip()

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    df.drop(columns=['customerID'], inplace=True)

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df