import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)


def handle_missing_values(df):
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def fix_datatypes(df):
    df['MSSubClass'] = df['MSSubClass'].astype(str) 
    return df


def remove_duplicates(df):
    return df.drop_duplicates()


def treat_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[column] >= lower) & (df[column] <= upper)]
    return df


def remove_irrelevant_columns(df):
    df = df.drop(columns=['Id'])
    return df


def transform_skewed_feature(df, column):
    df[column] = np.log1p(df[column])
    return df