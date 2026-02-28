import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

def one_hot_encoding(df, columns):
    return pd.get_dummies(df, columns=columns)


def label_encoding(df, column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    return df


def ordinal_encoding(df, column, categories_order):
    oe = OrdinalEncoder(categories=[categories_order])
    df[[column]] = oe.fit_transform(df[[column]])
    return df


def frequency_encoding(df, column):
    freq = df[column].value_counts()
    df[column + "_freq"] = df[column].map(freq)
    return df


def target_encoding(df, column, target):
    target_mean = df.groupby(column)[target].mean()
    df[column + "_target"] = df[column].map(target_mean)
    return df