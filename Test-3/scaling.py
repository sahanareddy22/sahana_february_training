from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, Normalizer

def min_max_scaling(df, column):
    scaler = MinMaxScaler()
    df[[column]] = scaler.fit_transform(df[[column]])
    return df


def max_abs_scaling(df, column):
    scaler = MaxAbsScaler()
    df[[column]] = scaler.fit_transform(df[[column]])
    return df


def standard_scaling(df, column):
    scaler = StandardScaler()
    df[[column]] = scaler.fit_transform(df[[column]])
    return df


def vector_normalization(df, column):
    scaler = Normalizer()
    df[[column]] = scaler.fit_transform(df[[column]])
    return df