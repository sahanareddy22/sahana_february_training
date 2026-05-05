import joblib


def save_model(model, path):
    """
    Save trained model to disk
    """
    joblib.dump(model, path)