import joblib
import os

def save_model(pipeline, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
