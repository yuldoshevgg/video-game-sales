import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Train & Evaluate")
st.title("Train & Evaluate Models")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "vg_sales_data.csv"

uploaded = st.file_uploader("Upload CSV to train on (or leave to use default)", type=['csv'], key="train_upload")
if uploaded:
    df = pd.read_csv(uploaded)
else:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        st.error("Dataset not found.")
        st.stop()

st.write("Dataset shape:", df.shape)

# Prepare simple train/test split
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
FEATURE_COLS = ['console','genre','publisher','developer','critic_score','na_sales','jp_sales','pal_sales','other_sales']
TARGET = 'total_sales'
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

if st.button("Create splits and train models"):
    X = df[FEATURE_COLS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    st.write("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    # Preprocessor
    num_cols = ['critic_score','na_sales','jp_sales','pal_sales','other_sales']
    cat_cols = ['console','genre','publisher','developer']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    # Models
    models = {
        'Ridge': Ridge(random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=150, random_state=42)
    }

    results = []
    for name, model in models.items():
        pipe = Pipeline([('preproc', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        rmse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results.append({'model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
        # Save model
        joblib.dump(pipe, f"../models/{name}_pipeline.joblib")
    st.write(pd.DataFrame(results).set_index('model'))
    st.success("Training finished and models saved to /models/")
