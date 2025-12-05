import streamlit as st
import pandas as pd
import os
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="Preprocessing")
st.title("Preprocessing options")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "vg_sales_data.csv"

uploaded = st.file_uploader("Upload CSV for preprocessing", type=['csv'], key="prep_upload")
if uploaded:
    df = pd.read_csv(uploaded)
else:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        st.error("Dataset not found.")
        st.stop()

st.write("Shape:", df.shape)

# Options
st.subheader("Missing value handling")
fill_critic = st.selectbox("Impute critic_score with:", ["median", "mean", "zero", "drop rows"])
fill_categorical = st.selectbox("Fill categorical missing values with:", ["Unknown", "drop rows"])

if st.button("Apply preprocessing preview"):
    df2 = df.copy()
    # convert dates
    df2['release_date'] = pd.to_datetime(df2['release_date'], errors='coerce')
    df2['last_update'] = pd.to_datetime(df2['last_update'], errors='coerce')

    # handle critic score
    if fill_critic == "median":
        df2['critic_score'] = df2['critic_score'].fillna(df2['critic_score'].median())
    elif fill_critic == "mean":
        df2['critic_score'] = df2['critic_score'].fillna(df2['critic_score'].mean())
    elif fill_critic == "zero":
        df2['critic_score'] = df2['critic_score'].fillna(0)
    else:
        df2 = df2[~df2['critic_score'].isna()]

    # categorical
    cat_cols = ['genre','publisher','developer','console']
    if fill_categorical == "Unknown":
        for c in cat_cols:
            df2[c] = df2[c].fillna('Unknown')
    else:
        df2 = df2.dropna(subset=cat_cols)

    st.write("Preview after preprocessing (first 200 rows):")
    st.dataframe(df2.head(200))
    st.write("New shape:", df2.shape)

    if st.button("Download processed CSV"):
        csv = df2.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "vg_sales_processed.csv", "text/csv")