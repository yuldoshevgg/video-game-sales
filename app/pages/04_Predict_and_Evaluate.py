import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

st.set_page_config(page_title="Predict & Evaluate")
st.title("Predict & Evaluate")

# Load saved models
model_files = [f for f in os.listdir("../models") if f.endswith("_pipeline.joblib")]
st.write("Found models:", model_files)

model_choice = st.selectbox("Choose a model", model_files)
model_path = os.path.join("../models", model_choice)
model = joblib.load(model_path)

st.subheader("Manual prediction")
# Provide simple inputs for required features
console = st.text_input("console", value="PS4")
genre = st.text_input("genre", value="Action")
publisher = st.text_input("publisher", value="Unknown")
developer = st.text_input("developer", value="Unknown")
critic_score = st.number_input("critic_score", min_value=0.0, max_value=10.0, value=7.0)
na_sales = st.number_input("na_sales (millions)", min_value=0.0, value=0.1)
jp_sales = st.number_input("jp_sales (millions)", min_value=0.0, value=0.0)
pal_sales = st.number_input("pal_sales (millions)", min_value=0.0, value=0.05)
other_sales = st.number_input("other_sales (millions)", min_value=0.0, value=0.01)

if st.button("Predict total_sales"):
    X_pred = pd.DataFrame([{
        'console': console, 'genre': genre, 'publisher': publisher, 'developer': developer,
        'critic_score': critic_score, 'na_sales': na_sales, 'jp_sales': jp_sales, 'pal_sales': pal_sales, 'other_sales': other_sales
    }])
    pred = model.predict(X_pred)[0]
    st.success(f"Predicted total_sales (millions): {pred:.4f}")
