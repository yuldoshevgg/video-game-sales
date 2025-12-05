import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

st.set_page_config(page_title="Data Explorer")
st.title("Data Explorer")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "vg_sales_data.csv"

st.markdown("Upload dataset or use default dataset in `data/`")

uploaded = st.file_uploader("Upload CSV", type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        st.error("No default dataset found at data/vg_sales_data.csv")
        st.stop()

st.write("Dataset shape:", df.shape)
if st.checkbox("Show raw data (first 200 rows)"):
    st.dataframe(df.head(200))

# Basic statistics
st.subheader("Summary statistics")
st.write(df.describe(include='all').T)

# Missing values
st.subheader("Missing values")
st.write(df.isna().sum().sort_values(ascending=False))

# Plots
st.subheader("Plots")
if st.button("Show basic plots"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid")
    numeric_cols = ['critic_score','total_sales','na_sales','jp_sales','pal_sales','other_sales']
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    sns.histplot(df['total_sales'].dropna(), bins=50, ax=ax[0])
    ax[0].set_title("total_sales distribution")
    sns.boxplot(x=df['critic_score'], ax=ax[1])
    ax[1].set_title("critic_score boxplot")
    st.pyplot(fig)

# Top genres
st.subheader("Top genres")
top_genres = df['genre'].value_counts().nlargest(10)
st.bar_chart(top_genres)
