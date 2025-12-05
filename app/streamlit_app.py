# streamlit_app.py
import streamlit as st

st.set_page_config(page_title="Video Game Sales ML", layout="wide")

st.title("Video Game Sales — ML Coursework")
st.markdown("""
Use the left-hand menu (or the pages in the Streamlit sidebar) to navigate:
- Data Explorer
- Preprocessing
- Train & Evaluate
- Predict & Evaluate
""")
st.write("Use the `pages/` folder files for the multi-page experience.")