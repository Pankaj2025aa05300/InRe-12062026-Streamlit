import streamlit as st
import pandas as pd

st.title("Information Retrieval System")

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success(f"Loaded {len(df)} documents")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Sample Document")

    if "abstract" in df.columns:
        st.write(df["abstract"].iloc[0])
