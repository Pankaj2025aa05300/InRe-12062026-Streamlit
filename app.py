import streamlit as st
import pandas as pd

st.set_page_config(page_title="Information Retrieval System")

st.title("Information Retrieval System")
st.write("BITS Pilani Assignment 1")

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.success(f"{len(df)} documents loaded")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Documents", len(df))

    with col2:
        if "category" in df.columns:
            st.metric(
                "Categories",
                df["category"].nunique()
            )

    if "category" in df.columns:

        st.subheader("Category Distribution")

        st.bar_chart(
            df["category"].value_counts()
        )

    st.subheader("Sample Abstract")

    if "abstract" in df.columns:
        st.write(df["abstract"].iloc[0])

    st.subheader("Preprocessing Comparison")
    st.dataframe(comparison_df)
