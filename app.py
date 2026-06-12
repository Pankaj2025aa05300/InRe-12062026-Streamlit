import streamlit as st
import pandas as pd
import nltk
import re

from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

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

    # Dataset Preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Dataset Information
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

    # Category Distribution
    if "category" in df.columns:

        st.subheader("Category Distribution")

        st.bar_chart(
            df["category"].value_counts()
        )

    # Sample Abstract
    if "abstract" in df.columns:

        st.subheader("Sample Abstract")

        text = str(df["abstract"].iloc[0])

        st.write(text)

        # Tokenization
        tokens = word_tokenize(text.lower())

        # Stopword Removal
        stop_words = set(stopwords.words("english"))

        filtered_tokens = [
            word
            for word in tokens
            if word.isalnum() and word not in stop_words
        ]

        # Stemming
        stemmer = PorterStemmer()

        stemmed_tokens = [
            stemmer.stem(word)
            for word in filtered_tokens
        ]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()

        lemmatized_tokens = [
            lemmatizer.lemmatize(word)
            for word in filtered_tokens
        ]

        # Comparison Table
        n = min(
            len(tokens),
            len(filtered_tokens),
            len(stemmed_tokens),
            len(lemmatized_tokens),
            20
        )

        comparison_df = pd.DataFrame({
            "Original": tokens[:n],
            "Stopword Removed": filtered_tokens[:n],
            "Stemmed": stemmed_tokens[:n],
            "Lemmatized": lemmatized_tokens[:n]
        })

        st.subheader("Preprocessing Comparison")
        st.dataframe(comparison_df)

        # Token Outputs

        st.subheader("Tokenized Output")
        st.write(tokens[:50])

        st.subheader("Stopword Removal Output")
        st.write(filtered_tokens[:50])

        st.subheader("Stemmed Output")
        st.write(stemmed_tokens[:50])

        st.subheader("Lemmatized Output")
        st.write(lemmatized_tokens[:50])

        # ====================================
        # INVERTED INDEX
        # ====================================

        st.subheader("Inverted Index")

        inverted_index = defaultdict(set)

        for doc_id, doc in enumerate(df["abstract"][:100]):

            words = re.findall(
                r'\b\w+\b',
                str(doc).lower()
            )

            for word in words:
                inverted_index[word].add(doc_id)

        sample_terms = list(inverted_index.keys())[:20]

        index_df = pd.DataFrame({
            "Term": sample_terms,
            "Document IDs": [
                list(inverted_index[t])[:10]
                for t in sample_terms
            ]
        })

        st.dataframe(index_df)
        # ====================================
# PHRASE QUERY PROCESSING
# ====================================

st.header("Phrase Query Processing")

query = st.text_input(
    "Enter Phrase Query",
    value="machine learning"
)

if query:

    # -------------------------------
    # BIWORD INDEX
    # -------------------------------

    biword_index = defaultdict(set)

    for doc_id, doc in enumerate(df["abstract"][:100]):

        words = re.findall(
            r'\b\w+\b',
            str(doc).lower()
        )

        for i in range(len(words) - 1):

            biword = words[i] + " " + words[i + 1]

            biword_index[biword].add(doc_id)

    biword_results = list(
        biword_index.get(
            query.lower(),
            set()
        )
    )

    # -------------------------------
    # POSITIONAL INDEX
    # -------------------------------

    positional_index = defaultdict(
        lambda: defaultdict(list)
    )

    for doc_id, doc in enumerate(df["abstract"][:100]):

        words = re.findall(
            r'\b\w+\b',
            str(doc).lower()
        )

        for pos, word in enumerate(words):

            positional_index[word][doc_id].append(pos)

    positional_results = []

    query_terms = query.lower().split()

    if len(query_terms) == 2:

        first_word = query_terms[0]
        second_word = query_terms[1]

        common_docs = set(
            positional_index[first_word].keys()
       
