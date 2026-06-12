import streamlit as st
import pandas as pd
import nltk
import re

from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# NLTK Downloads
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Page Config
st.set_page_config(page_title="Information Retrieval System")

st.title("Information Retrieval System")
st.write("BITS Pilani Assignment 1")

# File Upload
uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    # Read Dataset
    df = pd.read_csv(uploaded_file)

    st.success(f"{len(df)} documents loaded")

    # ==================================
    # DATASET PREVIEW
    # ==================================

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ==================================
    # DATASET INFORMATION
    # ==================================

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

    # ==================================
    # CATEGORY DISTRIBUTION
    # ==================================

    if "category" in df.columns:

        st.subheader("Category Distribution")

        st.bar_chart(
            df["category"].value_counts()
        )

    # ==================================
    # SAMPLE ABSTRACT
    # ==================================

    if "abstract" in df.columns:

        text = str(df["abstract"].iloc[0])

        st.subheader("Sample Abstract")
        st.write(text)

        # ==================================
        # TOKENIZATION
        # ==================================

        tokens = word_tokenize(text.lower())

        # ==================================
        # STOPWORD REMOVAL
        # ==================================

        stop_words = set(stopwords.words("english"))

        filtered_tokens = [
            word
            for word in tokens
            if word.isalnum() and word not in stop_words
        ]

        # ==================================
        # STEMMING
        # ==================================

        stemmer = PorterStemmer()

        stemmed_tokens = [
            stemmer.stem(word)
            for word in filtered_tokens
        ]

        # ==================================
        # LEMMATIZATION
        # ==================================

        lemmatizer = WordNetLemmatizer()

        lemmatized_tokens = [
            lemmatizer.lemmatize(word)
            for word in filtered_tokens
        ]

        # ==================================
        # PREPROCESSING COMPARISON
        # ==================================

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

        # ==================================
        # OUTPUTS
        # ==================================

        st.subheader("Tokenized Output")
        st.write(tokens[:50])

        st.subheader("Stopword Removal Output")
        st.write(filtered_tokens[:50])

        st.subheader("Stemmed Output")
        st.write(stemmed_tokens[:50])

        st.subheader("Lemmatized Output")
        st.write(lemmatized_tokens[:50])

        # ==================================
        # INVERTED INDEX
        # ==================================

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

        # ==================================
        # PHRASE QUERY PROCESSING
        # ==================================

        st.header("Phrase Query Processing")

        query = st.text_input(
            "Enter Phrase Query",
            "machine learning"
        )

        # BIWORD INDEX

        biword_index = defaultdict(set)

        for doc_id, doc in enumerate(df["abstract"][:100]):

            words = re.findall(
                r'\b\w+\b',
                str(doc).lower()
            )

            for i in range(len(words) - 1):

                biword = (
                    words[i]
                    + " "
                    + words[i + 1]
                )

                biword_index[biword].add(doc_id)

        biword_results = list(
            biword_index.get(
                query.lower(),
                set()
            )
        )

        # POSITIONAL INDEX

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

            common_docs = (
                set(positional_index[first_word].keys())
                &
                set(positional_index[second_word].keys())
            )

            for doc_id in common_docs:

                first_positions = positional_index[first_word][doc_id]
                second_positions = positional_index[second_word][doc_id]

                for pos in first_positions:

                    if (pos + 1) in second_positions:

                        positional_results.append(doc_id)
                        break

        # RESULTS

        st.subheader("Biword Index Result")
        st.write(biword_results)

        st.subheader("Positional Index Result")
        st.write(positional_results)

        comparison_phrase_df = pd.DataFrame({
            "Method": [
                "Biword Index",
                "Positional Index"
            ],
            "Retrieved Documents": [
                len(biword_results),
                len(positional_results)
            ]
        })

        st.subheader("Phrase Query Comparison")
        st.dataframe(comparison_phrase_df)

        st.info(
            "Positional Index preserves word positions and "
            "therefore provides more accurate phrase matching."
        )
