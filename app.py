import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    popular_df = pickle.load(open('popular.pkl', 'rb'))
    pt = pickle.load(open('pivot.pkl', 'rb'))
    books = pickle.load(open('books.pkl', 'rb'))

    similarity_scores = cosine_similarity(pt)

    return popular_df, pt, books, similarity_scores


popular_df, pt, books, similarity_scores = load_data()

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend(book_name):
    try:
        index = np.where(pt.index == book_name)[0][0]

        similar_items = sorted(
            list(enumerate(similarity_scores[index])),
            key=lambda x: x[1],
            reverse=True
        )[1:6]

        recommendations = []

        for item in similar_items:
            recommendations.append(pt.index[item[0]])

        return recommendations

    except:
        return []


# -----------------------------
# Sidebar
# -----------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Popular Books", "Book Recommendation"]
)

# -----------------------------
# Popular Books
# -----------------------------
if menu == "Popular Books":

    st.title("📚 Popular Books")

    st.dataframe(
        popular_df,
        use_container_width=True
    )

# -----------------------------
# Recommendation Page
# -----------------------------
if menu == "Book Recommendation":

    st.title("📖 Book Recommendation System")

    selected_book = st.selectbox(
        "Select a Book",
        sorted(pt.index.tolist())
    )

    if st.button("Recommend"):

        recommendations = recommend(selected_book)

        if recommendations:

            st.subheader("Recommended Books")

            for i, book in enumerate(recommendations, start=1):
                st.write(f"**{i}. {book}**")

        else:
            st.error("No recommendations found.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "Built using Collaborative Filtering and Cosine Similarity"
)
