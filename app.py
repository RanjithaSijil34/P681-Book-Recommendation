import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# =========================
# Load Models
# =========================

popular_df = pickle.load(open("popular.pkl", "rb"))
pt = pickle.load(open("pivot.pkl", "rb"))
similarity_scores = pickle.load(open("similarity.pkl", "rb"))
books = pickle.load(open("books.pkl", "rb"))
content_similarity = pickle.load(open("content_similarity.pkl", "rb"))

# =========================
# Helper Functions
# =========================

indices = pd.Series(
    books.index,
    index=books['Book-Title']
).drop_duplicates()


def collaborative_recommend(book_name):

    try:
        index = np.where(pt.index == book_name)[0][0]

        similar_items = sorted(
            list(enumerate(similarity_scores[index])),
            key=lambda x: x[1],
            reverse=True
        )[1:6]

        recommendations = []

        for item in similar_items:
            recommendations.append(
                pt.index[item[0]]
            )

        return recommendations

    except:
        return []


def content_recommend(book_name):

    try:

        idx = indices[book_name]

        sim_score = list(
            enumerate(content_similarity[idx])
        )

        sim_score = sorted(
            sim_score,
            key=lambda x: x[1],
            reverse=True
        )[1:6]

        book_indices = [
            i[0]
            for i in sim_score
        ]

        return list(
            books['Book-Title'].iloc[book_indices]
        )

    except:
        return []


def hybrid_recommend(book_name):

    collab = collaborative_recommend(book_name)

    content = content_recommend(book_name)

    final = list(
        dict.fromkeys(
            collab + content
        )
    )

    return final[:10]


# =========================
# UI
# =========================

st.title("📚 Hybrid Book Recommendation System")

menu = st.sidebar.selectbox(
    "Select Option",
    [
        "Popular Books",
        "Book Recommendation"
    ]
)

# =========================
# Popular Books
# =========================

if menu == "Popular Books":

    st.subheader("🔥 Top Popular Books")

    display_df = popular_df.reset_index()

    st.dataframe(
        display_df.head(20),
        use_container_width=True
    )

# =========================
# Recommendation Section
# =========================

elif menu == "Book Recommendation":

    st.subheader("📖 Find Similar Books")

    selected_book = st.selectbox(
        "Choose a Book",
        sorted(pt.index.tolist())
    )

    if st.button("Recommend Books"):

        recommendations = hybrid_recommend(
            selected_book
        )

        st.success(
            f"Recommendations for '{selected_book}'"
        )

        if len(recommendations) > 0:

            for i, book in enumerate(
                recommendations,
                start=1
            ):
                st.write(f"{i}. {book}")

        else:
            st.warning(
                "No recommendations found."
            )

# =========================
# Footer
# =========================

st.markdown("---")
st.write(
    "Developed using Collaborative Filtering + Content-Based Filtering"
)
