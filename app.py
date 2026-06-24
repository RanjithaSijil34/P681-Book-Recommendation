import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# ----------------------------
# Load Files
# ----------------------------
@st.cache_resource
def load_data():

    popular_df = pickle.load(open("popular.pkl", "rb"))
    pt = pickle.load(open("pivot.pkl", "rb"))

    similarity_scores = cosine_similarity(pt)

    return popular_df, pt, similarity_scores

popular_df, pt, similarity_scores = load_data()

# ----------------------------
# Recommendation Function
# ----------------------------
def recommend(book_name):

    try:

        index = np.where(pt.index == book_name)[0][0]

        similar_books = sorted(
            list(enumerate(similarity_scores[index])),
            key=lambda x: x[1],
            reverse=True
        )[1:6]

        recommendations = []

        for book in similar_books:
            recommendations.append(
                pt.index[book[0]]
            )

        return recommendations

    except Exception as e:
        st.error(f"Error: {e}")
        return []

# ----------------------------
# Sidebar
# ----------------------------
menu = st.sidebar.selectbox(
    "Select Option",
    ["Popular Books", "Book Recommendation"]
)

# ----------------------------
# Popular Books Page
# ----------------------------
if menu == "Popular Books":

    st.title("📚 Popular Books")

    st.dataframe(
        popular_df,
        use_container_width=True
    )

# ----------------------------
# Recommendation Page
# ----------------------------
if menu == "Book Recommendation":

    st.title("📖 Book Recommendation System")

    selected_book = st.selectbox(
        "Choose a Book",
        pt.index.tolist()
    )

    if st.button("Recommend"):

        books = recommend(selected_book)

        st.subheader("Recommended Books")

        for i, book in enumerate(books, start=1):
            st.write(f"{i}. {book}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.write("Book Recommendation System using Collaborative Filtering")
