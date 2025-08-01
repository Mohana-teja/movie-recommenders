import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load dataset
movies_data = pd.read_csv('movies data.csv')

# Preprocess
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = (
    movies_data['genres'] + ' ' +
    movies_data['keywords'] + ' ' +
    movies_data['tagline'] + ' ' +
    movies_data['cast'] + ' ' +
    movies_data['director']
)

vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vector)

# Streamlit UI
st.title("🎬 Movie Recommender App")
st.write("Get movie recommendations based on your favorite movie!")

# Movie selector
movie_list = movies_data['title'].tolist()
selected_movie = st.selectbox("Select a movie", movie_list)

if st.button("Recommend"):
    index = movies_data[movies_data.title == selected_movie].index[0]
    similarity_scores = list(enumerate(similarity[index]))
    sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:]

    st.subheader("Top 10 Movie Recommendations:")
    for i, movie in enumerate(sorted_similar_movies[:10]):
        movie_index = movie[0]
        st.write(f"{i+1}. {movies_data.iloc[movie_index].title}")
