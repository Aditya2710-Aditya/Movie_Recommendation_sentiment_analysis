import os
import pickle
import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Get the TMDB API key from environment variable
api_key = os.getenv('TMDB_API_KEY')

if not api_key:
    raise ValueError('TMDB_API_KEY environment variable is not set')

# Load models and data
movies = pickle.load(open('movies.pkl', 'rb'))
nlp_model = pickle.load(open('nlp_model.pkl', 'rb'))
transform_vectorizer = pickle.load(open('tranform.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Function to fetch reviews from TMDB APIS
def fetch_reviews(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={tmdb_api_key}&language=en-US&page=1"
    response = requests.get(url)
    data = response.json()
    reviews_data = data['results']
    if not reviews_data:
        return [], []
    reviews = [review['content'] for review in reviews_data]
    authors = [review['author'] for review in reviews_data]
    return reviews, authors

# Function to perform sentiment analysis
def analyze_sentiment(reviews):
    transformed_reviews = transform_vectorizer.transform(reviews)
    sentiments = nlp_model.predict(transformed_reviews)
    sentiment_labels = ['Positive' if sentiment == 1 else 'Negative' for sentiment in sentiments]
    return sentiment_labels

# Function to fetch poster
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={tmdb_api_key}&language=en-US"
    data = requests.get(url).json()
    poster_path = data['poster_path']
    return f"https://image.tmdb.org/t/p/w500/{poster_path}"

# Function to recommend movies
def recommend(movie):
    index = movies[movies['movie_title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].movie_title)
    return recommended_movie_names, recommended_movie_posters

# Streamlit app code
st.header('Movie Recommendation and Reviews Sentiment Analysis')

selected_movie = st.selectbox("Type or select a movie from the dropdown", movies['movie_title'].values)

if st.button('Show Recommendation'):
    st.subheader('Recommended Movies:')
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(recommended_movie_posters[0])
        st.write(recommended_movie_names[0])
    with col2:
        st.image(recommended_movie_posters[1])
        st.write(recommended_movie_names[1])
    with col3:
        st.image(recommended_movie_posters[2])
        st.write(recommended_movie_names[2])
    with col4:
        st.image(recommended_movie_posters[3])
        st.write(recommended_movie_names[3])
    with col5:
        st.image(recommended_movie_posters[4])
        st.write(recommended_movie_names[4])
    
    # Fetching reviews, authors, and performing sentiment analysis
    st.header('Reviews Sentiment Analysis:')
    movie_id = movies[movies['movie_title'] == selected_movie].iloc[0]['movie_id']
    reviews, authors = fetch_reviews(movie_id)
    if not reviews:
        st.warning("Reviews for this movie are not available.")
    else:
        sentiments = analyze_sentiment(reviews)
        # Displaying sentiment analysis results in a table format without horizontal scrolling
        df_sentiments = pd.DataFrame({'Reviews': reviews, 'Author': authors, 'Sentiment': sentiments})
        st.table(df_sentiments)
