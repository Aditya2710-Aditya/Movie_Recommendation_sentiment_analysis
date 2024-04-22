import os
import joblib
import streamlit as st
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import random
import urllib.request
from bs4 import BeautifulSoup

# # Get the TMDB API key from environment variable
# tmdb_api_key = os.getenv('TMDB_API_KEY')

# if not tmdb_api_key:
#     raise ValueError('TMDB_API_KEY environment variable is not set')

# TMDB API key
tmdb_api_key = '613b9e66c1e1b3fee798437e9803e1b5'
# Load models and data
movies = joblib.load('movies.pkl')
nlp_model = joblib.load('nlp_lr_model.pkl')
transform_vectorizer = joblib.load('transform.pkl')

def get_imdb_id(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={tmdb_api_key}&language=en-US"
    response = requests.get(url)
    data = response.json()
    return data.get('imdb_id')

def scrape_imdb_reviews(imdb_id):
    try:
        sauce = urllib.request.urlopen(f'https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ov_rt').read()
        soup = BeautifulSoup(sauce, 'lxml')
        soup_result = soup.find_all("div", {"class": "text show-more__control"})
        
        reviews_list = [review.text.strip() for review in soup_result if review.text]
        
        if len(reviews_list) > 10:
            reviews_list = random.sample(reviews_list, 10)
        
        return reviews_list
    except Exception as e:
        print(f"Error occurred while scraping IMDb reviews: {str(e)}")
        return []

def analyze_sentiment(reviews):
    transformed_reviews = transform_vectorizer.transform(reviews)
    sentiments = nlp_model.predict(transformed_reviews)
    sentiment_labels = ['Positive' if sentiment == 1 else 'Negative' for sentiment in sentiments]
    return sentiment_labels

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={tmdb_api_key}&language=en-US"
    data = requests.get(url).json()
    poster_path = data['poster_path']
    return f"https://image.tmdb.org/t/p/w500/{poster_path}"

def recommend(movie):
    index = movies[movies['movie_title'] == movie].index[0]
    cv = CountVectorizer()
    vectors = cv.fit_transform(movies['comb'])
    similarity = cosine_similarity(vectors)  
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].movie_title)
    return recommended_movie_names, recommended_movie_posters

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

    st.header('Reviews Sentiment Analysis:')
    movie_id = movies[movies['movie_title'] == selected_movie].iloc[0]['movie_id']
    imdb_id = get_imdb_id(movie_id)
    if imdb_id:
        reviews = scrape_imdb_reviews(imdb_id)
        if reviews:
            sentiments = analyze_sentiment(reviews)
            df_sentiments = pd.DataFrame({'Review': reviews, 'Sentiment': sentiments})
            st.table(df_sentiments)
        else:
            st.warning("No reviews found for this movie on IMDb.")
    else:
        st.error("Failed to fetch IMDb ID for the selected movie.")
