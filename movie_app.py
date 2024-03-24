import pickle
import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Get the TMDB API key from environment variable
# api_key = os.getenv('TMDB_API_KEY')

# if not api_key:
#     raise ValueError('TMDB_API_KEY environment variable is not set')

api_key = '613b9e66c1e1b3fee798437e9803e1b5'

# Load models and data
movies = pickle.load(open('movies.pkl', 'rb'))
nlp_model = pickle.load(open('nlp_model.pkl', 'rb'))
transform_vectorizer = pickle.load(open('tranform.pkl', 'rb'))

# Function to fetch poster
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    data = requests.get(url).json()
    poster_path = data['poster_path']
    return f"https://image.tmdb.org/t/p/w500/{poster_path}"

# Function to recommend movies
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

# Function to fetch reviews and perform sentiment analysis
def fetch_reviews(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"
    data = requests.get(url).json()
    if data['total_results'] > 0:
        movie_id = data['results'][0]['id']
        reviews_url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={api_key}&language=en-US&page=1"
        reviews_data = requests.get(reviews_url).json()
        reviews = [{'Review': review['content'], 'Author': review['author']} for review in reviews_data['results']]
        return reviews
    else:
        return []

# Streamlit app code
st.header('Movie Recommender System')

selected_movie = st.selectbox("Type or select a movie from the dropdown", movies['movie_title'].values)

if st.button('Show Recommendation'):
    recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])

    # Fetch reviews and perform sentiment analysis
    reviews = fetch_reviews(selected_movie)
    if reviews:
        st.text('')
        st.text('')
        st.subheader('Reviews and Sentiment Analysis:')
        reviews_df = pd.DataFrame(reviews)
        transformed_reviews = transform_vectorizer.transform(reviews_df['Review'])
        predictions = nlp_model.predict(transformed_reviews)
        reviews_df['Sentiment'] = ['Positive' if prediction else 'Negative' for prediction in predictions]
        st.table(reviews_df[['Review', 'Author', 'Sentiment']].style.hide_index())
    else:
        st.write("No reviews found for this movie.")
