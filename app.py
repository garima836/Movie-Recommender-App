import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaning
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load and prepare data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/rashida048/Some-NLP-Projects/master/movie_dataset.csv"
    df = pd.read_csv(url)

    # Drop duplicates and fill NaNs
    df.drop_duplicates(subset='title', inplace=True)
    features = ['keywords', 'cast', 'genres', 'director', 'tagline']
    df[features] = df[features].fillna('')

    # Combine and normalize features
    def combine_features(row):
        return f"{row['keywords']} {row['cast']} {row['genres']} {row['director']} {row['tagline']}"
    df['combined_features'] = df.apply(combine_features, axis=1)
    df['normalized_features'] = df['combined_features'].apply(normalize_text)

    # Vectorize and compute cosine similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['normalized_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()

    return df, cosine_sim, indices

df, cosine_sim, indices = load_data()

# Recommendation logic
def recommend_movies(title, num_recommendations=5):
    title = title.lower()
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].reset_index(drop=True)

# Streamlit UI
st.title("🎬 Movie Recommendation App")
st.markdown("Enter a movie title to get recommendations based on content similarity!")

user_input = st.text_input("Enter Movie Title", placeholder="e.g. Avatar")
if st.button("Get Recommendations"):
    if user_input:
        recommendations = recommend_movies(user_input)
        if recommendations is not None:
            st.success("Recommended Movies:")
            for i, title in enumerate(recommendations, 1):
                st.write(f"{i}. {title}")
        else:
            st.error("Movie not found. Try another title.")
