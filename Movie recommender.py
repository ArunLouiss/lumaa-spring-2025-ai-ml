import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#nltk.download('stopwords')
#nltk.download('punkt_tab')
#nltk.download('wordnet')


# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
#stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """
    Preprocesses text by:
    1. Removing special characters
    2. Tokenizing and converting to lowercase
    3. Removing stop words
    4. Lemmatization
    """

    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove all special characters
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()] 

    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens) 

# Read the movie dataset from a CSV file and combines relevant text features.
def read_data(file_path):

    movies_df = pd.read_csv(file_path)

    # Combine relevant text features for better recommendations
    movies_df['combined_text'] = (
        movies_df['Genre'].fillna('') + ' ' + 
        movies_df['Overview'].fillna('') + ' ' + 
        movies_df['Director'].fillna('') + ' ' + 
        movies_df['Star1'].fillna('') + ' ' + 
        movies_df['Star2'].fillna('')
    )

    # Perform preprocessing
    movies_df['processed_text'] = movies_df['combined_text'].apply(preprocess_text)
    
    return movies_df

# TF-IDF vectorization on the processed dataset.
def tfidf(movies_df):

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df["processed_text"])
    return tfidf_vectorizer, tfidf_matrix

# Returns the top N movie recommendations based on cosine similarity.
def recommendation_system(input_text, tfidf_vectorizer, tfidf_matrix, movies_df):

    processed_text = preprocess_text(input_text)
    input_tfidf = tfidf_vectorizer.transform([processed_text])

    # Compute cosine similarity between input and movie dataset
    cosine_similarities = cosine_similarity(input_tfidf, tfidf_matrix).flatten()

    top_indices = cosine_similarities.argsort()[-5:][::-1]

    return movies_df.iloc[top_indices][["Series_Title"]]


# Load movie dataset
movies_path = r"C:\Users\arunl\Documents\ASU\Uni\Lumaa\imdb_top_500.csv"
movies_df = read_data(movies_path)

# TF-IDF Vectorization for the movie dataset
tfidf_vectorizer, tfidf_movie_matrix = tfidf(movies_df)


# User input
user_input = "I love thrilling action movies set in space, with a comedic twist."
#recommendations = recommendation_system(user_input, tfidf_vectorizer, tfidf_movie_matrix, movies_df)

# Display recommendations
#print(recommendations)

print(pd.__version__)
print(np.__version__)
print(nltk.__version__)
print(sklearn.__version__)

