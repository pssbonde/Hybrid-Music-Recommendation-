import numpy as np
import pandas as pd
import joblib
import os

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import save_npz
from Data_Cleaning import data_for_content_filtering

# ✅ Set the working directory
os.chdir("D:/DATA_SCIENCE/CampusX/Projects/Hybrid Recommonder p4/Actual Work")

# Cleaned Data Path
CLEANED_DATA_PATH = "cleaned_data.csv"

# Columns to transform
frequency_encode_cols = ['year']
ohe_cols = ['artist', 'time_signature', 'key']
tfidf_col = 'tags'
standard_scale_cols = ['duration_ms', 'loudness', 'tempo']
min_max_scale_cols = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness']

def train_transformer(data):
    """
    Trains a ColumnTransformer on the provided data and saves the transformer to a file.
    Applies:
    - Frequency encoding using CountEncoder.
    - One-Hot Encoding.
    - TF-IDF Vectorization.
    - Standard Scaling.
    - Min-Max Scaling.
    Saves:
    transformer.joblib : The trained ColumnTransformer Object.
    """
    transformer = ColumnTransformer(
        transformers=[
            ('frequency_encode', CountEncoder(normalize=True, return_df=True), frequency_encode_cols),
            ('ohe', OneHotEncoder(handle_unknown='ignore'), ohe_cols),
            ('tfidf', TfidfVectorizer(max_features=85), tfidf_col),
            ('standard_scaler', StandardScaler(), standard_scale_cols),
            ('min_max_scaler', MinMaxScaler(), min_max_scale_cols)
        ],
        remainder='passthrough',
        n_jobs=-1,
        verbose=False
    )

    # Fit the transformer
    transformer.fit(data)

    # Save the transformer
    joblib.dump(transformer, "transformer.joblib")

def transform_data(data):
    """
    Transforms the input data using a pre-trained transformer.
    Args:
        data (array-like): The data to be transformed.
    Returns:
        array-like: The transformed data.
    """
    transformer = joblib.load("transformer.joblib")
    transformed_data = transformer.transform(data)
    return transformed_data

def save_transformed_data(transformed_data, save_path):
    """
    Save the transformed data to a specified file path.
    Parameters:
    transformed_data (scipy.sparse.csr_matrix): The transformed data to be saved.
    save_path (str): The file path where the transformed data will be saved.
    """
    save_npz(save_path, transformed_data)

def calculate_similarity_scores(input_vector, data):
    """
    Calculate similarity scores between an input vector and a dataset using cosine similarity.
    Args:
        input_vector (array_like): The input vector for which similarity scores are to be calculated.
        data (array-like): The dataset against which the similarity scores are to be calculated.
    Returns:
        array-like: An array of similarity scores.
    """
    similarity_scores = cosine_similarity(input_vector, data)
    return similarity_scores

def recommend(song_name, songs_data, transformed_data, k=10):
    """
    Recommends top K songs similar to the given song based on content-based filtering.
    Parameters:
        song_name (str): Name of the song for which recommendations are to be generated.
        songs_data (DataFrame): The DataFrame containing song information.
        transformed_data (ndarray): The transformed data matrix for similarity calculations.
        k (int, Optional): Number of similar songs to recommend. Defaults to 10.
    Returns:
        DataFrame: A DataFrame containing the top K similar songs.
    """
    song_name = song_name.lower()
    song_row = songs_data.loc[songs_data['name'].str.lower() == song_name, :]

    if song_row.empty:
        print("Song not found in the dataset.")
    else:
        song_index = song_row.index[0]
        print(song_index)

        input_vector = transformed_data[song_index].reshape(1, -1)
        similarity_scores = cosine_similarity(input_vector, transformed_data)
        print(similarity_scores.shape)

        top_k_songs_indexes = np.argsort(similarity_scores.ravel())[-k:][::-1]
        print(top_k_songs_indexes)

        top_k_songs_names = songs_data.iloc[top_k_songs_indexes]
        top_k_list = top_k_songs_names[['name', 'artist', 'spotify_preview_url']].reset_index(drop=True)
        return top_k_list

def test_recommendations(data_path, song_name, k=10):
    """
    Test Recommendations for a given song using content-based filtering.
    Parameters:
    data_path (str): The path to the CSV file containing the song data.
    song_name (str): The name of the song for which recommendations are to be generated.
    k (int, Optional): The number of top similar songs to return. Default is 10.
    Returns:
    DataFrame: Top k recommended songs based on content similarity.
    """
    # Convert song name to lower case
    song_name = song_name.lower()

    # Load data
    data = pd.read_csv(data_path)

    # Clean the data 
    data_content_filtering = data_for_content_filtering(data)

    # Train the transformer
    train_transformer(data_content_filtering)

    # Transform the data using trained transformer
    transformed_data = transform_data(data_content_filtering)

    # Save transformed data
    save_transformed_data(transformed_data, "transformed_data.npz")

    # Find the matching song row
    song_row = data.loc[data["name"].str.lower() == song_name]

    if song_row.empty:
        print("Song not found in the dataset.")
        return

    print(song_name)

    # Get the index of the song
    song_index = song_row.index[0]

    # Generate the Input Vector
    input_vector = transformed_data[song_index].reshape(1, -1)

    # Calculate Similarity Scores
    similarity_scores = calculate_similarity_scores(input_vector, transformed_data)

    # Get top K recommendations
    top_k_songs_indexes = np.argsort(similarity_scores.ravel())[-k-1:-1][::-1]

    # Get the top k songs name
    top_k_songs = data.iloc[top_k_songs_indexes]

    # print the top k songs name
    print(top_k_songs)


if __name__ == "__main__":
    test_recommendations(CLEANED_DATA_PATH, "in the end")
