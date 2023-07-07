# Import libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import csr_matrix
import streamlit as st

# Load the dataset
movies = pd.read_csv("movies_metadata.csv",low_memory=False)

# Drop the rows with missing values
movies.dropna(inplace=True)

# Extract the overview column
overviews = movies["overview"]

# Create a TF-IDF vectorizer object
tfidf = TfidfVectorizer(stop_words="english")

# Fit and transform the overviews
tfidf_matrix = tfidf.fit_transform(overviews)

# Compute the cosine similarity matrix for content-based filtering
cosine_sim_content = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a pivot table of ratings for collaborative filtering
ratings = pd.read_csv("ratings_small.csv")
ratings_pivot = ratings.pivot(index="movieId", columns="userId", values="rating").fillna(0)
ratings_matrix = csr_matrix(ratings_pivot.values)

# Compute the cosine similarity matrix for collaborative filtering
cosine_sim_collab = linear_kernel(ratings_matrix, ratings_matrix)

# Create a function to get the title from the index
def get_title(index):
  return movies[movies.index == index]["title"].values[0]

# Create a function to get the index from the title
def get_index(title):
  return movies[movies.title == title].index.values[0]

# Create a function to get the movieId from the title
def get_movieId(title):
  return movies[movies.title == title]["id"].values[0]

# Create a function to get the recommendations based on the title and the hybrid method
def get_recommendations(title, method):
  # Get the index and movieId of the movie
  index = get_index(title)
  movieId = get_movieId(title)
  # Get the similarity scores of all movies with that movie based on the method
  if method == "content":
    sim_scores = list(enumerate(cosine_sim_content[index]))
  elif method == "collab":
    sim_scores = list(enumerate(cosine_sim_collab[movieId]))
  elif method == "hybrid":
    sim_scores = list(enumerate(cosine_sim_content[index] + cosine_sim_collab[movieId]))
  # Sort the movies based on the similarity scores
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
  # Get the indices of the 10 most similar movies
  sim_indices = [i[0] for i in sim_scores[1:11]]
  # Get the titles of the 10 most similar movies
  sim_titles = [get_title(i) for i in sim_indices]
  # Return the titles
  return sim_titles

# Create a streamlit app
st.title("Movie Recommendation System")

# Ask the user for a movie title
user_input = st.text_input("Enter a movie title:")

# Ask the user for a recommendation method
user_method = st.selectbox("Select a recommendation method:", ["content", "collab", "hybrid"])

# Check if the user input is valid
if user_input in movies["title"].values:
  # Get the recommendations
  recommendations = get_recommendations(user_input, user_method)
  # Display the recommendations
  st.write(f"Here are 10 movies that are similar to {user_input} using {user_method} method:")
  for movie in recommendations:
    st.write(movie)
else:
  # Display an error message
  st.write("Invalid input. Please enter a valid movie title.")
