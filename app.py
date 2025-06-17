import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import pickle
import numpy as np
from surprise import Dataset, Reader
from surprise import SVD

# Load data
ratings = pd.read_csv("Data/ratings.csv")
movies = pd.read_csv("Data/movies.csv")

# Load trained model
with open("models/svd_model.pkl", "rb") as f:
    svd_model = pickle.load(f)

# Get unique users and movies
unique_users = ratings['userId'].unique()
unique_movies = movies['movieId'].unique()

# Create Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Movie Recommender System"

# Layout
app.layout = html.Div([
    dcc.Tabs(id="tabs", value='user', children=[
        dcc.Tab(label='User Page', value='user'),
        dcc.Tab(label='Item Page', value='item'),
    ]),
    html.Div(id='tabs-content')
])

# User page layout
user_page = html.Div([
    html.H3("User Recommender"),
    dcc.Dropdown(
        id='user-dropdown', 
        options=[{'label': str(int(u)), 'value': int(u)} for u in unique_users], 
        placeholder="Select a user"
    ),
    html.Div(id='user-history'),
    dcc.Input(id='num-recs', type='number', placeholder='Number of recommendations', value=5, min=1, max=20),
    html.Button("Get Recommendations", id='rec-button'),
    html.Div(id='user-recommendations'),
])

# Item page layout
item_page = html.Div([
    html.H3("Item Similarity"),
    dcc.Dropdown(
        id='item-dropdown', 
        options=[{'label': str(row['title']), 'value': int(row['movieId'])} for _, row in movies.iterrows()], 
        placeholder="Select a movie"
    ),
    html.Div(id='item-profile'),
    dcc.Input(id='num-similar', type='number', placeholder='Number of similar items', value=5, min=1, max=20),
    html.Button("Get Similar Items", id='sim-button'),
    html.Div(id='item-similarity'),
])

# Callback to render tabs
@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    if tab == 'user':
        return user_page
    elif tab == 'item':
        return item_page

# Callback: User History
@app.callback(Output('user-history', 'children'), Input('user-dropdown', 'value'))
def show_user_history(user_id):
    if user_id is None:
        return ""
    user_data = ratings[ratings['userId'] == user_id].merge(movies, on='movieId')
    return html.Ul([
        html.Li(f"{row['title']} (Rating: {row['rating']})") 
        for _, row in user_data.iterrows()
    ])

# Callback: Recommendations
@app.callback(
    Output('user-recommendations', 'children'),
    Input('rec-button', 'n_clicks'),
    State('user-dropdown', 'value'),
    State('num-recs', 'value')
)
def recommend_movies(n_clicks, user_id, n):
    if not n_clicks or user_id is None:
        return ""
    
    # Ensure user_id is int
    user_id = int(user_id) if not isinstance(user_id, int) else user_id
    
    # Get rated movies
    rated_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    
    # Get candidate movies
    candidate_movies = [int(m) for m in unique_movies if m not in rated_movies]
    
    # Get predictions
    predictions = []
    for mid in candidate_movies:
        try:
            pred = svd_model.predict(user_id, mid).est
            predictions.append((mid, pred))
        except:
            continue
    
    # Sort and get top N
    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    
    # Get movie titles
    movie_ids = [m[0] for m in top_n]
    top_titles = movies[movies['movieId'].isin(movie_ids)]
    
    # Create recommendation list with scores
    recommendations = []
    for movie_id, score in top_n:
        movie_title = movies[movies['movieId'] == movie_id]['title'].iloc[0]
        recommendations.append(html.Li(f"{movie_title} (Score: {score:.2f})"))
    
    return html.Ul(recommendations)

# Callback: Item Profile
@app.callback(Output('item-profile', 'children'), Input('item-dropdown', 'value'))
def show_item_profile(movie_id):
    if movie_id is None:
        return ""
    
    # Ensure movie_id is int
    movie_id = int(movie_id) if not isinstance(movie_id, int) else movie_id
    
    movie = movies[movies['movieId'] == movie_id].iloc[0]
    
    # Calculate average rating for the movie
    movie_ratings = ratings[ratings['movieId'] == movie_id]['rating']
    avg_rating = movie_ratings.mean() if len(movie_ratings) > 0 else 0
    num_ratings = len(movie_ratings)
    
    return html.Div([
        html.P(f"Title: {movie['title']}"),
        html.P(f"Genres: {movie['genres']}"),
        html.P(f"Average Rating: {avg_rating:.2f} ({num_ratings} ratings)")
    ])

# Improved item similarity function based on genres
def get_similar_items(movie_id, top_n=5):
    """Get similar movies based on genre overlap"""
    # Ensure movie_id is int
    movie_id = int(movie_id) if not isinstance(movie_id, int) else movie_id
    
    # Get target movie genres
    target_movie = movies[movies['movieId'] == movie_id].iloc[0]
    target_genres = set(target_movie['genres'].split('|'))
    
    # Calculate similarity scores for all other movies
    similarity_scores = []
    
    for _, movie in movies.iterrows():
        if movie['movieId'] != movie_id:
            movie_genres = set(movie['genres'].split('|'))
            # Jaccard similarity
            similarity = len(target_genres.intersection(movie_genres)) / len(target_genres.union(movie_genres))
            similarity_scores.append((movie['movieId'], similarity))
    
    # Sort by similarity and get top N
    similar_movie_ids = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:top_n]
    similar_movie_ids = [m[0] for m in similar_movie_ids]
    
    return movies[movies['movieId'].isin(similar_movie_ids)]

# Callback: Similar Items
@app.callback(
    Output('item-similarity', 'children'),
    Input('sim-button', 'n_clicks'),
    State('item-dropdown', 'value'),
    State('num-similar', 'value')
)
def similar_items(n_clicks, movie_id, n):
    if not n_clicks or movie_id is None:
        return ""
    
    similar = get_similar_items(movie_id, top_n=n)
    return html.Ul([
        html.Li(f"{row['title']} ({row['genres']})") 
        for _, row in similar.iterrows()
    ])

if __name__ == "__main__":
    app.run(debug=True)