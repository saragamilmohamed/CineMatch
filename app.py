import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import pickle
import numpy as np
from surprise import Dataset, Reader
from surprise import SVD
import plotly.express as px
import plotly.graph_objects as go

# Load data
ratings = pd.read_csv("Data/ratings.csv")
movies = pd.read_csv("Data/movies.csv")

# Load trained model
with open("models/svd_model.pkl", "rb") as f:
    svd_model = pickle.load(f)

# Get unique users and movies
unique_users = ratings['userId'].unique()
unique_movies = movies['movieId'].unique()

# Create some analytics
total_users = len(unique_users)
total_movies = len(unique_movies)
total_ratings = len(ratings)
avg_rating = ratings['rating'].mean()

# Create Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "🎬 CineMatch - Movie Recommender"

# Main Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1("🎬 CineMatch", className="header-title"),
            html.P("Discover Your Next Favorite Movie", className="header-subtitle"),
        ], className="header-content"),
        
        # Statistics Cards
        html.Div([
            html.Div([
                html.H3(f"{total_users:,}", className="stat-number"),
                html.P("Active Users", className="stat-label")
            ], className="stat-card"),
            
            html.Div([
                html.H3(f"{total_movies:,}", className="stat-number"),
                html.P("Movies", className="stat-label")
            ], className="stat-card"),
            
            html.Div([
                html.H3(f"{total_ratings:,}", className="stat-number"),
                html.P("Total Ratings", className="stat-label")
            ], className="stat-card"),
            
            html.Div([
                html.H3(f"{avg_rating:.1f}⭐", className="stat-number"),
                html.P("Avg Rating", className="stat-label")
            ], className="stat-card"),
        ], className="stats-container"),
        
    ], className="header-section"),
    
    # Navigation Tabs
    html.Div([
        dcc.Tabs(id="tabs", value='user', className="custom-tabs", children=[
            dcc.Tab(label='👤 User Recommendations', value='user', className="custom-tab"),
            dcc.Tab(label='🎭 Movie Similarity', value='item', className="custom-tab"),
            dcc.Tab(label='📊 Analytics', value='analytics', className="custom-tab"),
        ]),
    ], className="tabs-container"),
    
    # Content Area
    html.Div(id='tabs-content', className="content-area")
    
], className="main-container")

# User page layout
user_page = html.Div([
    html.Div([
        html.Div([
            html.H2("🎯 Get Personalized Recommendations", className="section-title"),
            html.P("Select a user to see their viewing history and get tailored movie suggestions", 
                   className="section-description"),
            
            html.Div([
                html.Label("Select User:", className="input-label"),
                dcc.Dropdown(
                    id='user-dropdown',
                    options=[{'label': f'User {int(u)}', 'value': int(u)} for u in unique_users[:100]], 
                    placeholder="Choose a user...",
                    className="custom-dropdown"
                ),
            ], className="input-group"),
            
            html.Div([
                html.Label("Number of Recommendations:", className="input-label"),
                dcc.Input(
                    id='num-recs', 
                    type='number', 
                    placeholder='5', 
                    value=5, 
                    min=1, 
                    max=20,
                    className="custom-input"
                ),
            ], className="input-group"),
            
            html.Button("🚀 Get Recommendations", id='rec-button', className="action-button"),
            
        ], className="control-panel"),
        
        html.Div([
            html.Div([
                html.H3("📚 Viewing History", className="subsection-title"),
                html.Div(id='user-history', className="history-container"),
            ], className="history-section"),
            
            html.Div([
                html.H3("🎬 Recommended Movies", className="subsection-title"),
                html.Div(id='user-recommendations', className="recommendations-container"),
            ], className="recommendations-section"),
        ], className="results-panel"),
        
    ], className="user-page-container")
], className="page-content")

# Item page layout
item_page = html.Div([
    html.Div([
        html.Div([
            html.H2("🔍 Discover Similar Movies", className="section-title"),
            html.P("Find movies similar to your favorites based on genres and characteristics", 
                   className="section-description"),
            
            html.Div([
                html.Label("Select Movie:", className="input-label"),
                dcc.Dropdown(
                    id='item-dropdown',
                    options=[{'label': f"{row['title']}", 'value': int(row['movieId'])} 
                            for _, row in movies.head(1000).iterrows()], 
                    placeholder="Search for a movie...",
                    className="custom-dropdown"
                ),
            ], className="input-group"),
            
            html.Div([
                html.Label("Number of Similar Movies:", className="input-label"),
                dcc.Input(
                    id='num-similar', 
                    type='number', 
                    placeholder='5', 
                    value=5, 
                    min=1, 
                    max=20,
                    className="custom-input"
                ),
            ], className="input-group"),
            
            html.Button("🔎 Find Similar Movies", id='sim-button', className="action-button"),
            
        ], className="control-panel"),
        
        html.Div([
            html.Div([
                html.H3("🎭 Movie Details", className="subsection-title"),
                html.Div(id='item-profile', className="movie-profile"),
            ], className="profile-section"),
            
            html.Div([
                html.H3("🎬 Similar Movies", className="subsection-title"),
                html.Div(id='item-similarity', className="similarity-container"),
            ], className="similarity-section"),
        ], className="results-panel"),
        
    ], className="item-page-container")
], className="page-content")

# Analytics page layout
analytics_page = html.Div([
    html.H2("📊 Platform Analytics", className="section-title"),
    
    html.Div([
        html.Div([
            dcc.Graph(id='rating-distribution'),
        ], className="chart-container"),
        
        html.Div([
            dcc.Graph(id='genre-popularity'),
        ], className="chart-container"),
    ], className="charts-row"),
    
    html.Div([
        html.Div([
            dcc.Graph(id='user-activity'),
        ], className="chart-container"),
        
        html.Div([
            html.H3("🏆 Top Rated Movies", className="subsection-title"),
            html.Div(id='top-movies-table'),
        ], className="table-container"),
    ], className="charts-row"),
    
], className="analytics-page")

# Callback to render tabs
@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    if tab == 'user':
        return user_page
    elif tab == 'item':
        return item_page
    elif tab == 'analytics':
        return analytics_page

# Callback: User History
@app.callback(Output('user-history', 'children'), Input('user-dropdown', 'value'))
def show_user_history(user_id):
    if user_id is None:
        return html.P("Please select a user to view their history.", className="loading")
    
    user_data = ratings[ratings['userId'] == user_id].merge(movies, on='movieId')
    if len(user_data) == 0:
        return html.P("No viewing history found for this user.", className="loading")
    
    # Sort by rating (highest first)
    user_data = user_data.sort_values('rating', ascending=False)
    
    return html.Div([
        html.Div([
            html.Div([
                html.Span("⭐" * int(row['rating']), className="rating-stars"),
                html.H4(row['title'], className="movie-title"),
                html.P(f"Genres: {row['genres']}", className="movie-details"),
                html.P(f"Your Rating: {row['rating']}/5", className="movie-details")
            ], className="movie-card")
        ]) for _, row in user_data.head(10).iterrows()
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
        return html.P("Click 'Get Recommendations' to see personalized movie suggestions.", className="loading")
    
    if n is None:
        n = 5
    
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
    
    if not top_n:
        return html.P("Unable to generate recommendations for this user.", className="loading")
    
    # Create recommendation list with scores
    recommendations = []
    for movie_id, score in top_n:
        try:
            movie_title = movies[movies['movieId'] == movie_id]['title'].iloc[0]
            movie_genres = movies[movies['movieId'] == movie_id]['genres'].iloc[0]
            
            # Get average rating for the movie
            movie_ratings = ratings[ratings['movieId'] == movie_id]['rating']
            avg_rating = movie_ratings.mean() if len(movie_ratings) > 0 else 0
            
            recommendations.append(
                html.Div([
                    html.Div([
                        html.H4(movie_title, className="movie-title"),
                        html.P(f"Genres: {movie_genres}", className="movie-details"),
                        html.P(f"Avg Rating: {avg_rating:.1f}/5", className="movie-details"),
                        html.Span(f"Predicted Rating: {score:.1f}/5", className="predicted-rating")
                    ], className="movie-card")
                ])
            )
        except:
            continue
    
    return html.Div(recommendations)

# Callback: Item Profile
@app.callback(Output('item-profile', 'children'), Input('item-dropdown', 'value'))
def show_item_profile(movie_id):
    if movie_id is None:
        return html.P("Please select a movie to view its details.", className="loading")
    
    # Ensure movie_id is int
    movie_id = int(movie_id) if not isinstance(movie_id, int) else movie_id
    
    movie = movies[movies['movieId'] == movie_id].iloc[0]
    
    # Calculate average rating for the movie
    movie_ratings = ratings[ratings['movieId'] == movie_id]['rating']
    avg_rating = movie_ratings.mean() if len(movie_ratings) > 0 else 0
    num_ratings = len(movie_ratings)
    
    return html.Div([
        html.H3(movie['title'], style={'color': '#2c2c2c', 'marginBottom': '1rem', 'borderBottom': '2px solid #DC143C', 'paddingBottom': '0.5rem'}),
        html.Div([
            html.P([
                html.Strong("Genres: ", style={'color': '#2c2c2c'}), 
                html.Span(movie['genres'], style={'color': '#DC143C', 'fontWeight': '500'})
            ]),
            html.P([
                html.Strong("Average Rating: ", style={'color': '#2c2c2c'}), 
                html.Span(f"{avg_rating:.1f}/5", style={'color': '#DC143C', 'fontWeight': '600'}),
                html.Span("⭐" * int(round(avg_rating)), style={'marginLeft': '10px', 'color': '#ffd700'})
            ]),
            html.P([
                html.Strong("Number of Ratings: ", style={'color': '#2c2c2c'}), 
                html.Span(f"{num_ratings:,}", style={'color': '#DC143C', 'fontWeight': '500'})
            ])
        ], style={'backgroundColor': '#f8f9fa', 'padding': '1rem', 'borderRadius': '10px', 'border': '1px solid rgba(220, 20, 60, 0.2)'})
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
            if len(target_genres.union(movie_genres)) > 0:
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
        return html.P("Click 'Find Similar Movies' to discover movies with similar genres.", className="loading")
    
    if n is None:
        n = 5
    
    similar = get_similar_items(movie_id, top_n=n)
    
    if len(similar) == 0:
        return html.P("No similar movies found.", className="loading")
    
    return html.Div([
        html.Div([
            html.Div([
                html.H4(row['title'], className="movie-title"),
                html.P(f"Genres: {row['genres']}", className="movie-details"),
                # Calculate similarity percentage
                html.P(f"Similarity: High", className="movie-details", style={'color': '#DC143C', 'fontWeight': '600'})
            ], className="movie-card")
        ]) for _, row in similar.iterrows()
    ])

# Analytics callbacks
@app.callback(Output('rating-distribution', 'figure'), Input('tabs', 'value'))
def update_rating_distribution(tab):
    if tab != 'analytics':
        return {}
    
    fig = px.histogram(ratings, x='rating', title='Rating Distribution',
                      color_discrete_sequence=['#DC143C'])
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#2c2c2c',
        title_font_color='#2c2c2c',
        title_font_size=18,
        title_font_family="Arial Black"
    )
    fig.update_traces(marker_line_color='#8B0000', marker_line_width=1)
    return fig

@app.callback(Output('genre-popularity', 'figure'), Input('tabs', 'value'))
def update_genre_popularity(tab):
    if tab != 'analytics':
        return {}
    
    # Extract genres
    genres = []
    for genre_list in movies['genres']:
        genres.extend(genre_list.split('|'))
    
    genre_counts = pd.Series(genres).value_counts().head(10)
    
    fig = px.bar(x=genre_counts.index, y=genre_counts.values, 
                title='Top 10 Movie Genres', color_discrete_sequence=['#8B0000'])
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#2c2c2c',
        xaxis_title='Genre',
        yaxis_title='Number of Movies',
        title_font_color='#2c2c2c',
        title_font_size=18,
        title_font_family="Arial Black"
    )
    fig.update_traces(marker_line_color='#DC143C', marker_line_width=1)
    return fig

@app.callback(Output('user-activity', 'figure'), Input('tabs', 'value'))
def update_user_activity(tab):
    if tab != 'analytics':
        return {}
    
    user_counts = ratings.groupby('userId').size().reset_index(name='num_ratings')
    
    fig = px.histogram(user_counts, x='num_ratings', title='User Activity Distribution',
                      color_discrete_sequence=['#DC143C'])
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#2c2c2c',
        xaxis_title='Number of Ratings per User',
        yaxis_title='Number of Users',
        title_font_color='#2c2c2c',
        title_font_size=18,
        title_font_family="Arial Black"
    )
    fig.update_traces(marker_line_color='#8B0000', marker_line_width=1)
    return fig

@app.callback(Output('top-movies-table', 'children'), Input('tabs', 'value'))
def update_top_movies(tab):
    if tab != 'analytics':
        return ""
    
    # Calculate movie statistics
    movie_stats = ratings.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    
    movie_stats.columns = ['movieId', 'avg_rating', 'num_ratings']
    movie_stats = movie_stats[movie_stats['num_ratings'] >= 50]  # Filter movies with at least 50 ratings
    movie_stats = movie_stats.sort_values('avg_rating', ascending=False).head(10)
    
    # Merge with movie titles
    top_movies = movie_stats.merge(movies[['movieId', 'title', 'genres']], on='movieId')
    
    return dash_table.DataTable(
        data=top_movies.to_dict('records'),
        columns=[
            {'name': 'Movie', 'id': 'title'},
            {'name': 'Genres', 'id': 'genres'},
            {'name': 'Avg Rating', 'id': 'avg_rating', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Num Ratings', 'id': 'num_ratings', 'type': 'numeric'}
        ],
        style_cell={
            'textAlign': 'left', 
            'padding': '12px',
            'fontFamily': 'Arial',
            'border': '1px solid rgba(220, 20, 60, 0.2)'},
                
        style_header={
            'backgroundColor': '#DC143C',
            'color': 'white',
            'fontWeight': 'bold',
            'border': '1px solid #8B0000',
            'textAlign': 'center'
        },
        style_data={
            'backgroundColor': 'white',
            'color': '#2c2c2c',
            'border': '1px solid rgba(220, 20, 60, 0.1)'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f8f9fa'
            },
            {
                'if': {'state': 'selected'},
                'backgroundColor': 'rgba(220, 20, 60, 0.1)',
                'border': '1px solid #DC143C'
            }
        ],
        style_table={
            'borderRadius': '10px',
            'overflow': 'hidden',
            'boxShadow': '0 3px 10px rgba(0,0,0,0.1)'
        },
        page_size=10,
        sort_action="native",
        filter_action="native"
    )

if __name__ == "__main__":
    app.run(debug=True, port=8050)