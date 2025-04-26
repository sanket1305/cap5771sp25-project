import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import re
import matplotlib.pyplot as plt

from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load saved datasets for UI visualization
df = pd.read_csv("saved_kmeans_models/movies_with_kmeans.csv")
df_hdbscan = pd.read_csv("saved_hdbscan_models/movies_with_hdbscan.csv")
X = pd.read_csv("saved_kmeans_models/kmeans_features.csv")
scaler = joblib.load("saved_kmeans_models/kmeans_scaler.pkl")
knn_model = joblib.load("saved_knn_model/knn_model.pkl")

# helper function for plotting graphs on UI
# returns the genre counts in sorted order
def get_genres_distribution(df):
    genre_counts = {}
    for genres in df['genres'].dropna():
        for genre in genres.split(','):
            genre = genre.strip()
            if genre in genre_counts:
                genre_counts[genre] += 1
            else:
                genre_counts[genre] = 1
    
    genre_df = pd.DataFrame({'Genre': list(genre_counts.keys()), 'Count': list(genre_counts.values())})
    genre_df = genre_df.sort_values('Count', ascending=False)
    
    return genre_df


# App Setup
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sub-header {
        font-size: 1.5rem !important;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.8;
    }
    .stButton > button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 8px 16px;
    }
    .stButton > button:hover {
        background-color: #E03C3C;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎥 Movie Recommender System")

# Database connection and data loading
DB_PATH = "../Data/movies.db"

@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    
    # Join imdb + genre + director + star tables
    query = """SELECT
        l.movieid,
        i.movie_name,
        i.description,
        i.rating AS imdb_rating,
        i.votes AS imdb_votes,
        i.runtime AS runtime,
        i.year AS year,
        t.vote_average AS tmdb_vote_average,
        t.vote_count AS tmdb_votes,
        t.original_language as language,
        t.popularity as popularity,
        t.release_year,
        t.budget as budget,
        t.revenue as revenue,
        GROUP_CONCAT(DISTINCT g.genre_name) AS genres,
        GROUP_CONCAT(DISTINCT d.director_name) AS directors,
        GROUP_CONCAT(DISTINCT s.star_name) AS stars,
        GROUP_CONCAT(DISTINCT p.production_companies_name) AS production_house
    FROM links l
    JOIN imdb i ON l.imdbid = i.movie_id
    LEFT JOIN tmdb t ON l.tmdbid = t.id
    LEFT JOIN genre_imdb gi ON i.movie_id = gi.movie_id
    LEFT JOIN genre g ON gi.genre_id = g.genre_id
    LEFT JOIN director_imdb di ON i.movie_id = di.movie_id
    LEFT JOIN director d ON di.director_id = d.director_id
    LEFT JOIN star_imdb si ON i.movie_id = si.movie_id
    LEFT JOIN star s ON si.star_id = s.star_id
    LEFT JOIN production_companies_tmdb pi ON t.id = pi.id
    LEFT JOIN production_companies p ON pi.production_companies_id = p.production_companies_id 
    WHERE language = 'English'
    GROUP BY l.movieid
    """
    
    df = pd.read_sql_query(query, conn)
    print(df.columns.tolist())
    conn.close()
    
    # Clean and prepare data
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
    df['imdb_rating'] = pd.to_numeric(df['imdb_rating'], errors='coerce')
    
    # Create a combined text field for recommendation system
    df['combined_features'] = df['movie_name'] + ' ' + df['description'].fillna('') + ' ' + \
                            df['genres'].fillna('') + ' ' + df['directors'].fillna('') + ' ' + \
                            df['stars'].fillna('')
    
    # Create decade column
    df['decade'] = (df['year'] // 10) * 10
    
    return df

# Contains DB data to be used for visualization
orig_df = load_data()

tabs = st.tabs(["Insights", "Movies Explorer", "Clusters", "Recommendations"])

# Data insights
with tabs[0]:
    col1, col2, col3 = st.columns(3)
    
    # first row 
    # display key stats like Total Movies, Year and Average IMDB rating
    with col1:
        st.markdown(f'<div class="metric-value">{len(orig_df):,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Movies</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown(f'<div class="metric-value">{int(orig_df["year"].max() - orig_df["year"].min())}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Years ({int(orig_df["year"].min())} - {int(orig_df["year"].max())})</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown(f'<div class="metric-value">{orig_df["imdb_rating"].mean():.1f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Average IMDb Rating</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 2nd row - plots begins
    # single plot - bar chart showing rating vs count distribution
    st.markdown('### Rating Distribution')
    
    fig = px.histogram(
        orig_df, 
        x="imdb_rating", 
        nbins=20,
        color_discrete_sequence=['#FF4B4B'],
        labels={"imdb_rating": "Rating", "count": "Number of Movies"},
        title=""
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 3rd - 4th row row - left plots 
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('### Movies By Decade')
        decade_counts = orig_df.groupby('decade').size().reset_index(name='count')
        decade_counts = decade_counts[decade_counts['decade'] >= 1900]
        
        # row 3 left plot - Movies by decade
        fig = px.bar(
            decade_counts, 
            x="decade", 
            y="count",
            labels={"decade": "Decade", "count": "Number of Movies"},
            color_discrete_sequence=['#1E88E5']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, tickmode='array', tickvals=decade_counts['decade']),
            yaxis=dict(showgrid=False),
            margin=dict(l=20, r=20, t=20, b=20),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

        # plot 4 left plot - Genre distribution
        st.markdown('### Genre Distribution')
        genre_df = get_genres_distribution(orig_df)
        top_genres = genre_df.head(10)
        
        fig = px.pie(
            top_genres, 
            values='Count', 
            names='Genre',
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.4
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=400
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
    
    with col2:
        st.markdown('### Runtime Distribution')
        
        # Filter out extreme outliers for better visualization
        runtime_df = orig_df[(orig_df['runtime'] > 0) & (orig_df['runtime'] < 300)]
        
        # row 3 right plot - runtime vs number of movies ditribution
        fig = px.histogram(
            runtime_df, 
            x="runtime", 
            nbins=30,
            color_discrete_sequence=['#A777E3'],
            labels={"runtime": "Runtime (minutes)", "count": "Number of Movies"}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            margin=dict(l=20, r=20, t=20, b=20),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('### Rating vs. Runtime')
            
        # Filter outliers for better visualization
        scatter_df = orig_df[(orig_df['runtime'] > 0) & (orig_df['runtime'] < 300) & (orig_df['imdb_rating'] > 0)]
        
        # row 4 right plot - rating vs runtime
        fig = px.scatter(
            scatter_df, 
            x="runtime", 
            y="imdb_rating",
            color="decade",
            color_continuous_scale='Viridis',
            opacity=0.7,
            # size="imdb_rating",
            hover_name="movie_name",
            labels={"runtime": "Runtime (minutes)", "imdb_rating": "IMDb Rating", "decade": "Decade"}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0.02)',
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
            margin=dict(l=20, r=20, t=20, b=20),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# Data Exploration
with tabs[1]:
    st.markdown('<h2 class="sub-header">Movie Explorer</h2>', unsafe_allow_html=True)
        
    # 1st section contains 2x3 grid
    # year range, genres, search field
    # rating, runtime, sort by
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        # year range
        min_year, max_year = int(orig_df['year'].min()), int(orig_df['year'].max())
        year_range = st.slider(
            "Year Range", 
            min_value=min_year, 
            max_value=max_year, 
            value=(min_year, max_year)
        )
        
        # rating
        min_rating, max_rating = 0.0, 10.0
        rating_range = st.slider(
            "IMDb Rating", 
            min_value=min_rating, 
            max_value=max_rating, 
            value=(5.0, max_rating), 
            step=0.1
        )
    
    with col2:
        # genre
        all_genres = sorted({g.strip() for sublist in orig_df['genres'].dropna().str.split(',') for g in sublist})
        selected_genres = st.multiselect("Select Genres", all_genres)
        
        # runtime
        runtime_options = ["Any", "Short (<60 min)", "Medium (60-120 min)", "Long (>120 min)"]
        selected_runtime = st.selectbox("Runtime", runtime_options)
    
    with col3:
        # search_field
        search_term = st.text_input("Search by Movie Title, Director, or Actor")
        
        # sort by
        sort_options = [
            "IMDb Rating (High to Low)", 
            "IMDb Rating (Low to High)", 
            "Year (Newest First)", 
            "Year (Oldest First)",
            "Runtime (Longest First)",
            "Runtime (Shortest First)"
        ]
        sort_by = st.selectbox("Sort by", sort_options)
    
    # Apply filters
    filtered_df = orig_df[(orig_df['year'] >= year_range[0]) & (orig_df['year'] <= year_range[1])]
    filtered_df = filtered_df[(filtered_df['imdb_rating'] >= rating_range[0]) & (filtered_df['imdb_rating'] <= rating_range[1])]
    
    # show results based on selected genres (there can be more than one)
    if selected_genres:
        filtered_df = filtered_df[filtered_df['genres'].apply(
            lambda x: any(g in x.split(',') for g in selected_genres) if pd.notnull(x) else False)]
    
    # show results based on runtime flag
    if selected_runtime != "Any":
        if selected_runtime == "Short (<60 min)":
            filtered_df = filtered_df[filtered_df['runtime'] < 60]
        elif selected_runtime == "Medium (60-120 min)":
            filtered_df = filtered_df[(filtered_df['runtime'] >= 60) & (filtered_df['runtime'] <= 120)]
        else:  # Long
            filtered_df = filtered_df[filtered_df['runtime'] > 120]
    
    # show results based on search term entered by user
    if search_term:
        search_lower = search_term.lower()
        filtered_df = filtered_df[
            filtered_df['movie_name'].str.lower().str.contains(search_lower, na=False) |
            filtered_df['directors'].str.lower().str.contains(search_lower, na=False) |
            filtered_df['stars'].str.lower().str.contains(search_lower, na=False)
        ]
    
    # Apply sorting based on drop down choice
    if sort_by == "IMDb Rating (High to Low)":
        filtered_df = filtered_df.sort_values('imdb_rating', ascending=False)
    elif sort_by == "IMDb Rating (Low to High)":
        filtered_df = filtered_df.sort_values('imdb_rating', ascending=True)
    elif sort_by == "Year (Newest First)":
        filtered_df = filtered_df.sort_values('year', ascending=False)
    elif sort_by == "Year (Oldest First)":
        filtered_df = filtered_df.sort_values('year', ascending=True)
    elif sort_by == "Runtime (Longest First)":
        filtered_df = filtered_df.sort_values('runtime', ascending=False)
    elif sort_by == "Runtime (Shortest First)":
        filtered_df = filtered_df.sort_values('runtime', ascending=True)
    
    # Display results with metrics
    result_count = len(filtered_df)
    st.markdown(f"### Found {result_count} movies matching your criteria")
    
    if result_count > 0:
        # calculate avg rating of the result
        avg_rating = filtered_df['imdb_rating'].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Rating", f"{avg_rating:.1f}")
        col2.metric("Average Runtime", f"{filtered_df['runtime'].mean():.0f} min")
        col3.metric("Most Common Year", f"{int(filtered_df['year'].mode().iloc[0])}")
        
        # Display movie grid
        display_count = min(100, result_count)  # Limit to 100 to avoid performance issues
        
        st.markdown(f"### Showing top {display_count} results")
        
        # Create a grid view of movies
        cols_per_row = 3
        rows = (display_count + cols_per_row - 1) // cols_per_row
        
        results_to_show = filtered_df.head(display_count)
        
        # build matrix grid to dsplay the results
        for row in range(rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                movie_idx = row * cols_per_row + col_idx
                if movie_idx < display_count:
                    movie = results_to_show.iloc[movie_idx]
                    
                    with cols[col_idx]:
                        st.markdown(f"""
                        <div style="padding: 10px; border-radius: 5px; border: 1px solid #ddd; height: 200px; overflow: hidden;">
                            <h4>{movie['movie_name']} ({int(movie['year'])})</h4>
                            <p><strong>⭐ {movie['imdb_rating']}</strong> | ⏱️ {int(movie['runtime'])} min</p>
                            <p><small>{movie['genres']}</small></p>
                            <p style="font-size: 0.8rem; color: #666; height: 80px; overflow: hidden;">
                                {movie['description'][:150]}...
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.info("No movies found matching your criteria. Try adjusting your filters.")

# Data Exploration by clusters
with tabs[2]:
    st.header("Explore Movies by Cluster")

    # Display kmeans cluster groupings
    cluster_option = st.selectbox("Select Cluster ID (KMeans):", sorted(df['kmeans_cluster'].unique()))
    cluster_df = df[df['kmeans_cluster'] == cluster_option][['title', 'rating', 'year', 'genres_raw']]
    st.write(f"Found {len(cluster_df)} movies in cluster {cluster_option}.")
    st.dataframe(cluster_df.sort_values(by='rating', ascending=False).reset_index(drop=True))

    # display hdbcan cluster groupings
    hdb_clusters = sorted(df_hdbscan['hdbscan_cluster'].unique())
    selected_hdb = st.selectbox("Select HDBSCAN Cluster ID:", hdb_clusters)

    # Filter out noise
    if selected_hdb == -1:
        st.warning("This is a noise cluster. May not contain meaningful groupings.")
    else:
        hdb_df = df_hdbscan[df_hdbscan['hdbscan_cluster'] == selected_hdb]
        st.write(f"{len(hdb_df)} movies found in HDBSCAN cluster {selected_hdb}")
        st.dataframe(hdb_df[['title', 'rating', 'year', 'genres_raw']])

# Recommendation System based on knn
with tabs[3]:
    st.header("Movie Recommendations")
    movie_input = st.selectbox("Select a movie to find similar ones:", sorted(df['title'].unique()))
    top_n = st.slider("Number of recommendations:", 3, 15, 5)

    if st.button("Get Recommendations"):
        idx = df[df['title'] == movie_input].index[0]
        query_vector = X.iloc[[idx]]
        distances, indices = knn_model.kneighbors(query_vector, n_neighbors=top_n + 1)
        result_indices = indices[0][1:]  # skip self
        recommendations = df.iloc[result_indices][['title', 'rating', 'year', 'genres_raw', 'stars_raw', 'directors_raw']]

        st.success(f"Here are {top_n} movies similar to '{movie_input}':")
        st.dataframe(recommendations.reset_index(drop=True))
