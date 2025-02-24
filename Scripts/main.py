import sqlite3
import pandas as pd
from analysis import imdb, movielens, tmdb
from visualization import movie_language_analysis, tmdb_vote_correlation, ratings_correlation, genre_analysis, tag_analysis

# connect to the SQLite database
db_name = '../Data/movies.db'
conn = sqlite3.connect(db_name)

# analysis for IMDB Dataset
df_imdb = imdb()
df_imdb.rename(columns={'movie_id': 'imdbId'}, inplace=True)

# insert IMDB data to output table
df_imdb.to_sql('imdb', conn, if_exists='replace', index=False)
conn.commit()

# analysis for TMDB Dataset
df_tmdb = tmdb()
df_tmdb.rename(columns={'id': 'tmdbId'}, inplace=True)
df_tmdb.rename(columns={'title': 'movie_name'}, inplace=True)

# insert TMDB data to output table
df_tmdb.to_sql('tmdb', conn, if_exists='replace', index=False)
conn.commit()

# analysis for movielens Dataset
df_movielens = movielens()

# insert Movie Lens data to output table
df_movielens.to_sql('movielens', conn, if_exists='replace', index=False)
conn.commit()
conn.close()

print("\n***** Displaying Dimensions of datasets after preprocessing")
print(df_imdb.columns)
print(df_tmdb.columns)
print(df_movielens.columns)

# merge imdb and tmdb datasets to plot ratings correplations
imdb_tmdb = pd.merge(df_imdb, df_tmdb, on='movie_name')

# bar plot of most popular movie language (except english)
movie_language_analysis(df_tmdb)

# scatter plot of vote avg vs vote count correlation for some sample of data
tmdb_vote_correlation(df_tmdb)

# scatter plot of rating (imdb) correlation with vote_avg (tmdb)
ratings_correlation(imdb_tmdb)

# pie chart of top 10 genres
genre_analysis(df_imdb)

# bar plot of most popular user tags for movies
tag_analysis(df_movielens)