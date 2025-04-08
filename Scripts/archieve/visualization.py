import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# map to be used for converting language codes to their language names
language_dict = {
    'fr': 'French',
    'es': 'Spanish',
    'de': 'German',
    'ja': 'Japanese',
    'zh': 'Chinese (zh)',
    'pt': 'Portuguese',
    'it': 'Italian',
    'nu': 'Nauruan',
    'ko': 'Korean',
    'cs': 'Czech',
    'nl': 'Dutch',
    'ar': 'Arabic',
    'sv': 'Swedish',
    'hi': 'Hindi',
    'tr': 'Turkish',
    'pl': 'Polish',
    'tl': 'Tagalog',
    'xx': 'Unknown',
    'da': 'Danish',
    'cn': 'Chinese (cn)'  # This can be treated the same as 'zh'
}

## bar plot for language distributions
def movie_language_analysis(df):
    # df copy is required, melse pandas may not work properly, which might lead to unexpected results
    language_counts = df[df['original_language'] != 'en'].copy()
    language_counts.loc[:, 'original_language'] = language_counts['original_language'].map(language_dict)
    language_counts = language_counts['original_language'].value_counts().head(20)

    plt.figure(figsize=(10,6))
    sns.barplot(x=language_counts.index, y=language_counts.values)

    plt.title('Distribution of Movies by Language')
    plt.xlabel('Language')
    plt.ylabel('Frequency')

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig('../Images/lang_counts.png')

## scatter plot for correlation between vote avg and vote count (tmdb_dataset)
def tmdb_vote_correlation(df):
    # get sample where rating avg is based on at least 5000 vote_count (adds credibility)
    df_vote_count_vs_avg_vote = df[df['vote_count'] > 5000]

    plt.figure(figsize=(10,6))
    sns.scatterplot(x=df_vote_count_vs_avg_vote['vote_average'], y=df_vote_count_vs_avg_vote['vote_count'])
    
    plt.title('Vote Average vs. Vote Count')
    plt.xlabel('Vote Average')
    plt.ylabel('Vote Count')

    plt.savefig('../Images/vote_avg_vs_vote_count.png')

## scatter plot for correlation between vote avg (tmdb) vs rating (imdb)
def ratings_correlation(df):
    # Although this plot comes up for random 50 records, each time it gives almost the same results
    # that the 2 pltforms ratings differs for couple of records in sample
    df = df.sample(n=150)
    correlation = df['rating'].corr(df['vote_average'])
    print(f"Correlation between rating and vote_avg: {correlation}")

    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='rating', y='vote_average')

    plt.title('Correlation between Rating and Vote Average')
    plt.xlabel('Rating')
    plt.ylabel('Vote Average')

    plt.savefig('../Images/vote_avg_tmdb_vs_rating_imdb.png')

## genre distrubtion pie chart
def genre_analysis(df):
    df['genre'] = df['genre'].str.split(', ')
    df_exploded = df.explode('genre')

    genre_counts = df_exploded['genre'].value_counts()

    # taking top 9 and 10th one will be everything else
    top_genres = genre_counts.head(9)
    other_genres = genre_counts.iloc[9:].sum()

    top_genres['other'] = other_genres

    plt.figure(figsize=(8, 6))
    top_genres.plot.pie(autopct='%1.1f%%', startangle=90, figsize=(7, 7))

    plt.title('Distribution of Genres in Movies')
    plt.ylabel('')  # Hides the 'Genres' label on the y-axis
    plt.savefig('../Images/Genre_analysis.png')

## most popular tags given by users
def tag_analysis(df):
    tag_counts = df['tag'].value_counts()
    # plotting graph only for top 15
    top_15 = tag_counts.head(15)

    plt.figure(figsize=(10, 6))
    top_15.plot(kind='bar', color='skyblue')

    plt.title('Top 15 Movie Tags')
    plt.xlabel('Tags')
    plt.ylabel('Count')
    plt.xticks(rotation=45) 
    plt.tight_layout()

    plt.savefig('../Images/most_popular_user_tags.png')