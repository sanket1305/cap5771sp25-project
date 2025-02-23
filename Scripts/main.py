import pandas as pd
import os
from analysis import get_dataset_dimensions, describe_data, get_df_dimensions, get_top5_records_df_columnwise
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

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

# TMDB Dataset
def tmdb():
    directory_path = '../Data/TMDB'
    get_dataset_dimensions(directory_path)

    # analyse the columns of TMDB dataset
    directory_path = '../Data/TMDB/TMDB.csv'
    df = pd.read_csv(f"{directory_path}")
    print(describe_data(df))

    # based on analysis so far, datatype for some columns needs to be changed
    # details given in Milestone1.pdf

    # drop columns which we won't be using
    print("***** Dropping unnecessary columns from TMDB dataset *****\n")
    print("Current Dimensions of TMDB Dataset")
    get_df_dimensions(df)
    print()
    cols_to_be_dropped = ['backdrop_path', 'homepage', 'keywords', 'overview', 'tagline',  'poster_path', 'imdb_id']
    df.drop(columns=cols_to_be_dropped, inplace=True)
    print("Dimensions of TMDB dataset after dropping unnecessary columns... 4 columns are dropped")
    get_df_dimensions(df)
    print()

    # let's analyse the data more by displaying couple of values of each column from dataframe
    print("***** Analyse columns based on data, identify their conversion type *****\n")
    get_top5_records_df_columnwise(df)

    # notice that datatype for some columns needs to be changed.
    # details of this is given in Milestone1.pdf
    print("***** Aligning dataype for columns in process... *****")
    # correcting date format
    df["release_date"] = pd.to_datetime(df["release_date"], format = "mixed" , errors='coerce')
    # correcting object to string datatype
    cols_to_string_list = ["original_language", "original_title", "title"]
    for col in cols_to_string_list:
        df[col] = df[col].astype(pd.StringDtype())
    # correcting object to category datatype
    df["status"] = df["status"].astype("category")
    print("***** Aligning dataype for columns completed... *****")
    
    # let's see the updated dataframe now
    print("***** Checking if all variables datatypes are correct now *****\n")
    print(describe_data(df))

    print(df.describe())

    # check null counts
    print("***** Displaying null counts *****\n")
    print(df.isnull().sum())

    # majority of the columns does not have data for revenue and budget. Drop them
    print("***** Dropping Revenue and Budget columns *****\n")
    df.drop(columns=['revenue', 'budget'], inplace=True)

    print(df.describe())
    print(df.isnull().sum())
    print(df.shape)

    print("***** Filling null values with default_values *****")
    df_filled = df.fillna({
        'title': 'Unknown',
        'original_title': 'Unknown',
        'genres': 'Unknown',
        'production_companies': 'Unknown',
        'production_countries': 'Unknown',
        'spoken_languages': 'Unknown'
    })

    print("Performing min-max scale for popularity column")
    scaler = MinMaxScaler(feature_range=(0, 10))
    df_filled['popularity'] = scaler.fit_transform(df[['popularity']])

    print("Removing noisy data from runtime column")
    df_filled['runtime'] = df['runtime'].apply(lambda x: 0 if x < 0 else x)

    print("The filled dataset is \n")
    print(df_filled.isnull().sum())
    print(df_filled.describe())

    print(df.columns)

    ## scatter plot for relationship between vote avg and vote count
    df_vote_count_vs_avg_vote = df[df['vote_count'] > 5000]
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=df_vote_count_vs_avg_vote['vote_average'], y=df_vote_count_vs_avg_vote['vote_count'])
    plt.title('Vote Average vs. Vote Count')
    plt.xlabel('Vote Average')
    plt.ylabel('Vote Count')
    plt.savefig('../Images/vote_avg_vs_cote_count.png')

    print("sanyeeeeeeeeeeee")

    ## bar plot for language distributions
    language_counts = df[df['original_language'] != 'en'].copy()  # Create a copy to avoid the warning
    language_counts.loc[:, 'original_language'] = language_counts['original_language'].map(language_dict)
    language_counts = language_counts['original_language'].value_counts().head(20)
    # print("chaavvvvaaaaaaaaaaaaaaaaaa")
    plt.figure(figsize=(10,6))
    sns.barplot(x=language_counts.index, y=language_counts.values)
    plt.title('Distribution of Movies by Language')
    plt.xlabel('Language')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../Images/lang_counts.png')


tmdb()