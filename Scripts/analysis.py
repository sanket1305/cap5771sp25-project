import os
import pandas as pd
from io import StringIO
from sklearn.preprocessing import MinMaxScaler

# prints quick dimensions of any dataframe
def get_df_dimensions(df):
    rows, cols = df.shape
    print(f"Number of rows {rows} and Number of cols {cols}")

# prints top 5 records for each cols in dataframe (each column is being printed separately, because there are 6-7+ columns)
def get_top5_records_df_columnwise(df):
    cols_list = df.columns
    for col in cols_list:
        print(df[col].head())
        print()

# returns dimensions for each file in given dataset and it's list of columns
def get_dataset_dimensions(directory_path):
    # List all files and directories in the current directory
    files_in_directory = os.listdir(directory_path)
    for filename in files_in_directory:
        if filename.endswith(".csv"):
            df = pd.read_csv(f"{directory_path}/{filename}")
            print(f"current file is {filename}")
            num_rows, num_cols = df.shape
            print(f"Number of rows {num_rows}, Number of cols {num_cols}")
            print(df.columns)

# function to describe dataset, including column types and different statistics for numerical dtype
def describe_data(df):
    output = StringIO()     # Create a StringIO object to capture

    # print("***Describing the data:***", file=output)
    num_rows, num_columns = df.shape

    print(f"Number of rows: {num_rows}", file=output)
    print(f"Number of columns: {num_columns}", file=output)

    print("\nColumn details:", file=output)
    for column in df.columns:
        col_data = df[column]
        col_dtype = col_data.dtype
        print(f"\nColumn: {column}, Type: {col_dtype}", file=output)

        # if data type is numeric, then calculate min, max, mean, median
        if pd.api.types.is_numeric_dtype(col_data):
            min_val = col_data.min()
            max_val = col_data.max()

            mean_val = col_data.mean()
            median_val = col_data.median()

            print(f"  Min: {min_val}", file=output)
            print(f"  Max: {max_val}", file=output)
            print(f"  Mean: {mean_val:.2f}", file=output)
            print(f"  Median: {median_val}", file=output)
        elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
            # Note 1 in Learning.md
            # some columns like Type, Method, postcode, YearBuilt, Regionname can be treated as category ???
            num_categories = col_data.nunique()
            print(f"  Number of categories: {num_categories}", file=output)

            if num_categories <= 10:
                print("  Counts per category:", file=output)
                category_counts = col_data.value_counts()
                for index, value in category_counts.items():
                    print(f"   {index}: {value}", file=output)
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            # why "Date" column was treated as object above ???? should we convert it??
            min_date = col_data.min()
            max_date = col_data.max()
            print(f"  Date Range: {min_date} to {max_date}", file=output)
            print(f"  Number of unique dates: {col_data.nunique()}", file=output)
        else:
            unique_vals = col_data.unique()
            if len(unique_vals) <= 10:
                print("  Unique values:", file=output)
                for val in unique_vals:
                    print(f"    {val}", file=output)
    
    return output.getvalue()

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
    print(f"Dimensions of TMDB dataset after dropping unnecessary columns... {len(cols_to_be_dropped)} columns are dropped")
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
    cols_to_string_list = ["original_language", "original_title", "title", "id"]
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

    print("\n***** vote_average (rating) is important for us and we have a lot of data, so drop data which does not have rating *****")
    df_filtered = df[(df['vote_average'] != 0) & (df['vote_average'].notna())].copy()
    df_filtered = df[(df['vote_average'] != 0) & (df['vote_average'].notna()) & (df['release_date'].notna())].copy()

    # check null counts
    print("***** Displaying null counts (after rating filtering) *****\n")
    print(df_filtered.isnull().sum())

    print(df_filtered.shape)

    # majority of the columns does not have data for revenue and budget. Drop them
    print("***** Dropping Revenue, Budget, Original title and Genres (we get this from IMDB) columns *****\n")
    df_filtered.drop(columns=['revenue', 'budget', 'original_title', 'genres'], inplace=True)

    print(df_filtered.describe())
    print(df_filtered.isnull().sum())
    print(df_filtered.shape)

    print("***** Filling null values with default_values *****")
    df_filled = df_filtered.fillna({
        'title': 'Unknown',
        'original_title': 'Unknown',
        'genres': 'Unknown',
        'production_companies': 'Unknown',
        'production_countries': 'Unknown',
        'spoken_languages': 'Unknown'
    })

    print("Performing min-max scale for popularity column")
    scaler = MinMaxScaler(feature_range=(0, 10))
    df_filled['popularity'] = scaler.fit_transform(df_filled[['popularity']])

    print("Removing noisy data from runtime column")
    df_filled['runtime'] = df_filled['runtime'].apply(lambda x: 0 if x < 0 else x)

    print("The filled dataset is \n")
    print(df_filled.isnull().sum())
    print(df_filled.describe())

    print("The Final columns of the Dataset are:")
    print(df_filled.columns)

    get_top5_records_df_columnwise(df_filled)

    print("\n\n ********** Now let's normalise our dataframe **********")
    print("\n***** We will separate Movies, Production_companies and production_country datasets *****")

    # Step 1: Normalize the 'production_companies' column
    companies_normalized = df_filled[['id', 'production_companies']].copy()
    companies_normalized['production_companies'] = companies_normalized['production_companies'].str.split(',')
    companies_normalized = companies_normalized[['id', 'production_companies']].explode('production_companies')
    # companies_normalized['production_companies'] = companies_normalized['production_companies'].str.strip()

    # Step 2: Normalize the 'production_countries' column
    countries_normalized = df_filled[['id', 'production_countries']].copy()
    countries_normalized['production_countries'] = countries_normalized['production_countries'].str.split(',')
    countries_normalized = countries_normalized[['id', 'production_countries']].explode('production_countries')
    # countries_normalized['production_countries'] = countries_normalized['production_countries'].str.strip()

    # Merge back with the original movie DataFrame (excluding the columns to be removed)
    df_normalized = df_filled.drop(columns=['production_companies', 'production_countries'])

    # Merge back with the normalized DataFrames
    df_companies = pd.merge(companies_normalized, df_normalized[['id', 'title']], on='id', how='left')
    df_countries = pd.merge(countries_normalized, df_normalized[['id', 'title']], on='id', how='left')

    # Print the final DataFrames
    print("Original DataFrame (after removing production columns):")
    print(df_normalized)

    print("\nNormalized Production Companies DataFrame:")
    print(df_companies)

    print("\nNormalized Production Countries DataFrame:")
    print(df_countries)

    return df_filled

def imdb():
    directory_path = '../Data/IMDB'
    get_dataset_dimensions(directory_path)

    # analyse the columns of IMDB dataset
    directory_path = '../Data/IMDB'

    print("\n***** All IMDB files are separated by genre, but contains same columns *****")
    print("\n***** So let's merge them, for easy processing. Note: they do have 'genre' column *****")
    # List all files and directories in the current directory
    files_in_directory = os.listdir(directory_path)
    df_list = []
    for filename in files_in_directory:
        if filename.endswith(".csv"):
            df_list.append(pd.read_csv(f"{directory_path}/{filename}"))
    merged_df = pd.concat(df_list, ignore_index = True)
    print("\n***** Merged dataset dimensions are: *****")
    total_rows, total_cols = merged_df.shape
    print(f"Number of rows {total_rows} and Number of cols {total_cols}")

    print("\n***** Describing Data *****")
    print(describe_data(merged_df))

    print("***** Dropping unnecessary columns from IMDB dataset *****\n")
    print("Current Dimensions of TMDB Dataset")
    get_df_dimensions(merged_df)
    print()
    cols_to_be_dropped = ['certificate', 'description']
    merged_df.drop(columns=cols_to_be_dropped, inplace=True)
    print(f"Dimensions of TMDB dataset after dropping unnecessary columns... {len(cols_to_be_dropped)} columns are dropped")
    get_df_dimensions(merged_df)
    print()

    # let's analyse the data more by displaying couple of values of each column from dataframe
    print("***** Analyse columns based on data, identify their conversion type *****\n")
    get_top5_records_df_columnwise(merged_df)

    # notice that datatype for some columns needs to be changed.
    # details of this is given in Milestone1.pdf
    print("***** Aligning dataype for columns in process... *****")
    # correcting object to string datatype
    cols_to_string_list = ["movie_id", "movie_name"]
    for col in cols_to_string_list:
        merged_df[col] = merged_df[col].astype(pd.StringDtype())
    # correcting object to int datatype
    merged_df["year"] = pd.to_numeric(merged_df['year'], errors='coerce')
    merged_df['runtime'] = pd.to_numeric(merged_df['runtime'].str.replace(' min', ''), errors='coerce')
    
    print("***** Aligning dataype for columns completed... *****")

    # let's see the updated dataframe now
    print("***** Checking if all variables datatypes are correct now *****\n")
    print(describe_data(merged_df))

    print(merged_df.describe())

    # check null counts
    print("***** Displaying null counts *****\n")
    print(merged_df.isnull().sum())

    print(merged_df.shape)

    print("\n***** Rating is imporatnat for us and we have a lot of data, so drop data which does not have rating *****")
    df_filtered = merged_df[(merged_df['rating'] != 0) & (merged_df['rating'].notna())]

    # check null counts
    print("***** Displaying null counts (after rating filtering) *****\n")
    print(df_filtered.isnull().sum())

    print(df_filtered.shape)

    # majority of the columns does not have data for revenue and budget. Drop them
    print("***** Dropping Gross column, as it's not useful *****\n")
    df_filtered.drop(columns=['gross(in $)'], inplace=True)

    print(df_filtered.describe())
    print(df_filtered.isnull().sum())
    print(df_filtered.shape)

    print("***** Filling null values with default_values *****")
    df_filled = df_filtered.fillna({
        'runtime': 0,
        'director': 'Unknown',
        'director_id': 'Unknown',
        'star': 'Unknown',
        'star_id': 'Unknown'
    })

    print(df_filled.describe())
    print(df_filled.isnull().sum())
    print(df_filled.shape)

    print(df_filled.columns)

    print(df_filled['movie_id'].head())

    df_filled['movie_id'] = df_filled['movie_id'].str[2:]

    print(df_filled['movie_id'].head())

    return df_filled


def movielens():
    directory_path = '../Data/movielens/links.csv'
    
    movie_df = pd.read_csv(f"{directory_path}")
    print(movie_df.shape)
    print(describe_data(movie_df))

    directory_path = '../Data/movielens/tags.csv'

    tags_df = pd.read_csv(f"{directory_path}")
    print(tags_df.shape)
    print(describe_data(tags_df))

    merged_df = pd.merge(movie_df, tags_df, on='movieId')

    return merged_df