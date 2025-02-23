import os
import pandas as pd

from io import StringIO

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
    
    return num_rows, num_columns, output.getvalue()

# function to analyse IMDB dataset
def imdb_analysis():
    directory_path = '../Data/IMDB'
    # List all files and directories in the current directory
    files_in_directory = os.listdir('../Data/IMDB')
    print("Files in IMDB Folder")
    print(files_in_directory)

    dic = {}
    total_rows = 0

    for filename in files_in_directory:
        if filename.endswith(".csv"):
            print(f"Describing Dataset {filename}")
            df = pd.read_csv(f"{directory_path}/{filename}")
            num_rows, num_cols, content = describe_data(df)
            # print(f"number of rows: {num_rows}, number of columns: {num_cols}")
            print(content)
            dic[filename] = num_rows
            total_rows += num_rows
    
    print(dic)
    print(total_rows)


# function to anlyse genre column of each file
def imdb_genre_analysis():
    # we will start with one file at first, let's take action.csv
    df = pd.read_csv(f"../Data/IMDB/action.csv")
    print(df["genre"].head(10))

    # notice that in describe_data we received this column as "object" type and now we can see it holds multiple genre values
    # we need to normalise it. But let's first check if all the records contains "action" as genre
    count = df["genre"].str.contains("Action", case=False).sum()
    filtered_df = df[~df["genre"].str.contains("Action", case=False)]
    rows, cols = df.shape

    print(count, rows, cols)
    print(filtered_df[["movie_name", "genre"]])
    # based on this analysis we found that even though files are separated by genre
    # so we need to explode this column and then merge all the datasets into one 
    # later we have to remove duplicates

# merge all datasets
def imdb_merge_datasets():
    directory_path = '../Data/IMDB'
    # List all files and directories in the current directory
    files_in_directory = os.listdir('../Data/IMDB')
    print("Files in IMDB Folder")
    print(files_in_directory)

    dic = {}
    total_rows = 0
    new_dic = {}
    new_total_rows = 0

    df_array = []

    for filename in files_in_directory:
        if filename.endswith(".csv"):
            print(f"Describing Dataset {filename}")
            df = pd.read_csv(f"{directory_path}/{filename}")
            # print(f"number of rows: {num_rows}, number of columns: {num_cols}")
            num_rows, num_cols = df.shape
            print(f"before {num_rows}")

            df["genre"] = df["genre"].str.split(',')
            exploded_df = df.explode('genre')
            new_rows, new_cols = exploded_df.shape
            print(f"after {new_rows}")

            dic[filename] = num_rows
            new_dic[filename] = new_rows

            total_rows += num_rows
            new_total_rows += new_rows

            df_array.append(exploded_df)
    
    merged_df = pd.concat(df_array, ignore_index=True)
    print(merged_df.shape)
    merged_df = merged_df.drop_duplicates()
    print(merged_df.shape)
    
    print(dic)
    print(total_rows)
    print(new_dic)
    print(new_total_rows)

    print(merged_df.head(20))
    print(merged_df.columns)
    print(merged_df[["star", "star_id"]])
    print(merged_df[["director", "director_id"]])

def imdb_merged_without_explode_analysis():
    directory_path = '../Data/IMDB'
    # List all files and directories in the current directory
    files_in_directory = os.listdir('../Data/IMDB')
    print("Files in IMDB Folder")
    print(files_in_directory)

    df_array = []

    for filename in files_in_directory:
        if filename.endswith(".csv"):
            print(f"Describing Dataset {filename}")
            df = pd.read_csv(f"{directory_path}/{filename}")
            df_array.append(df)
    
    merged_df = pd.concat(df_array, ignore_index=True)
    merged_df = merged_df.dropna(subset=['rating'])

    null_counts_columns = merged_df.isnull().sum()
    print("Null Counts per variable:")
    print(null_counts_columns)
    print(merged_df.shape)