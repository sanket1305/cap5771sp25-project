import os
import pandas as pd
from io import StringIO

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
    directory_path = '../Data/TMDB'
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