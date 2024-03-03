import pandas as pd
import chardet
import os

file_paths = [
    'reviews_202106.csv', 'reviews_202107.csv', 'reviews_202108.csv', 'reviews_202109.csv',
    'reviews_202110.csv', 'reviews_202111.csv', 'reviews_202112.csv', 'sales_202106.csv',
    'sales_202107.csv', 'sales_202108.csv', 'sales_202109.csv', 'sales_202110.csv', 
    'sales_202111.csv', 'sales_202112.csv', 'stats_crashes_202106_overview.csv', 
    'stats_crashes_202107_overview.csv', 'stats_crashes_202108_overview.csv', 
    'stats_crashes_202109_overview.csv', 'stats_crashes_202110_overview.csv', 
    'stats_crashes_202111_overview.csv', 'stats_crashes_202112_overview.csv', 
    'stats_ratings_202106_country.csv', 'stats_ratings_202106_overview.csv', 
    'stats_ratings_202107_country.csv', 'stats_ratings_202107_overview.csv', 
    'stats_ratings_202108_country.csv', 'stats_ratings_202108_overview.csv', 
    'stats_ratings_202109_country.csv', 'stats_ratings_202109_overview.csv', 
    'stats_ratings_202110_country.csv', 'stats_ratings_202110_overview.csv', 
    'stats_ratings_202111_country.csv', 'stats_ratings_202111_overview.csv', 
    'stats_ratings_202112_country.csv', 'stats_ratings_202112_overview.csv'
]
    # Your file paths


# This dictionary will hold preprocessed DataFrames
preprocessed_dfs = {}

for file_path in file_paths:
    # Detect the encoding of the current file
    with open(file_path, 'rb') as file:
        encoding_result = chardet.detect(file.read())  # Read the entire file
        file_encoding = encoding_result['encoding']  # Extract the encoding
        print(f"File: {os.path.basename(file_path)}, Encoding: {file_encoding}")
    
    # Read the file into a DataFrame using the detected encoding
    df = pd.read_csv(file_path, encoding=file_encoding)

    # Extract file name without extension for use as a key in the dictionary
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    try:
    # Convert all column names to lowercase for case-insensitive matching
        df.columns = df.columns.str.lower()
        
        # Define a helper function to find a column name containing a specific substring
        def find_column_with_substring(df, substring):
            for column in df.columns:
                if substring in column:
                    return column
            return None  # Return None if no matching column is found

        # Apply conditional preprocessing based on file_name
        if 'stats_ratings_' in file_name and 'country' in file_name:
            df = df[['date', 'package name', 'country', 'daily average rating', "total average rating"]]
        elif 'sales' in file_name:
            # Find a column that contains "date"
            date_column = find_column_with_substring(df, "date")
            if date_column is None:
                raise KeyError("date")
            df = df[[date_column, 'transaction type', 'product id', 'sku id', 
                    'buyer country', 'buyer postal code', 'amount (merchant currency)']]
        elif 'stats_crashes' in file_name:
            df = df[['date', 'package name', 'daily crashes', 'daily anrs']]

        # Store the preprocessed DataFrame in the dictionary
        # Assuming preprocessed_dfs is a dictionary you've defined earlier
        preprocessed_dfs[file_name] = df

    except KeyError as e:
        print(f"KeyError: {e} in file {file_name}. Please check if the columns exist in the DataFrame.")
    

     # Convert date column to datetime format
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'order charged date' in df.columns:
        df['order charged date'] = pd.to_datetime(df['order charged date'])
    elif 'transaction date' in df.columns:
        df['transaction date'] = pd.to_datetime(df['transaction date'])

print(preprocessed_dfs)

#in welke column komt charged of google fee voor 
#als et refund moet je row weghalen