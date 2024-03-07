import chardet
import os
import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource
from bokeh.transform import dodge
import numpy as np


#Amount(merchant currency) vs Charged amount : 
#moeten we alleen de euro values van charged amount nemen of alles converten maar dan hoe

#Transaction type vs Financial status : 
# charged vs google_fee vs google_fee_refund vs Refund. 
# welke moeten we precies gebruiken, alleen charged van beide columnen ?

#Hoe plaatsen we meerdere figuren op de html file

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
        # Preprocess rating country files
        if 'stats_ratings_' in file_name and 'country' in file_name:
            df = df[['date', 'package name', 'country', 'daily average rating', "total average rating"]]
            
        # Preprocess sale files
        elif 'sales' in file_name:
            # Find a column that contains "date"
            date_column = find_column_with_substring(df, "date")
            if date_column is None:
                raise KeyError("date")
            
            # Determine the actual transaction column name in the DataFrame
            transaction_column = next((col for col in ['transaction type', 'financial status'] if col in df.columns), None)
            if transaction_column is None:
                raise KeyError("Neither 'transaction type' nor 'financial status' found in index.")

            # Filter rows based on the transaction column
            df = df[df[transaction_column].isin(['Charge', 'Charged'])]
            # Filter rows based on the product ID
            df = df[df['product id'] == 'com.vansteinengroentjes.apps.ddfive']
            # Filter rows for the specific SKU IDs ('unlockcharactermanager' and 'premium') for the ddfive app
            valid_skus = ['unlockcharactermanager', 'premium']
            df = df[df['sku id'].isin(valid_skus)]
            # Determine the actual column name for buyer country
            buyer_country_column = next((col for col in ['buyer country', 'country of buyer'] if col in df.columns), None)
            # KeyError 
            if buyer_country_column is None:
                raise KeyError("Neither 'buyer country' nor 'country of buyer' found in index.")
            # Determine the actual column name for buyer postal code
            postal_code_column = next((col for col in ['buyer postal code', 'postal code of buyer'] if col in df.columns), None)
            # KeyError 
            if postal_code_column is None:
                raise KeyError("Neither 'buyer postal code' nor 'postal code of buyer' found in index.")
            # Determine the actual column name for amount
            amount_in_euros = next((col for col in ['amount (merchant currency)', 'charged amount'] if col in df.columns), None)  
            # if the column is 'charged amount' select only the rows that are in EUR
            if amount_in_euros is 'charged amount' :
                df = df[df['currency of sale'].isin (['EUR'])]
            # KeyError  
            if amount_in_euros is None:
                raise KeyError("Neither 'amount (merchant currency)' nor 'charged amount' found in index.")
            # Cleaned Dataset
            df = df[[date_column, transaction_column, 'product id', 'sku id', 
                buyer_country_column, postal_code_column, amount_in_euros]]
            
        # Preprocess crash files
        elif 'stats_crashes' in file_name:
            df = df[['date', 'package name', 'daily crashes', 'daily anrs']]

        # Store the preprocessed DataFrame in the dictionary
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

# Dictionary to store sums of 'amount (merchant currency)' for each sales file
# Dictionary to store sums of 'amount (merchant currency)' for each sales file
amount_sums = {}

# Dictionary to store number of transactions in each sales file
transactions = {}

# Initialize counter at 6 because we only have data files from the 6th month to 12th
counter = 6

# Iterate through preprocessed_dfs to calculate sum and count transactions for each sales file
for file_name, df in preprocessed_dfs.items():
    if 'sales' in file_name:
        # Check if 'amount (merchant currency)' column exists in the DataFrame
        if 'amount (merchant currency)' in df.columns:
            # Calculate the sum of 'amount (merchant currency)' column
            amount_sums[f"amount{counter}"] = int(df['amount (merchant currency)'].sum())
            # Count the number of transactions
            transactions[f"transactions{counter}"] = len(df)
            counter += 1
        # Check if 'charged amount' column exists in the DataFrame
        elif 'charged amount' in df.columns:
            # Remove commas from 'charged amount' column and convert to float
            df['charged amount'] = df['charged amount'].replace(',', '', regex=True).astype(float)
            # Calculate the sum of 'charged amount' column
            amount_sums[f"amount{counter}"] = int(df['charged amount'].sum())
            # Count the number of transactions
            transactions[f"transactions{counter}"] = len(df)
            counter += 1
    
months = [6,7,8,9,10,11,12]

Amount_EUR = [amount_sums['amount6'],amount_sums['amount7'],amount_sums['amount8'],
          amount_sums['amount9'],amount_sums['amount10'],amount_sums['amount11'],
          amount_sums['amount12']]

Transactions = [transactions['transactions6'],transactions['transactions7'],
                transactions['transactions8'],transactions['transactions9'],
                transactions['transactions10'],transactions['transactions11'],
                transactions['transactions12']]


# Output the visualization directly in the notebook
output_file('index.html')

# Create a figure with a datetime type x-axis
fig = figure(title='Sales Data - Amount(EUR)',
             height=400, width=700,
             x_axis_label='Months', y_axis_label='Amount(EUR)',
             x_minor_ticks=2, y_range=(0, 1500),
             toolbar_location=None)

fig.vbar(x=months, bottom=0, top=Amount_EUR, 
         color='blue', width=0.25,legend_label='Amount')

fig.line(x=months, y=Transactions, 
         color='red', line_width=1, legend_label='Transactions')


# Put the legend in the upper left corner
fig.legend.location = 'top_right'

show(fig)

# Dictionary to store total sum of 'amount (merchant currency)' for each country
amount_sums_per_country = {}

# Dictionary to store average rating for each country
average_rating_per_country = {}

# Iterate through preprocessed_dfs to calculate sum and average rating for each country
for file_name, df in preprocessed_dfs.items():
    if 'sales' in file_name:
        # Check if 'amount (merchant currency)' column exists in the DataFrame
        if 'amount (merchant currency)' in df.columns:
            # Group by country and calculate the sum of 'amount (merchant currency)' column
            country_sums = df.groupby(df['buyer country'].fillna(df['buyer country']))['amount (merchant currency)'].sum()
            # Update the total sum for each country
            for country, total_sum in country_sums.items():
                amount_sums_per_country[country] = amount_sums_per_country.get(country, 0) + total_sum
        # Check if 'charged amount' column exists in the DataFrame
        elif 'charged amount' in df.columns:
            # Remove commas from 'charged amount' column and convert to float
            df['charged amount'] = df['charged amount'].replace(',', '', regex=True).astype(float)
            # Group by country and calculate the sum of 'charged amount' column
            country_sums = df.groupby(df['country of buyer'].fillna(df['country of buyer']))['charged amount'].sum()
            # Update the total sum for each country
            for country, total_sum in country_sums.items():
                amount_sums_per_country[country] = amount_sums_per_country.get(country, 0) + total_sum
    elif 'ratings' in file_name and 'country' in df.columns:
        # Group by country and calculate the mean of 'total average rating' column
        country_avg_ratings = df.groupby('country')['total average rating'].mean()
        # Update the average rating for each country
        for country, avg_rating in country_avg_ratings.items():
            average_rating_per_country[country] = avg_rating


print(average_rating_per_country)
