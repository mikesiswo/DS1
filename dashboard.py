import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column

# Define file paths for all CSV files
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

# Read CSV files into pandas DataFrames and store them in a dictionary
dfs = {}
for file_path in file_paths:
    file_name = file_path.split('.')[0]  # Extract the file name without extension
    dfs[file_name] = pd.read_csv(file_path)

# Now we have all the DataFrames stored in the 'dfs' dictionary
# We can proceed to create the dashboard using these DataFrames
# Below is a basic structure for creating a simple Bokeh dashboard

# Preprocces the data : 
# important columns for sales file : Transaction Date; Transaction Type; Product id; Sku Id; Buyer Country; Buyer Postal Code; Amount (Merchant Currency)
# important columns for crashes file : Date; Package Name; Daily Crashes; Daily ANRs
# important columns for retings_country file : Date; Package Name; Country; Daily Average Rating

# Dictionary to store preprocessed DataFrames
preprocessed_dfs = {}

# Preprocess each file
for file_path in file_paths:
    # Extract file name without extension
    file_name = file_path.split('.')[0]
    
    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File '{file_path}' not found. Skipping.")
        continue
    
    # Filter important columns based on file type
    if 'reviews' in file_name:
        df = df[['Date', 'Package Name', 'Country', 'Daily Average Rating']]
    elif 'sales' in file_name:
        df = df[['Transaction Date', 'Transaction Type', 'Product id', 'Sku Id', 
                 'Buyer Country', 'Buyer Postal Code', 'Amount (Merchant Currency)']]
    elif 'stats_crashes' in file_name:
        df = df[['Date', 'Package Name', 'Daily Crashes', 'Daily ANRs']]
    
    # Convert date column to datetime format
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    elif 'Transaction Date' in df.columns:
        df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    
    # Store preprocessed DataFrame in dictionary
    preprocessed_dfs[file_name] = df

# 1. Create Bokeh plots based on the data in 'dfs'
# For example:
# sales_plot = figure(title="Sales Volume over Time", x_axis_label="Date", y_axis_label="Sales")

# 2. Organize the plots into a layout
# layout = column(sales_plot)

# 3. Output the dashboard to an HTML file
# output_file("dashboard.html")
# show(layout)
