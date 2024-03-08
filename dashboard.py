import chardet
import os
import pandas as pd
import numpy as np
from bokeh.plotting import figure, save, show, output_file
from bokeh.transform import dodge,cumsum
from math import pi
from bokeh.models import AnnularWedge, ColumnDataSource, Legend, LegendItem, Plot, Range1d,LinearColorMapper
from bokeh.palettes import Category10,Category20c
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool, LabelSet
from math import pi, sin, cos
import geopandas as gpd
from bokeh.layouts import row
from bokeh.models import GeoJSONDataSource
from bokeh.models import WheelZoomTool

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
        
        # Preprocess rating overview files
        elif 'stats_ratings_' in file_name and 'overview' in file_name:
            # Convert 'daily average rating' column to numeric, coercing non-convertible values to NaN
            df['daily average rating'] = pd.to_numeric(df['daily average rating'], errors='coerce')
            # Remove rows with NaN values in both columns
            df = df.dropna(subset=['daily average rating'], how='any')
            df = df[['date', 'package name', 'daily average rating', "total average rating"]]

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
            if amount_in_euros == 'charged amount' :
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

# TASK 1 - SALES VOLUME

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


# # Put the legend in the upper left corner
fig.legend.location = 'top_right'
fig.legend.border_line_color ='black'
fig.legend.border_line_width = 1

# TASK 2 - ATTRIBUTE SEGMENTATION AND FILTERING

# Initialize an empty set to store unique countries
# Function to find column names that include "country"
def find_country_columns(columns):
    return [col for col in columns if 'country' in col]

# Function to find column name for sales amount
def find_sales_amount_column(columns):
    for col_name in ['amount (merchant currency)', 'charged amount']:
        if col_name in columns:
            return col_name
    return None

# Dictionary to store total sales amount by country
country_sales_totals = {}

for file_name, df in preprocessed_dfs.items():
    if 'sales' in file_name:
        # Identify the country column
        country_columns = find_country_columns(df.columns)
        if not country_columns:
            continue  # Skip if no country column found
        country_column = country_columns[0]  # Use the first matching country column
        
        # Identify the sales amount column
        amount_column = find_sales_amount_column(df.columns)
        if amount_column is None:
            continue  # Skip if no sales amount column found

        # Aggregate sales by country
        for _, row in df.iterrows():
            country = row[country_column]
            sale_amount = row[amount_column]
            if country in country_sales_totals:
                country_sales_totals[country] += sale_amount
            else:
                country_sales_totals[country] = sale_amount
 # Convert to DataFrame and sort
df = pd.DataFrame(list(country_sales_totals.items()), columns=['Country', 'Sales']).sort_values(by='Sales', ascending=False)

# Top 9 countries + "Other"
top_countries = df[:9].copy()
other_sales_total = df[9:]['Sales'].sum()
other = pd.DataFrame([["Other", other_sales_total]], columns=['Country', 'Sales'])
df_combined = pd.concat([top_countries, other], ignore_index=True)

# Calculate angles
total_sales = df_combined['Sales'].sum()
df_combined['Angle'] = df_combined['Sales'] / total_sales * 2 * pi

# Assign colors from Category10 palette
df_combined['Color'] = Category10[10]  # This palette has exactly 10 unique colors

# Plot setup
xdr = Range1d(start=-2, end=2)
ydr = Range1d(start=-2, end=2)
plot = Plot(x_range=xdr, y_range=ydr, title="Sales by Country (Top 9 + Other)", toolbar_location=None)

angles = df_combined.Angle.cumsum().tolist()
country_source = ColumnDataSource(dict(
    start=[0] + angles[:-1],
    end=angles,
    colors=df_combined['Color'],
))

glyph = AnnularWedge(x=0, y=0, inner_radius=1.3, outer_radius=1.8,
                     start_angle="start", end_angle="end",
                     line_color="white", line_width=3, fill_color="colors")
r = plot.add_glyph(country_source, glyph)

legend = Legend(location="center")
for i, country in enumerate(df_combined['Country']):
    legend.items.append(LegendItem(label=country, renderers=[r], index=i))
plot.add_layout(legend, "center")

valid_skus = ['unlockcharactermanager', 'premium']

# Initialize an empty DataFrame for aggregated sales volume
sku_sales_volume = pd.DataFrame(columns=['SKU', 'SalesVolume'])

# Function to find column name for sales amount
def find_sales_amount_column(columns):
    sales_columns = ['amount (merchant currency)', 'charged amount']
    found_columns = [col_name for col_name in sales_columns if col_name in columns]
    return found_columns

for file_name, df in preprocessed_dfs.items():
    if 'sales' in file_name:
        # Filter the DataFrame for only those rows where the 'sku id' matches the valid SKUs
        filtered_df = df[df['sku id'].isin(valid_skus)]
        
        # Check if the filtered_df is not empty
        if not filtered_df.empty:
            # Identify sales amount column(s)
            sales_columns = find_sales_amount_column(filtered_df.columns)
            
            # Sum the relevant sales amount columns if they exist
            if len(sales_columns) > 1:
                # If both columns exist, sum them
                filtered_df['TotalSales'] = filtered_df[sales_columns[0]] + filtered_df[sales_columns[1]]
            elif len(sales_columns) == 1:
                # If only one column exists, use it as is
                filtered_df['TotalSales'] = filtered_df[sales_columns[0]]
            else:
                # If neither column is present, proceed without modifying the DataFrame
                continue  # Or handle the case as needed
            
            # Aggregate sales by SKU
            agg_sales = filtered_df.groupby('sku id')['TotalSales'].sum().reset_index()
            agg_sales.columns = ['SKU', 'SalesVolume']
            sku_sales_volume = pd.concat([sku_sales_volume, agg_sales], ignore_index=True)

# Further aggregate in case of data from multiple files or duplicate SKUs across them
sku_sales_volume = sku_sales_volume.groupby('SKU')['SalesVolume'].sum().reset_index()


# Calculate total sales and angles for the pie (now donut) chart
total_sales = sku_sales_volume['SalesVolume'].sum()
sku_sales_volume['angle'] = sku_sales_volume['SalesVolume'] / total_sales * 2 * np.pi


# Manually assign colors to the SKUs
sku_color_map = {
    'unlockcharactermanager': 'blue',  # Example color
    'premium': 'red'  # Example color
}
sku_sales_volume['color'] = sku_sales_volume['SKU'].map(sku_color_map)

# Create a ColumnDataSource for the main chart
source = ColumnDataSource(sku_sales_volume)

# Create figure
p = figure(height=500, title="Sales Volume by SKU", toolbar_location=None,
           tools="", tooltips=None, x_range=(-0.5, 1.0))

# Adjust here for donut plot - Add annular wedges instead of wedges
p.annular_wedge(x=0.2, y=2, inner_radius=0.3, outer_radius=0.4,
                start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                line_color="white", fill_color='color', legend_field='SKU', source=source)

# Hide axis and grid
p.axis.axis_label=None
p.axis.visible=False
p.grid.grid_line_color = None

p.legend.orientation = "horizontal"
p.legend.location = "bottom_center"
p.legend.padding = 10  # Adds padding around the legend
p.legend.margin = 10  # Adds margin around the legend box
p.legend.label_text_font_size = '10pt'  # Adjust the font size of the legend text
p.legend.background_fill_alpha = 0.5  # Adds transparency to the legend background
p.legend.background_fill_color = "white"  # Sets the background color of the legend

# Optionally, if you want to make the legend interactive to hide the slices when clicked, you can do:
p.legend.click_policy = "hide"


# TASK 3 - RATING VS STABILITY

# Initialize lists to store dates, daily crashes, and daily average ratings
dates = []
crashes = []
ratings = []

# Iterate through preprocessed_dfs to extract data
for file_name, df in preprocessed_dfs.items():
    if 'rating' in file_name and 'overview' in file_name:
        if 'daily average rating' in df.columns:
            # Extract date and daily average rating data
            rating_data = df[['date', 'daily average rating']]
            # Store the daily average ratings data
            ratings.append(rating_data)
    
    elif 'crashes' in file_name:
        if 'daily crashes' in df.columns:
            # Extract date and daily crashes data
            crashes_data = df[['date', 'daily crashes']]
            # Store the daily crashes data
            crashes.append(crashes_data)

# Merge all crashes and ratings data frames based on the 'date' column
merged_crashes = pd.concat(crashes)
merged_ratings = pd.concat(ratings)
merged_data = pd.merge(merged_crashes, merged_ratings, on='date', how='inner')

# Extract crashes and ratings from the merged data
crashes = merged_data['daily crashes']
ratings = merged_data['daily average rating']

#Define the second figure
fig3 = figure(width=400, height=400, x_axis_label='Daily Crashes', y_axis_label='Daily Average Ratings',
              background_fill_color="#fafafa")
fig3.y_range.start = 0

#Change from circle to cross, and set color to red
fig3.circle(crashes, ratings, size=10, alpha=0.8, color='red', line_color="black", legend_label='Rating based on Crashes')

#Add and configure the legend
fig3.legend.location = 'bottom_right'
fig3.legend.border_line_color = 'black'
fig3.legend.border_line_width = 1

# TASK 4 - GEOGRAPHICAL DEVELOPMENT 

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


# Set the SHAPE_RESTORE_SHX option to YES to attempt restoration of .shx file
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# Set the path to the shapefile
world_shapefile_path = 'country_shapes/country_shapes.shp'

try:
    # Load the shapefile
    world = gpd.read_file(world_shapefile_path)

    # Create a figure
    p_top = figure(title='Top 10 Countries by Total Amount and Average Rating', tools="pan,wheel_zoom,box_zoom,reset,save")
    
    # Convert dictionaries to DataFrames
    amount_df = pd.DataFrame.from_dict(amount_sums_per_country, orient='index', columns=['total_amount'])
    rating_df = pd.DataFrame.from_dict(average_rating_per_country, orient='index', columns=['average_rating'])

    # Get top 15 countries for total amount
    top_amount_countries = amount_df.nlargest(10, 'total_amount')

    # Get ratings for the top 15 countries with the highest total amount
    top_rating_countries = rating_df.loc[top_amount_countries.index]

    # Merge top amount and rating data with shapefile using ISO2 codes
    world_merged = world.merge(top_amount_countries, how='inner', left_on='iso_a2', right_index=True)
    world_merged = world_merged.merge(top_rating_countries, how='inner', left_on='iso_a2', right_index=True)

    # Create GeoJSONDataSource for Bokeh plot
    world_source = GeoJSONDataSource(geojson=world_merged.to_json())
    world_source_uncolored = GeoJSONDataSource(geojson=world.to_json())
    
    # Define a color mapper based on the average rating
    color_mapper = LinearColorMapper(palette='Turbo256', 
                    low=top_rating_countries['average_rating'].min(), 
                    high=top_rating_countries['average_rating'].max())

    # Plot world map with color mapper
    world_plot = p_top.patches('xs', 'ys', source=world_source, 
                fill_color={'field': 'average_rating', 'transform': color_mapper}, 
                line_color='black', line_width=0.5)
    
    # Plot uncolored countries
    p_top.patches('xs', 'ys', source=world_source_uncolored,
                fill_color=None, line_color='black', line_width=0.5)
    
    # Create legend
    legend_items = []
    for i, (country, row) in enumerate(top_amount_countries.iterrows()):
        total_amount_int = int(row['total_amount'])  # Convert total amount to integer
        avg_rating_one_decimal = round(top_rating_countries.loc[country, 'average_rating'], 1)  # Round average rating to 1 decimal place
        legend_items.append(LegendItem(label=f"{country} (Total Amount: {total_amount_int}, Avg Rating: {avg_rating_one_decimal})", renderers=[world_plot], index=i))
    legend = Legend(items=legend_items, location='bottom_left')
    p_top.add_layout(legend)

except Exception as e:
    print("Error occurred:", e)
    

# Create a layout grid with both figures and their legends
layout = gridplot([[fig, fig3] ,[plot, p], [p_top]])

# Save the layout to the HTML file
output_file('index.html')
save(layout)
