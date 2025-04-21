from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import matplotlib.pyplot as plt

# Initialize Spark Session
spark = SparkSession.builder.appName("StockMarketAnalysis").getOrCreate()

# Define the path to the stock data CSV file
stock_data_path = 'datasets/AAPL(Apple)StockAnalysis.csv'

# Load the stock data into a DataFrame
#  - header=True:  Indicates that the CSV file has a header row
#  - inferSchema=True:  Automatically infers the data type for each column
stock_df = spark.read.csv(stock_data_path, header=True, inferSchema=True)

# --- Data Preparation for Plotting ---
# Select Date, Close Price, and Volume columns
plot_data = stock_df.select("Date", "Close", "Volume").rdd.collect()

# Extract data for plotting
dates = [row["Date"] for row in plot_data]         # Extract dates for x-axis
close_prices = [row["Close"] for row in plot_data]   # Extract closing prices for y-axis (price trend)
volumes = [row["Volume"] for row in plot_data]     # Extract volumes for y-axis (volume trend)

# --- Create Subplots for Price and Volume Analysis ---
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True) # Create a figure and a set of subplots (2 rows, 1 column), sharing the x-axis
fig.suptitle('Stock Price and Volume Analysis') # Set the main title for the entire figure

# --- Plotting Close Price Trend ---
axs[0].plot(dates, close_prices, label='Close Price') # Plot the closing prices against dates
axs[0].set_ylabel('Close Price') # Set the y-axis label for the price subplot
axs[0].set_title('Price Trend') # Set the title for the price subplot
axs[0].grid(True) # Enable grid lines for better readability
axs[0].legend() # Display the legend to identify the plotted line

# --- Plotting Trading Volume ---
axs[1].bar(dates, volumes, label='Volume', color='skyblue') # Create a bar plot for trading volume against dates
axs[1].set_ylabel('Volume') # Set the y-axis label for the volume subplot
axs[1].set_xlabel('Date') # Set the x-axis label for the volume subplot
axs[1].set_title('Trading Volume') # Set the title for the volume subplot
axs[1].grid(True) # Enable grid lines for better readability
axs[1].legend() # Display the legend to identify the bars
plt.xticks(rotation=45) # Rotate date labels to prevent overlapping
plt.tight_layout() # Adjust subplot parameters to provide reasonable spacing
plt.show() # Display the generated plots

# --- Function to Calculate Simple Moving Average (SMA) ---
def calculate_sma(df, column_name='Close', window=20):
    """
    Calculates the Simple Moving Average (SMA) for a specified column over a given window.
    """
    window_spec = Window.orderBy("Date").rowsBetween(-window + 1, 0) # Define a window spec - last 'window' days including current day, ordered by Date
    return df.withColumn(f'SMA_{window}', avg(column_name).over(window_spec)) # Calculate SMA using the defined window

# Calculate SMA for the 'Close' price with a 20-day window
stock_df_with_sma = calculate_sma(stock_df)

# Collect SMA values for plotting
sma_prices = stock_df_with_sma.select("SMA_20").rdd.flatMap(lambda x: x).collect() # Extract SMA values from DataFrame for plotting

# --- Function to Calculate Exponential Moving Average (EMA) ---
def calculate_ema(df, column_name='Close', window=20):
    """
    Calculates the Exponential Moving Average (EMA) for a specified column over a given window.
    """
    window_spec = Window.orderBy("Date").rowsBetween(-window + 1, 0) # Define a window spec - last 'window' days including current day, ordered by Date
    return df.withColumn(f'EMA_{window}', expr(f'avg({column_name}) OVER (ORDER BY Date ASC ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW)')) # Calculate EMA using SQL expression over the window

# Calculate EMA for the 'Close' price with a 20-day window
stock_df_with_ema = calculate_ema(stock_df)

# Collect EMA values for plotting
ema_prices = stock_df_with_ema.select("EMA_20").rdd.flatMap(lambda x: x).collect() # Extract EMA values from DataFrame for plotting

# --- Plotting Close Price, SMA, and EMA ---
plt.figure(figsize=(12, 6)) # Create a new figure for plotting
plt.plot(dates, close_prices, label='Close Price') # Plot closing prices
plt.plot(dates, sma_prices, label=f'SMA (20 days)', color='red') # Plot SMA values, color-coded red
plt.plot(dates, ema_prices, label=f'EMA (20 days)', color='green') # Plot EMA values, color-coded green
plt.title('Close Price vs. 20-Day SMA vs. 20-Day EMA') # Set the title for the plot
plt.xlabel('Date') # Set the x-axis label
plt.ylabel('Price') # Set the y-axis label
plt.xticks(rotation=45) # Rotate date labels for better readability
plt.legend() # Display the legend to identify each line
plt.grid(True) # Enable grid lines for better readability
plt.tight_layout() # Adjust subplot parameters for reasonable spacing
plt.show() # Display the plot

# Stop Spark Session
spark.stop()
