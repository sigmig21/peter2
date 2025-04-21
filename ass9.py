from pyspark.sql import SparkSession
from pyspark.sql.functions import year, col, max, to_date

# Step 1: Start Spark Session
spark = SparkSession.builder \
    .appName("MaxSnowfallFinder") \
    .master("local[*]") \
    .getOrCreate()

# Step 2: Load local CSV data
df = spark.read.option("header", True).option("inferSchema", True).csv("weather.csv")

# Step 3: Parse date column if necessary
df = df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))

# Step 4: Filter for a specific year
target_year = 2022
df_filtered = df.filter(year(col("date")) == target_year)

# Step 5: Get the maximum snowfall value
max_snowfall = df_filtered.agg(max("snowfall").alias("max_snow")).collect()[0]["max_snow"]

# Step 6: Filter the record(s) with the max snowfall
result_df = df_filtered.filter(col("snowfall") == max_snowfall)

# Step 7: Show result
result_df.select("station_id", "date", "snowfall").show()
