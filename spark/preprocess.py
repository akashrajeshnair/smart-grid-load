from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, stddev
from pyspark.sql.types import DoubleType, TimestampType

spark = SparkSession.builder.appName("SmartGridDataPreprocessing").getOrCreate()

# Read local CSV
df = spark.read.csv("file:///home/akashnair/Projects/smart-grid-load/data/smart_grid_dataset.csv", header=True, inferSchema=True)

# Print columns for sanity
print("Columns:", df.columns)

# Handle missing data
numeric_cols = [c for c, t in df.dtypes if t in ['double', 'int']]
for c in numeric_cols:
    mean_val = df.select(mean(col(c))).collect()[0][0]
    df = df.fillna({c: mean_val})

# Handle outliers (cap at ±3σ)
for c in numeric_cols:
    stats = df.select(mean(col(c)).alias('mean'), stddev(col(c)).alias('std')).collect()[0]
    lower, upper = stats['mean'] - 3 * stats['std'], stats['mean'] + 3 * stats['std']
    df = df.withColumn(c, when(col(c) < lower, lower).when(col(c) > upper, upper).otherwise(col(c)))

# Normalize numeric features
for c in numeric_cols:
    stats = df.select(mean(col(c)).alias('mean'), stddev(col(c)).alias('std')).collect()[0]
    df = df.withColumn(c, (col(c) - stats['mean']) / stats['std'])

# Write cleaned data to HDFS for the ML phase
df.write.mode("overwrite").parquet("hdfs:///user/akash/cleaned_smartgrid_data/")

spark.stop()

