from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    hour, dayofweek, month, avg, corr, col
)
import sys

# Initialize Spark
spark = SparkSession.builder.appName("SmartGridEDA").getOrCreate()

# === Load cleaned or raw dataset ===
# Prefer cleaned parquet from HDFS if available
try:
    df = spark.read.parquet("hdfs://localhost:9000/user/akash/cleaned_smartgrid_data/")
    print("[OK] Loaded cleaned data from HDFS.")
except Exception as e:
    print("[WARNING] Could not find cleaned HDFS data, falling back to local CSV.")
    df = spark.read.csv(
        "C:\\Users\\USER\\Desktop\\Projects\\smart-grid-load\\data\\smart_grid_dataset.csv",
        header=True,
        inferSchema=True
    )

print("Columns:", df.columns)
print("Total records:", df.count())

# Ensure timestamp column exists and is parsed
if "Timestamp" in df.columns:
    df = df.withColumn("Timestamp", col("Timestamp").cast("timestamp"))
else:
    print("[ERROR] 'Timestamp' column not found. Check dataset headers.")
    sys.exit(1)

# === Basic descriptive stats ===
print("\n=== Summary Statistics ===")
df.describe().show()

# === Hourly Load Analysis ===
if "Power Consumption (kW)" in df.columns:
    hourly_load = (
        df.groupBy(hour("Timestamp").alias("Hour"))
          .agg(avg("Power Consumption (kW)").alias("Avg_Load_kW"))
          .orderBy("Hour")
    )
    print("\n=== Average Load by Hour ===")
    hourly_load.show(24)

    # Save result to HDFS
    hourly_load.write.mode("overwrite").csv("hdfs://localhost:9000/user/akash/eda_results/hourly_load/")
else:
    print("[WARNING] 'Power Consumption (kW)' column not found for hourly analysis.")

# === Daily Load Pattern (Day of Week) ===
daily_load = (
    df.groupBy(dayofweek("Timestamp").alias("DayOfWeek"))
      .agg(avg("Power Consumption (kW)").alias("Avg_Load_kW"))
      .orderBy("DayOfWeek")
)
print("\n=== Average Load by Day of Week ===")
daily_load.show()
daily_load.write.mode("overwrite").csv("hdfs://localhost:9000/user/akash/eda_results/daily_load/")

# === Monthly Load Trend ===
monthly_load = (
    df.groupBy(month("Timestamp").alias("Month"))
      .agg(avg("Power Consumption (kW)").alias("Avg_Load_kW"))
      .orderBy("Month")
)
print("\n=== Monthly Average Load ===")
monthly_load.show()
monthly_load.write.mode("overwrite").csv("hdfs://localhost:9000/user/akash/eda_results/monthly_load/")

# === Correlations (Power vs Environmental/Generation factors) ===
corr_targets = [
    ("Power Consumption (kW)", "Temperature (°C)"),
    ("Power Consumption (kW)", "Solar Power (kW)"),
    ("Power Consumption (kW)", "Wind Power (kW)"),
    ("Power Consumption (kW)", "Humidity (%)"),
    ("Power Consumption (kW)", "Electricity Price (USD/kWh)")
]

print("\n=== Correlations with Power Consumption ===")
for a, b in corr_targets:
    if a in df.columns and b in df.columns:
        val = df.stat.corr(a, b)
        print(f"Correlation({a}, {b}) = {val:.4f}")
    else:
        print(f"[WARNING] Missing column for correlation: {a} or {b}")

# Stop Spark
spark.stop()
