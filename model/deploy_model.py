from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, when
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.feature import VectorAssembler

# --------------------------------------------------
# 1. Spark Session Setup
# --------------------------------------------------
spark = SparkSession.builder \
    .appName("SmartGridModelDeployment") \
    .config("spark.sql.adaptive.enabled", "false") \
    .config("spark.streaming.stopGracefullyOnShutdown", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# --------------------------------------------------
# 2. Load Trained Spark ML Model
# --------------------------------------------------
model_path = "file:///C:/Users/USER/Desktop/Projects/smart-grid-load/model/models"
model = LinearRegressionModel.load(model_path)
print("[OK] Model loaded successfully from Spark ML.")
# Log expected feature size to help catch mismatches early
try:
    print(f"[INFO] Model expects numFeatures = {model.numFeatures}")
except Exception:
    pass

# --------------------------------------------------
# 3. Define Input Schema (same as before)
# --------------------------------------------------
schema = StructType([
    StructField("Timestamp", StringType(), True),
    StructField("Voltage (V)", DoubleType(), True),
    StructField("Current (A)", DoubleType(), True),
    StructField("Power Consumption (kW)", DoubleType(), True),
    StructField("Reactive Power (kVAR)", DoubleType(), True),
    StructField("Power Factor", DoubleType(), True),
    StructField("Solar Power (kW)", DoubleType(), True),
    StructField("Wind Power (kW)", DoubleType(), True),
    StructField("Grid Supply (kW)", DoubleType(), True),
    StructField("Voltage Fluctuation (%)", DoubleType(), True),
    StructField("Overload Condition", StringType(), True),
    StructField("Transformer Fault", StringType(), True),
    StructField("Temperature (°C)", DoubleType(), True),
    StructField("Humidity (%)", DoubleType(), True),
    StructField("Electricity Price (USD/kWh)", DoubleType(), True),
])

# --------------------------------------------------
# 4. Kafka Source
# --------------------------------------------------
kafka_topic = "smartgrid"
kafka_bootstrap = "localhost:9092"

stream_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_bootstrap) \
    .option("subscribe", kafka_topic) \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load()

# Convert Kafka JSON to structured DataFrame
json_df = stream_df.selectExpr("CAST(value AS STRING) as json_value") \
    .select(from_json(col("json_value"), schema).alias("data")) \
    .select("data.*")

# --------------------------------------------------
# 5. Prepare Features (align to training)
#    Cast the two categorical flags to numeric (0/1) to match training.
# --------------------------------------------------
json_df_casted = (
    json_df
    .withColumn(
        "Overload Condition",
        when(col("Overload Condition").isin("1", "Yes", "YES", "True", "true", "Y"), 1.0).otherwise(0.0)
    )
    .withColumn(
        "Transformer Fault",
        when(col("Transformer Fault").isin("1", "Yes", "YES", "True", "true", "Y"), 1.0).otherwise(0.0)
    )
)

# IMPORTANT: Keep the column order the same as used in training.
# This list now includes the two binary flags, giving 14 features.
numeric_cols = [
    "Voltage (V)", "Current (A)", "Power Consumption (kW)",
    "Reactive Power (kVAR)", "Power Factor", "Solar Power (kW)",
    "Wind Power (kW)", "Grid Supply (kW)", "Voltage Fluctuation (%)",
    "Overload Condition", "Transformer Fault",
    "Temperature (°C)", "Humidity (%)", "Electricity Price (USD/kWh)"
]

json_df_filled = json_df_casted.fillna(0, subset=numeric_cols)

assembler = VectorAssembler(
    inputCols=numeric_cols,
    outputCol="features",
    handleInvalid="skip"
)

feature_df = assembler.transform(json_df_filled)

# --------------------------------------------------
# 6. Apply Model for Prediction
# --------------------------------------------------
pred_df = model.transform(feature_df) \
    .select("Timestamp", *numeric_cols, col("prediction").alias("Predicted Load (kW)"))

# --------------------------------------------------
# 7. Write Predictions to HDFS
# --------------------------------------------------
query = pred_df.writeStream \
    .outputMode("append") \
    .format("json") \
    .option("checkpointLocation", "hdfs://localhost:9000/user/akash/checkpoints/predictions/") \
    .option("path", "hdfs://localhost:9000/user/akash/predictions/") \
    .start()

query.awaitTermination()