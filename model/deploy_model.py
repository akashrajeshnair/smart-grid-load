from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, udf
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, FloatType
import joblib
import pandas as pd

# -------------------------------
# 1. Spark Session Setup
# -------------------------------
spark = SparkSession.builder \
    .appName("SmartGridModelDeployment") \
    .config("spark.streaming.stopGracefullyOnShutdown", "true") \
    .getOrCreate()

# -------------------------------
# 2. Load Trained Model
# -------------------------------
model_path = "models/smartgrid_load_model.pkl"
model = joblib.load(model_path)
print("✅ Model loaded successfully.")

# -------------------------------
# 3. Define Input Schema
# -------------------------------
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
    StructField("Electricity Price (USD/kWh)", DoubleType(), True)
])

# -------------------------------
# 4. Kafka Source
# -------------------------------
kafka_topic = "smartgrid"
kafka_bootstrap = "localhost:9092"

stream_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_bootstrap) \
    .option("subscribe", kafka_topic) \
    .option("startingOffsets", "latest") \
    .load()

# Convert Kafka JSON → structured columns
json_df = stream_df.selectExpr("CAST(value AS STRING) as json_value") \
    .select(from_json(col("json_value"), schema).alias("data")) \
    .select("data.*")

# -------------------------------
# 5. Define UDF for Prediction
# -------------------------------
def predict_load(*cols):
    # Prepare DataFrame with all columns (including categorical)
    data = pd.DataFrame([cols], columns=[
        "Voltage (V)", "Current (A)", "Power Consumption (kW)",
        "Reactive Power (kVAR)", "Power Factor", "Solar Power (kW)",
        "Wind Power (kW)", "Grid Supply (kW)", "Voltage Fluctuation (%)",
        "Overload Condition", "Transformer Fault",
        "Temperature (°C)", "Humidity (%)", "Electricity Price (USD/kWh)"
    ])

    # Encode categorical variables as during training
    data["Overload Condition"] = data["Overload Condition"].map({"Yes": 1, "No": 0})
    data["Transformer Fault"] = data["Transformer Fault"].map({"Yes": 1, "No": 0})

    # Fill any unknown/null values with 0 (safe fallback)
    data.fillna(0, inplace=True)

    # Predict
    pred = model.predict(data)[0]
    return float(pred)

predict_udf = udf(predict_load, FloatType())

# -------------------------------
# 6. Apply Model Predictions
# -------------------------------
predict_df = json_df.withColumn(
    "Predicted Load (kW)",
    predict_udf(
        col("Voltage (V)"), col("Current (A)"), col("Power Consumption (kW)"),
        col("Reactive Power (kVAR)"), col("Power Factor"), col("Solar Power (kW)"),
        col("Wind Power (kW)"), col("Grid Supply (kW)"), col("Voltage Fluctuation (%)"),
        col("Overload Condition"), col("Transformer Fault"),
        col("Temperature (°C)"), col("Humidity (%)"), col("Electricity Price (USD/kWh)")
    )
)

# -------------------------------
# 7. Write Predictions to HDFS
# -------------------------------
query = predict_df.writeStream \
    .outputMode("append") \
    .format("parquet") \
    .option("checkpointLocation", "/user/akash/checkpoints/predictions/") \
    .option("path", "/user/akash/predictions/") \
    .trigger(processingTime="30 seconds") \
    .start()

query.awaitTermination()
