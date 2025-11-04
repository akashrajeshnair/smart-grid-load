from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
import os

# ----------------------------
# 1. Initialize Spark
# ----------------------------
spark = SparkSession.builder \
    .appName("SmartGridLoadForecasting") \
    .getOrCreate()

# ----------------------------
# 2. Load cleaned data from HDFS
# ----------------------------
# Adjust to the actual output path from Phase 3
hdfs_path = "hdfs://localhost:9000/user/akash/cleaned_smartgrid_data/"
df_spark = spark.read.parquet(hdfs_path)

print(f"✅ Loaded {df_spark.count()} records from {hdfs_path}")

# ----------------------------
# 3. Convert to Pandas for ML
# ----------------------------
df = df_spark.toPandas()

# Drop any remaining nulls
df = df.dropna()

# ----------------------------
# 4. Prepare features and target
# ----------------------------
target_col = "Predicted Load (kW)"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found. Available columns: {list(df.columns)}")

X = df.drop(columns=[target_col, "Timestamp"], errors="ignore")
y = df[target_col]

# ----------------------------
# 5. Split data
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 6. Train model
# ----------------------------
model = RandomForestRegressor(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ----------------------------
# 7. Evaluate
# ----------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"✅ Model trained successfully!")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# ----------------------------
# 8. Save model
# ----------------------------
os.makedirs("models", exist_ok=True)
model_path = "models/smartgrid_load_model.pkl"
joblib.dump(model, model_path)
print(f"💾 Model saved at {model_path}")

# ----------------------------
# 9. Stop Spark
# ----------------------------
spark.stop()
