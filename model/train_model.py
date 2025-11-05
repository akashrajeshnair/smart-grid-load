from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, stddev
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import os
import shutil

# -------------------------------
# 1. Spark Session Setup
# -------------------------------
spark = SparkSession.builder.appName("SmartGridDataPreprocessing").getOrCreate()

# -------------------------------
# 2. Load Dataset
# -------------------------------
local_csv_path = r"file:///C:/Users/USER/Desktop/Projects/smart-grid-load/data/smart_grid_dataset.csv"
df = spark.read.csv(local_csv_path, header=True, inferSchema=True)

print(f"Columns Loaded: {df.columns}")

# -------------------------------
# 3. Identify Numeric Columns
# -------------------------------
numeric_cols = [c for c, t in df.dtypes if t in ['double', 'int']]
if not numeric_cols:
    raise RuntimeError("No numeric columns found in the dataset.")

# -------------------------------
# 4. Handle Missing Data
# -------------------------------
for col_name in numeric_cols:
    mean_val = df.select(mean(col(col_name))).first()[0]
    df = df.fillna({col_name: mean_val})

# -------------------------------
# 5. Cap Outliers at ±3σ
# -------------------------------
for col_name in numeric_cols:
    stats = df.select(mean(col(col_name)).alias('mean'), stddev(col(col_name)).alias('std')).first()
    lower, upper = stats['mean'] - 3 * stats['std'], stats['mean'] + 3 * stats['std']
    df = df.withColumn(col_name, when(col(col_name) < lower, lower)
                                  .when(col(col_name) > upper, upper)
                                  .otherwise(col(col_name)))

# -------------------------------
# 6. Normalize Numeric Features
# -------------------------------
for col_name in numeric_cols:
    stats = df.select(mean(col(col_name)).alias('mean'), stddev(col(col_name)).alias('std')).first()
    std = stats['std'] or 1.0  # prevent div by zero
    df = df.withColumn(col_name, (col(col_name) - stats['mean']) / std)

# -------------------------------
# 7. Save Cleaned Data to HDFS
# -------------------------------
hdfs_output_path = "hdfs://localhost:9000/user/akash/cleaned_smartgrid_data/"
df.write.mode("overwrite").parquet(hdfs_output_path)
print(f"Cleaned data saved to HDFS at: {hdfs_output_path}")

# -------------------------------
# 8. Train a Linear Regression Model
# -------------------------------
# Determine target column
target_candidates = ['target', 'Global_active_power', 'global_active_power', 'consumption', 'load']
target_col = next((col for col in target_candidates if col in df.columns), df.columns[-1])

if target_col not in df.columns:
    raise RuntimeError(f"Target column {target_col} not found in dataset columns.")

print(f"Using target column: {target_col}")

df = df.withColumn(target_col, col(target_col).cast(DoubleType()))

feature_cols = [c for c in numeric_cols if c != target_col]
if not feature_cols:
    raise RuntimeError("No valid feature columns found for training.")

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
dataset = assembler.transform(df).select("features", target_col).na.drop()

# Train/test split
train_df, test_df = dataset.randomSplit([0.8, 0.2], seed=42)

lr = LinearRegression(featuresCol="features", labelCol=target_col, maxIter=100, regParam=0.1)
lr_model = lr.fit(train_df)

# -------------------------------
# 9. Evaluate Model
# -------------------------------
predictions = lr_model.transform(test_df)
evaluator_rmse = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print(f"Linear Regression RMSE: {rmse:.4f}")
print(f"Linear Regression R2: {r2:.4f}")

# -------------------------------
# 10. Save Model Locally
# -------------------------------
model_dir = r"file:///C:/Users/USER/Desktop/Projects/smart-grid-load/model/models"

lr_model.save(model_dir)
print(f"Model saved at: {model_dir}")

# -------------------------------
# Finalize
# -------------------------------
spark.stop()
