from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# -----------------------------
# Spark Session
# -----------------------------
spark = (SparkSession.builder
         .appName("MedicalCostsTraining")
         .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
         .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                 "com.amazonaws.auth.InstanceProfileCredentialsProvider")
         .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                 "com.amazonaws.auth.EnvironmentVariableCredentialsProvider")
         .getOrCreate())

# -----------------------------
# Load Data From S3
# -----------------------------
df = spark.read.csv("s3a://medical-costs-project/medical_costs.csv",
                    header=True, inferSchema=True)

# -----------------------------
# Categorical Conversion
# -----------------------------
categorical_cols = ["Sex", "Smoker", "Region"]
indexers = [StringIndexer(inputCol=c, outputCol=c + "_index").fit(df)
            for c in categorical_cols]

for indexer in indexers:
    df = indexer.transform(df)

# -----------------------------
# Assemble Features
# -----------------------------
feature_cols = ["Age", "BMI", "Children"] + [c + "_index" for c in categorical_cols]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)
print(df)

# -----------------------------
# Train/Test Split
# -----------------------------
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# -----------------------------
# Train And Evaluate Model
# -----------------------------
lr = LinearRegression(labelCol="Medical Cost", featuresCol="features")
model = lr.fit(train_df)

predictions = model.transform(test_df)


evaluator = RegressionEvaluator(labelCol="Medical Cost", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("RMSE:", rmse)

# -----------------------------
# Save Output For Spark API
# -----------------------------
predictions.select(
    "Age", "BMI", "Children", "Sex", "Smoker", "Region", "Medical Cost", "prediction"
).write.mode("overwrite").parquet("s3a://medical-costs-project/pred_output/")

model.write().overwrite().save("s3a://medical-costs-project/model/")

print("Training complete. Files saved.")