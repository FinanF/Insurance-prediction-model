import pandas as pd
from flask import request, jsonify, Flask
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegressionModel
from pyspark.sql import SparkSession

# -----------------------------
# Flask API
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Spark Session
# -----------------------------
spark = (SparkSession.builder
         .appName("MedicalCostsApi")
         .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
         .config(
    "spark.hadoop.fs.s3a.aws.credentials.provider",
    "com.amazonaws.auth.EnvironmentVariableCredentialsProvider,"
    "com.amazonaws.auth.InstanceProfileCredentialsProvider"
)
         .getOrCreate())

# -----------------------------
# Model And Prediction Output Loading
# -----------------------------

model = LinearRegressionModel.load("s3a://medical-costs-project/model/")
pred_pd = pd.read_parquet("s3://medical-costs-project/pred_output/")
pred_output_json = pred_pd.to_dict(orient="records")

# -----------------------------
# Categorical Feature Embedding
# -----------------------------

sex_indexer = StringIndexer(inputCol="Sex", outputCol="Sex_index").fit(spark.createDataFrame([
    ("male",), ("female",)
], ["Sex"]))

smoker_indexer = StringIndexer(inputCol="Smoker", outputCol="Smoker_index").fit(spark.createDataFrame([
    ("yes",), ("no",)
], ["Smoker"]))

region_indexer = StringIndexer(inputCol="Region", outputCol="Region_index").fit(spark.createDataFrame([
    ("northwest",), ("northeast",), ("southwest",), ("southeast",)
], ["Region"]))

feature_cols = ["Age", "BMI", "Children", "Sex_index", "Smoker_index", "Region_index"]

# -----------------------------
# API Requests
# -----------------------------
@app.route("/",methods=["GET"])
def output():
    return jsonify(pred_output_json)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    df = spark.createDataFrame([[
        int(data["age"]),
        float(data["bmi"]),
        int(data["children"]),
        data["sex"],
        data["smoker"],
        data["region"]
    ]], ["Age", "BMI", "Children", "Sex", "Smoker", "Region"])

    # Transform categorical features
    df = sex_indexer.transform(df)
    df = smoker_indexer.transform(df)
    df = region_indexer.transform(df)

    # Assemble features
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)

    # Predict
    pred = model.transform(df).collect()[0]["prediction"]
    return jsonify({"prediction": float(pred)})

app.run(host="0.0.0.0", port=5000)