from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import streamlit as st

spark = SparkSession.builder \
    .appName("MedicalCostsCloud") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.InstanceProfileCredentialsProvider") \
    .getOrCreate()

df = spark.read.csv("s3a://medical-costs-project/medical_costs.csv", header=True, inferSchema=True)
df.printSchema()
df.show(5)

# -----------------------------
# Convert categorical features
# -----------------------------
categorical_cols = ["Sex", "Smoker", "Region"]
indexers = [StringIndexer(inputCol=c, outputCol=c + "_index") for c in categorical_cols]
for indexer in indexers:
    df = indexer.fit(df).transform(df)

# -----------------------------
# Assemble features
# -----------------------------
feature_cols = ["Age", "BMI", "Children"] + [c + "_index" for c in categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)
df.select("features", "Medical Cost").show(5)

# -----------------------------
# Train-test split
# -----------------------------
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# -----------------------------
# Linear Regression
# -----------------------------
lr = LinearRegression(labelCol="Medical Cost", featuresCol="features")
model = lr.fit(train_df)
predictions = model.transform(test_df)

# -----------------------------
# Evaluate Regression
# -----------------------------
re = RegressionEvaluator(labelCol="Medical Cost", predictionCol="prediction", metricName="rmse")
rmse = re.evaluate(predictions)
print("Root Mean Square Error:", rmse)

# -----------------------------
# Streamlit Dashboard
# -----------------------------
pred_pd = predictions.select(
    "Age", "BMI", "Children", "Sex", "Smoker", "Region", "Medical Cost", "prediction"
).toPandas()

st.title("Medical Cost Prediction Dashboard")
st.subheader("Predictions Table")
st.dataframe(pred_pd.head(20))

st.subheader("Scatter plot: Actual vs Predicted")
st.scatter_chart(pred_pd[["Medical Cost", "prediction"]])

st.subheader("Filter by Smoker Status")
smoker_status = st.selectbox("Smoker?", ["all"] + pred_pd["Smoker"].unique().tolist())
if smoker_status != "all":
    st.dataframe(pred_pd[pred_pd["Smoker"] == smoker_status])

st.subheader("Filter by Age")
age_selected = st.slider(
    "Select Age Range", int(pred_pd["Age"].min()), int(pred_pd["Age"].max()), (20, 60)
)
filtered_df = pred_pd[
    (pred_pd["Age"] >= age_selected[0]) & (pred_pd["Age"] <= age_selected[1])
]
st.dataframe(filtered_df)
