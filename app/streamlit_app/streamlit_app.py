import requests
import streamlit as st
import pandas as pd

# -----------------------------
# Predicted Output Data Frame
# -----------------------------
# Load predictions saved by spark_job.py
pred_pd = pd.DataFrame(requests.get("http://35.168.59.52:5000/").json())

# -----------------------------
# Streamlit Dashboard
# -----------------------------

st.title("Medical Cost Prediction Dashboard")
st.subheader("Predictions Table")
st.dataframe(pred_pd.head(20))

st.subheader("Scatter plot: Actual vs Predicted")
st.scatter_chart(pred_pd[["Medical Cost", "prediction"]],color=["#f09b13","#4da9d6"])

st.subheader("Your probable insurance")
age = st.number_input("Age", min_value=1, max_value=120, step=1)
bmi = st.slider("BMI", min_value=15.0, max_value=55.0)
children = st.number_input("Children", min_value=0, max_value=10, step=1)

sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])

region = st.selectbox(
    "Region",
    ["northwest", "southwest", "northeast", "southeast"]
)

request_data = {
    "age": int(age),
    "bmi": float(bmi),
    "children": int(children),
    "sex": sex,
    "smoker": smoker,
    "region": region
}

# -----------------------------
# User Insurance Prediction
# -----------------------------
if st.button("Predict Cost"):
    try:
        response = requests.post("http://35.168.59.52:5000/predict", json=request_data)
        if response.status_code == 200:
            pred = response.json()["prediction"]
            st.success(f"Predicted Medical Cost: ${pred:,.2f}")
        else:
            st.error(f"Server error: {response.text}")

    except Exception as e:
        st.error(f"Failed to connect to prediction server: {e}")

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
