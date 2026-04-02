FROM python:3.10-slim

# Install system tools
RUN apt-get update && apt-get install -y \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install streamlit pandas pyarrow boto3 fsspec s3fs

# Copy Streamlit app
WORKDIR /app
COPY streamlit_app.py /app/streamlit_app.py
EXPOSE 8501
# Run Streamlit on container start
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
