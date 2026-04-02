# Dockerfile.spark
FROM apache/spark:3.5.3

USER root

# Install Python3, pip, and other dependencies
RUN apt-get update && apt-get install -y python3 python3-pip && \
    pip3 install pandas numpy

RUN mkdir -p /opt/spark/jars

RUN curl -L -o /opt/spark/jars/hadoop-aws-3.3.4.jar \
    https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar && \
    curl -L -o /opt/spark/jars/aws-java-sdk-bundle-1.12.530.jar \
    https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.530/aws-java-sdk-bundle-1.12.530.jar
# Set working directory
WORKDIR /opt/spark/work-dir

# Copy the PySpark job into the container
COPY spark_job.py /opt/spark/work-dir/spark_job.py

# Make sure Spark scripts can be run
ENTRYPOINT ["/opt/spark/bin/spark-submit", "/opt/spark/work-dir/spark_job.py"]
