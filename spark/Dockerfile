# Based on ubunto Docker image
FROM ubuntu:latest as builder

# Install necessary packages
RUN apt-get update
RUN apt-get install -y wget
RUN apt-get install -y default-jdk-headless
RUN apt-get install -y pip

# Specify environment variables
ENV SPARK_VERSION=3.2.1
ENV HADOOP_VERSION=3.2
ENV SPARK_HOME=/opt/spark
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=${JAVA_HOME}/bin:${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${PATH}

# Download and install Apache Spark
RUN wget --no-verbose -O apache-spark.tgz "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
  && mkdir -p /opt/spark \
  && tar -xf apache-spark.tgz -C /opt/spark --strip-components=1 \
  && rm apache-spark.tgz

# Create working directory
RUN mkdir /workspace

FROM builder as apache-spark

# Copy log4j configuration into spark config folder
COPY log4j.properties /opt/spark/conf/log4j.properties

# Set starting directory
WORKDIR /workspace

# Copy Python requirements file
COPY requirements.txt requirements.txt

# Install requirements
RUN pip3 install -r requirements.txt

# Expose ports
EXPOSE 4040 6066 7077

# Ensure container remains active
ENTRYPOINT ["tail", "-f", "/dev/null"]