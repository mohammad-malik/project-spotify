from pyspark.sql import SparkSession
from pyspark.ml.feature import MinHashLSHModel, VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, DoubleType
import os

# Initialize Spark Session
spark = (
    SparkSession.builder.appName("Music Recommendation Model")
    .config("spark.executor.memory", "16g")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memoryOverhead", "4096")
    .config("spark.driver.memoryOverhead", "4096")
    .config("spark.executor.heartbeatInterval", "120s")
    .config(
        "spark.executor.extraJavaOptions",
        "-XX:MaxDirectMemorySize=512M -XX:+HeapDumpOnOutOfMemoryError -Dio.netty.noUnsafe=true")
    .config(
        "spark.driver.extraJavaOptions",
        "-XX:MaxDirectMemorySize=512M -XX:+HeapDumpOnOutOfMemoryError -Dio.netty.noUnsafe=true")
    .config("spark.network.timeout", "800s")
    .config("spark.default.parallelism", "200")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.storage.level", "MEMORY_AND_DISK_SER")
    .config(
        "spark.mongodb.input.uri",
        "mongodb://localhost:27017/music_database.audio_features_small",
    )
    .config(
        "spark.mongodb.output.uri",
        "mongodb://localhost:27017/music_database.transformed_tracks",
    )
    .getOrCreate()
)

# Define UDFs
def flatten_and_mean(features):
    if features and isinstance(features[0], list):
        flattened = [item for sublist in features for item in sublist]
        return float(sum(flattened) / len(flattened))
    elif features:
        return float(sum(features) / len(features))
    return 0.0


get_mean_udf = udf(flatten_and_mean, DoubleType())
flatten_mfcc_udf = udf(
    lambda mfccs: [float(sum(col) / len(col)) for col in mfccs if mfccs],
    ArrayType(DoubleType()),
)
array_to_vector_udf = udf(lambda x: Vectors.dense(x), VectorUDT())

# Read and prepare data
df = spark.read.format("mongo").load()
df = df.repartition(200)  # Increase the number of partitions

# Apply transformations
df = (
    df.withColumn(
        "mfcc_flat", flatten_mfcc_udf("mfcc"))
    .withColumn(
        "mfcc_vector", array_to_vector_udf("mfcc_flat"))
    .withColumn(
        "spectral_centroid_mean", get_mean_udf("spectral_centroid"))
    .withColumn(
        "zero_crossing_rate_mean", get_mean_udf("zero_crossing_rate"))
    .filter(
        col("mfcc_vector").isNotNull()
        & col("spectral_centroid_mean").isNotNull()
        & col("zero_crossing_rate_mean").isNotNull()
    )
)

# Assemble features
assembler = VectorAssembler(
    inputCols=[
        "mfcc_vector",
        "spectral_centroid_mean",
        "zero_crossing_rate_mean"
    ],
    outputCol="features",
)
df = assembler.transform(df).na.drop(subset=["features"])

# Persist the DataFrame
df.persist()

# Load the pre-saved model
model_dir = "file:///home/mohammad/Desktop/test/"
model_path = os.path.join(model_dir, "minhash_lsh_model")
model = MinHashLSHModel.load(model_path)

# Transform the DataFrame with the loaded model
df = model.transform(df)

# Get the first entry in the database
first_entry = df.first()

# Find 5 similar items for the first entry
similar_items = model.approxNearestNeighbors(df, first_entry["features"], 5)

# Show the similar items
similar_items.show()

# Unpersist the DataFrame
df.unpersist()

# Stop Spark session
spark.stop()
