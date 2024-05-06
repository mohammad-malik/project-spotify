from pyspark.sql import SparkSession
from pyspark.ml.feature import MinHashLSH, VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf, col, hash, expr, broadcast
from pyspark.sql.types import ArrayType, DoubleType, StructType, StructField
import os

# Initialize Spark Session
spark = (
    SparkSession.builder.appName("Music Recommendation Model")
    .config("spark.executor.memory", "16g")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.heartbeatInterval", "120s")
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

# Determine the number of batches
num_batches = 15

# Define the schema for the transformed data
transformed_data_schema = StructType([
    StructField("hashes", ArrayType(VectorUDT()), True),
    StructField("features", VectorUDT(), True)
])

# Initialize an empty DataFrame to store transformed data
transformed_data = spark.createDataFrame([], transformed_data_schema)

# Process each batch
for i in range(num_batches):
    batch_df = df.filter((hash("file_name") % num_batches) == i)

    # Apply transformations
    batch_df = (
        batch_df.withColumn(
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
    features_batch_df = assembler.transform(batch_df) \
        .na.drop(subset=["features"])

    # Apply MinHash LSH
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
    model = mh.fit(features_batch_df)
    transformed_batch_df = model.transform(features_batch_df)

    # Store transformed data
    transformed_data = transformed_data.union(
        transformed_batch_df.select("hashes", "features"))

    # Show results for the batch
    transformed_batch_df.show()

# Drop the 'hashes' column from the DataFrame
transformed_data = transformed_data.drop("hashes")

# Transform the entire DataFrame with the last model
transformed_data = model.transform(transformed_data)

# Compute Jaccard Similarity on the final model
jaccard_df = model.approxSimilarityJoin(
    broadcast(transformed_data),  # Broadcast smaller DataFrame
    transformed_data,
    0.5,
    distCol="JaccardDistance"
)

jaccard_similarity = jaccard_df.select(
    expr("1 - JaccardDistance as JaccardSimilarity")
)

# Compute average Jaccard Similarity Score across all batches
avg_similarity_score = jaccard_similarity.agg(
    {"JaccardSimilarity": "avg"}).collect()[0][0]
print(f"Average Jaccard Similarity Score: {avg_similarity_score}")

# Save the model
model_dir = "file:///home/mohammad/Desktop/test/"
model_path = os.path.join(model_dir, "minhash_lsh_model")
model.write().overwrite().save(model_path)

# Stop Spark session
spark.stop()
