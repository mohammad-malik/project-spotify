from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml.feature import MinHashLSH, VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import ArrayType, DoubleType, BooleanType
import os

# Initialize Spark Session
spark = (
    SparkSession.builder.appName("Music Recommendation Model")
    .config(
        "spark.mongodb.input.uri",
        "mongodb://localhost:27017/music_database.audio_features_small",
    )
    .config(
        "spark.mongodb.output.uri",
        "mongodb://localhost:27017/music_database.transformed_tracks",
    )
    .config("spark.master", "local")
    .getOrCreate()
)

# Define UDFs
flatten_mfcc_udf = F.udf(
    lambda mfccs: [
        float(sum(col) / len(col)) for col in mfccs] if mfccs else [],
    ArrayType(DoubleType()),
)
array_to_vector_udf = F.udf(lambda x: Vectors.dense(x), VectorUDT())


def is_non_zero_vector(v):
    return any(e != 0 for e in v)


is_non_zero_vector_udf = F.udf(is_non_zero_vector, BooleanType())


# Read and repartition data
df = spark.read.format("mongo").load()

# Filter out rows with missing features early
df = df.filter(
    F.col("mfcc").isNotNull()
    & F.col("spectral_centroid").isNotNull()
    & F.col("zero_crossing_rate").isNotNull()
)

# Add a row number to the DataFrame
window_spec = Window.orderBy("_id")
df = df.withColumn("row_num", F.row_number().over(window_spec))

# Checkpoint directory setup
spark.sparkContext.setCheckpointDir("file:///tmp/spark-checkpoint")


# Function to process data in batches
def process_batch(batch_df, batch_name):
    # Apply transformations
    batch_df = (
        batch_df.withColumn("mfcc_flat", flatten_mfcc_udf("mfcc"))
        .withColumn("mfcc_vector", array_to_vector_udf("mfcc_flat"))
        .withColumn(
            "spectral_centroid_mean",
            F.expr(
                "aggregate(spectral_centroid, 0D, (acc, x) -> acc + x[0]) \
                    / size(spectral_centroid)"
            ),
        )
        .withColumn(
            "zero_crossing_rate_mean",
            F.expr(
                "aggregate(zero_crossing_rate, 0D, (acc, x) -> acc + x[0]) \
                    / size(zero_crossing_rate)"
            ),
        )
        .filter(
            F.col("mfcc_vector").isNotNull()
            & F.col("spectral_centroid_mean").isNotNull()
            & F.col("zero_crossing_rate_mean").isNotNull()
            & is_non_zero_vector_udf(F.col("mfcc_vector"))
        )
    )

    # Assemble features
    assembler = VectorAssembler(
        inputCols=[
            "mfcc_vector",
            "spectral_centroid_mean",
            "zero_crossing_rate_mean",
        ],
        outputCol="features",
    )
    features_data = assembler.transform(batch_df).na.drop(subset=["features"])

    # Apply MinHash LSH
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
    model = mh.fit(features_data)
    transformed_data = model.transform(features_data)

    # Compute Jaccard Similarity using approxSimilarityJoin
    jaccard_df = model.approxSimilarityJoin(
        transformed_data,
        transformed_data,
        0.1,
        distCol="JaccardDistance",
    )

    jaccard_similarity = jaccard_df.select(
        F.expr("1 - JaccardDistance as JaccardSimilarity")
    )

    # Use approxQuantile instead of collect to avoid large data collection
    avg_similarity_score = jaccard_similarity.approxQuantile(
        "JaccardSimilarity", [0.5], 0.01
    )[0]
    print(
        f"Average Jaccard Similarity Score for {batch_name} "
        + f"batch: {avg_similarity_score}"
    )

    return model


# Process data in batches
batch_size = 100  # Adjust batch size accordingly

total_rows = df.count()
num_batches = (total_rows // batch_size) + 1

for batch_num in range(num_batches):
    start = batch_num * batch_size
    end = start + batch_size
    batch_df = df.filter(
        (F.col("row_num") > start) & (F.col("row_num") <= end))
    model = process_batch(batch_df, f"batch_{batch_num + 1}")

    # # Save the model only for the final batch
    if batch_num == num_batches - 1:
        # Get the current working directory.
        cwd = os.getcwd()

        # Construct the path for the model.
        model_path = "file://" + os.path.join(cwd, "minhash_lsh_model")
        model.write().overwrite().save(model_path)
        break
    else:
        # freeing up memory
        del model
        batch_df.unpersist()

# Stop Spark session
spark.stop()
