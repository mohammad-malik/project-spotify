from pyspark.sql import SparkSession
from pyspark.ml.feature import MinHashLSH, VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import (
    udf,
    col,
    hash,
    expr,
    pandas_udf,
    PandasUDFType,
)
from pyspark.sql.types import ArrayType, StructType, StructField
import os

# Initialize Spark Session optimized for Azure Blob Storage
spark = (
    SparkSession.builder.appName("Music Recommendation Model")
    .config("spark.executor.memory", "8g")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.instances", "16")
    .config("spark.executor.cores", "4")
    .config("spark.driver.maxResultSize", "2g")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.default.parallelism", "100")
    .config("spark.memory.fraction", "0.6")
    .config("spark.memory.storageFraction", "0.5")
    .config("spark.scheduler.mode", "FAIR")
    .config("fs.azure", "org.apache.hadoop.fs.azure.NativeAzureFileSystem")
    .config(
        "fs.azure.account.key.<your-storage-account-name>.blob.core.windows.net",
        "<your-storage-account-key>",
    )
    .getOrCreate()
)

# Configure Blob storage URI for input data
dataUri = "wasb://<your-container-name>@<your-storage-account-name>.blob.core.windows.net/<path-to-your-data>"


# Define optimized UDFs using Pandas UDFs for better performance
@pandas_udf("double", PandasUDFType.SCALAR)
def flatten_and_mean(features):
    return features.mean()


get_mean_udf = flatten_and_mean


@pandas_udf("array<double>", PandasUDFType.SCALAR)
def flatten_mfcc(mfccs):
    return [mfcc.mean() for mfcc in mfccs.transpose()]


array_to_vector_udf = udf(lambda x: Vectors.dense(x), VectorUDT())

# Read and prepare data from Azure Blob Storage
df = spark.read.format("mongo").load(dataUri)
df = df.repartition(200)  # Optimized repartition

# Process each batch
num_batches = 15
transformed_data_schema = StructType(
    [
        StructField("hashes", ArrayType(VectorUDT()), True),
        StructField("features", VectorUDT(), True),
    ]
)
transformed_data = spark.createDataFrame([], transformed_data_schema)

for i in range(num_batches):
    batch_df = df.filter((hash("file_name") % num_batches) == i)
    batch_df = (
        batch_df.withColumn("mfcc_flat", flatten_mfcc("mfcc"))
        .withColumn("mfcc_vector", array_to_vector_udf("mfcc_flat"))
        .withColumn("spectral_centroid_mean", get_mean_udf("spectral_centroid"))
        .withColumn("zero_crossing_rate_mean", get_mean_udf("zero_crossing_rate"))
        .filter(col("mfcc_vector").isNotNull())
    )

    assembler = VectorAssembler(
        inputCols=["mfcc_vector", "spectral_centroid_mean", "zero_crossing_rate_mean"],
        outputCol="features",
    )
    features_batch_df = assembler.transform(batch_df).na.drop(subset=["features"])

    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
    model = mh.fit(features_batch_df)
    transformed_batch_df = model.transform(features_batch_df)
    transformed_data = transformed_data.union(
        transformed_batch_df.select("hashes", "features")
    )

# Final processing steps
transformed_data = transformed_data.drop("hashes")
transformed_data = model.transform(transformed_data)
jaccard_df = model.approxSimilarityJoin(
    transformed_data, transformed_data, 0.5, distCol="JaccardDistance"
)
jaccard_similarity = jaccard_df.select(expr("1 - JaccardDistance as JaccardSimilarity"))
avg_similarity_score = jaccard_similarity.groupBy().avg().collect()[0][0]

print(f"Average Jaccard Similarity Score: {avg_similarity_score}")

# Save the model
model_dir = "file:///home/username/Desktop/test/"
model_path = os.path.join(model_dir, "minhash_lsh_model")
model.write().overwrite().save(model_path)

# Stop Spark session
spark.stop()
