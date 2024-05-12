from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml.feature import MinHashLSH, MinHashLSHModel, VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import ArrayType, DoubleType
from pyspark import StorageLevel
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
    .getOrCreate()
)

# Define UDFs
flatten_mfcc_udf = F.udf(
    lambda mfccs: [float(sum(col) / len(col)) for col in mfccs] if mfccs else [],
    ArrayType(DoubleType()),
)
array_to_vector_udf = F.udf(lambda x: Vectors.dense(x), VectorUDT())

# Function to load and preprocess data
def load_and_preprocess_data():
    # Read data
    df = spark.read.format("mongo").load()

    # Add a row number to the DataFrame
    window_spec = Window.orderBy("_id")
    df = df.withColumn("row_num", F.row_number().over(window_spec))

    # Apply transformations
    df = (
        df.withColumn("mfcc_flat", flatten_mfcc_udf("mfcc"))
        .withColumn("mfcc_vector", array_to_vector_udf("mfcc_flat"))
        .withColumn(
            "spectral_centroid_mean",
            F.expr(
                "aggregate(spectral_centroid, 0D, (acc, x) -> acc + x[0]) / size(spectral_centroid)"
            ),
        )
        .withColumn(
            "zero_crossing_rate_mean",
            F.expr(
                "aggregate(zero_crossing_rate, 0D, (acc, x) -> acc + x[0]) / size(zero_crossing_rate)"
            ),
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
    df = assembler.transform(df).na.drop(subset=["features"])
    return df

# Function to build and save MinHash LSH model
def build_and_save_model(df, model_path):
    # Apply MinHash LSH
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
    model = mh.fit(df)
    model.write().overwrite().save(model_path)
    return model

# Function to find the closest tracks
def find_closest_tracks(track_id, model, df):
    # Extract features for the given track_id
    track_df = df.filter(F.col("track_id") == track_id)

    # Debugging: Check if track is found
    if track_df.count() == 0:
        print(f"Track ID '{track_id}' not found.")
        return []

    # Retrieve the features for the specific track
    query_features = track_df.first()["features"]

    # Find the 5 closest tracks using approxNearestNeighbors
    neighbors = model.approxNearestNeighbors(df, query_features, 6)

    # Exclude original track and return closest track IDs
    return [
        row["datasetA.track_id"]
        for row in neighbors.collect()
        if row["datasetA.track_id"] != track_id
    ][:5]

# Encapsulated function to build the recommendation system and find similar tracks
def recommend_tracks(track_id):
    # Define model path
    model_dir = "file:///home/mohammad/Desktop/manal/"
    model_path = os.path.join(model_dir, r"minhash_lsh_model")

    # Load and preprocess data
    df = load_and_preprocess_data()

    # Check if model exists
    if not os.path.exists(model_dir.replace("file://", "")):
        os.makedirs(model_dir.replace("file://", ""))

    if not os.path.exists(model_path.replace("file://", "")):
        # Build and save the model
        model = build_and_save_model(df, model_path)
    else:
        # Load the existing model
        model = MinHashLSHModel.load(model_path)

    # Add MinHash LSH transformation to the DataFrame
    df = model.transform(df)
    df.persist(StorageLevel.DISK_ONLY)

    # Find and return the closest tracks
    return find_closest_tracks(track_id, model, df)

def stop_spark_session():
    spark.stop()