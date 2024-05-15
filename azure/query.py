from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml.feature import MinHashLSH, MinHashLSHModel, VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import ArrayType, DoubleType, BooleanType
from pyspark import StorageLevel
import os
import json

# Read data from a json file in current directory.
with open('data.json', 'r') as file:
    data = json.load(file)

# Initialize Spark Session
spark = (
    SparkSession.builder.appName("Music Recommendation Model")
    .config(
        "spark.mongodb.input.uri",
        f"{data['mongo_uri']}/music_database.audio_features",
    )
    .config(
        "spark.mongodb.output.uri",
        f"{data['mongo_uri']}/music_database.transformed_tracks",
    )
    .config(
        "spark.jars.packages",
        "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1"
    )
    .getOrCreate()
)

# Define UDFs
flatten_mfcc_udf = F.udf(
    lambda mfccs: [
        float(sum(col) / len(col)) for col in mfccs]
    if mfccs else [],
    ArrayType(DoubleType()),
)
array_to_vector_udf = F.udf(lambda x: Vectors.dense(x), VectorUDT())


# Function to load and preprocess data.
def load_and_preprocess_data():
    # Read data.
    df = spark.read.format("mongo").load()

    # Add a row number to the DataFrame.
    window_spec = Window.orderBy("_id")
    df = df.withColumn("row_num", F.row_number().over(window_spec))

    # Apply transformations.
    df = (
        df.withColumn("mfcc_flat", flatten_mfcc_udf("mfcc"))
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

    # Debugging: Check for zero vectors
    zero_vector_udf = F.udf(
        lambda vec: all(value == 0 for value in vec.toArray()), BooleanType()
    )
    df = df.withColumn("is_zero_vector", zero_vector_udf("features"))
    zero_vector_count = df.filter(
        F.col("is_zero_vector") == True).count()
    print(f"Number of zero vectors: {zero_vector_count}")

    # Filter out zero vectors
    df = df.filter(
        F.col("is_zero_vector") == False).drop("is_zero_vector")
    return df


# Function to build and save MinHash LSH model.
def build_and_save_model(df, model_path):
    # Apply MinHash LSH
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
    model = mh.fit(df)
    model.write().overwrite().save(model_path)
    return model


# Function to find the closest tracks.
def find_closest_tracks(track_id, model, df):
    # Append .mp3 extension to the track_id.
    track_id = str(track_id).zfill(6).encode().decode("utf-8") + ".mp3"

    # Extract features for the given track_id.
    track_df = df.filter(F.col("track_id") == track_id)

    # Debugging: Check if track is found.
    if track_df.count() == 0:
        print(f"Track ID '{track_id}' not found.")
        return []

    # Retrieve the features for the specific track.
    query_features = track_df.first()["features"]

    # Find the 5 closest tracks using approxNearestNeighbors
    neighbors = model.approxNearestNeighbors(df, query_features, 6)

    # Exclude original track and return closest track IDs.
    return [
        row["track_id"]
        for row in neighbors.collect()
        if row["track_id"] != track_id
    ][:5]


# Encapsulated function to build the recommendation system and return tracks.
def recommend_tracks(track_id):
    print(f"Starting recommend_tracks for track ID: {track_id}")
    # Define model path
    cwd = os.getcwd()
    model_dir = os.path.abspath(cwd)
    model_path = "file://" + os.path.join(cwd, "minhash_lsh_model")

    # Load and preprocess data.
    df = load_and_preprocess_data()

    # Check if model directory exists, if not, create it.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Check if model exists
    model_file_path = os.path.join(model_path, "metadata")
    if not os.path.exists(model_file_path):
        # Build and save the model
        print("Building and saving the MinHash LSH model.")
        model = build_and_save_model(df, model_path)
    else:
        # Load the existing model
        print("Loading existing MinHash LSH model.")
        model = MinHashLSHModel.load(model_path)

    # Add MinHash LSH transformation to the DataFrame.
    df = model.transform(df)
    df.persist(StorageLevel.DISK_ONLY)

    # Find and return the closest tracks
    recommended_tracks = find_closest_tracks(track_id, model, df)
    print(f"Recommended tracks for track ID {track_id}: {recommended_tracks}")
    return recommended_tracks
