import librosa
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pymongo import MongoClient
from confluent_kafka import Consumer
import concurrent.futures
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATABASE_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "music_database"
COLLECTION_NAME = "audio_features_small"

# Database setup
client = MongoClient(DATABASE_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Kafka details
bootstrap_servers = "10.0.0.4:9092, 10.0.0.5:9092, 192.168.0.4:9092, 192.168.0.5:9092"
topic = "streamed_music_azure_blob"
group_id = "audio_feature_extraction_group"

# Create a Kafka consumer
consumer_conf = {
    'bootstrap.servers': bootstrap_servers,
    'group.id': group_id,
    'auto.offset.reset': 'earliest'
}
consumer = Consumer(consumer_conf)
consumer.subscribe([topic])

# Function to process individual audio files
def process_and_insert_features(file_name, audio_data, n_components):
    try:
        y, sr = librosa.load(audio_data, sr=None, format='mp3')
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

        scaler_mfcc = StandardScaler()
        mfcc_scaled = scaler_mfcc.fit_transform(mfcc.T).T

        pca = PCA(n_components=n_components)
        mfcc_reduced = pca.fit_transform(mfcc_scaled)

        scaler_feature = MinMaxScaler()
        spectral_centroid_normalized = scaler_feature.fit_transform(spectral_centroid.T).T
        zero_crossing_rate_normalized = scaler_feature.fit_transform(zero_crossing_rate.T).T

        document = {
            "file_name": file_name,
            "mfcc": mfcc_reduced.tolist(),
            "spectral_centroid": spectral_centroid_normalized.tolist(),
            "zero_crossing_rate": zero_crossing_rate_normalized.tolist(),
        }
        collection.insert_one(document)
        logging.info(f"Processed and inserted: {file_name}")
    except Exception as e:
        logging.error(f"Error processing file {file_name}: {e}")

# Function to consume and process audio data from Kafka
def consume_and_process_audio(n_components):
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            logging.error(f"Kafka error: {msg.error()}")
            continue

        try:
            # Assuming the message key contains the filename and value contains the audio data
            file_name = msg.key().decode('utf-8')
            audio_data = msg.value()
            process_and_insert_features(file_name, audio_data, n_components)
        except Exception as e:
            logging.error(f"Error processing Kafka message: {e}")

# Dummy audio data to fit PCA
def get_dummy_audio_data():
    y, sr = librosa.load(librosa.example('trumpet'), sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return mfcc

# Fit PCA to determine optimal components
mfcc_features = get_dummy_audio_data()
scaler = StandardScaler()
mfcc_scaled = scaler.fit_transform(mfcc_features.T).T

pca = PCA()
pca.fit(mfcc_scaled)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
optimal_components = np.where(cumulative_variance >= 0.95)[0][0] + 1

logging.info(f"Optimal number of PCA components: {optimal_components}")

# Consume messages from Kafka and process them
try:
    consume_and_process_audio(optimal_components)
except KeyboardInterrupt:
    logging.info("Stopping audio feature extraction process.")
finally:
    consumer.close()
    client.close()
    logging.info("MongoDB connection closed.")