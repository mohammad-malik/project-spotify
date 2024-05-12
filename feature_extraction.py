import librosa
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pymongo import MongoClient, InsertOne
import logging
from confluent_kafka import Consumer, KafkaError, TopicPartition
import concurrent.futures
from pydub import AudioSegment
import io
import numpy as np

# Kafka Consumer Configuration
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "streamed_music_local"
KAFKA_GROUP_ID = "audio_feature_extraction_group"

# MongoDB Configuration
DATABASE_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "music_database"
COLLECTION_NAME = "audio_features_small"

# Database setup
client = MongoClient(DATABASE_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# PCA Components
FIXED_COMPONENTS = 18

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Kafka Consumer setup
consumer_conf = {
    "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
    "group.id": KAFKA_GROUP_ID,
    "auto.offset.reset": "earliest",
}
consumer = Consumer(consumer_conf)
tp = TopicPartition(KAFKA_TOPIC, 0, 0)
consumer.assign([tp])

# Initialize scikit-learn transformers and PCA model
scaler_mfcc = StandardScaler()
scaler_feature = MinMaxScaler()
pca = PCA(n_components=FIXED_COMPONENTS)


def process_audio_features(track_id, audio_data):
    try:
        # Create a BytesIO object from the audio data
        audio_file = io.BytesIO(audio_data)

        # Load audio data with pydub
        audio = AudioSegment.from_file(audio_file)

        # Convert to mono and get raw data
        audio_mono = audio.set_channels(1)
        raw_data = np.array(audio_mono.get_array_of_samples())

        # Normalize raw data to between -1 and 1
        normalized_data = raw_data / np.iinfo(raw_data.dtype).max

        # Process with librosa
        y = librosa.to_mono(normalized_data)
        sr = audio.frame_rate

        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

        mfcc_scaled = scaler_mfcc.fit_transform(mfcc.T).T
        mfcc_reduced = pca.fit_transform(mfcc_scaled)

        spectral_centroid_normalized = scaler_feature.fit_transform(
            spectral_centroid.T
        ).T
        zero_crossing_rate_normalized = scaler_feature.fit_transform(
            zero_crossing_rate.T
        ).T

        document = {
            "track_id": track_id,
            "mfcc": mfcc_reduced.tolist(),
            "spectral_centroid": spectral_centroid_normalized.tolist(),
            "zero_crossing_rate": zero_crossing_rate_normalized.tolist(),
        }
        return document

    except Exception as e:
        logging.error(f"Error processing audio data for track {track_id}: {e}")
        return None


def batch_insert_documents(documents):
    try:
        if documents:
            collection.bulk_write(
                [InsertOne(doc) for doc in documents], ordered=False)
            logging.info(f"Inserted batch of {len(documents)} documents.")
    except Exception as e:
        logging.error(f"Error inserting batch documents: {e}")


def consume_audio_messages():
    batch_size = 10
    batch = []

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            while True:
                msg = consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logging.error(f"ERROR: {msg.error()}")
                        break

                track_id = msg.key().decode("utf-8")
                audio_data = msg.value()

                future = executor.submit(
                    process_audio_features, track_id, audio_data)
                batch.append(future)

                if len(batch) >= batch_size:
                    results = [
                        f.result() for f in batch if f.result() is not None
                    ]
                    batch_insert_documents(results)
                    batch.clear()

    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        client.close()
        logging.info("Kafka consumer and MongoDB connection closed.")


if __name__ == "__main__":
    consume_audio_messages()
