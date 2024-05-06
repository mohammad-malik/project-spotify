from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import librosa
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pymongo import MongoClient
import numpy as np
import concurrent.futures
from tqdm import tqdm
from joblib import Memory, Parallel, delayed
import logging
import io

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Read file containing vm details.
with open("vm_details.txt", "r") as f:
    DATABASE_URI = f.readline().strip()
    DATABASE_NAME = f.readline().strip()
    COLLECTION_NAME = f.readline().strip()

DATABASE_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "music_database"
COLLECTION_NAME = "audio_features_small"

# Database setup
client = MongoClient(DATABASE_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Read file containing azure details.
with open("azure_details.txt", "r") as f:
    connect_str = f.readline().strip()
    container_name = f.readline().strip()

connect_str = "your_connection_string_here"
container_name = "your_container_name_here"
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_client = blob_service_client.get_container_client(container_name)

# Azure Blob Storage setup.
audio_files = []
prefix = "fma_large/"
blob_list = container_client.list_blobs(name_starts_with=prefix)

for blob in blob_list:
    # Check if the blob's name ends with ".mp3" to filter out audio files
    if blob.name.endswith(".mp3"):
        audio_files.append(blob.name)


# Create a memory object for caching
memory = Memory("cache_directory", verbose=0)


@memory.cache
def process_file(blob_name):
    try:
        blob_client = container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob().readall()
        y, sr = librosa.load(io.BytesIO(blob_data), sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        return mfcc
    except Exception as e:
        logging.error(f"Error loading file {blob_name}: {e}")
        return np.array([])


# Use joblib to process the files in parallel
mfcc_features_list = Parallel(n_jobs=-1)(
    delayed(process_file)(file) for file in tqdm(
        audio_files, total=len(audio_files))
)

# Concatenate all the MFCC features into a single 2D array
mfcc_features = np.concatenate(
    [f.T for f in mfcc_features_list if f.size > 0],
    axis=0
)

# Standardize the features
scaler = StandardScaler()
mfcc_scaled = scaler.fit_transform(mfcc_features)

# Fit PCA on the standardized features without reducing dimensionality.
pca = PCA()
pca.fit(mfcc_scaled)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
optimal_components = np.where(cumulative_variance >= 0.95)[0][0] + 1

# Logging optimal components
logging.info(f"Optimal number of PCA components: {optimal_components}")


# Processing each file in ThreadPoolExecutor
def insert_features(file, n_components):
    try:
        y, sr = librosa.load(file, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

        scaler_mfcc = StandardScaler()
        mfcc_scaled = scaler_mfcc.fit_transform(mfcc.T).T

        pca = PCA(n_components=n_components)
        mfcc_reduced = pca.fit_transform(mfcc_scaled)

        scaler_feature = MinMaxScaler()
        spectral_centroid_normalized = scaler_feature.fit_transform(
            spectral_centroid.T
        ).T
        zero_crossing_rate_normalized = scaler_feature.fit_transform(
            zero_crossing_rate.T
        ).T

        document = {
            "file_name": file,
            "mfcc": mfcc_reduced.tolist(),
            "spectral_centroid": spectral_centroid_normalized.tolist(),
            "zero_crossing_rate": zero_crossing_rate_normalized.tolist(),
        }
        collection.insert_one(document)
        logging.info(f"Processed and inserted: {file}")
    except Exception as e:
        logging.error(f"Error processing file {file}: {e}")


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    for file in audio_files:
        executor.submit(insert_features, file, optimal_components)

client.close()
logging.info("MongoDB connection closed.")
