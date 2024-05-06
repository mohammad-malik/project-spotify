import librosa
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pymongo import MongoClient
import numpy as np
import os
import concurrent.futures
from tqdm import tqdm
from joblib import Memory, Parallel, delayed
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DATABASE_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "music_database"
COLLECTION_NAME = "audio_features_small"

# Database setup
client = MongoClient(DATABASE_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Directory with datasets
audio_dir_prefix = r"/media/mohammad/Small boi/Project_Datasets"

# Select dataset comes from collection name after removing "audio_features_".
selected = "fma_" + COLLECTION_NAME.split("_")[-1]

file_read_limits = {
    "fma_tiny": 3,
    "fma_small": 156,
    "fma_medium": 156,
    "fma_large": 156,
}

# Generate folder names from 000 to as many as in dataset
folder_names = [f"{i:03d}" for i in range(file_read_limits[selected])]
audio_files = []
for folder in folder_names:
    folder_path = os.path.join(audio_dir_prefix, selected, folder)
    if os.path.exists(folder_path):
        audio_files.extend(
            [
                os.path.join(folder_path, file)
                for file in os.listdir(folder_path)
                if file.endswith(".mp3")
            ]
        )

# Create a memory object for caching
memory = Memory("cache_directory", verbose=0)


@memory.cache
def process_file(file):
    try:
        y, sr = librosa.load(file, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        return mfcc
    except Exception as e:
        logging.error(f"Error loading file {file}: {e}")
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
