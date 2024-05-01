import librosa
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from tqdm import tqdm
from joblib import Memory, Parallel, delayed

# Directory with datasets.
audio_dir_prefix = r"/mnt/d/Project_Datasets/"

# Dataset to use.
selected = "fma_small"
upper_limit = 153

# Generate folder names from 000 to as many as in dataset.
folder_names = [f"{i:03d}" for i in range(upper_limit)]

audio_files = []
for folder in folder_names:
    folder_path = os.path.join(audio_dir_prefix, selected, folder)
    if os.path.exists(folder_path):
        audio_files.extend(
            [os.path.join(folder_path, file)
             for file in os.listdir(folder_path)]
        )

# Convert audio_files to a numpy array.
audio_files = np.array(audio_files)

# Select a random sample of audio files
indices = np.random.choice(len(audio_files), size=100, replace=False)
sample_files = audio_files[indices]

# Create a memory object for caching
memory = Memory("cache_directory", verbose=0)


@memory.cache
def process_file(file):
    # Load the audio file.
    y, sr = librosa.load(file)

    # Extract MFCC features.
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    # Standardize the MFCC features.
    scaler = StandardScaler()
    mfcc_scaled = scaler.fit_transform(mfcc.T).T

    return mfcc_scaled


# Use joblib to process the files in parallel
mfcc_features_list = Parallel(n_jobs=-1)(
    delayed(process_file)(file) for file in tqdm(
        sample_files, total=len(sample_files))
)

# Concatenate all the MFCC features into a single 2D array
mfcc_features = np.concatenate(mfcc_features_list, axis=1)

# Fit PCA on the standardized features without reducing dimensionality.
pca = PCA()
mfcc_pca = pca.fit_transform(
    mfcc_features.T)  # Transpose to have samples as rows

# Calculate cumulative explained variance ratio.
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot the cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(cumulative_variance, marker="o")
plt.title("Cumulative Explained Variance by PCA Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.axhline(y=0.95, color="r", linestyle="--")  # Line at 95% variance
plt.axvline(
    x=np.where(cumulative_variance >= 0.95)[0][0], color="r", linestyle="--"
)  # Line at the optimal component count
plt.show()

# Print optimal number of components for at least 95% variance
optimal_components = np.where(cumulative_variance >= 0.95)[0][0] + 1
print("Optimal number of components:", optimal_components)
