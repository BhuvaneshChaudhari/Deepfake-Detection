import os
import librosa
import numpy as np
import pickle
from concurrent.futures import ThreadPoolExecutor

AUDIO_DIR = "D:/Project/backend/dataset"
OUTPUT_PICKLE = "D:/Project/backend/audio_features_parallel.pkl"
SAMPLE_RATE = 16000
DURATION = 2  # in seconds
N_MFCC = 13

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        label = 1 if "real" in file_path.lower() else 0
        return mfcc_scaled, label
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def get_all_files(base_dir):
    files = []
    for root, dirs, filenames in os.walk(base_dir):
        for f in filenames:
            if f.endswith(".wav"):
                files.append(os.path.join(root, f))
    return files

def main():
    all_files = get_all_files(AUDIO_DIR)
    print(f"Found {len(all_files)} audio files. Extracting features...")

    features = []
    with ThreadPoolExecutor() as executor:
        for result in executor.map(extract_features, all_files):
            if result:
                features.append(result)

    X, y = zip(*features)
    with open(OUTPUT_PICKLE, "wb") as f:
        pickle.dump((np.array(X), np.array(y)), f)

    print(f"Saved extracted features to {OUTPUT_PICKLE}")

if __name__ == "__main__":
    main()
