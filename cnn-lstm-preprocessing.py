import mne
import numpy as np
import os
from scipy.signal import welch
from sklearn.utils import shuffle

# Paths
BASE_PATH = "/content/drive/My Drive/FYP/eeg_data"
SAVE_PATH = "/content/drive/My Drive/FYP"

TRAIN_PATH_NORMAL = os.path.join(BASE_PATH, "normal", "train")
EVAL_PATH_NORMAL = os.path.join(BASE_PATH, "normal", "eval")
TRAIN_PATH_ABNORMAL = os.path.join(BASE_PATH, "abnormal", "train")
EVAL_PATH_ABNORMAL = os.path.join(BASE_PATH, "abnormal", "eval")

# Processing settings
BATCH_SIZE = 50  # Number of EEG files per batch
WINDOW_SIZE = 100  # Window size for segmenting EEG signals
MIN_SEGMENT_SIZE = 80  # Reduced from 100 to allow shorter abnormal EEGs

#Overlap Adjustments
OVERLAP_NORMAL = 0.25  # Reduced overlap for normal EEGs
OVERLAP_ABNORMAL = 0.60  #Increased overlap for abnormal EEGs

#Computes Welch PSD features
def compute_psd_features(eeg_segment, sfreq):
    """ Computes Power Spectral Density (PSD) for EEG segment using Welch's method. """
    nperseg = min(eeg_segment.shape[0], sfreq)  # Avoids issues with short segments
    freqs, psd = welch(eeg_segment, sfreq, nperseg=nperseg, axis=0)

    freq_bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50)
    }

    band_power = []
    for low, high in freq_bands.values():
        band_indices = np.where((freqs >= low) & (freqs <= high))[0]
        if band_indices.size > 0:
            band_power.append(np.mean(psd[band_indices], axis=0))
        else:
            band_power.append(np.zeros(eeg_segment.shape[1]))  # Zero if no valid indices

    return np.array(band_power).T  # Shape: (channels, 5)

#EEG preprocessing function
def process_eeg_with_features(folder_path, label, category):
    """ Loads EEG files, applies Welch method, and saves extracted features in batches. """
    all_features, all_labels = [], []
    eeg_files = [f for f in os.listdir(folder_path) if f.endswith(".edf")]

    # Shuffle files to prevent bias**
    eeg_files = shuffle(eeg_files, random_state=42)

    for i, file in enumerate(eeg_files):
        file_path = os.path.join(folder_path, file)
        print(f"Processing ({i+1}/{len(eeg_files)}): {file}")

        try:
            eeg_data = mne.io.read_raw_edf(file_path, preload=True)
            eeg_data.filter(1, 50, verbose=False)
            eeg_data.set_eeg_reference('average')

            eeg_array = eeg_data.get_data() * 1e6  # Convert to microvolts
            eeg_array = eeg_array.T  # Shape: (samples, channels)
            sfreq = eeg_data.info["sfreq"]

            num_samples = eeg_array.shape[0]

            #Applies Different Overlaps for Normal and Abnormal Segments
            overlap = OVERLAP_NORMAL if category == "normal" else OVERLAP_ABNORMAL
            step_size = int(WINDOW_SIZE * (1 - overlap))

            num_windows = max(1, (num_samples - MIN_SEGMENT_SIZE) // step_size)

            eeg_segments = np.array([
                eeg_array[i * step_size:i * step_size + WINDOW_SIZE]
                for i in range(num_windows) if (i * step_size + WINDOW_SIZE) <= num_samples
            ])  #Shape: (windows, samples, channels)

            if eeg_segments.size == 0:
                print(f"Skipping {file}: Not enough samples.")
                continue

            psd_features = np.array([compute_psd_features(seg, sfreq) for seg in eeg_segments])
            psd_features = psd_features.reshape(psd_features.shape[0], -1)  # Flatten into (windows, channels * 5)

            labels = np.full((psd_features.shape[0],), label)

            all_features.append(psd_features)
            all_labels.append(labels)

            #Ensures Last Batch is Saved
            if (i + 1) % BATCH_SIZE == 0 or (i + 1) == len(eeg_files):
                batch_num = (i + 1) // BATCH_SIZE + 1
                np.save(f"{SAVE_PATH}/preprocessed_v3_{category}_batch{batch_num}_data.npy", np.vstack(all_features))
                np.save(f"{SAVE_PATH}/preprocessed_v3_{category}_batch{batch_num}_labels.npy", np.hstack(all_labels))
                print(f"Saved batch {batch_num} ({category})")

                all_features, all_labels = [], []  #Resets batch list

        except Exception as e:
            print(f"Skipping {file}: {e}")

#Run EEG Preprocessing 
process_eeg_with_features(TRAIN_PATH_NORMAL, 0, "train_normal")
process_eeg_with_features(TRAIN_PATH_ABNORMAL, 1, "train_abnormal")
process_eeg_with_features(EVAL_PATH_NORMAL, 0, "eval_normal")
process_eeg_with_features(EVAL_PATH_ABNORMAL, 1, "eval_abnormal")

print("Preprocessing Completed")
