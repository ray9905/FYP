import mne
import numpy as np
import os

# Dataset paths
BASE_PATH = "eeg_data" 
TRAIN_PATH_NORMAL = os.path.join(BASE_PATH, "normal", "train")
EVAL_PATH_NORMAL = os.path.join(BASE_PATH, "normal", "eval")
TRAIN_PATH_ABNORMAL = os.path.join(BASE_PATH, "abnormal", "train")
EVAL_PATH_ABNORMAL = os.path.join(BASE_PATH, "abnormal", "eval")


#Processes EEG files with artifact removal 
def process_eeg_with_features(folder_path, label):
    all_features, all_labels = [], []
    eeg_files = [f for f in os.listdir(folder_path) if f.endswith(".edf")]

    for file in eeg_files:
        file_path = os.path.join(folder_path, file)
        print(f"Processing {file_path}...")

        # Loads the EEG data from the edf file
        eeg_data = mne.io.read_raw_edf(file_path, preload=True)
        print(eeg_data.info)  
        
        # Apply Notch Filter (50Hz) to remove powerline noise
        eeg_data.notch_filter(freqs=50)

        # Apply Bandpass Filter (1-50Hz) to remove irrelevant frequencies
        eeg_data.filter(1, 50)

        # Apply ICA to remove artifacts (blinks, muscle movements, etc.)
        ica = mne.preprocessing.ICA(n_components=20, random_state=42)
        ica.fit(eeg_data)
        eeg_data = ica.apply(eeg_data)

        # Converts data from volts to microvolts
        eeg_array = eeg_data.get_data() * 1e6
        # Swaps axes to (samples, channels)
        eeg_array = eeg_array.T
        sfreq = eeg_data.info["sfreq"]


        #Extracts the Power Spectral Density (PSD) Features
        from scipy.signal import welch
        def compute_psd_features(eeg_segment, sfreq):
            freqs, psd = welch(eeg_segment, sfreq, nperseg=sfreq)
            freq_bands = {
                "delta": (0.5, 4),
                "theta": (4, 8),
                "alpha": (8, 13),
                "beta": (13, 30),
                "gamma": (30, 50)
            }
            band_power = []
            for band, (low, high) in freq_bands.items():
                band_indices = (freqs >= low) & (freqs <= high)
                band_power.append(np.mean(psd[:, band_indices], axis=1))
            return np.concatenate(band_power, axis=-1)


       


        

# Process Training Data (Normal = 0, Abnormal = 1)
train_normal_segments, train_normal_labels = process_eeg_with_features(TRAIN_PATH_NORMAL, 0)
train_abnormal_segments, train_abnormal_labels = process_eeg_with_features(TRAIN_PATH_ABNORMAL, 1)

# Process Evaluation Data (Normal = 0, Abnormal = 1)
eval_normal_segments, eval_normal_labels = process_eeg_with_features(EVAL_PATH_NORMAL, 0)
eval_abnormal_segments, eval_abnormal_labels = process_eeg_with_features(EVAL_PATH_ABNORMAL, 1)

# Combine Training Data
X_train = np.vstack([train_normal_segments, train_abnormal_segments])
y_train = np.hstack([train_normal_labels, train_abnormal_labels])

# Combine Evaluation Data
X_val = np.vstack([eval_normal_segments, eval_abnormal_segments])
y_val = np.hstack([eval_normal_labels, eval_abnormal_labels])

# Save as NumPy arrays
np.save("train_eeg.npy", X_train)
np.save("train_labels.npy", y_train)
np.save("eval_eeg.npy", X_val)
np.save("eval_labels.npy", y_val)

print(f"Saved Training EEG data: {X_train.shape}")
print(f"Saved Training Labels: {y_train.shape}")
print(f"Saved Evaluation EEG data: {X_val.shape}")
print(f"Saved Evaluation Labels: {y_val.shape}")
