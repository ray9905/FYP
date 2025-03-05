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
        print(eeg_data.info)  # Displays EEG file info

        # Filters relevant data between 1 and 50 Hz
        eeg_data.filter(1, 50)

        # Applies average reference to the data
        eeg_data.set_eeg_reference('average')

        # Converts data from volts to microvolts
        eeg_array = eeg_data.get_data() * 1e6

        # Swaps axes to (samples, channels)
        eeg_array = eeg_array.T

        # Splits into 1-second windows
        sampling_rate = int(eeg_data.info['sfreq'])
        window_samples = sampling_rate * 1

        eeg_segments = [
            eeg_array[i:i + window_samples]
            for i in range(0, eeg_array.shape[0] - window_samples, window_samples)
        ]

        # Adds padding if final segment is smaller than window_samples
        remaining_samples = eeg_array.shape[0] % window_samples
        if remaining_samples != 0:
            final_segment = eeg_array[-remaining_samples:]
            padded_segment = np.zeros((window_samples, eeg_array.shape[1]))
            padded_segment[:remaining_samples, :] = final_segment
            eeg_segments.append(padded_segment)

        # Converts to numpy array
        eeg_segments = np.array(eeg_segments)

        # Creates labels for all segments
        labels = np.full((eeg_segments.shape[0],), label)

        # Appends to the dataset
        all_segments.append(eeg_segments)
        all_labels.append(labels)

    # Combines segments and labels
    if all_segments:
        final_segment = np.vstack(all_segments)
        labels = np.hstack(all_labels)
        return final_segment, labels
    else:
        return None, None

# Process Training Data (Normal = 0, Abnormal = 1)
train_normal_segments, train_normal_labels = process_eeg(TRAIN_PATH_NORMAL, 0)
train_abnormal_segments, train_abnormal_labels = process_eeg(TRAIN_PATH_ABNORMAL, 1)

# Process Evaluation Data (Normal = 0, Abnormal = 1)
eval_normal_segments, eval_normal_labels = process_eeg(EVAL_PATH_NORMAL, 0)
eval_abnormal_segments, eval_abnormal_labels = process_eeg(EVAL_PATH_ABNORMAL, 1)

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
