import mne
import numpy as np
import os
from scipy.stats import skew, kurtosis

#preprocessing function

def preprocess_eeg(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.filter(0.5, 40., fir_design='firwin', verbose=False)
    epochs = mne.make_fixed_length_epochs(raw, duration=5, overlap=2.5, verbose=False)
    data = epochs.get_data()

    features = []
    for epoch in data:
        epoch_features = []
        for channel_data in epoch:
            epoch_features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75),
                skew(channel_data),
                kurtosis(channel_data),
                np.median(channel_data),
                np.sqrt(np.mean(channel_data**2)),  # RMS
            ])
        features.append(epoch_features)

    return np.array(features)

#Generates dataset in batches to manage memory

def create_dataset(base_directory, save_directory, dataset_type='train', batch_size=50):
    X, y, file_mapping = [], [], []
    batch_count = 0

    for label_type in ['normal', 'abnormal']:
        current_dir = os.path.join(base_directory, label_type, dataset_type)
        files = [f for f in os.listdir(current_dir) if f.endswith('.edf')]

        #Balances the dataset training
        if dataset_type == 'train' and label_type == 'normal':
            np.random.seed(42)
            files = np.random.choice(files, size=min(len(files), 400), replace=False)

        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            for file_name in batch_files:
                file_path = os.path.join(current_dir, file_name)
                label = 1 if label_type == 'abnormal' else 0

                print(f"Processing file: {file_path}")
                features = preprocess_eeg(file_path)
                X.extend(features)
                y.extend([label] * features.shape[0])
                file_mapping.extend([file_name] * features.shape[0])

            #Save each batch separately to manage memory
            os.makedirs(save_directory, exist_ok=True)
            np.save(os.path.join(save_directory, f"X_{dataset_type}_eeg_batch_{batch_count}.npy"), np.array(X))
            np.save(os.path.join(save_directory, f"y_{dataset_type}_eeg_batch_{batch_count}.npy"), np.array(y))
            np.save(os.path.join(save_directory, f"file_mapping_{dataset_type}_batch_{batch_count}.npy"), np.array(file_mapping))
            print(f"Saved batch {batch_count} with {len(X)} samples to {save_directory}.")
            X, y, file_mapping = [], [], []
            batch_count += 1


base_directory = "/content/drive/MyDrive/FYP/eeg_data"
save_directory = "/content/drive/MyDrive/FYP/new-preprocessed"

create_dataset(base_directory, save_directory, 'train')
create_dataset(base_directory, save_directory, 'eval')