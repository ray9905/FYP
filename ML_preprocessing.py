import mne
import numpy as np
import os
from scipy.stats import skew, kurtosis

#Machine Learning Preprocessing for EEG files

def preprocess_eeg(file_path):
    #Loads EEG data
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    #Applies bandpass filter (1-50 Hz) and sets EEG reference to average
    raw.filter(1, 50, verbose=False)  
    raw.set_eeg_reference('average', verbose=False)  

    #Splits EEG data into 5-second epochs with 2.5 seconds overlap
    epochs = mne.make_fixed_length_epochs(raw, duration=5, overlap=2.5, verbose=False)
    data = epochs.get_data()

    features = []
    for epoch in data:
        epoch_features = []
        for channel_data in epoch:
            #Extracts statistical features for each channel in each epoch
            epoch_features.extend([
                np.mean(channel_data),  #Mean
                np.std(channel_data),   #Standard deviation
                np.min(channel_data),   #Minimum
                np.max(channel_data),   # Maximum
                np.percentile(channel_data, 25),  #25th percentile
                np.percentile(channel_data, 75),  #75th percentile
                skew(channel_data),     #Skewness
                kurtosis(channel_data), #Kurtosis
                np.median(channel_data), #Median
                np.sqrt(np.mean(channel_data**2)),  #Root Mean Square 
            ])
        features.append(epoch_features)

    return np.array(features)

def create_dataset(base_directory, save_directory, dataset_type='train', batch_size=50):
    X, y, file_mapping = [], [], []
    batch_count = 0

    #Loops through normal and abnormal labels
    for label_type in ['normal', 'abnormal']:
        current_dir = os.path.join(base_directory, label_type, dataset_type)
        files = [f for f in os.listdir(current_dir) if f.endswith('.edf')]

        #Processes the files in batches
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            for file_name in batch_files:
                file_path = os.path.join(current_dir, file_name)
                label = 1 if label_type == 'abnormal' else 0

                print(f"Processing file: {file_path}")
                features = preprocess_eeg(file_path)  #Extracts features from each file
                X.extend(features)
                y.extend([label] * features.shape[0])  #Adds corresponding label
                file_mapping.extend([file_name] * features.shape[0])  #Keeps track of the file name

            #Saves batch of features and labels 
            os.makedirs(save_directory, exist_ok=True)
            np.save(os.path.join(save_directory, f"X_{dataset_type}_eeg_batch_{batch_count}.npy"), np.array(X))
            np.save(os.path.join(save_directory, f"y_{dataset_type}_eeg_batch_{batch_count}.npy"), np.array(y))
            np.save(os.path.join(save_directory, f"file_mapping_{dataset_type}_batch_{batch_count}.npy"), np.array(file_mapping))
            print(f"Saved batch {batch_count} with {len(X)} samples to {save_directory}.")
            X, y, file_mapping = [], [], []  #Resets lists for the next batch
            batch_count += 1

#Sets base and save directories
base_directory = "/content/drive/MyDrive/FYP/eeg_data"
save_directory = "/content/drive/MyDrive/FYP/new-preprocessed"

#Processes training data 
create_dataset(base_directory, save_directory, 'train')

#Processes eval data
create_dataset(base_directory, save_directory, 'eval')
