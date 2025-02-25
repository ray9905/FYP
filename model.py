import numpy as np
from sklearn.model_selection import train_test_split

#Loads the preprocessed EEG data
final_segment = np.load("processed_eeg.npy")

#Loads the labels
labels = np.load("labels.npy") 

#Prints shape of EEG data and labels
print(f"EEG Data Shape: {final_segment.shape}")  
print(f"Labels Shape: {labels.shape}")  
