import numpy as np
from sklearn.model_selection import train_test_split

#Loads the preprocessed EEG data
final_segment = np.load("processed_eeg.npy")

#Loads the labels
labels = np.load("labels.npy") 

#Prints shape of EEG data and labels
print(f"EEG Data Shape: {final_segment.shape}")  
print(f"Labels Shape: {labels.shape}")  

#Splits the data into 80% for training and 20% for testing sets
X_train, X_val, y_train, y_val = train_test_split(final_segment, labels, test_size=0.2, random_state=42)

#Print the shapes of the training and validation data
print(f"Training Data Shape: {X_train.shape}, Training Labels Shape: {y_train.shape}")
print(f"Validation Data Shape: {X_val.shape}, Validation Labels Shape: {y_val.shape}")