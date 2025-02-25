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

#CNN-LSTM Model
model = Sequential([

    #First convolutional layer detects EEG patterns across different channels
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    
    #First pooling layer reduces size 
    MaxPooling1D(pool_size=2),

    #Second convolutional layer extracts deeper EEG patterns
    Conv1D(128, kernel_size=3, activation='relu'),

    #Second pooling layer further reduces size
    MaxPooling1D(pool_size=2),

    #First LSTM layer detects temporal patterns in EEG data
    LSTM(64, return_sequences=True),

    #Second LSTM layer extracts deeper time dependent patterns in EEG data
    LSTM(32),

    #Fully connected dense layer converts EEG patterns into meaningful values
    Dense(64, activation='relu'),

    #Dropout Layer prevents overfitting
    Dropout(0.3),

    #Output Layer outputs 0 (normal) or 1 (abnormal)
    Dense(1, activation='sigmoid')
])