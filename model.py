import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt

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
model = Sequential()

#Input layer
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

#First convolutional layer detects EEG patterns across different channels
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())

#First pooling layer reduces size 
model.add(MaxPooling1D(pool_size=2))

#Second convolutional layer extracts deeper EEG patterns
model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())

#Second pooling layer further reduces size
model.add(MaxPooling1D(pool_size=2))

#First LSTM layer detects temporal patterns in EEG data
model.add(LSTM(64, return_sequences=True))

#Second LSTM layer extracts deeper time dependent patterns in EEG data
model.add(LSTM(32))

#Fully connected dense layer converts EEG patterns into meaningful values
model.add(Dense(64, activation='relu'))

#Dropout Layer prevents overfitting
model.add(Dropout(0.3))

#Output Layer outputs 0 (normal) or 1 (abnormal)
model.add(Dense(1, activation='sigmoid'))

#Compiles the model and assesses its accuracy
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Prints summary
model.summary()

#Trains the model
training_results = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

#Evaluates the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy is: {accuracy:.2f}")

#Saves the model
model.save("eeg_model.h5")
