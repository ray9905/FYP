import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt

# 🔹 Load the preprocessed EEG data (Updated filenames)
X_train = np.load("train_eeg.npy")
y_train = np.load("train_labels.npy")
X_val = np.load("eval_eeg.npy")
y_val = np.load("eval_labels.npy")

#Prints the dataset shapes
print(f"Training Data Shape: {X_train.shape}, Training Labels Shape: {y_train.shape}")
print(f"Validation Data Shape: {X_val.shape}, Validation Labels Shape: {y_val.shape}")

#Makes sure the data is converted to 0-1 range
X_train = X_train / np.max(np.abs(X_train), axis=0)
X_val = X_val / np.max(np.abs(X_val), axis=0)

#Makes sure shape works with CNN-LSTM 
print(f"Adjusted Training Data Shape: {X_train.shape}")
print(f"Adjusted Validation Data Shape: {X_val.shape}")

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
