import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Loads the preprocessed EEG data 
X_train = np.load("train_eeg.npy")
y_train = np.load("train_labels.npy")
X_val = np.load("eval_eeg.npy")
y_val = np.load("eval_labels.npy")

#Makes sures labels are binary (1,0)
y_train = (y_train > 0.5).astype(int)
y_val = (y_val > 0.5).astype(int)

#Makes sure that data is normalised
X_train = X_train / (np.max(np.abs(X_train), axis=0) + 1e-8)
X_val = X_val / (np.max(np.abs(X_val), axis=0) + 1e-8)

#Prints dataset shapes
print(f"Training Data Shape: {X_train.shape}, Training Labels Shape: {y_train.shape}")
print(f"Validation Data Shape: {X_val.shape}, Validation Labels Shape: {y_val.shape}")

#Checks if shape is correct for CNN-LSTM model
if len(X_train.shape) != 3:
    raise ValueError("X_train must have shape (samples, timesteps, features).")

#CNN-LSTM Model
model = Sequential()

#Input layer: Takes EEG signals in shape (time, features)
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

#First convolutional layer detects EEG patterns across different channels
model.add(Conv1D(64, kernel_size=3))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))  # 🛠️ Using LeakyReLU for better gradient flow

#First pooling layer reduces size to retain important patterns
model.add(MaxPooling1D(pool_size=2))

#Second convolutional layer extracts deeper EEG patterns
model.add(Conv1D(128, kernel_size=3))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))

#Second pooling layer further reduces size for better feature extraction
model.add(MaxPooling1D(pool_size=2))

#First LSTM layer detects temporal patterns in EEG data
model.add(LSTM(64, return_sequences=True))

#Second LSTM layer extracts deeper time-dependent patterns in EEG data
model.add(LSTM(32))

#Fully connected dense layer converts EEG patterns into meaningful values
model.add(Dense(64, activation='relu'))

#Dropout Layer prevents overfitting 
model.add(Dropout(0.4))

#Output Layer outputs 0 (normal) or 1 (abnormal)
model.add(Dense(1, activation='sigmoid'))

#Compiles the model and assess its accuracy
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Prints model summary 
model.summary()

#Trains the model 
training_results = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

#Evaluates the model performance
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {accuracy:.2f}")

#Saves the trained model
model.save("eeg_model.h5")
