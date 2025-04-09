import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# ✅ Set Paths
SAVE_PATH = "/content/drive/My Drive/FYP"

#Loads Training Data (Perfectly Balanced)
X_train = np.load(f"{SAVE_PATH}/preprocessed_v3_train_perfectly_balanced_data.npy")
y_train = np.load(f"{SAVE_PATH}/preprocessed_v3_train_perfectly_balanced_labels.npy")

#Train Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

#Reshaped for CNN-LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

#Converts labels 
if len(y_train.shape) == 1:
    y_train = to_categorical(y_train, num_classes=2)
if len(y_val.shape) == 1:
    y_val = to_categorical(y_val, num_classes=2)

print(f"Training Data Shape: {X_train.shape}, y_train: {y_train.shape}")
print(f"Validation Data Shape: {X_val.shape}, y_val: {y_val.shape}")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

#Defines the Hybrid Model 
model = Sequential([
    Conv1D(filters=64, kernel_size=5, activation='relu', padding='same', input_shape=(X_train.shape[1], 1)),
    LayerNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
    LayerNormalization(),
    MaxPooling1D(pool_size=2),

    LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
    LSTM(32, dropout=0.3, recurrent_dropout=0.3),

    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dropout(0.3),

    Dense(2, activation='softmax')
])

#Compile Hybrid Model
model.compile(optimizer=Adam(learning_rate=0.0005), loss="categorical_crossentropy", metrics=['accuracy'])

print("Hybrid Model Compiled")
#Hybrid Model Save Path
model_save_path = f"{SAVE_PATH}/model2_full_train.keras"

#Callbacks
checkpoint = ModelCheckpoint(model_save_path, monitor="val_loss", save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

#Train for 20 Epochs
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=256,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stopping]
)

print(f"Hybrid Full Training Complete {model_save_path}")
