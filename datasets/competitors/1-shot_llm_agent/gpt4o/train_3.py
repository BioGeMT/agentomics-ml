# train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Embedding
from tensorflow.keras.callbacks import EarlyStopping

# Paths for data and model
train_data_path = "/home/eddy/Documents/Agentomics-ML/datasets/human_non_tata_promoters/human_nontata_promoters_train.csv"
model_save_path = "/home/eddy/Documents/Agentomics-ML/datasets/competitors/1-shot_llm_agent/gpt4o/model_3.h5"

# Read and preprocess data
data = pd.read_csv(train_data_path)
sequences = data['sequence'].values
labels = data['class'].values

# One-hot encoding
def one_hot_encode(seq):
    mapping = {'A': [1, 0, 0, 0, 0], 'G': [0, 1, 0, 0, 0], 'T': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0], 'N': [0, 0, 0, 0, 1]}
    return np.array([mapping[nuc] for nuc in seq])

sequences_encoded = np.array([one_hot_encode(seq) for seq in sequences])

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(sequences_encoded, labels, test_size=0.2, random_state=42)

# Model architecture
model = Sequential([
    Conv1D(32, 5, activation='relu', input_shape=(251, 5)),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate the model
y_val_pred = model.predict(X_val).flatten()
auc = roc_auc_score(y_val, y_val_pred)
print(f'Validation AUC: {auc:.4f}')

# Save the model
model.save(model_save_path)

