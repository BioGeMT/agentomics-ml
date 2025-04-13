# train.py

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Constants
TRAIN_FILE = "/home/eddy/Documents/Agentomics-ML/datasets/human_non_tata_promoters/human_nontata_promoters_train.csv"
MODEL_PATH = "/home/eddy/Documents/Agentomics-ML/datasets/competitors/1-shot_llm_agent/gpt4o/model_1.h5"

# Function to one-hot encode DNA sequences
def one_hot_encode(sequences):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    encoded = np.zeros((len(sequences), 251, 5), dtype=int)
    for i, sequence in enumerate(sequences):
        for j, nucleotide in enumerate(sequence):
            if nucleotide in mapping:
                encoded[i, j, mapping[nucleotide]] = 1
    return encoded

# Load and preprocess data
data = pd.read_csv(TRAIN_FILE)
sequences = data['sequence'].values
labels = data['class'].values

X = one_hot_encode(sequences)
y = labels

# Create train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(251, 5)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Save the model
model.save(MODEL_PATH)

# Evaluate the model using AUC
y_pred = model.predict(X_val).flatten()
auc = roc_auc_score(y_val, y_pred)
print(f'Validation AUC: {auc:.4f}')
