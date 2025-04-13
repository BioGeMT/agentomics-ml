# train.py

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load and preprocess the data
train_file = "/home/eddy/Documents/Agentomics-ML/datasets/human_non_tata_promoters/human_nontata_promoters_train.csv"
data = pd.read_csv(train_file)

# One-hot encode the sequences
def one_hot_encode_sequences(sequences, max_len):
    mapping = {'A': [1, 0, 0, 0, 0], 'G': [0, 1, 0, 0, 0], 'T': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0], 'N': [0, 0, 0, 0, 1]}
    encoded_sequences = np.array([[mapping[char] for char in seq] for seq in sequences])
    return encoded_sequences

sequences = data['sequence'].values
labels = data['class'].values
encoded_sequences = one_hot_encode_sequences(sequences, 251)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(encoded_sequences, labels, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(251, 5)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the model
model.save("/home/eddy/Documents/Agentomics-ML/datasets/competitors/1-shot_llm_agent/gpt4o/model_2.h5")

