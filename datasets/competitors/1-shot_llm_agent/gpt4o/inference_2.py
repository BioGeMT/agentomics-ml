# inference.py

import numpy as np
import pandas as pd
import tensorflow as tf
import sys
from sklearn.metrics import roc_auc_score

# Accept command-line arguments for input and output paths
input_file = sys.argv[sys.argv.index("--input") + 1]
output_file = sys.argv[sys.argv.index("--output") + 1]

# Load the model
model = tf.keras.models.load_model("/home/eddy/Documents/Agentomics-ML/datasets/competitors/1-shot_llm_agent/gpt4o/model_2.h5")

# Load and preprocess the test data
test_data = pd.read_csv(input_file)

# One-hot encode the sequences
def one_hot_encode_sequences(sequences, max_len):
    mapping = {'A': [1, 0, 0, 0, 0], 'G': [0, 1, 0, 0, 0], 'T': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0], 'N': [0, 0, 0, 0, 1]}
    encoded_sequences = np.array([[mapping.get(char, [0,0,0,0,0]) for char in seq] for seq in sequences])
    return encoded_sequences

test_sequences = test_data['sequence'].values
encoded_test_sequences = one_hot_encode_sequences(test_sequences, 251)

# Make predictions
predictions = model.predict(encoded_test_sequences)

# Save the predictions
output_df = pd.DataFrame({'prediction': predictions.flatten()})
output_df.to_csv(output_file, index=False)

# Optionally, if ground truth labels are available for evaluation, compute AUC
# Assuming true labels are provided in a file for evaluation purposes
# true_labels = pd.read_csv(true_labels_file)['class'].values
# auc_score = roc_auc_score(true_labels, predictions)
# print(f"AUC Score: {auc_score}")

