# inference.py

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import argparse

# Paths for model
model_load_path = "/home/eddy/Documents/Agentomics-ML/datasets/competitors/1-shot_llm_agent/gpt4o/model_3.h5"

# Argument parser
parser = argparse.ArgumentParser(description="Inference script for DNA sequence classification")
parser.add_argument('--input', required=True, help='Input file path')
parser.add_argument('--output', required=True, help='Output file path')

args = parser.parse_args()

# Load model
model = tf.keras.models.load_model(model_load_path)

# Read and preprocess data
input_data_path = args.input
output_data_path = args.output

try:
    data = pd.read_csv(input_data_path)
except Exception as e:
    raise RuntimeError("Error reading input file. Ensure it is a CSV file with a 'sequence' column.") from e

sequences = data['sequence'].values

# One-hot encoding
def one_hot_encode(seq):
    mapping = {'A': [1, 0, 0, 0, 0], 'G': [0, 1, 0, 0, 0], 'T': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0], 'N': [0, 0, 0, 0, 1]}
    return np.array([mapping.get(nuc, [0, 0, 0, 0, 1]) for nuc in seq])

sequences_encoded = np.array([one_hot_encode(seq) for seq in sequences])

# Make predictions
predictions = model.predict(sequences_encoded).flatten()

# Save predictions
output_df = pd.DataFrame({'prediction': predictions})
output_df.to_csv(output_data_path, index=False)

print(f"Predictions saved to {output_data_path}")

