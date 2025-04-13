# inference.py

import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from sklearn.metrics import roc_auc_score

# Constants
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

def main(input_file, output_file):
    # Load the test data
    test_data = pd.read_csv(input_file)
    test_sequences = test_data['sequence'].values

    # One-hot encode the test sequences
    X_test = one_hot_encode(test_sequences)

    # Load the trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Make predictions
    predictions = model.predict(X_test)

    # Save the predictions
    output = pd.DataFrame({'prediction': predictions.flatten()})
    output.to_csv(output_file, index=False)

    # For demonstration purposes, if labels are available, calculate AUC
    if 'class' in test_data.columns:
        y_true = test_data['class'].values
        auc = roc_auc_score(y_true, predictions)
        print(f'Test AUC: {auc:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script for DNA sequence classification')
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to the output CSV file')
    
    args = parser.parse_args()
    main(args.input, args.output)
