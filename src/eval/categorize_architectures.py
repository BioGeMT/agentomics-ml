#!/usr/bin/env python3
"""
Categorize model architectures from the extracted CSV.

This script reads the ablation_architectures.csv file and categorizes each
architecture into predefined categories using an LLM.
"""

import os
import pandas as pd
import argparse
from ollama import Client
from dotenv import load_dotenv

# Categories for architecture classification
CATEGORIES = [
    "XGBoost",
    "Random Forest",
    "Custom Ensemble",
    "CNN",
    "Linear Regression",
    "Hybrid Model",
    "Transformer",
    "MLP",
    "SVM",
    "Log. Regression",
    "FM Fine-Tuning",
    "Naive Bayes"
]


def init_ollama_client():
    """Initialize Ollama client with credentials"""
    load_dotenv()

    api_key = os.environ.get('OLLAMA_API_KEY')
    if not api_key:
        raise ValueError("OLLAMA_API_KEY must be set in .env")

    client = Client(
        host="https://ollama.com",
        headers={'Authorization': f'Bearer {api_key}'}
    )

    return client


def categorize_architecture(client, architecture_text: str, model: str = 'gpt-oss:120b', timeout: int = 5) -> str:
    """
    Categorize a single architecture description using LLM.

    Args:
        client: Ollama client
        architecture_text: The architecture description to categorize
        model: The model to use for categorization
        timeout: Timeout in seconds for the API call

    Returns:
        The category name
    """
    import time

    categories_str = '", "'.join(CATEGORIES)

    messages = [
        {
            'role': 'system',
            'content': f'Reply with only one of the following categories: "{categories_str}". Do not provide any additional text or explanation. Reply with exactly one category name.',
        },
        {
            'role': 'user',
            'content': f'Categorize this architecture description into one of the following categories: "{categories_str}".\n\nArchitecture description: "{architecture_text}"',
        },
    ]

    # Add a short delay between requests to avoid rate limiting
    time.sleep(timeout)

    # Get response from LLM (non-streaming for cleaner parsing)
    response = client.chat(model, messages=messages, stream=False)
    category = response['message']['content'].strip()

    # Clean up response (remove quotes, extra whitespace)
    category = category.strip('"').strip("'").strip()

    return category


def main():
    """Main function to categorize architectures from CSV"""

    parser = argparse.ArgumentParser(description="Categorize architectures from extracted CSV")
    parser.add_argument(
        '--input-file',
        type=str,
        default='outputs/ablation_architectures.csv',
        help='Path to the input CSV file with architectures (default: outputs/ablation_architectures.csv)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='outputs/ablation_architectures_categorized.csv',
        help='Path to the output CSV file with categories (default: outputs/ablation_architectures_categorized.csv)'
    )
    args = parser.parse_args()

    # Input and output file paths
    input_file = args.input_file
    output_file = args.output_file

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        print("Please run extract_architectures.py first")
        return

    # Load the CSV
    print(f"Loading architectures from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Found {len(df)} architectures to categorize\n")

    # Initialize Ollama client
    print("Initializing Ollama client...")
    client = init_ollama_client()

    # Categorize each architecture
    categories = []
    for idx, row in df.iterrows():
        agent_id = row['agent_id']
        iteration = row['iteration']
        architecture = row['architecture']

        print(f"[{idx+1}/{len(df)}] Categorizing {agent_id} iteration {iteration}...", end=' ')

        try:
            category = categorize_architecture(client, architecture)
            categories.append(category)
            print(f"→ {category}")
        except Exception as e:
            print(f"ERROR: {e}")
            categories.append("Error")

    # Add category column
    df['category'] = categories

    # Save to output CSV
    df.to_csv(output_file, index=False)

    print(f"\n✓ Saved categorized architectures to {output_file}")
    print(f"  Total rows: {len(df)}")

    # Print category distribution
    print(f"\nCategory distribution:")
    print(df['category'].value_counts())


if __name__ == "__main__":
    main()