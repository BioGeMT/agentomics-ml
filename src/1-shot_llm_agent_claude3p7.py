#!/usr/bin/env python3
"""
Simple one-shot LLM test script using both OpenAI and Anthropic libraries.
This script calls an LLM to generate train.py and inference.py scripts in one shot.
The response is printed to the console for manual copy-pasting.
"""

import os
import argparse
import dotenv
from openai import OpenAI
import anthropic


def main():
    # Load environment variables from .env file
    dotenv.load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate ML code with one-shot LLM")
    parser.add_argument("--dataset", default="human_non_tata_promoters",
                        help="Dataset name")
    parser.add_argument("--model", default="claude-3-7-sonnet-20250219",
                        help="Model to use")
    parser.add_argument("--temp", type=float, default=0.7,
                        help="Temperature for LLM generation")
    parser.add_argument("--api", default="anthropic", choices=["openai", "openrouter", "anthropic"],
                        help="API to use (openai, openrouter, or anthropic)")
    args = parser.parse_args()

    # Load API key from environment
    api_key = None
    base_url = None

    if args.api == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            # Try OPENAI_API_KEY as fallback for OpenRouter (common mistake)
            api_key = os.getenv("OPENAI_API_KEY")
        base_url = "https://openrouter.ai/api/v1"
    elif args.api == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    else:  # openai
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: No API key found in environment variables")
        print("Please set your OPENAI_API_KEY, OPENROUTER_API_KEY, or ANTHROPIC_API_KEY environment variable")
        return

    print(f"Using model: {args.model} with {args.api} API")

    # Create the prompt
    prompt = f"""
    You are an expert bioinformatics ML engineer. I need you to create a machine learning model for a classification task.

    - Create a machine learning classifier for the dataset: {args.dataset}
    - Training file: /home/eddy/Documents/Agentomics-ML/datasets/{args.dataset}/human_nontata_promoters_train.csv
    - Test file: /home/eddy/Documents/Agentomics-ML/datasets/{args.dataset}/human_nontata_promoters_test.no_label.csv

    DATASET INFO:
    - This dataset contains DNA sequences for classification
    - Format: CSV with columns 'sequence' (DNA sequence) and 'class' (0 or 1)
    - The sequences represent non-TATA promoters (class=1) or non-promoters (class=0)
    - Each sequence is about 251 nucleotides long, specifically: the sequences have a uniform length of 251 characters
    - The dataset is balanced (50% positive, 50% negative examples)
    - The dataset contains sequences of nucleotides 'A', 'G', 'T', 'C' and 'N'

    REQUIREMENTS:
    1. You must write TWO Python scripts:
       - train.py: to train and save the model
       - inference.py: to load the model and make predictions

    2. The inference.py script must:
       - Accept command-line arguments: 
            --input (input file path: /home/eddy/Documents/Agentomics-ML/datasets/competitors/1-shot_llm_agent/gpt4o)
            --output (output file path: /home/eddy/Documents/Agentomics-ML/datasets/competitors/1-shot_llm_agent/gpt4o)
       - Output a CSV file with a column named 'prediction'
       - Use ABSOLUTE PATHS for any file references
       - Handle input file format issues appropriately
       - Must compute and report the AUC (Area Under the ROC Curve) as the final score to evaluate the performance of the model.

    3. You should create a machine learning model that:
       - Is appropriate for DNA sequence classification
       - Has good generalization ability
       - Uses appropriate sequence encoding (one-hot, k-mer, etc.)

    4. IMPORTANT: The inference.py script must save raw probability scores (NOT binary classifications) to the prediction column.
       - DO NOT convert predictions to binary before saving
       - INCORRECT: output = pd.DataFrame({{'prediction': (predictions > 0.5).astype(int)}})
       - CORRECT: output = pd.DataFrame({{'prediction': predictions.flatten()}})
       - The raw probabilities are needed to calculate AUC and other metrics properly

    5. Ensure that the final tensor shape after preprocessing matches the expected input dimension for the Dense layers. 
       If using one-hot encoding or a flattening layer, verify that the product of the sequence length and number of features 
       matches the Dense layer's input. If there is any discrepancy between the computed shape and the expected input dimension, 
       adjust the preprocessing pipeline or modify the model architecture to avoid dimension mismatches.

    6. VERY IMPORTANT: Save the trained model file with a specific path and filename:
       - The model MUST be saved to: {args.output_dir}/model_{run_number}.h5
       - The inference.py script MUST load the model from this exact path
       - DO NOT save the model to the default models directory
       - Example code for saving: model.save("{args.output_dir}/model_{run_number}.h5")
       - Example code for loading: model = tf.keras.models.load_model("{args.output_dir}/model_{run_number}.h5")

    First, explain your approach - what model architecture you'll use and why, how you'll encode the DNA sequences, etc.

    Then, provide the complete code for both train.py and inference.py scripts.
    Use clear headers like "# train.py" and "# inference.py" to separate the scripts.
    """

    # Call the LLM
    print("Sending request to LLM...")

    try:
        if args.api == "anthropic":
            # Initialize Anthropic client
            client = anthropic.Anthropic(api_key=api_key)

            # Call Claude
            response = client.messages.create(
                model=args.model,
                max_tokens=4000,
                temperature=args.temp,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract content from the response
            output = response.content[0].text

        elif args.api == "openrouter":
            # Initialize OpenAI client for OpenRouter
            client = OpenAI(api_key=api_key, base_url=base_url)

            # Set headers for OpenRouter
            headers = {
                "HTTP-Referer": "https://localhost",  # Your URL
                "X-Title": "1-Shot LLM Test"  # Optional, for analytics
            }

            # Call OpenRouter
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temp,
                max_tokens=4000,
                headers=headers
            )

            # Extract content from the response
            output = response.choices[0].message.content

        else:  # openai
            # Initialize OpenAI client
            client = OpenAI(api_key=api_key)

            # Call OpenAI
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temp,
                max_tokens=4000
            )

            # Extract content from the response
            output = response.choices[0].message.content

        # Print the response for copying
        print("\n" + "=" * 80)
        print("LLM RESPONSE - COPY FROM BELOW:")
        print("=" * 80 + "\n")

        print(output)

        print("\n" + "=" * 80)
        print("END OF RESPONSE - COPY UNTIL ABOVE")
        print("=" * 80)

    except Exception as e:
        print(f"Error during LLM call: {e}")


if __name__ == "__main__":
    main()