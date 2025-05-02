import asyncio
import os
from metagpt.roles.di.data_interpreter import DataInterpreter

async def main(requirement: str):
    # Create DataInterpreter with all tools enabled
    role = DataInterpreter(tools=["<all>"])
    # Run DataInterpreter with the requirement
    await role.run(requirement)

if __name__ == "__main__":
    # Define the paths relative to the workspace
    train_csv = "human_nontata_promoters_train.csv"
    test_csv = "human_nontata_promoters_test.csv"
    output_dir = "output"
    model_path = f"{output_dir}/model.h5"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the requirement
    requirement = f"""
    You are an expert bioinformatics ML engineer. Create a machine learning model for DNA sequence classification.

    DATASET:
    - Training file: {train_csv}
    - Test file: {test_csv}
    - Format: CSV with columns 'sequence' (DNA sequence, 251 nucleotides long) and 'class' (0 or 1)
    - Contains sequences of nucleotides 'A', 'G', 'T', 'C' and 'N'
    - Classifies non-TATA promoters (class=1) vs non-promoters (class=0)


    REQUIREMENTS:
    1. Load and explore the training dataset: {train_csv}

    2. Load and explore the testing dataset: {test_csv}

    3. Create a python script train.py which will carry out the following steps:
    - Preprocess the DNA sequences by encoding them appropriately.
    - Build a neural network classifier (CNN or LSTM) with 20% validation split
    - Save the trained model to this path: {model_path}

    4. Create a python script inference.py which will carry out the following steps:
    - Accept arguments: --input and --output
    - Load the model from this path: {model_path}
    - Output a CSV with column 'prediction' containing RAW PROBABILITIES (not binary classes)
    - Use pd.DataFrame({{'prediction': predictions.flatten()}}) to save predictions
    - Compute and report AUC

    Provide complete code for both scripts with "# train.py" and "# inference.py" headers.
    """
    
    # Run the main function with the requirement
    asyncio.run(main(requirement))
