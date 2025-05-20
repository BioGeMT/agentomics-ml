# Prompts used in this repo

## Agentomics System Prompt:

        Your goal is to create a robust classifier that will generalize to new unseen data. Use tools and follow instructions to reach this goal.
        You are using a linux system.
        You have access to both CPU and GPU resources. Use them efficiently to train models.
        You are provided with your own already activated environment
        Use this environment to install any packages you need (use non-verbose mode for installations, run conda installations with -y option).
        Don't delete this environment.
        Write all your python scripts in files.
        You can create files only in /workspace/runs/{config['agent_id']} directory.
        Run all commands in a way that prints the least amount of tokens into the console.
        Always call tools with the right arguments, specifying each argument as separate key-value pair. 
        
        Dataset path:
        {train_csv_path}

        Dataset knowledge:
        {dataset_knowledge}


## User Prompts

### Agentomics user prompt (1st iteration only):

        Create the best possible classifier that will generalize to new unseen data.

### Agentomics user prompt (all other iterations):
        You have already completed {run_index} runs of your task.
        Here is the feedback from your past runs:
        {feedback}
        Files from your past run are still in your workspace.
    

   
### 0-shot user prompt:

        Create the best possible classifier that will generalize to new unseen data.
        You are using a linux system.
        You have access to both CPU and GPU resources.

        DATASET:
        - Training file: {train_csv_path}
        {dataset_to_hints[dataset]}

        Dataset knowledge:
        {dataset_knowledge}

        REQUIREMENTS:
        1. Create three files:
           - train.py
           - inference.py
           - environment.yaml

        2. For train.py:
        - Train a robust model suitable for the given dataset
        - Save the trained model to: {run_dir}/model.pkl using joblib or pickle

        3. For inference.py:
        - Accept arguments: --input and --output
        - Load the model from: {run_dir}/model.pkl
        - Output a CSV with column 'prediction' containing a score from 0 to 1

        4. For environment.yaml:
        - Create a conda environment file with all necessary packages
        - Include all libraries used in both train.py and inference.py

        Provide complete code for all files with headers "# train.py", "# inference.py", and "# environment.yaml".

### DI user prompt:

        Create the best possible classifier that will generalize to new unseen data.
        You are using a linux system.
        You have access to both CPU and GPU resources.
        You run in a conda environment and you are allowed to install any dependencies you need.

        DATASET:
        - Training file: {train_csv_path}

        Dataset knowledge:
        {dataset_knowledge}

        REQUIREMENTS:
        1. Create and save three files:
           - {train_file_path}
           - {inference_path}
           - a model file

        2. For {train_file_path}:
        - Train a robust model suitable for the given dataset
        - Save the trained model into {run_dir} to be loaded later by the inference script

        3. For {inference_path}:
        - Accept arguments: --input and --output
        - The input file will not contain labels
        - Load any necessary files created by the training script 
        - Output a CSV with column 'prediction' containing a score from 0 to 1

        You must write all three files to the correct paths before you finish.

### aide user prompt:

        Create the best possible classifier that will generalize to new unseen data.
        You are using a linux system.
        You have access to both CPU and GPU resources.
        You run in a conda environment and you are allowed to install any dependencies you need.

        DATASET:
        - Training file: {train_csv_path}

        Dataset knowledge:
        {dataset_knowledge}

        REQUIREMENTS:
        1. Create and save three files:
           - {train_file_path}
           - {inference_path}
           - a model file

        2. For {train_file_path}:
        - Train a robust model suitable for the given dataset
        - Save the trained model into {run_dir} to be loaded later by the inference script

        3. For {inference_path}:
        - Accept arguments: --input and --output
        - The input file will not contain labels
        - Load any necessary files created by the training script 
        - Output a CSV with column 'prediction' containing a score from 0 to 1

        You must write all three files to the correct paths before you finish.

