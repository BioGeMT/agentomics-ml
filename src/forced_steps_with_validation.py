import dotenv
import os
import asyncio

from rich.console import Console
from pydantic import Field, BaseModel, field_validator
from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior, UsageLimitExceeded, capture_run_messages
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits
import wandb

from prompts.prompts_utils import load_prompts
from tools.bash import create_bash_tool
from tools.write_python_tool import create_write_python_tool
from run_logging.evaluate_log_run import evaluate_log_run
from run_logging.wandb import setup_logging
from utils.create_user import create_new_user_and_rundir
from utils.models import MODELS

dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
wandb_key = os.getenv("WANDB_API_KEY")
console = Console()

async def run_agent(agent: Agent, user_prompt: str, max_steps: int, result_type: BaseModel, message_history: list | None):
    """
    Executes the agent with the given prompt and returns the structured result or exception.
    """
    with capture_run_messages() as messages:
        try:
            async with agent.iter(
                user_prompt=user_prompt,
                usage_limits=UsageLimits(request_limit=max_steps),
                result_type=result_type,
                message_history=message_history,
            ) as agent_run:
                async for node in agent_run:
                    console.log(node)
                return agent_run.result.all_messages()
        except (UnexpectedModelBehavior, UsageLimitExceeded) as e:
            console.log("Exception occurred", e)
            console.log('Cause:', repr(e.__cause__))
            console.log("Messages:", messages)
            return e

async def main():
    # Store the full context from the previous run
    previous_run_context = None

    # Initialize a new agent environment
    agent_id = create_new_user_and_rundir()
    config = {
                "agent_id": agent_id,
                "model": MODELS.GPT4o,
                "temperature": 1,
                "max_steps": 30,
                "max_run_retries": 1,
                "max_validation_retries": 5,
                "tags": ["double_run"],
                "dataset": "human_non_tata_promoters",
                "prompt": "toolcalling_agent.yaml",
            }
    model = OpenAIModel(
                config['model'],
                provider=OpenAIProvider(api_key=api_key)
            )

    agent = Agent(
        model=model,
        system_prompt=load_prompts(config["prompt"])["system_prompt"],
        tools=[
            create_bash_tool(
                agent_id=config['agent_id'],
                timeout=60 * 15,
                autoconda=True,
                max_retries=1
            ),
            create_write_python_tool(
                agent_id=config['agent_id'],
                timeout=60 * 5,
                add_code_to_response=False,
                max_retries=1
            ),
        ],
        model_settings={'temperature': config['temperature']},
        retries=config["max_run_retries"],
        result_retries=config["max_validation_retries"],
    )
    setup_logging(config, api_key=wandb_key)
    for run_index in range(2):

        # Construct the prompt, including prior context if available
        if run_index == 0:
            base_prompt = f"""
You are working on a Linux system with CPU only (no GPU).
Your working directory is /workspace/runs/{config['agent_id']}.
You have an activated environment named {config['agent_id']}_env.
Install any required packages using non-verbose mode; for conda use '-y'.
Write all Python scripts to files and execute them within this environment.
Your final deliverable is an inference.py script at /workspace/runs/{config['agent_id']}/inference.py.
Ensure the trained model file exists beforehand.
Paths in inference.py must be absolute (e.g., /workspace/runs/{config['agent_id']}/model.pth).
The script must accept arguments:
    --input (path to input CSV, same format as training data without labels)
    --output (path to output CSV with one column named 'prediction')
Explore data under workspace/datasets/{config['dataset']}/ using the '*_train.csv' file.
Build the best generalizing classifier.
Run commands with minimal console output.
"""
        else:
            base_prompt = f"""
Here is the full context from the previous run:
{previous_run_context}
Ensure that there is an 'inference.py' script at /workspace/runs/{config['agent_id']}/inference.py.
- If the previous run failed to create this script, generate it from scratch following the same input/output specs.
- If the script exists, inspect its contents, describe what improvements you'd make, then implement those changes.
Use the existing context to guide your refinements.
"""

        try:
            run_context = await run_architecture(agent, config, base_prompt, iteration=run_index)
        except Exception as e:
            console.log("Error during pipeline execution:", e)
            run_context = {"error": str(e), "traceback": repr(e)}

        if run_index == 0:
            previous_run_context = run_context

    evaluate_log_run(config)
    wandb.finish()

async def run_architecture(agent: Agent, config: dict, base_prompt: str, iteration: int):
    """
    Run the agent through data exploration, train/validation split, representation,
    model design, and validation. Returns the final context from validation.
    """
    class DataExplorationReasoning(BaseModel):
        data_description: str = Field(
            description="Description of the data, including summary statistics and domain insights."
        )
    messages_data_exploration = await run_agent(
        agent=agent,
        user_prompt=base_prompt + "\nYour first task: explore the dataset.",
        max_steps=config["max_steps"],
        result_type=DataExplorationReasoning,
        message_history=None,
    )

    class DataSplitReasoning(BaseModel):
        train_path: str = Field(description="Path to generated train.csv file")
        val_path: str = Field(description="Path to generated validation.csv file")


    if iteration == 0:
        split_prompt = f"""
Split the training dataset into training and validation sets:
- Save 'train.csv' and 'validation.csv' in /workspace/runs/{config['agent_id']}/.
Return the absolute paths to these files.
"""
        messages_split = await run_agent(
            agent=agent,
            user_prompt=split_prompt,
            max_steps=config["max_steps"],
            result_type=DataSplitReasoning,
            message_history=messages_data_exploration,
        )
    else:
        messages_split = messages_data_exploration

    class RepresentationReasoning(BaseModel):
        representation: str = Field(description="Instructions for how to represent the data before modeling.")
        reasoning: str = Field(description="Reasoning behind your chosen representation.")

    messages_representation = await run_agent(
        agent=agent,
        user_prompt="Next task: define the data representation. (Do not implement yet.)",
        max_steps=config["max_steps"],
        result_type=RepresentationReasoning,
        message_history=messages_split,
    )

    class ModelArchitectureReasoning(BaseModel):
        architecture: str = Field(description="Chosen model type and architecture.")
        reasoning: str = Field(description="Reasoning behind architectural choices.")

    messages_architecture = await run_agent(
        agent=agent,
        user_prompt="Next task: choose the model architecture. (Do not implement yet.)",
        max_steps=config["max_steps"],
        result_type=ModelArchitectureReasoning,
        message_history=messages_representation,
    )

    class InferenceScriptReasoning(BaseModel):
        path_to_inference: str = Field(description="Absolute path to the inference.py script")

    inference_file = f"/workspace/runs/{config['agent_id']}/inference.py"
    if not os.path.exists(inference_file):
        inference_prompt = f"""
The file '{inference_file}' was not found.
Generate a new 'inference.py' script at this path following spec:
- Accept --input and --output arguments
- Load the trained model from '/workspace/runs/{config['agent_id']}/model.pth'
- Write predictions to CSV with column 'prediction'
"""
    else:
        inference_prompt = f"""
The file '{inference_file}' exists.
Inspect its contents, describe any improvements, then implement those changes in place.
Follow the same input/output specs and update the script accordingly.
"""

    messages_inference = await run_agent(
        agent=agent,
        user_prompt=inference_prompt,
        max_steps=config["max_steps"],
        result_type=InferenceScriptReasoning,
        message_history=messages_architecture,
    )

    class ValidationReasoning(BaseModel):
        auprc: float = Field(description="AUPRC on validation set.")
        auroc: float = Field(description="AUROC on validation set.")
        returncode: int = Field(description="Exit code of the inference script.")

    validation_context = await run_agent(
        agent=agent,
        user_prompt=f"""
Run 'inference.py' on the validation.csv file and report:
- AUPRC
- AUROC
- Exit code
""",
        max_steps=config["max_steps"],
        result_type=ValidationReasoning,
        message_history=messages_inference,
    )

    return validation_context

if __name__ == "__main__":
    asyncio.run(main())

