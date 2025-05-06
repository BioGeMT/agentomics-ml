import dotenv
import os
import asyncio
import traceback

from rich.console import Console
from pydantic import Field, BaseModel
from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior, UsageLimitExceeded, capture_run_messages
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits
import wandb
from openai import AsyncOpenAI

from prompts.prompts_utils import load_prompts
from tools.bash import create_bash_tool
from tools.write_python_tool import create_write_python_tool
from run_logging.evaluate_log_run import evaluate_log_run, dry_run_evaluate_log_run
from run_logging.wandb import setup_logging
from utils.create_user import create_new_user_and_rundir
from utils.models import MODELS

dotenv.load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
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
    previous_run_context = None

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
        "prompt": "toolcalling_agent.yaml"
    }

    client = AsyncOpenAI(
        base_url='https://openrouter.ai/api/v1',
        api_key=os.getenv('OPENROUTER_API_KEY')
    )

    model = OpenAIModel(
        config['model'],
        provider=OpenAIProvider(openai_client=client)
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

    for run_index in range(2):
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
            run_context = {"error": str(e), "traceback": traceback.format_exc()}

        if run_index == 0:
            previous_run_context = run_context

    wandb.finish()

async def run_architecture(agent: Agent, config: dict, base_prompt: str, iteration: int):
    """
    Run the agent through data exploration, train/validation split, representation,
    model design, and validation. After creating inference.py, export the conda environment
    and then validate the script.
    Returns the final context from validation.
    """
    # 1) Data exploration
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

    # 2) Train/validation split (only on first iteration)
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

    # 3) Representation reasoning
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

    # 4) Model architecture reasoning
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

    # 5) Final script generation (sólo construcción del modelo de datos)
    class FinalOutcome(BaseModel):
        path_to_inference_file: str = Field(description="Absolute path to the inference.py file")

        @staticmethod
        def validate_inference(path: str) -> str:
            if not os.path.isfile(path):
                raise ModelRetry(f"Inference script not found: {path}")
            return path

    # Ejecutar al agente
    messages = await run_agent(
        agent=agent,
        user_prompt="Continue with your task",
        max_steps=config["max_steps"],
        result_type=FinalOutcome,
        message_history=messages_architecture,
    )

            # Exportar el entorno de conda usando subprocess async nativo
    console.log("Exportando entorno de conda a YAML…")
    export_cmd = (
        f"conda env export > "
        f"/workspace/runs/{config['agent_id']}/{config['agent_id']}_env.yaml"
    )
    try:
        # Usar asyncio para ejecutar el comando sin depender de Agent tools
        proc = await asyncio.create_subprocess_shell(
            export_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        console.log("stdout:", stdout.decode().strip())
        if stderr:
            console.log("stderr:", stderr.decode().strip())
        if proc.returncode != 0:
            raise ModelRetry(f"Conda export failed with code {proc.returncode}")
        console.log("Entorno exportado con éxito")
    except Exception as e:
        console.log("Error exportando entorno de conda:", e)
        console.log(traceback.format_exc())
        raise

    # Validar inference.py mediante dry_run mediante dry_run
    console.log("Validando inference.py…")
    eval_output = dry_run_evaluate_log_run(config)
    console.log(eval_output)
    if eval_output.returncode != 0:
        console.log("Validation failed, retrying...")
        raise ModelRetry(str(eval_output))
    console.log("Validando inference.py…")
    eval_output = dry_run_evaluate_log_run(config)
    console.log(eval_output)
    if eval_output.returncode != 0:
        console.log("Validation failed, retrying...")
        raise ModelRetry(str(eval_output))

    return messages

if __name__ == "__main__":
    asyncio.run(main())

