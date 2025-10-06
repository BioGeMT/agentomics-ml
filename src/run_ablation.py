import asyncio
import argparse
import dotenv
from pathlib import Path
from run_agent import run_experiment
from utils.providers.provider import get_provider_from_string, Provider

ABLATION_CONFIGS = [
    {'name': 'baseline', 'steps_to_skip': []},
    {'name': 'no_data_exploration', 'steps_to_skip': ['data_exploration']},
    {'name': 'no_data_split', 'steps_to_skip': ['data_split']},
    {'name': 'no_data_representation', 'steps_to_skip': ['data_representation']},
    {'name': 'no_model_architecture', 'steps_to_skip': ['model_architecture']},
    {'name': 'no_model_training', 'steps_to_skip': ['model_training']},
    {'name': 'no_final_outcome', 'steps_to_skip': ['final_outcome']}
]

def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation study across models and datasets")
    parser.add_argument('--models', nargs='+', required=True, help='List of models to test')
    parser.add_argument('--datasets', nargs='+', required=True, help='List of datasets to test')
    parser.add_argument('--provider', required=True, help=f'LLM provider. Available: {Provider.get_available_providers()}')
    parser.add_argument('--val-metric', required=True, help='Validation metric (ACC, F1, AUROC, etc.)')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations per run')
    parser.add_argument('--user-prompt', type=str, default="Create the best possible machine learning model that will generalize to new unseen data.", help='User prompt for the agent')
    parser.add_argument('--tags', nargs='*', default=['ablation_study'], help='Additional W&B tags')
    parser.add_argument('--workspace-dir', type=Path, default=Path('../workspace').resolve())
    parser.add_argument('--prepared-datasets-dir', type=Path, default=Path('../repository/prepared_datasets').resolve())
    parser.add_argument('--agent-datasets-dir', type=Path, default=Path('../workspace/datasets').resolve())
    parser.add_argument('--repetitions', type=int, default=1, help='Number of repetitions for each ablation setting')
    parser.add_argument('--timeout', type=int, default=60*60*24, help='Timeout in seconds per experiment (default: 24 hours)')
    return parser.parse_args()

async def main():
    args = parse_args()
    dotenv.load_dotenv()

    provider = get_provider_from_string(args.provider)

    for model in args.models:
        for dataset_name in args.datasets:
            for ablation in ABLATION_CONFIGS:
                for repetition in range(args.repetitions):
                    print(f"\n{'='*60}")
                    print(f"Starting ablation: {ablation['name']}")
                    print(f"Model: {model}, Dataset: {dataset_name}")
                    print(f"Repetition: {repetition + 1}/{args.repetitions}")
                    print(f"{'='*60}\n")

                    run_tags = args.tags + [
                        f"ablation:{ablation['name']}",
                        f"repetition:{repetition + 1}"
                    ]

                    try:
                        await asyncio.wait_for(
                            run_experiment(
                                model=model,
                                dataset_name=dataset_name,
                                val_metric=args.val_metric,
                                prepared_datasets_dir=args.prepared_datasets_dir,
                                agent_datasets_dir=args.agent_datasets_dir,
                                workspace_dir=args.workspace_dir,
                                user_prompt=args.user_prompt,
                                iterations=args.iterations,
                                tags=run_tags,
                                no_root_privileges=False,
                                provider=provider,
                                steps_to_skip=ablation['steps_to_skip']
                            ),
                            timeout=args.timeout
                        )
                    except asyncio.TimeoutError:
                        print(f"\n TIMEOUT after {args.timeout}s")
                        print(f"Continuing to next experiment...\n")

if __name__ == "__main__":
    asyncio.run(main())