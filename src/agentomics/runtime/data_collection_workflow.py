from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from agentomics.agents.data_collection.collector import DataCollectionStep
from agentomics.agents.data_collection.spec import DataCollectionSpec
from agentomics.cli.run import CONTAINER_CODEX_DIRECTORY
from agentomics.runtime.collected_data_directory import (
    initialize_collected_data_directory,
    validate_collected_data_directory,
)
from agentomics.runtime.filesystem import restore_host_ownership
from agentomics.utils.data_module_config import DataModuleConfig
from agentomics.utils.providers.codex_auth import prepare_codex_credentials
from agentomics.utils.providers.provider import Provider, get_provider_and_api_key


def run_data_collection_workflow(
    spec_path: Path,
    collected_data_dir: Path,
    provider_name: str | None,
    collector_model_name: str,
    reviewer_model_name: str,
) -> None:
    try:
        spec = DataCollectionSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
        config = DataModuleConfig(
            collected_data_dir=collected_data_dir,
            collector_model_name=collector_model_name,
            reviewer_model_name=reviewer_model_name,
            max_download_gb=spec.max_download_gb,
        )
        prepare_codex_credentials(CONTAINER_CODEX_DIRECTORY)
        api_key, provider_name = get_provider_and_api_key(provider_name)
        provider = Provider.create_provider(provider_name, api_key)
        collector_model = provider.create_model(config.collector_model_name, config.use_proxy, config.llm_response_timeout)
        reviewer_model = provider.create_model(config.reviewer_model_name, config.use_proxy, config.llm_response_timeout)
        initialize_collected_data_directory(config)
        validate_collected_data_directory(config)
        asyncio.run(DataCollectionStep(config, spec, collector_model, reviewer_model).run())
    finally:
        restore_host_ownership(collected_data_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect public data for a machine-learning task.")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--collected-data-dir", type=Path, required=True)
    parser.add_argument("--provider")
    parser.add_argument("--collector-model", required=True)
    parser.add_argument("--reviewer-model", required=True)
    arguments = parser.parse_args()
    run_data_collection_workflow(
        arguments.spec,
        arguments.collected_data_dir,
        arguments.provider,
        arguments.collector_model,
        arguments.reviewer_model,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
