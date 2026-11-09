from __future__ import annotations

import sys
from argparse import Namespace

from agentomics.agents.data_collection import DataRequest, collect_data
from agentomics.cli.collect_data import build_parser
from agentomics.runtime.filesystem import restore_host_ownership
from agentomics.utils.config import Config
from agentomics.utils.providers.codex_auth import prepare_codex_credentials
from agentomics.utils.providers.provider import Provider, get_provider_and_api_key


def run_collect_data_workflow(arguments: Namespace) -> int:
    request = DataRequest.model_validate_json(arguments.request.read_text())
    prepare_codex_credentials()
    api_key, provider_name = get_provider_and_api_key(arguments.provider)
    provider = Provider.create_provider(provider_name, api_key)
    model = provider.create_model_from_settings(
        arguments.model, Config.DEFAULT_USE_PROXY, Config.DEFAULT_LLM_RESPONSE_TIMEOUT,
    )
    try:
        return collect_data(request, arguments.collected_data_dir, model)
    finally:
        restore_host_ownership(arguments.collected_data_dir)


def main() -> int:
    return run_collect_data_workflow(build_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
