from utils.foundation_models_utils import format_foundation_model_catalog, get_foundation_model_family_info
from pydantic_ai import Tool

def create_foundation_models_info_tool(foundation_model_to_desc, max_retries):
    def _get_foundation_models_info(family: str):
        """
        This tool lists foundation models available in this system and ready to be used.
        Returns brief infos of the model families, along with the different model sizes.
        If the family argument is specified, returns detailed info, code snippets, available models, and README of a specific family.

        Args:
            family: The foundation model family of interest. If 'All' is passed, brief info about all available families is returned.
        """
        if family.strip().lower() == 'all':
            return format_foundation_model_catalog(foundation_model_to_desc)
        return get_foundation_model_family_info(family)

    foundation_models_info_tool = Tool(
        function=_get_foundation_models_info,
        takes_ctx=False,
        max_retries=max_retries,
        name="get_foundation_models_info",
        sequential=True,
    )

    return foundation_models_info_tool