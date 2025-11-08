from utils.foundation_models_utils import format_foundation_model_catalog
from pydantic_ai import Tool

def create_foundation_models_info_tool(foundation_model_to_desc, max_retries):
    def _get_foundation_models_info():
        """
        Tool for retrieving information about the foundation models available in this system and ready to be used.
        Returns brief infos of the model families, along with the different model sizes and path to complete documentation which includes code snippets.
        """
        return format_foundation_model_catalog(foundation_model_to_desc)

    foundation_models_info_tool = Tool(
        function=_get_foundation_models_info,
        takes_ctx=False,
        max_retries=max_retries,
        name="get_foundation_models_info"
    )

    return foundation_models_info_tool